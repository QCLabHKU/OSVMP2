import os
import ctypes
import gc
import h5py
import numpy as np
import cupy
import cupyx
#from opt_einsum import contract as opt_contract
#opt_contract.backend = 'cupy'
from pyscf import gto, df, lib
from pyscf.scf import _vhf
from gpu4pyscf.scf.int4c2e import BasisProdCache, libgvhf, libgint
from gpu4pyscf.df.int3c2e import make_fake_mol
from gpu4pyscf.lib.cupy_helper import (block_c2s_diag, cart2sph, contract, get_avail_mem, 
                                       libcupy_helper)
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.mole import cart2sph_by_l
from osvmp2.__config__ import ngpu
from osvmp2.loc.loc_addons import slice_fit, get_fit_domain, get_bfit_domain
from osvmp2.int_2c2e import get_j2c_low
from osvmp2.gpu.cuda_utils import (avail_gpu_mem, ave_gpu_memory, equisum_partition, get_cupy_buffers, dgemm_cupy)
from osvmp2.pbc.gpu.gamma_int_3c2e_cuda import get_gamma_feri_cuda
from osvmp2.lib import linalgUtilsCuda, int3c2eCuda
from osvmp2.osvutil import *
from osvmp2.mpi_addons import *

_num_devices = cupy.cuda.runtime.getDeviceCount()

from mpi4py import MPI


#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
#inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//nrank_shm
inode = irank // nrank_shm

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)
ngpu_shm = ngpu // nnode
nrank_per_gpu = nrank_shm // ngpu_shm
igpu = irank // nrank_per_gpu
igpu_shm = irank_shm // nrank_per_gpu
irank_gpu = irank % nrank_per_gpu
cupy.cuda.runtime.setDevice(igpu_shm)

# Set up process pairs for double buffering
ipair_rank = irank // 2
irank_pair = irank % 2
comm_pair = comm.Split(color=ipair_rank, key=irank)

LMAX_ON_GPU = 8
FREE_CUPY_CACHE = True
STACK_SIZE_PER_THREAD = 8192 * 4
BLKSIZE = 128
NROOT_ON_GPU = 7

def basis_seg_contraction(mol, allow_replica=False):
    '''transform generally contracted basis to segment contracted basis
    Kwargs:
        allow_replica:
            transform the generally contracted basis to replicated
            segment-contracted basis
    '''
    bas_templates = {}
    _bas = []
    _env = mol._env.copy()

    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        key = tuple(mol._bas[ib0:ib1,gto.PTR_EXP])
        if key in bas_templates:
            bas_of_ia = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:,gto.ATOM_OF] = ia
        else:
            # Generate the template for decontracted basis
            bas_of_ia = []
            for shell in mol._bas[ib0:ib1]:
                l = shell[gto.ANG_OF]
                nctr = shell[gto.NCTR_OF]
                if nctr == 1:
                    bas_of_ia.append(shell)
                    continue

                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[gto.NPRIM_OF]
                pcoeff = shell[gto.PTR_COEFF]
                if allow_replica:
                    bs = np.repeat(shell[np.newaxis], nctr, axis=0)
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr, nprim)
                    bas_of_ia.append(bs)
                else:
                    pexp = shell[gto.PTR_EXP]
                    exps = _env[pexp:pexp+nprim]
                    norm = gto.gto_norm(l, exps)
                    # remove normalization from contraction coefficients
                    _env[pcoeff:pcoeff+nprim] = norm
                    bs = np.repeat(shell[np.newaxis], nprim, axis=0)
                    bs[:,gto.NPRIM_OF] = 1
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_EXP] = np.arange(pexp, pexp+nprim)
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim)
                    bas_of_ia.append(bs)

            bas_of_ia = np.vstack(bas_of_ia)
            bas_templates[key] = bas_of_ia
        _bas.append(bas_of_ia)

    pmol = mol.copy()
    pmol.output = mol.output
    pmol.verbose = mol.verbose
    pmol.stdout = mol.stdout
    pmol.cart = True #mol.cart
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env
    return pmol

def get_shells_ranks(uniq_l_ctr, l_ctr_counts, ranks=nrank, weight_nprim=False):
    full_l_ctr = np.repeat(uniq_l_ctr, l_ctr_counts, axis=0)
    full_ls = full_l_ctr[:, 0]
    nfs_ls = 2 * full_ls + 1

    assert len(nfs_ls) >= ranks, f"Too many cores: there are {len(nfs_ls)} shells and {ranks} cores"

    if weight_nprim:
        nfs_ls *= full_l_ctr[:, 1]
    nfs_gpus = equisum_partition(nfs_ls, ranks)

    job_offsets = cum_offset([len(nfs) for nfs in nfs_gpus])
    
    uniq_l_ctr_ranks = []
    l_ctr_counts_ranks = []
    for rank_i in range(ranks):
        idx0, idx1 = job_offsets[rank_i:rank_i+2]
        full_ls_rank = full_l_ctr[idx0:idx1]

        uniq_l_ctr_rank, l_ctr_counts_rank = np.unique(full_ls_rank, 
                                                            return_counts=True, 
                                                            axis=0)
        uniq_l_ctr_ranks.append(uniq_l_ctr_rank)
        l_ctr_counts_ranks.append(l_ctr_counts_rank)

    return np.vstack(uniq_l_ctr_ranks), np.concatenate(l_ctr_counts_ranks)


class VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, auxmol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        self.mol = mol              # original mol
        self.auxmol = auxmol        # original auxiliary mol
        self.is_cart = self.mol.cart
        self.is_aux_cart = self.auxmol.cart

        self._sorted_mol = None     # sorted mol
        self._sorted_auxmol = None  # sorted auxilary mol

        self._ao_idx = None
        self._aux_ao_idx = None

        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

        self.cart_ao_loc = []
        self.cart_aux_loc = []
        self.sph_ao_loc = []
        self.sph_aux_loc = []

        self.angular = None
        self.aux_angular = None

        self.cp_idx = None
        self.cp_jdx = None

        self.log_qs = None
        self.aux_log_qs = None

        # pre-generate variables
        self.initialize()

    init_cvhf_direct = _vhf.VHFOpt.init_cvhf_direct

    def initialize(self):
        _mol = self.mol
        _auxmol = self.auxmol

        mol = basis_seg_contraction(_mol,allow_replica=True)
        auxmol = basis_seg_contraction(_auxmol, allow_replica=True)

        log = logger.new_logger(_mol, _mol.verbose)

        _sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(mol, log=log)
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_counts = l_ctr_counts
        self.sorted_ao_loc = _sorted_mol.ao_loc_nr(cart=_mol.cart)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.init_l_ctr_offsets = l_ctr_offsets

        uniq_l_ctr_ranks, l_ctr_counts_ranks = get_shells_ranks(uniq_l_ctr, l_ctr_counts, nrank)
        self.uniq_l_ctr_ranks = uniq_l_ctr_ranks
        self.l_ctr_counts_ranks = l_ctr_counts_ranks

        # sort fake mol
        fake_mol = make_fake_mol()
        _, _, self.fake_uniq_l_ctr, fake_l_ctr_counts = sort_mol(fake_mol, log=log)
        self.fake_l_ctr_offsets = np.append(0, np.cumsum(fake_l_ctr_counts)) + l_ctr_offsets[-1]

        # sort auxiliary mol
        _sorted_auxmol, sorted_aux_idx, self.init_aux_uniq_l_ctr, self.init_aux_l_ctr_counts = sort_mol(auxmol, log=log)

        aux_ls = self.init_aux_uniq_l_ctr[:, 0]
        self.init_aux_ncaos = self.init_aux_l_ctr_counts * (aux_ls + 1) * (aux_ls + 2) // 2

        self._sorted_mol = _sorted_mol
        self._sorted_auxmol = _sorted_auxmol

        _tot_mol = _sorted_mol + fake_mol + _sorted_auxmol
        _tot_mol.cart = True
        self._tot_mol = _tot_mol

        total_ao_loc = _tot_mol.ao_loc_nr(cart=True)
        self.c_total_ao_loc = total_ao_loc.ctypes.data_as(ctypes.c_void_p)
        self.c_total_atm = _tot_mol._atm.ctypes.data_as(ctypes.c_void_p)
        self.c_total_natm = ctypes.c_int(_tot_mol.natm)
        self.c_total_bas = _tot_mol._bas.ctypes.data_as(ctypes.c_void_p)
        self.c_total_nbas = ctypes.c_int(_tot_mol.nbas)
        self.c_total_env = _tot_mol._env.ctypes.data_as(ctypes.c_void_p)


        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, _sorted_mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        
        self.int_q_cond = self.get_q_cond()
        #cput1 = log.timer_debug1('Initialize q_cond', *cput0)

        self.sorted_cart_ao_loc = _sorted_mol.ao_loc_nr(cart=True)
        self.sorted_sph_ao_loc = _sorted_mol.ao_loc_nr(cart=False)

        self.angular = uniq_l_ctr[:, 0] 
        self.angular_shell = np.repeat(self.angular, l_ctr_counts)
        
        # Sorted AO indices
        ao_loc = mol.ao_loc_nr(cart=_mol.cart)
        unsort_ao_idx = np.arange(_mol.nao)
        ao_idx = np.array_split(unsort_ao_idx, ao_loc[1:-1])
        self._ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        self.unsorted_ao_ids = np.empty(_mol.nao, dtype=np.int32)
        self.unsorted_ao_ids[self._ao_idx] = unsort_ao_idx

        #cput1 = log.timer_debug1('AO indices', *cput1)

        self.sorted_cart_aux_loc = _sorted_auxmol.ao_loc_nr(cart=True)
        self.sorted_sph_aux_loc = _sorted_auxmol.ao_loc_nr(cart=False)

        aux_loc = _auxmol.ao_loc_nr(cart=_auxmol.cart)
        unsort_aux_idx = np.arange(_auxmol.nao)
        ao_idx = np.array_split(unsort_aux_idx, aux_loc[1:-1])
        self._aux_ao_idx = np.hstack([ao_idx[i] for i in sorted_aux_idx])    
        self.unsorted_aux_ids = np.empty(_auxmol.nao, dtype=np.int32)
        self.unsorted_aux_ids[self._aux_ao_idx] = unsort_aux_idx
        #cput1 = log.timer_debug1('Aux AO indices', *cput1)

        
    
    def free_bpcache(self):
        for n, bpcache in self._bpcache.items():
            libgvhf.GINTdel_basis_prod(ctypes.byref(bpcache))

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        for n, bpcache in self._bpcache.items():
            libgvhf.GINTdel_basis_prod(ctypes.byref(bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

    def build(self, cutoff=1e-14, group_size_aoi=None, group_size_aoj=None,
              group_size_aux=None, slice_aoi=False, slice_aoj=False, 
              diag_block_with_triu=False, aosym=False):
        '''
        int3c2e is based on int2e with (ao,ao|aux,1)
        a tot_mol is created with concatenating [mol, fake_mol, aux_mol]
        we will pair (ao,ao) and (aux,1) separately.
        '''

        if nrank > 1:
            assert (slice_aoi ^ slice_aoj), "Either i or j AOs must be sliced for multiple GPUs"
        
        log = logger.new_logger(self.mol, self.mol.verbose)
        cput0 = log.init_timer()
        #_sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(mol, log=log)

        uniq_l_ctr = self.uniq_l_ctr
        l_ctr_counts = self.l_ctr_counts

        uniq_li_ctr = self.uniq_l_ctr_ranks if slice_aoi else uniq_l_ctr
        li_ctr_counts = self.l_ctr_counts_ranks if slice_aoi else l_ctr_counts

        uniq_lj_ctr = self.uniq_l_ctr_ranks if slice_aoj else uniq_l_ctr
        lj_ctr_counts = self.l_ctr_counts_ranks if slice_aoj else l_ctr_counts

        if group_size_aoi is not None :
            uniq_li_ctr, li_ctr_counts = _split_l_ctr_groups(uniq_li_ctr, li_ctr_counts, group_size_aoi)

        if group_size_aoj is not None :
            uniq_lj_ctr, lj_ctr_counts = _split_l_ctr_groups(uniq_lj_ctr, lj_ctr_counts, group_size_aoj)

        self.nctr = len(uniq_l_ctr)
   
        aux_uniq_l_ctr = self.init_aux_uniq_l_ctr
        aux_l_ctr_counts = self.init_aux_l_ctr_counts

        if group_size_aux is not None:
            aux_uniq_l_ctr, aux_l_ctr_counts = _split_l_ctr_groups(aux_uniq_l_ctr, aux_l_ctr_counts, group_size_aux)
        
        self.aux_l_ctr_counts = aux_l_ctr_counts
        self.direct_scf_tol = cutoff

        # TODO:is it more accurate to filter with overlap_cond (or exp_cond)?
        l_ctr_offsets = self.init_l_ctr_offsets
        li_ctr_offsets = np.append(0, np.cumsum(li_ctr_counts))
        lj_ctr_offsets = np.append(0, np.cumsum(lj_ctr_counts))

        log_qs, pair2bra, pair2ket = get_pairing(
            li_ctr_offsets, lj_ctr_offsets, self.int_q_cond,
            diag_block_with_triu=diag_block_with_triu, aosym=aosym)
        self.log_qs = log_qs.copy()

        # contraction coefficient for ao basis
        self.cart_ao_loc_si = np.asarray([self.sorted_cart_ao_loc[cp] for cp in li_ctr_offsets])
        self.sph_ao_loc_si = np.asarray([self.sorted_sph_ao_loc[cp] for cp in li_ctr_offsets])
        self.cart_ao_loc_sj = np.asarray([self.sorted_cart_ao_loc[cp] for cp in lj_ctr_offsets])
        self.sph_ao_loc_sj = np.asarray([self.sorted_sph_ao_loc[cp] for cp in lj_ctr_offsets])

        self.angular_si = [self.angular_shell[s] for s in li_ctr_offsets[:-1]]
        self.angular_sj = [self.angular_shell[s] for s in lj_ctr_offsets[:-1]]


        # pairing auxiliary basis with fake basis set
        fake_l_ctr_offsets = self.fake_l_ctr_offsets 
        aux_l_ctr_offsets = np.append(0, np.cumsum(aux_l_ctr_counts))

        # contraction coefficient for auxiliary basis
        self.cart_aux_loc = np.asarray([self.sorted_cart_aux_loc[cp] for cp in aux_l_ctr_offsets])
        self.sph_aux_loc = np.asarray([self.sorted_sph_aux_loc[cp] for cp in aux_l_ctr_offsets])
        self.aux_angular = [l[0] for l in aux_uniq_l_ctr]

        #ao_loc = _sorted_mol.ao_loc_nr(cart=_mol.cart)
        self.ao_pairs_row, self.ao_pairs_col = get_ao_pairs(pair2bra, pair2ket, self.sorted_ao_loc)
        cderi_row = np.hstack(self.ao_pairs_row)
        cderi_col = np.hstack(self.ao_pairs_col)
        self.cderi_row = cderi_row
        self.cderi_col = cderi_col
        self.cderi_diag = np.argwhere(cderi_row == cderi_col)[:,0]
        #cput1 = log.timer_debug1('Get AO pairs', *cput1)

        aux_pair2bra = []
        aux_pair2ket = []
        aux_log_qs = []
        aux_l_ctr_offsets += fake_l_ctr_offsets[-1]
        for p0, p1 in zip(aux_l_ctr_offsets[:-1], aux_l_ctr_offsets[1:]):
            aux_pair2bra.append(np.arange(p0,p1,dtype=np.int32))
            aux_pair2ket.append(fake_l_ctr_offsets[0] * np.ones(p1-p0, dtype=np.int32))
            aux_log_qs.append(np.ones(p1-p0, dtype=np.int32))

        self.aux_log_qs = aux_log_qs.copy()
        pair2bra += aux_pair2bra
        pair2ket += aux_pair2ket

        self.aux_pair2bra = aux_pair2bra
        self.aux_pair2ket = aux_pair2ket

        uniq_l_ctr = np.concatenate([uniq_l_ctr, self.fake_uniq_l_ctr, aux_uniq_l_ctr])
        l_ctr_offsets = np.concatenate([
            l_ctr_offsets,
            fake_l_ctr_offsets[1:],
            aux_l_ctr_offsets[1:]])

        self.pair2bra = pair2bra
        self.pair2ket = pair2ket
        self.l_ctr_offsets = l_ctr_offsets
        bas_pair2shls = np.hstack(pair2bra + pair2ket).astype(np.int32).reshape(2,-1)
        bas_pairs_locs = np.append(0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        log_qs = log_qs + aux_log_qs
        #ao_loc = _tot_mol.ao_loc_nr(cart=True)
        ncptype = len(log_qs)

        self._bpcache = {}
        #for n in range(_num_devices):
        with cupy.cuda.Device(igpu_shm):
            bpcache = ctypes.POINTER(BasisProdCache)()
            
            scale_shellpair_diag = 1.
            libgint.GINTinit_basis_prod(
                ctypes.byref(bpcache), ctypes.c_double(scale_shellpair_diag),
                self.c_total_ao_loc,
                bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
                self.c_total_atm, self.c_total_natm,
                self.c_total_bas, self.c_total_nbas,
                self.c_total_env)
            self._bpcache[igpu_shm] = bpcache

        #cput1 = log.timer_debug1('Initialize GPU cache', *cput1)
        self.bas_pairs_locs = bas_pairs_locs
        ncptype = len(self.log_qs)
        self.aosym = aosym
        if aosym:
            self.cp_idx, self.cp_jdx = np.tril_indices(ncptype)
        else:
            #nl = int(round(np.sqrt(ncptype)))
            npi = len(li_ctr_offsets) - 1
            npj = len(lj_ctr_offsets) - 1
            self.cp_idx, self.cp_jdx = np.unravel_index(np.arange(ncptype), (npi, npj))

        if self.is_cart:
            self.ao_loc_si = self.cart_ao_loc_si
            self.ao_loc_sj = self.cart_ao_loc_sj
        else:
            self.ao_loc_si = self.sph_ao_loc_si
            self.ao_loc_sj = self.sph_ao_loc_sj
        if self.is_aux_cart:
            self.aux_ao_loc = self.cart_aux_loc
        else:
            self.aux_ao_loc = self.sph_aux_loc

        self.c2s_coeff = {}
        uniq_ls = np.unique(np.concatenate((self.angular, self.init_aux_uniq_l_ctr[:, 0])))
        for l in uniq_ls:
            self.c2s_coeff[l] = cupy.asarray(gto.mole.cart2sph(l, normalized='sp'))

        #self._sorted_mol = _sorted_mol
        #self._sorted_auxmol = _sorted_auxmol
        self.t_feri = create_timer()
        self.t_cart = create_timer()
        self.t_prep = create_timer()
    
    @property
    def bpcache(self):
        device_id = igpu_shm #cupy.cuda.Device().id
        bpcache = self._bpcache[device_id]
        return bpcache

    def sort_orbitals(self, mat, axis=[], aux_axis=[]):
        ''' Transform given axis of a matrix into sorted AO,
        and transform given auxiliary axis of a matrix into sorted auxiliary AO
        '''
        idx = self._ao_idx
        aux_idx = self._aux_ao_idx
        shape_ones = (1,) * mat.ndim
        fancy_index = []
        for dim, n in enumerate(mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            elif dim in aux_axis:
                assert n == len(aux_idx)
                indices = aux_idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        return mat[tuple(fancy_index)]

    def unsort_orbitals(self, sorted_mat, axis=[], aux_axis=[]):
        ''' Transform given axis of a matrix into sorted AO,
        and transform given auxiliary axis of a matrix into original auxiliary AO
        '''
        idx = self._ao_idx
        aux_idx = self._aux_ao_idx
        shape_ones = (1,) * sorted_mat.ndim
        fancy_index = []
        for dim, n in enumerate(sorted_mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            elif dim in aux_axis:
                assert n == len(aux_idx)
                indices = aux_idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        if isinstance(sorted_mat, cupy.ndarray):
            mat = cupy.empty_like(sorted_mat)
        else:
            mat = np.empty_like(sorted_mat)
        mat[tuple(fancy_index)] = sorted_mat
        return mat
    
    @property
    def cart2sph(self):
        return block_c2s_diag(self.angular, self.l_ctr_counts)
    
    @property
    def aux_cart2sph(self):
        return block_c2s_diag(self.aux_angular, self.aux_l_ctr_counts)
    
    @property
    def coeff(self):
        nao = self.mol.nao
        if self.mol.cart:
            coeff = cupy.eye(nao)
            self._coeff = self.unsort_orbitals(coeff, axis=[1])
        else:
            self._coeff = self.unsort_orbitals(self.cart2sph, axis=[1])
        return self._coeff

    @property
    def aux_coeff(self):
        naux = self.auxmol.nao
        if self.auxmol.cart:
            coeff = cupy.eye(naux)
            self._aux_coeff = self.unsort_orbitals(coeff, aux_axis=[1])
        else:
            self._aux_coeff = self.unsort_orbitals(self.aux_cart2sph, aux_axis=[1])
        return self._aux_coeff


def sort_mol(mol0, cart=True, log=None):
    '''
    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    '''
    if log is None:
        log = logger.new_logger(mol0, mol0.verbose)
    mol = mol0.copy(deep=True)
    l_ctrs = mol._bas[:,[gto.ANG_OF, gto.NPRIM_OF]]
    uniq_l_ctr, _, inv_idx, l_ctr_counts = np.unique(
        l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)

    if mol.verbose >= logger.DEBUG:
        log.debug1('Number of shells for each [l, nctr] group')
        for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
            log.debug('    %s :%s', l_ctr, n)

    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    # Sort basis inplace
    mol._bas = mol._bas[sorted_idx]
    return mol, sorted_idx, uniq_l_ctr, l_ctr_counts

def get_pairing(p_offsets, q_offsets, q_cond,
                cutoff=1e-14, diag_block_with_triu=True, aosym=True):
    '''
    pair shells and return pairing indices
    '''
    log_qs = []
    pair2bra = []
    pair2ket = []

    for p0, p1 in zip(p_offsets[:-1], p_offsets[1:]):
        for q0, q1 in zip(q_offsets[:-1], q_offsets[1:]):
            if aosym and q0 < p0 or not aosym:
                q_sub = q_cond[p0:p1,q0:q1].ravel()
                mask = q_sub > cutoff
                ishs, jshs = np.indices((p1-p0,q1-q0))
                ishs = ishs.ravel()[mask]
                jshs = jshs.ravel()[mask]
                ishs += p0
                jshs += q0
                pair2bra.append(ishs)
                pair2ket.append(jshs)
                log_q = np.log(q_sub[mask])
                log_q[log_q > 0] = 0
                log_qs.append(log_q)
            elif aosym and p0 == q0 and p1 == q1:
                q_sub = q_cond[p0:p1,p0:p1].ravel()
                ishs, jshs = np.indices((p1-p0, p1-p0))
                ishs = ishs.ravel()
                jshs = jshs.ravel()
                mask = q_sub > cutoff
                if not diag_block_with_triu:
                    # Drop the shell pairs in the upper triangle for diagonal blocks
                    mask &= ishs >= jshs

                ishs = ishs[mask]
                jshs = jshs[mask]
                ishs += p0
                jshs += p0
                if len(ishs) == 0 and len(jshs) == 0:continue

                pair2bra.append(ishs)
                pair2ket.append(jshs)

                log_q = np.log(q_sub[mask])
                log_q[log_q > 0] = 0
                log_qs.append(log_q)
    return log_qs, pair2bra, pair2ket

def _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size):
    '''
    Splits l_ctr patterns into small groups with group_size the maximum
    number of AOs in each group
    '''
    '''l = uniq_l_ctr[:,0]
    nf = l * (l + 1) // 2'''
    _l_ctrs = []
    _l_ctr_counts = []
    for l_ctr, counts in zip(uniq_l_ctr, l_ctr_counts):
        l = l_ctr[0]
        nf = (l + 1) * (l + 2) // 2
        aligned_size = (group_size // nf // 1) * 1
        max_shells = max(aligned_size, 2)
        assert max_shells * nf <= group_size
        if l > LMAX_ON_GPU or counts <= max_shells:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(counts)
            continue

        nsubs, rests = counts.__divmod__(max_shells)
        _l_ctrs.extend([l_ctr] * nsubs)
        _l_ctr_counts.extend([max_shells] * nsubs)
        if rests > 0:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(rests)
    uniq_l_ctr = np.vstack(_l_ctrs)
    l_ctr_counts = np.hstack(_l_ctr_counts)
    return uniq_l_ctr, l_ctr_counts

def get_ao_pairs(pair2bra, pair2ket, ao_loc):
    """
    Compute the AO-pairs for the given pair2bra and pair2ket
    """
    bra_ctr = []
    ket_ctr = []
    for bra_shl, ket_shl in zip(pair2bra, pair2ket):
        if len(bra_shl) == 0 or len(ket_shl) == 0:
            bra_ctr.append(np.array([], dtype=np.int64))
            ket_ctr.append(np.array([], dtype=np.int64))
            continue

        i = bra_shl[0]
        j = ket_shl[0]
        indices = np.mgrid[:ao_loc[i+1]-ao_loc[i], :ao_loc[j+1]-ao_loc[j]]
        ao_bra = indices[0].reshape(-1,1) + ao_loc[bra_shl]
        ao_ket = indices[1].reshape(-1,1) + ao_loc[ket_shl]
        mask = ao_bra >= ao_ket
        bra_ctr.append(ao_bra[mask])
        ket_ctr.append(ao_ket[mask])
    return bra_ctr, ket_ctr

def get_slice_gpu(intopt, ranks, slice_aoj=True):
    
    n_cp_jdx = len(intopt.ao_loc_sj) - 1

    if slice_aoj:
        aop_segs = {}
        for cp_ij_id in range(len(intopt.log_qs)):
            cpj = intopt.cp_jdx[cp_ij_id]
            be0, be1 = intopt.ao_loc_sj[cpj:cpj+2]

            if cp_ij_id < n_cp_jdx:
                aop_segs[(be0, be1)] = [cp_ij_id]
            else:
                aop_segs[(be0, be1)].append(cp_ij_id)
    else:

        aop_segs = {}
        for cp_ij_id in range(len(intopt.log_qs)):
            cpi = intopt.cp_idx[cp_ij_id]
            be0, be1 = intopt.ao_loc_si[cpi:cpi+2]

            #if cp_ij_id < n_cp_idx:
            if cp_ij_id % n_cp_jdx == 0:
                aop_segs[(be0, be1)] = [cp_ij_id]
            else:
                aop_segs[(be0, be1)].append(cp_ij_id)
    
    if ranks == 1:
        return [[[ikey, aop_segs[ikey]] for ikey in aop_segs.keys()]]

    aop_keys = list(aop_segs.keys())
    be_segs = np.asarray([ikey for ikey in aop_segs.keys()])
    nbe_segs = be_segs[:, 1] - be_segs[:, 0]
    nbe_gpus = equisum_partition(nbe_segs, ranks)

    slice_gpus = []

    idx0 = 0
    for nbe in nbe_gpus:
        idx1 = idx0 + len(nbe)

        slice_gpu = []
        for be_idx in range(idx0, idx1):
            ikey = aop_keys[be_idx]
            slice_gpu.append([ikey, aop_segs[ikey]])
        slice_gpus.append(slice_gpu)

        idx0 = idx1

    return slice_gpus

def cart2sph(intopt, t, axis=0, ang=1, out=None, stream=None):
    '''
    transform 'axis' of a tensor from cartesian basis into spherical basis
    '''
    
    if(ang <= 1):
        if(out is not None): out[:] = t
        return t
    size = list(t.shape)
    c2s = intopt.c2s_coeff[ang] #cart2sph_by_l(ang)
    if(not t.flags['C_CONTIGUOUS']): t = cupy.asarray(t, order='C')
    li_size = c2s.shape
    nli = size[axis] // li_size[0]
    i0 = max(1, np.prod(size[:axis]))
    i3 = max(1, np.prod(size[axis+1:]))
    out_shape = size[:axis] + [nli*li_size[1]] + size[axis+1:]

    t_cart = t.reshape([i0*nli, li_size[0], i3])
    if(out is not None):
        out = out.reshape([i0*nli, li_size[1], i3])
    else:
        out = cupy.empty(out_shape)
    count = i0*nli*i3
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    
    err = libcupy_helper.cart2sph(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(t_cart.data.ptr, ctypes.c_void_p),
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.c_int(i3),
        ctypes.c_int(count),
        ctypes.c_int(ang)
    )

    if err != 0:
        raise RuntimeError('failed in cart2sph kernel')
    return out.reshape(out_shape)

def get_int3c2e_slice(intopt, cp_ij_id, cp_aux_id, cart=False, 
                      aosym=None, out=None, out_int=None, omega=None, stream=None, 
                      c2s_coeff=None):
    '''
    Generate one int3c2e block for given ij, k
    '''
    if stream is None:stream = cupy.cuda.get_current_stream()
    if omega is None:omega = 0.0
    
    t0 = get_current_time()
    nao_cart = intopt._sorted_mol.nao
    naux_cart = intopt._sorted_auxmol.nao
    norb_cart = nao_cart + naux_cart + 1

    cpi = intopt.cp_idx[cp_ij_id]
    cpj = intopt.cp_jdx[cp_ij_id]
    cp_kl_id = cp_aux_id + len(intopt.log_qs)

    log_q_ij = intopt.log_qs[cp_ij_id]
    log_q_kl = intopt.aux_log_qs[cp_aux_id]

    nbins = 1
    bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
    bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)
    
    #cart_ao_loc = intopt.cart_ao_loc
    cart_aux_loc = intopt.cart_aux_loc
    '''i0, i1 = cart_ao_loc[cpi], cart_ao_loc[cpi+1]
    j0, j1 = cart_ao_loc[cpj], cart_ao_loc[cpj+1]'''
    i0, i1 = intopt.cart_ao_loc_si[cpi:cpi+2]
    j0, j1 = intopt.cart_ao_loc_sj[cpj:cpj+2]
    k0, k1 = cart_aux_loc[cp_aux_id:cp_aux_id+2]

    ni = i1 - i0
    nj = j1 - j0
    nk = k1 - k0

    si0, si1 = intopt.sph_ao_loc_si[cpi:cpi+2]
    sj0, sj1 = intopt.sph_ao_loc_sj[cpj:cpj+2]
    nsi = si1 - si0
    nsj = sj1 - sj0

    li = intopt.angular_si[cpi]
    lj = intopt.angular_sj[cpj]
    lk = intopt.aux_angular[cp_aux_id]

    ao_offsets = np.array([i0,j0,nao_cart+1+k0,nao_cart], dtype=np.int32)
    '''
    # if possible, write the data into the given allocated space
    # otherwise, need a temporary space for cart2sph
    '''
    if out is None: # or (lk > 1 and not intopt.auxmol.cart):
        int3c_blk = cupy.zeros([nk,nj,ni])
        strides = np.array([1, ni, ni*nj, 1], dtype=np.int32)
    else:
        if out_int is None or li > 1 or lj > 1 or lk > 1:
            int3c_blk = out[:nk*nj*ni].reshape((nk,nj,ni))
        else:
            int3c_blk = out_int.reshape((nk,nj,ni))
        int3c_blk.fill(0.0)
        strides = np.array([1, ni, ni*nj, 1], dtype=np.int32)
    #accumulate_time(intopt.t_prep, t0)
    #stream.synchronize()
    
    err = libgint.GINTfill_int3c2e(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        intopt.bpcache,
        ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
        ctypes.c_int(norb_cart),
        strides.ctypes.data_as(ctypes.c_void_p),
        ao_offsets.ctypes.data_as(ctypes.c_void_p),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbins),
        ctypes.c_int(cp_ij_id),
        ctypes.c_int(cp_kl_id),
        ctypes.c_double(omega))
    #accumulate_time(intopt.t_feri, t0)

    #stream.synchronize()

    if err != 0:
        print(int3c_blk.shape)
        raise RuntimeError('GINT_fill_int2e failed')
    
    # intopt.c2s_coeff
    #gto.mole.cart2sph(l, normalized='sp')

    bidx0 = nk*nj*ni
    if not intopt.mol.cart:
        if li > 1:
            if out_int is None or lj > 1 or lk > 1:
                out2 = out[bidx0:bidx0+nsi*nj*nk].reshape(nk, nj, nsi)
                bidx0 = 0
            else:
                out2 = out_int.reshape(nk, nj, nsi)
            int3c_blk = cart2sph(intopt, int3c_blk, axis=2, ang=li, out=out2)
            
        
        if lj > 1:
            if out_int is None or lk > 1:
                out1 = out[bidx0:bidx0+nsi*nsj*nk].reshape(nk, nsj, nsi)
                bidx0 = nsi*nsj*nk if bidx0 == 0 else 0
            else:
                out1 = out_int.reshape(nk, nsj, nsi)
            int3c_blk = cart2sph(intopt, int3c_blk, axis=1, ang=lj, out=out1)
            

    # move this operation to j2c?
    if lk > 1 and intopt.auxmol.cart == 0:
        sk0, sk1 = intopt.sph_aux_loc[cp_aux_id:cp_aux_id+2]
        nsk = sk1 - sk0
        if out_int is None:
            out = out[bidx0:bidx0+nsk*nsj*nsi].reshape(nsk, nsj, nsi)
        else:
            out = out_int.reshape(nsk, nsj, nsi)
        int3c_blk = cart2sph(intopt, int3c_blk, axis=0, ang=lk, out=out)
        #bidx0 = nsk*nsj*nsi if bidx0 == 0 else 0
    #accumulate_time(intopt.t_cart, t0)
    #stream.synchronize()

    return int3c_blk



def aux_e2_gpu(intopt, cp_ij_id, omega=None, out=None, buf_aux=None, feri_shape="pji", c2s_coeff=None, 
               stream=None):

    naux = intopt.auxmol.nao
    cpi = intopt.cp_idx[cp_ij_id]
    cpj = intopt.cp_jdx[cp_ij_id]
    i0, i1 = intopt.ao_loc_si[cpi:cpi+2]
    j0, j1 = intopt.ao_loc_sj[cpj:cpj+2]
    ci0, ci1 = intopt.cart_ao_loc_si[cpi:cpi+2]
    cj0, cj1 = intopt.cart_ao_loc_sj[cpj:cpj+2]

    if feri_shape == "ijp":
        shape_slice = ((i1-i0), (j1-j0), naux)
    elif feri_shape == "jip":
        shape_slice = ((j1-j0), (i1-i0), naux)
    else:
        shape_slice = (naux, (j1-j0), (i1-i0))

    if out is None:
        int3c_slice = cupy.empty(shape_slice)
    else:
        int3c_slice = out.reshape(shape_slice)
    
    if buf_aux is None:
        max_ncaux = np.max(intopt.cart_aux_loc[1:] - intopt.cart_aux_loc[:-1])
        buf_aux = cupy.empty(2*(ci1-ci0)*(cj1-cj0)*max_ncaux)
    
    cp_aux_indices = np.arange(len(intopt.aux_log_qs))
    
    if stream is None:
        stream = cupy.cuda.get_current_stream()

    for cp_kl_id in cp_aux_indices:
        k0, k1 = intopt.aux_ao_loc[cp_kl_id:cp_kl_id+2]

        if feri_shape == "pji":
            int_slice = get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, 
                                        omega=omega, stream=stream, 
                                        out=buf_aux, out_int=int3c_slice[k0:k1], 
                                        c2s_coeff=c2s_coeff)
            #int3c_slice[k0:k1] = int_slice
        else:

            int_slice = get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, 
                                        omega=omega, stream=stream, 
                                        out=buf_aux, c2s_coeff=c2s_coeff)
        
            if feri_shape == "ijp":
                int3c_slice[:, :, k0:k1] = int_slice.transpose(2, 1, 0)
            elif feri_shape == "jip":
                int3c_slice[:, :, k0:k1] = int_slice.transpose(1, 2, 0)
        
    return int3c_slice

def get_nao_range(intopt):
    uniq_l_ctr = intopt.uniq_l_ctr
    l_ctr_counts = intopt.l_ctr_counts

    ls = uniq_l_ctr[:, 0]
    nfs = (ls + 1) * (ls + 2) // 2
    min_nao = 2 * np.max(nfs)
    max_nao = np.max(nfs * l_ctr_counts)
    return min_nao, max_nao

def get_mol_feri_cuda(self, df_obj, qc_method, log, ijp_shape=True, fitting=True):
    tt = get_current_time()
    t_cal = create_timer()
    t_write = create_timer()
    t_data = create_timer()
    self.debug = False

    if fitting:
        t0 = get_current_time()
        file_j2c = f"j2c_{qc_method}.tmp"
        win_low, low_node = get_j2c_low(self, file_j2c, use_lowT=True)
        print_time(["j2c", get_elapsed_time(t0)], log)


    max_memory = get_mem_spare(self.mol, 0.9)
    max_memory_f8 = max_memory * 1e6 // 8
    gpu_memory_f8 = ave_gpu_memory(self.gpu_memory) * 1e6 // 8
    nao = self.nao
    naoaux = self.intopt.auxmol.nao

    if fitting:
        low_gpu, low_ptr = get_shared_cupy((naoaux, naoaux), numpy_array=low_node)

    feri_mem_f8 = gpu_memory_f8 - naoaux * naoaux
    feri_mem_f8 /= nrank_per_gpu
    max_ncaux_slice = np.max(self.intopt.init_aux_ncaos)
    min_nao, max_nao = get_nao_range(self.intopt)
    max_l = max(self.mol._bas[:, gto.ANG_OF])
    max_c2s = ((max_l+1)*(max_l+2)//2) / (2*max_l + 1)
    max_nao0 = min_nao #min(max_nao, max(min_nao, int(0.3*feri_mem_f8 / (2*nao*naoaux))))
    mem_left = 0.8*feri_mem_f8 - 2*max_nao0*nao*naoaux
    max_nao1 = int(mem_left / (2*(max_c2s**2)*max_nao0*max_ncaux_slice))
    if max_nao1 > max_nao:
        max_nao1 = max_nao
    elif max_nao1 < min_nao:
        max_nao1 = min_nao
    max_nao0 = min(max_nao, int(0.8 * feri_mem_f8 / (2*max_nao1*max_ncaux_slice + 2*nao*naoaux)))
    mem_scheme = 0
    self.unsort_alpha = False

    if max_nao0 < min_nao:
        # Memory management 1
        mem_scheme = 1
        self.unsort_beta = False
        fake_naux = max(naoaux, 2*(max_c2s**2)*max_ncaux_slice)
        max_nao0 = min(max_nao, max(min_nao, int(0.8*feri_mem_f8 / (max_nao*(naoaux+fake_naux)))))
        max_nao1 = min(max_nao, max(min_nao, int(0.8*feri_mem_f8 / (max_nao0*(naoaux+fake_naux)))))
        max_nao0 = min(max_nao, int(0.8 * feri_mem_f8 / (max_nao1*(naoaux+fake_naux))))

        #print(min_nao, max_nao, max_nao0, max_nao1);sys.exit()
        if max_nao0 < min_nao:
            raise MemoryError("Insufficient GPU memory!")

    self.intopt.build(1e-9, diag_block_with_triu=True, group_size_aoi=max_nao0, group_size_aoj=max_nao1, slice_aoi=True)

    shell_slice_ranks = get_slice_gpu(self.intopt, nrank, slice_aoj=False)
    shell_slice_rank = shell_slice_ranks[irank]

    if ijp_shape:
        shape_feri = (self.nao, self.nao, self.naoaux)
    else:
        shape_feri = (self.naoaux, self.nao, self.nao)

    if self.int_storage in {0, 3}:# 0: CPU Incore; 3: GPU incore 
        self.win_feri, self.feri_node = get_shared(shape_feri)#, set_zeros=True)
        if nnode > 1:
            als_off_node = []
            for rank_i, (shslice_rank) in shell_slice_ranks:
                if rank_i // nrank_shm != inode:
                    for (al0, al1), _ in shslice_rank:
                        als_off_node.append(np.arange(al0, al1))
            if len(als_off_node) > 0:
                als_off_node = np.concatenate(als_off_node)
                als_slice = get_slice(range(nrank_shm), job_list=als_off_node)[irank_shm]
                if als_slice is not None:
                    if ijp_shape:
                        self.feri_node[als_slice] = 0.0
                    else:
                        self.feri_node[:, als_slice] = 0.0
        feri_data = self.feri_node
    elif self.int_storage == 1:# Outcore
        self.file_feri = f"feri_{qc_method}.tmp"
        file_feri = h5py.File(self.file_feri, 'w', driver='mpio', comm=comm)
        feri_data = file_feri.create_dataset("feri", shape_feri, dtype=np.float64)

    if shell_slice_rank is not None:
        max_nao_gpu = np.max([al1-al0 for (al0, al1), _ in shell_slice_rank])
        max_ncal_gpu = np.max(self.intopt.cart_ao_loc_si[1:] - self.intopt.cart_ao_loc_si[:-1])
        max_ncbe_gpu = np.max(self.intopt.cart_ao_loc_sj[1:] - self.intopt.cart_ao_loc_sj[:-1])
        
        if mem_scheme == 0:
            #feri_buf_cpu = np.empty((max_nao_gpu*nao*naoaux))
            feri_buf_cpu = cupyx.empty_pinned((max_nao_gpu*nao*naoaux,))

            feri_buf0_gpu = cupy.empty((max_nao_gpu*nao*naoaux))
            feri_buf1_gpu = cupy.empty((max_nao_gpu*nao*naoaux))
            feri_aux_buf_gpu = cupy.empty(2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice)
        elif mem_scheme == 1:
            max_nsal_gpu = np.max(self.intopt.ao_loc_si[1:] - self.intopt.ao_loc_si[:-1])
            max_nsbe_gpu = np.max(self.intopt.ao_loc_sj[1:] - self.intopt.ao_loc_sj[:-1])

            #feri_buf_cpu = np.empty((max_nao_gpu*nao*naoaux))
            #feri_buf_cpu = np.empty((max_nsal_gpu*max_nsbe_gpu*naoaux))
            feri_buf_cpu = cupyx.empty_pinned((max_nsal_gpu*max_nsbe_gpu*naoaux,))

            feri_buf1_gpu = cupy.empty((max_nsal_gpu*max_nsbe_gpu*naoaux))
            '''feri_buf1_gpu = cupy.empty((max_nsal_gpu*max_nsbe_gpu*naoaux))
            feri_aux_buf_gpu = cupy.empty(2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice)'''
            buf0_size = max(max_nsal_gpu*max_nsbe_gpu*naoaux, 2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice)
            feri_buf0_gpu = feri_aux_buf_gpu = cupy.empty(buf0_size)

        unsort_aux_idx = np.empty(naoaux, dtype=np.int32)
        unsort_aux_idx[self.intopt._aux_ao_idx] = np.arange(naoaux, dtype=np.int32)
        unsort_aux_idx_gpu = cupy.asarray(unsort_aux_idx, dtype=cupy.int32)

        unsort_ao_idx = np.empty(nao, dtype=np.int32)
        unsort_ao_idx[self.intopt._ao_idx] = np.arange(nao, dtype=np.int32)
        unsort_ao_idx_gpu = cupy.asarray(unsort_ao_idx, dtype=cupy.int32)

        #print(irank, "NOW!!"); sys.exit()
        for (al0, al1), cpidx_list in shell_slice_rank:
            nao0 = al1 - al0
            #feri_gpu = cupy.empty((nao0, nao, naoaux))
            #feri_gpu = cupy.empty((naoaux, nao, nao0))
            #feri_buf_gpu = cupy.empty(nao0*max_nao1*naoaux)

            if mem_scheme == 0:
                feri0_gpu = feri_buf0_gpu[:naoaux*nao*nao0].reshape(naoaux, nao, nao0)
            else:
                '''if self.int_storage == 1:
                    if ijp_shape:
                        feri_cpu = feri_buf_cpu[:naoaux*nao*nao0].reshape(nao0, nao, naoaux)
                    else:
                        feri_cpu = feri_buf_cpu[:naoaux*nao*nao0].reshape(naoaux, nao0, nao)'''
                #feri_cpu = feri_buf_cpu[:naoaux*nao*nao0].reshape(nao, nao0, naoaux)
            
            

            t1 = get_current_time()
            for cp_ij_id in cpidx_list:
                t1 = get_current_time()

                cpj = self.intopt.cp_jdx[cp_ij_id]
                be0, be1 = self.intopt.ao_loc_sj[cpj], self.intopt.ao_loc_sj[cpj+1]
                nao1 = be1 - be0
                feri_slice_gpu = feri_buf1_gpu[:nao1*nao0*naoaux].reshape(naoaux, nao1, nao0)
                aux_e2_gpu(self.intopt, cp_ij_id, feri_shape="pji", out=feri_slice_gpu, buf_aux=feri_aux_buf_gpu)
                if mem_scheme == 0:
                    feri0_gpu[:, be0:be1] = feri_slice_gpu
                elif mem_scheme == 1:
                    feri1_gpu = feri_slice_gpu
                    feri0_gpu = feri_buf0_gpu[:naoaux*nao1*nao0].reshape(naoaux, nao1, nao0)
                    cupy.take(feri1_gpu, unsort_aux_idx_gpu, axis=0, out=feri0_gpu)

                    if fitting:
                        feri1_gpu = feri1_gpu.reshape(nao0, nao1, naoaux)
                        feri1_gpu[:] = feri0_gpu.transpose(2, 1, 0) #naoaux, nao1, nao0 -> nao0, nao1, naoaux

                        cupyx.scipy.linalg.solve_triangular(low_gpu.T, feri1_gpu.reshape(-1, naoaux).T, lower=True,
                                                            overwrite_b=True, check_finite=False)
                        if ijp_shape:
                            feri_slice_cpu = feri_buf_cpu[:naoaux*nao1*nao0].reshape(nao0, nao1, naoaux)
                            t_cal += get_elapsed_time(t1)

                            t1 = get_current_time()
                            cupy.asnumpy(feri1_gpu, out=feri_slice_cpu)
                            t_data += get_elapsed_time(t1)

                            t1 = get_current_time()
                            feri_data[al0:al1, be0:be1] = feri_slice_cpu
                            t_data += get_elapsed_time(t1)
                        else:
                            feri_slice_cpu = feri_buf_cpu[:naoaux*nao1*nao0].reshape(naoaux, nao0, nao1)
                            feri0_gpu = feri_buf0_gpu[:naoaux*nao1*nao0].reshape(naoaux, nao0, nao1)
                            feri0_gpu[:] = feri1_gpu.transpose(2, 0, 1)

                            t_cal += get_elapsed_time(t1)
                            cupy.asnumpy(feri0_gpu, out=feri_slice_cpu)
                            t_data += get_elapsed_time(t1)

                            t1 = get_current_time()
                            feri_data[:, al0:al1, be0:be1] = feri_slice_cpu
                            t_data += get_elapsed_time(t1)

                    else:
                        raise NotImplementedError
            
            
            if mem_scheme == 0:
                feri1_gpu = feri_buf1_gpu[:naoaux*nao*nao0].reshape(naoaux, nao, nao0)
                cupy.take(feri0_gpu, unsort_aux_idx_gpu, axis=0, out=feri1_gpu)
                cupy.take(feri1_gpu, unsort_ao_idx_gpu, axis=1, out=feri0_gpu)

                
                if fitting:
                    feri1_gpu = feri1_gpu.reshape(nao0, nao, naoaux)
                    feri1_gpu[:] = feri0_gpu.transpose(2, 1, 0) #naoaux, nao, nao0 -> nao0, nao, naoaux

                    cupyx.scipy.linalg.solve_triangular(low_gpu.T, feri1_gpu.reshape(-1, naoaux).T, lower=True,
                                                        overwrite_b=True, check_finite=False)

                    if ijp_shape:
                        feri_gpu = feri1_gpu
                    else:
                        feri0_gpu = feri0_gpu.reshape(naoaux, nao0, nao)
                        feri0_gpu[:] = feri1_gpu.transpose(2, 0, 1)
                        feri_gpu = feri0_gpu
                    

                else:
                    #TODO
                    if ijp_shape:
                        feri1_gpu = feri1_gpu.reshape(nao0, nao, naoaux)
                        feri1_gpu[:] = feri0_gpu.transpose(2, 1, 0) #naoaux, nao, nao0 -> nao0, nao, naoaux
                        feri_gpu = feri1_gpu
                    else:
                        feri1_gpu = feri1_gpu.reshape(naoaux, nao0, nao)
                        feri1_gpu[:] = feri0_gpu.transpose(0, 2, 1) #naoaux, nao, nao0 -> naoaux, nao0, nao
                        feri_gpu = feri0_gpu

                t_cal += get_elapsed_time(t1)
                                                                    
                #feri_gpu = cupy.ascontiguousarray(feri_gpu.transpose(1, 2, 0))
                

                t1 = get_current_time()
                if ijp_shape:
                    #feri_gpu = cupy.ascontiguousarray(feri_gpu.transpose(0, 1, 2))
                    #feri_gpu = cupy.ascontiguousarray(feri_gpu.transpose(1, 0, 2))
                    feri_cpu = feri_buf_cpu[:nao0*nao*naoaux].reshape(nao0, nao, naoaux)
                else:
                    #feri_gpu = cupy.ascontiguousarray(feri_gpu.transpose(0, 2, 1))
                    #feri_gpu = cupy.ascontiguousarray(feri_gpu.transpose(2, 1, 0))
                    feri_cpu = feri_buf_cpu[:nao0*nao*naoaux].reshape(naoaux, nao0, nao)
                cupy.asnumpy(feri_gpu, out=feri_cpu)
                t_data += get_elapsed_time(t1)

                t1 = get_current_time()
 
                #unsort_ao_idx_seg = self.intopt._ao_idx[al0:al1]
                if ijp_shape:
                    #self.feri_node[unsort_ao_idx_seg] = feri_cpu
                    feri_data[al0:al1] = feri_cpu
                else:
                    #self.feri_node[:, unsort_ao_idx_seg] = feri_cpu
                    feri_data[:, al0:al1] = feri_cpu #.transpose(2, 0, 1)
                t_write += get_elapsed_time(t1)
            elif mem_scheme == 1:
                '''t1 = get_current_time()
                # feri_cpu.shape: (nao, nao0, naoaux)
                if ijp_shape:
                    #self.feri_node[unsort_ao_idx_seg] = feri_cpu
                    feri_data[al0:al1] = feri_cpu.transpose(1, 0, 2)
                else:
                    #self.feri_node[:, unsort_ao_idx_seg] = feri_cpu.transpose(2, 0, 1)
                    feri_data[:, al0:al1] = feri_cpu.transpose(2, 1, 0) #.transpose(2, 0, 1)
                t_write += get_elapsed_time(t1)'''
        low_gpu = None

    comm.Barrier()

    
    if self.int_storage in {0, 3}:
        Acc_and_get_GA(self.feri_node)
        comm.Barrier()

        if self.int_storage == 3:
            aux_slice = get_slice(range(nrank), job_size=naoaux)[irank]
            if aux_slice is not None:
                p0 = aux_slice[0]
                p1 = aux_slice[-1] + 1
                if ijp_shape:
                    self.feri_gpu = cupy.asarray(self.feri_node[:, :, p0:p1])
                else:
                    self.feri_gpu = cupy.asarray(self.feri_node[p0:p1])
            comm_shm.Barrier()
            free_win(self.win_feri)
            self.feri_node = None

    elif self.int_storage == 1:
        file_feri.close()
    
    if fitting:
        free_win(win_low)
        close_ipc_handle(low_ptr)
    
    self.intopt.free_bpcache()
    
    time_list = [["writing", t_write], ["GPU-CPU data", t_data], 
                ['computing', t_cal], [f"{qc_method} feri", get_elapsed_time(tt)]]
    time_list = get_max_rank_time_list(time_list)
    if irank == 0:
        print_time(time_list, log)


def get_feri_cuda(self, df_obj, qc_method, log, ijp_shape=True, ovlp=None, 
                  fitting=True, with_long_range=True):
    self.unsort_alpha = True
    self.unsort_beta = True
    if self.mol.pbc:
        get_gamma_feri_cuda(self, df_obj, qc_method, log, ijp_shape, ovlp,
                            fitting=fitting, with_long_range=with_long_range)
    else:
        get_mol_feri_cuda(self, df_obj, qc_method, log, ijp_shape, fitting=fitting)



def get_ialp_direct_cuda(self, df_obj, qc_method, log, unsort_ao=True):
    
    if self.double_buffer:
        irank_cal = 0
        irank_io = 1
        is_cal_rank = irank_pair == irank_cal
        is_io_rank = irank_pair == irank_io
        istream_data = cupy.cuda.Stream(non_blocking=True)
        istream_comp = cupy.cuda.Stream(non_blocking=True)
    else:
        is_cal_rank = is_io_rank = True
        istream_data = istream_comp = cupy.cuda.Stream.null

    #cupy.get_default_memory_pool().free_all_blocks()

    recorder_feri = []
    recorder_ialp = []
    recorder_cal = []
    recorder_data = []
    recorder_write = []

    nao = self.mol.nao_nr()
    nocc = self.mol.nelectron//2
    nmo = len(self.mo_list)
    nocc_core = nocc - nmo
    occ_coeff = self.o[:, nocc_core:]
    naoaux = self.naoaux
    mol = self.mol
    shape_ialp = (nmo, self.nao, self.naoaux)
    
    file_j2c = f"j2c_{qc_method}.tmp"
    t1 = get_current_time()
    win_low, low_node = get_j2c_low(self, file_j2c, use_lowT=True)
    print_time(['Fitting int', get_elapsed_time(t1)], log)


    t1 = get_current_time()
    shm_buf_gpu, shm_ptr = get_shared_cupy(occ_coeff.size+naoaux**2)

    occ_coeff_gpu = shm_buf_gpu[:occ_coeff.size].reshape(occ_coeff.shape)
    low_gpu = shm_buf_gpu[occ_coeff.size:].reshape(naoaux, naoaux)
    if irank_gpu == 0:
        occ_coeff_gpu.set(self.intopt.sort_orbitals(occ_coeff, axis=[0]))
        low_gpu.set(low_node)

    accumulate_time(self.t_data, t1)

    comm_shm.Barrier()
    free_win(win_low)
    low_node = None

    max_cpu_memory = ave_mem_spare(mol, 0.9)
    max_nao_cpu = max_cpu_memory * 1e6 / (8 * nmo * naoaux)

    gpu_memory_f8 = 0.9 * ave_gpu_memory(max_mem=self.gpu_memory) * 1e6 // 8 - (nao*nmo + naoaux**2)
    gpu_memory_f8 /= nrank_per_gpu
    if self.double_buffer:
        gpu_memory_f8 *= 2
    max_ncaux_slice = np.max(self.intopt.init_aux_ncaos)
    min_nao, max_nao = get_nao_range(self.intopt)
    max_l = max(mol._bas[:, gto.ANG_OF])
    max_c2s = ((max_l+1)*(max_l+2)//2) / (2*max_l + 1) 
    nbuf_ialp = 3 if self.double_buffer else 2
    max_nao1 = min_nao #min(min(max_nao, max_nao_cpu), max(min_nao, int(0.3*gpu_memory_f8 / (2*nocc*naoaux))))
    mem_left = 0.9*gpu_memory_f8 - nbuf_ialp*nocc*max_nao1*naoaux
    max_nao0 = max(min_nao, int(mem_left / ((max_c2s**2)*max_nao1*(naoaux + 2*max_ncaux_slice))))
    if max_nao0 < min_nao:
        max_nao0 = min_nao
    elif max_nao0 > max_nao:
        max_nao0 = max_nao
    max_nao1 = min(max_nao, int(0.9 * gpu_memory_f8 / ((max_c2s**2)*max_nao0*(naoaux + 2*max_ncaux_slice) + nbuf_ialp*nocc*naoaux)))
    if max_nao0 < min_nao or max_nao1 < min_nao:
        print(max_nao0, max_nao1)
        raise MemoryError("Insufficient GPU memory!")

    self.intopt.build(1e-9, diag_block_with_triu=True, group_size_aoi=max_nao0, group_size_aoj=max_nao1, slice_aoj=True)
        
    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=True)
    if self.int_storage in {0, 3}:
        shell_slice_ranks = get_slice_gpu(self.intopt, nrank)
        self.win_ialp, self.ialp_node = get_shared(shape_ialp)#, set_zeros=True)
        if nnode > 1:
            als_off_node = []
            for rank_i, (shslice_rank) in shell_slice_ranks:
                if rank_i // nrank_shm != inode:
                    for (al0, al1), _ in shslice_rank:
                        als_off_node.append(np.arange(al0, al1))
            if len(als_off_node) > 0:
                als_off_node = np.concatenate(als_off_node)
                als_slice = get_slice(range(nrank_shm), job_list=als_off_node)[irank_shm]
                if als_slice is not None:
                    self.ialp_node[:, als_slice] = 0.0

        ialp_data = self.ialp_node
        use_ialp_file = False
    else:
        t0 = get_current_time()
        self.file_ialp = f"ialp_{qc_method}.tmp"
        file_ialp = h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm)
        nao_chunk = min(536870912 // naoaux, nao) # chunks must be smaller than 4GB
        chunks = (1, nao_chunk, naoaux)
        ialp_data = create_h5py_dataset(file_ialp, "ialp", shape_ialp, dtype=np.float64, 
                                         chunks=chunks)
        use_ialp_file = True
        #print_time(["creating ialp dataset", get_elapsed_time(t0)], log=log)

    if self.double_buffer:
        shell_slice_rank = get_slice_gpu(self.intopt, nrank//2)[ipair_rank]
    else:
        shell_slice_rank = get_slice_gpu(self.intopt, nrank)[irank]

    if shell_slice_rank is not None:

        ialp_buf_size = nmo*max_nao1*naoaux
        if is_cal_rank:
            aux_ratio = np.zeros((nocc, naoaux))

            unsort_aux_idx = np.empty(naoaux, dtype=np.int32)
            unsort_aux_idx[self.intopt._aux_ao_idx] = np.arange(naoaux, dtype=np.int32)
            unsort_aux_idx_gpu = cupy.asarray(unsort_aux_idx, dtype=cupy.int32)

            max_ncal_gpu = np.max(self.intopt.cart_ao_loc_si[1:] - self.intopt.cart_ao_loc_si[:-1])
            max_ncbe_gpu = np.max(self.intopt.cart_ao_loc_sj[1:] - self.intopt.cart_ao_loc_sj[:-1])
            max_nal_gpu = np.max(self.intopt.ao_loc_si[1:] - self.intopt.ao_loc_si[:-1])
            max_nbe_gpu = np.max(self.intopt.ao_loc_sj[1:] - self.intopt.ao_loc_sj[:-1])

            '''feri_buf_gpu = cupy.empty((max_nal_gpu*max_nbe_gpu*naoaux)) # This is crucial
            feri_aux_buf_gpu = cupy.empty((2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice))
            aux_ratio_gpu = cupy.zeros((nmo, naoaux))

            ialp_buf0_gpu = cupy.empty(ialp_buf_size)
            ialp_buf1_gpu = cupy.empty(ialp_buf_size)'''

            buf_sizes = [max_nal_gpu*max_nbe_gpu*naoaux, 
                         2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice, 
                         nbuf_ialp*ialp_buf_size, nmo*naoaux]
            
            (feri_buf_gpu, feri_aux_buf_gpu,
             ialp_buf_gpu, aux_ratio_gpu) = get_cupy_buffers(buf_sizes)
            
            aux_ratio_gpu.fill(0.0)
            aux_ratio_gpu = aux_ratio_gpu.reshape(nmo, naoaux)

            ialp_buf_gpu = ialp_buf_gpu.reshape(nbuf_ialp, -1)

        if self.double_buffer:
            win_ialp_buf, double_ialp_buf_cpu = get_shared((2, ialp_buf_size), rank_shm_buf=irank_cal, 
                                                            comm_shared=comm_pair, use_pin=True)

            comp_events = [cupy.cuda.Event(), cupy.cuda.Event()]
            data_events = [cupy.cuda.Event(), cupy.cuda.Event()]
        else:
            ialp_buf_cpu = cupyx.empty_pinned((ialp_buf_size,))
        
        alpha = cupy.array(1.0, dtype=cupy.float64)
        beta = alpha

        for bidx, ((al0, al1), cpidx_list) in enumerate(shell_slice_rank):
            nao0 = al1 - al0
            if self.double_buffer:
                io_idx = 0 if bidx % 2 else 1
                cal_idx = 1 if bidx % 2 else 0
            else:
                ialp_slice_cpu = ialp_buf_cpu[:nmo*nao0*naoaux].reshape(nmo, nao0, naoaux)
            
            if is_cal_rank:
                if self.double_buffer:
                    data_idx = 2 if bidx % 2 else 0
                    comp_idx = 0 if bidx % 2 else 1

                    data_event_idx = 0 if data_idx == 0 else 1
                    comp_event_idx = comp_idx

                    ialp_data_gpu = ialp_buf_gpu[data_idx]
                    ialp_comp_gpu = ialp_buf_gpu[comp_idx:comp_idx+2]
                    
                else:
                    ialp_comp_gpu = ialp_buf_gpu
                    
                
                with istream_comp:
                    if self.double_buffer and bidx > 1:
                        istream_comp.wait_event(data_events[comp_event_idx])

                    if unsort_ao:
                        buf_idx0 = 1 if bidx % 2 else 0
                        buf_idx1 = 0 if bidx % 2 else 1
                    else:
                        buf_idx0 = 0 if bidx % 2 else 1
                        buf_idx1 = 1 if bidx % 2 else 0

                    ialp0_gpu = ialp_comp_gpu[buf_idx0][:naoaux*nao0*nmo].reshape(naoaux, nao0, nmo)
                    ialp1_gpu = ialp_comp_gpu[buf_idx1][:naoaux*nao0*nmo].reshape(naoaux, nao0, nmo)
                    ialp0_gpu.fill(0.0)

                    t2 = get_current_time()
                    for cp_ij_id in cpidx_list:
                        cpi = self.intopt.cp_idx[cp_ij_id]
                        be0, be1 = self.intopt.ao_loc_si[cpi], self.intopt.ao_loc_si[cpi+1]
                        nao1 = be1 - be0
                        t1 = get_current_time()
                        int3c_slice = feri_buf_gpu[:(nao0*nao1*naoaux)]
                        #int3c_slice.fill(0.0)
                        int3c_slice = aux_e2_gpu(self.intopt, cp_ij_id, feri_shape="pji", out=int3c_slice, buf_aux=feri_aux_buf_gpu)
                        #accumulate_time(self.t_feri, t1)
                        recorder_feri.append(record_elapsed_time(t1))

                        t1 = get_current_time()
                        #ialp_gpu += cupy.dot(int3c_slice.reshape(-1, nao1), occ_coeff_gpu[be0:be1]).reshape(self.naoaux, nao0, nmo).swapaxes(0, 2)
                        #dgemm_cupy(1, 0, occ_coeff_gpu[be0:be1], int3c_slice.reshape(nao1, -1), ialp_gpu.reshape(nmo, -1), 1.0, 1.0)
                        dgemm_cupy(0, 0, int3c_slice.reshape(-1, nao1), occ_coeff_gpu[be0:be1], ialp0_gpu, 1.0, 1.0)
                        #dgemm_cupy(0, 0, int3c_slice.reshape(-1, nao1), occ_coeff_gpu[be0:be1], ialp0_gpu, alpha, beta)
                        #accumulate_time(self.t_ialp, t1)
                        recorder_ialp.append(record_elapsed_time(t1))
                    recorder_cal.append(record_elapsed_time(t2))
            
            if unsort_ao:
                ori_ao_idx = self.intopt._ao_idx[al0:al1]
                sort_indices = np.argsort(ori_ao_idx)
                sorted_ao_idx = np.ascontiguousarray(ori_ao_idx[sort_indices])

                if is_cal_rank:
                    with istream_comp:
                        if self.double_buffer and bidx > 1:
                            istream_comp.wait_event(data_events[comp_event_idx])

                        t1 = get_current_time()
                        t2 = get_current_time()
                        sort_indices_gpu = cupy.asarray(sort_indices, dtype=cupy.int32)

                        cupy.take(ialp0_gpu, unsort_aux_idx_gpu, axis=0, out=ialp1_gpu)
                        cupy.take(ialp1_gpu, sort_indices_gpu, axis=1, out=ialp0_gpu)
                        ialp1_gpu = ialp1_gpu.reshape(nmo, nao0, naoaux)
                        ialp1_gpu[:] = ialp0_gpu.transpose(2, 1, 0)

                        cupyx.scipy.linalg.solve_triangular(low_gpu.T, ialp1_gpu.reshape(-1, naoaux).T, 
                                                            lower=True, overwrite_b=True, check_finite=False)
                        
                        ialp0_gpu = ialp0_gpu.reshape(ialp1_gpu.shape)
                        cupy.multiply(ialp1_gpu, ialp1_gpu, out=ialp0_gpu)
                        aux_ratio_gpu += cupy.sum(ialp0_gpu, axis=1)

                        recorder_ialp.append(record_elapsed_time(t1))
                        recorder_cal.append(record_elapsed_time(t2))

                        if self.double_buffer:
                            comp_events[comp_event_idx].record()

                    with istream_data:
                        if self.double_buffer:
                            if bidx > 0:
                                istream_data.wait_event(comp_events[data_event_idx])

                                t1 = get_current_time()
                                (prev_al0, prev_al1), _ = shell_slice_rank[bidx - 1]
                                prev_nao0 = prev_al1 - prev_al0
                                ialp_gpu = ialp_data_gpu[:nmo*prev_nao0*naoaux].reshape(nmo, prev_nao0, naoaux)
                                ialp_slice_cpu = double_ialp_buf_cpu[cal_idx][:nmo*prev_nao0*naoaux].reshape(nmo, prev_nao0, naoaux)
                                cupy.asnumpy(ialp_gpu, out=ialp_slice_cpu)
                                recorder_data.append(record_elapsed_time(t1))

                                data_events[data_event_idx].record()
                                istream_data.synchronize()
                        else:
                            t1 = get_current_time()
                            cupy.asnumpy(ialp1_gpu, out=ialp_slice_cpu)
                            recorder_data.append(record_elapsed_time(t1))

                            istream_data.synchronize()

                if is_io_rank:
                    t1 = get_current_time()
                    if self.double_buffer:
                        if bidx > 1:
                            (pprev_al0, pprev_al1), _ = shell_slice_rank[bidx-2]
                            pprev_nao0 = pprev_al1 - pprev_al0
                            pprev_sorted_ao_idx = np.sort(self.intopt._ao_idx[pprev_al0:pprev_al1])
                            ialp_io_cpu = double_ialp_buf_cpu[io_idx][:nmo*pprev_nao0*naoaux].reshape(nmo, pprev_nao0, naoaux)
                            if use_ialp_file:
                                ialp_data.write_direct(ialp_io_cpu, dest_sel=np.s_[:, pprev_sorted_ao_idx])
                            else:
                                ialp_data[:, pprev_sorted_ao_idx] = ialp_io_cpu
                    else:
                        if use_ialp_file:
                            ialp_data.write_direct(ialp_slice_cpu, dest_sel=np.s_[:, sorted_ao_idx])
                        else:
                            ialp_data[:, sorted_ao_idx] = ialp_slice_cpu
                    recorder_write.append(record_elapsed_time(t1))

                
            else:
                if is_cal_rank:
                    with istream_comp:
                        if self.double_buffer and bidx > 1:
                            istream_comp.wait_event(data_events[comp_event_idx])

                        t2 = get_current_time()
                        cupy.take(ialp0_gpu, unsort_aux_idx_gpu, axis=0, out=ialp1_gpu)
                        ialp0_gpu = ialp0_gpu.reshape(nmo, nao0, naoaux)
                        ialp0_gpu[:] = ialp1_gpu.transpose(2, 1, 0)

                        cupyx.scipy.linalg.solve_triangular(low_gpu.T, ialp0_gpu.reshape(-1, naoaux).T, 
                                                            lower=True, overwrite_b=True, check_finite=False)

                        ialp1_gpu = ialp1_gpu.reshape(ialp0_gpu.shape)
                        cupy.multiply(ialp0_gpu, ialp0_gpu, out=ialp1_gpu)
                        aux_ratio_gpu += cupy.sum(ialp1_gpu, axis=1)

                        recorder_ialp.append(record_elapsed_time(t1))
                        recorder_cal.append(record_elapsed_time(t2))

                        if self.double_buffer:
                            comp_events[comp_event_idx].record()

                    with istream_data:
                        if self.double_buffer:
                            if bidx > 0:
                                istream_data.wait_event(comp_events[data_event_idx])

                                t1 = get_current_time()
                                (prev_al0, prev_al1), _ = shell_slice_rank[bidx - 1]
                                prev_nao0 = prev_al1 - prev_al0
                                ialp_gpu = ialp_data_gpu[:nmo*prev_nao0*naoaux].reshape(nmo, prev_nao0, naoaux)
                                ialp_slice_cpu = double_ialp_buf_cpu[cal_idx][:nmo*prev_nao0*naoaux].reshape(nmo, prev_nao0, naoaux)
                                cupy.asnumpy(ialp_gpu, out=ialp_slice_cpu)
                                recorder_data.append(record_elapsed_time(t1))

                                data_events[data_event_idx].record()
                                istream_data.synchronize()
                        else:
                            t1 = get_current_time()
                            cupy.asnumpy(ialp0_gpu, out=ialp_slice_cpu)
                            recorder_data.append(record_elapsed_time(t1))

                            istream_data.synchronize()

                
                if is_io_rank:
                    t1 = get_current_time()
                    if self.double_buffer:
                        if bidx > 1:
                            (pprev_al0, pprev_al1), _ = shell_slice_rank[bidx-2]
                            pprev_nao0 = pprev_al1 - pprev_al0
                            ialp_io_cpu = double_ialp_buf_cpu[io_idx][:nmo*pprev_nao0*naoaux].reshape(nmo, pprev_nao0, naoaux)
                            if use_ialp_file:
                                ialp_data.write_direct(ialp_io_cpu, dest_sel=np.s_[:, pprev_al0:pprev_al1])
                            else:
                                ialp_data[:, pprev_al0:pprev_al1] = ialp_io_cpu
                    else:
                        if use_ialp_file:
                            ialp_data.write_direct(ialp_slice_cpu, dest_sel=np.s_[:, al0:al1])
                        else:
                            ialp_data[:, al0:al1] = ialp_slice_cpu
                    recorder_write.append(record_elapsed_time(t1))

            if self.double_buffer:
                comm_pair.Barrier()

        if is_cal_rank:
            with istream_comp:
                cupy.asnumpy(aux_ratio_gpu, out=aux_ratio[nocc_core:])

        if self.double_buffer:
            for bidx in [bidx+1, bidx+2]:
                io_idx = 0 if bidx % 2 else 1
                cal_idx = 1 if bidx % 2 else 0

                if is_cal_rank and bidx <= len(shell_slice_rank):
                    with istream_data:
                        data_idx = 2 if bidx % 2 else 0
                        data_event_idx = 0 if data_idx == 0 else 1

                        istream_data.wait_event(comp_events[data_event_idx])

                        t1 = get_current_time()
                        (prev_al0, prev_al1), _ = shell_slice_rank[bidx - 1]
                        prev_nao0 = prev_al1 - prev_al0

                        ialp_buf_cpu = double_ialp_buf_cpu[cal_idx]
                        ialp_cpu = ialp_buf_cpu[:prev_nao0*nmo*naoaux].reshape(nmo, prev_nao0, naoaux)
                        ialp_gpu = ialp_buf_gpu[data_idx][:prev_nao0*nmo*naoaux].reshape(nmo, prev_nao0, naoaux)
                        cupy.asnumpy(ialp_gpu, out=ialp_cpu)
                        recorder_data.append(record_elapsed_time(t1))

                        data_events[data_event_idx].record()

                        istream_data.synchronize()


                if is_io_rank:
                    t1 = get_current_time()
                    (pprev_al0, pprev_al1), _ = shell_slice_rank[bidx-2]
                    pprev_nao0 = pprev_al1 - pprev_al0
                    ialp_io_cpu = double_ialp_buf_cpu[io_idx][:nmo*pprev_nao0*naoaux].reshape(nmo, pprev_nao0, naoaux)
                    
                    if unsort_ao:
                        pprev_sorted_ao_idx = np.sort(self.intopt._ao_idx[pprev_al0:pprev_al1])
                        if use_ialp_file:
                            ialp_data.write_direct(ialp_io_cpu, dest_sel=np.s_[:, pprev_sorted_ao_idx])
                        else:
                            ialp_data[:, pprev_sorted_ao_idx] = ialp_io_cpu
                    else:
                        if use_ialp_file:
                            ialp_data.write_direct(ialp_io_cpu, dest_sel=np.s_[:, pprev_al0:pprev_al1])
                        else:
                            ialp_data[:, pprev_al0:pprev_al1] = ialp_io_cpu

                    recorder_write.append(record_elapsed_time(t1))
                
                comm_pair.Barrier()

            if is_cal_rank:
                unregister_pinned_memory(double_ialp_buf_cpu)
            win_ialp_buf.Free()

    batch_accumulate_time(self.t_feri, recorder_feri)
    batch_accumulate_time(self.t_ialp, recorder_ialp)
    batch_accumulate_time(self.t_cal, recorder_cal)
    batch_accumulate_time(self.t_data, recorder_data)
    batch_accumulate_time(self.t_write, recorder_write)

    if is_cal_rank:
        if self.double_buffer:
            istream_comp.synchronize()
        Accumulate_GA_shm(win_aux_ratio, aux_ratio_node, aux_ratio)

    comm.Barrier()

    close_ipc_handle(shm_ptr)

    if self.int_storage in {0, 3}:
        if nnode > 1:
            Acc_and_get_GA(self.ialp_node)
            comm.Barrier()
        if self.int_storage == 3:
            mo_slice = get_slice(range(nrank), job_size=nmo)[irank]
            if mo_slice is not None:
                i0 = mo_slice[0]
                i1 = mo_slice[-1] + 1
                self.ialp_gpu = cupy.asarray(self.ialp_node[i0:i1])
            comm_shm.Barrier()
            free_win(self.win_ialp)
            self.ialp_node = None
    else:
        file_ialp.close()

    if (qc_method == 'hf') and (self.chkfile_ialp is None):
        free_win(win_j2c)
    
    self.intopt.free_bpcache()
    
    return win_aux_ratio, aux_ratio_node

def get_ialp_incore_cuda(self, df_obj, qc_method, ijp_shape):
    nao = self.mol.nao_nr()
    nocc = self.mol.nelectron//2
    nmo = len(self.mo_list)
    nocc_core = nocc - nmo
    occ_coeff = self.o[:, nocc_core:]
    naoaux = self.naoaux
    shape_ialp = (nmo, self.nao, self.naoaux)

    set_zeros = True if nnode > 1 else False
    self.win_ialp, self.ialp_node = get_shared(shape_ialp, set_zeros=set_zeros)
    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=set_zeros)
    

    occ_coeff_gpu, occ_coeff_ptr = get_shared_cupy((nao, nocc), numpy_array=occ_coeff)

    aux_slice = get_slice(range(nrank), job_size=naoaux)[irank]

    if aux_slice is not None:
        p0_gpu = aux_slice[0]
        #occ_coeff_gpu = cupy.asarray(occ_coeff)

        naux_slice = len(aux_slice)
        max_naux = get_ncols_for_memory(0.8*avail_gpu_mem(igpu_shm)/nrank_per_gpu, nao*(nao + nmo), naux_slice)

        if self.int_storage == 0:
            feri_gpu_buffer = cupy.empty((max_naux*nao*nao))
        ialp_gpu_buffer = cupy.empty((max_naux*nao*nmo))

        for pidx0 in np.arange(naux_slice, step=max_naux):
            pidx1 = min(pidx0+max_naux, naux_slice)
            naux_seg = pidx1 - pidx0
            p0 = aux_slice[pidx0]
            p1 = aux_slice[pidx1-1] + 1

            feri_size = naux_seg * nao**2
            ialp_size = naux_seg * nao * nmo
            if ijp_shape:
                if self.int_storage == 0:
                    feri_gpu = feri_gpu_buffer[:feri_size].reshape(nao, nao, naux_seg)
                    feri_gpu.set(self.feri_node[:, :, p0:p1])
                elif self.int_storage == 3:
                    feri_gpu = self.feri_gpu[:, :, (p0-p0_gpu):(p1-p0_gpu)]
                ialp_gpu = ialp_gpu_buffer[:ialp_size].reshape(nmo, -1)
                cupy.dot(occ_coeff_gpu.T, feri_gpu.reshape(nao, -1), out=ialp_gpu)
                ialp_gpu = ialp_gpu.reshape(nmo, nao, naux_seg)
                aux_ratio_gpu = cupy.einsum("ijk,ijk->ik", ialp_gpu, ialp_gpu)

                self.ialp_node[:, :, p0:p1] = cupy.asnumpy(ialp_gpu)
                aux_ratio_node[:, p0:p1] = cupy.asnumpy(aux_ratio_gpu)
            else:
                if self.int_storage == 0:
                    feri_gpu = feri_gpu_buffer[:feri_size].reshape(naux_seg, nao, nao)
                    feri_gpu.set(self.feri_node[p0:p1])
                elif self.int_storage == 3:
                    feri_gpu = self.feri_gpu[(p0-p0_gpu):(p1-p0_gpu)]
                ialp_gpu = ialp_gpu_buffer[:ialp_size].reshape(-1, nmo)
                cupy.dot(feri_gpu.reshape(-1, nao), occ_coeff_gpu, out=ialp_gpu)
                ialp_gpu = ialp_gpu.reshape(naux_seg, nao, nmo)
                aux_ratio_gpu = cupy.einsum("ijk,ijk->ik", ialp_gpu, ialp_gpu)

                self.ialp_node[:, :, p0:p1] = cupy.asnumpy(ialp_gpu).transpose(2, 1, 0)
                aux_ratio_node[:, p0:p1] = cupy.asnumpy(aux_ratio_gpu).T
        
    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(self.ialp_node)
        comm.Barrier()
    
    close_ipc_handle(occ_coeff_ptr)

    if self.int_storage == 3:
        mo_slice = get_slice(range(nrank), job_size=nmo)[irank]
        if mo_slice is not None:
            i0 = mo_slice[0]
            i1 = mo_slice[-1] + 1
            self.ialp_gpu = cupy.asarray(self.ialp_node[i0:i1])
        comm_shm.Barrier()
        free_win(self.win_ialp)
        self.ialp_node = None
        
    return win_aux_ratio, aux_ratio_node


def get_ialp_outcore_cuda(self, df_obj, qc_method, ijp_shape=True):
    #TODO
    nao = self.mol.nao_nr()
    nocc = self.mol.nelectron//2
    nmo = len(self.mo_list)
    nocc_core = nocc - nmo
    occ_coeff = self.o[:, nocc_core:]
    naoaux = self.naoaux
    mol = self.mol
    auxmol = df_obj.auxmol
    shape_ialp = (nmo, self.nao, self.naoaux)
    ao_loc = make_loc(self.mol._bas, 'sph')

    self.file_ialp = f"ialp_{qc_method}.tmp"
    file_ialp = h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm)
    nao_chunk = min(536870912 // naoaux, nao) # chunks must be smaller than 4GB
    '''ialp_data = file_ialp.create_dataset("ialp", shape_ialp, dtype=np.float64, 
                                         chunks=(1, nao_chunk, naoaux))'''
    ialp_data = create_h5py_dataset(file_ialp, "ialp", shape_ialp, dtype=np.float64, 
                                         chunks=(1, nao_chunk, naoaux))

    set_zeros = True if nnode > 1 else False
    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=set_zeros)

    occ_coeff_gpu, occ_coeff_ptr = get_shared_cupy((nao, nocc), numpy_array=occ_coeff)

    aux_slice = get_slice(range(nrank), job_size=naoaux)[irank]
    if aux_slice is not None:
        #occ_coeff_gpu = cupy.asarray(occ_coeff)

        max_memory = get_mem_spare(mol, 0.9)
        naux_slice = len(aux_slice)
        max_naux = get_ncols_for_memory(0.8*max_memory, nao*(nao + nmo), naux_slice)
        max_naux_gpu = get_ncols_for_memory(0.8*avail_gpu_mem()/nrank_per_gpu, nao*(nao + nmo), naux_slice)

        min_aux, max_aux = aux_slice[0], aux_slice[-1]+1

        with h5py.File(self.file_feri, 'r') as file_feri:
            feri_data = file_feri["feri"]
            for p0 in np.arange(min_aux, max_aux, step=max_naux):
                p1 = min(p0+max_naux, max_aux)
                naux_seg = p1 - p0

                '''if ijp_shape:
                    ialp_slice = np.dot(occ_coeff.T, feri_data[:, :, p0:p1].reshape(nao, -1)).reshape(nmo, nao, naux_seg)
                    ialp_data[:, :, p0:p1] = ialp_slice
                    aux_ratio_node[nocc_core:, p0:p1] = np.einsum("ijp,ijp->ip", ialp_slice, ialp_slice, optimize="optimal")
                    
                else:
                    #ialp_slice = np.dot(feri_data[p0:p1].reshape(-1, nao), occ_coeff).reshape(naux_seg, nao, nmo)
                    ialp_slice = np.tensordot(feri_data[p0:p1], occ_coeff, axes=([1], [0]))
                    ialp_data[:, :, p0:p1] = ialp_slice.transpose(2, 1, 0)
                    aux_ratio_node[nocc_core:, p0:p1] = np.einsum("pji,pji->ip", ialp_slice, ialp_slice, optimize="optimal")'''
                
                feri_cpu = feri_data[:, :, p0:p1] if ijp_shape else feri_data[p0:p1]
                ialp_cpu = np.empty((nmo, nao, naux_seg))
                for pidx0 in np.arange(naux_seg, step=max_naux_gpu):
                    pidx1 = min(pidx0+max_naux_gpu, naux_seg)
                    naux_slice = pidx1 - pidx0

                    if ijp_shape:
                        feri_gpu = cupy.asarray(feri_cpu[:, :, pidx0:pidx1])
                        ialp_gpu = cupy.dot(occ_coeff_gpu.T, feri_gpu.reshape(nao, -1)).reshape(nmo, nao, naux_slice)
                        aux_ratio_gpu = cupy.einsum("iap,iap->ip", ialp_gpu, ialp_gpu, optimize="optimal")

                        ialp_cpu[:, :, pidx0:pidx1] = cupy.asnumpy(ialp_gpu)
                    else:
                        feri_gpu = cupy.asarray(feri_cpu[pidx0:pidx1])
                        pali_gpu = np.dot(feri_gpu.reshape(-1, nao), occ_coeff_gpu).reshape(naux_slice, nao, nmo)
                        aux_ratio_gpu = cupy.einsum("pai,pai->ip", pali_gpu, pali_gpu, optimize="optimal")

                        ialp_cpu[:, :, pidx0:pidx1] = cupy.asnumpy(pali_gpu).transpose(2, 1, 0)
                    
                    aux_ratio_node[nocc_core:, p0+pidx0:p0+pidx1] = cupy.asnumpy(aux_ratio_gpu)
                
                ialp_data[:, :, p0:p1] = ialp_cpu
                
    comm_shm.Barrier()
    close_ipc_handle(occ_coeff_ptr)
    file_ialp.close()

    return win_aux_ratio, aux_ratio_node


def get_loc_ialp_cuda(self, df_obj, qc_method):
    nao = self.mol.nao_nr()
    nocc = self.mol.nelectron//2
    nmo = len(self.mo_list)
    naoaux = self.naoaux
    mol = self.mol
    # in hf, (ial|P) is computed with occupation number weighted MO coefficients. 
    # The weights must be taken away for MP2 (ial|P)
    uo_bar = self.uo / self.mo_occ[self.mo_occ>0]**0.5 

    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=True)

    uo_bar_gpu, uo_bar_ptr = get_shared_cupy((nocc, nocc), numpy_array=uo_bar)

    ao_slices = get_slice(range(nrank), job_size=nao)
    ao_slice = ao_slices[irank]
    if self.int_storage in {0, 3}:
        ialp_data = self.ialp_node
        use_ialp_file = False
    elif self.int_storage in {1, 2}:
        file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
        ialp_data = file_ialp["ialp"]
        use_ialp_file = True

    if ao_slice is not None:
        al0_rank, al1_rank = ao_slice[0], ao_slice[-1]+1
        nao_slice = len(ao_slice)

        gpu_mem_left = avail_gpu_mem(self.gpu_memory) - nocc*nocc*8*1e-6
        max_nao_gpu = get_ncols_for_memory(0.8*gpu_mem_left/nrank_per_gpu, naoaux*nocc, nao_slice) // 2

        #uo_bar_gpu = cupy.asarray(uo_bar)
        #ialp_buff_gpu = cupy.empty((nocc*naoaux*max_nao_gpu))
        #ialp_buff_cpu = np.empty((nocc*naoaux*max_nao_gpu))
        ialp_buff_cpu = cupyx.empty_pinned((nocc*naoaux*max_nao_gpu,))
        

        aux_ratio_gpu = cupy.zeros((nocc, naoaux))
        for al0 in np.arange(al0_rank, al1_rank, step=max_nao_gpu):
            al1 = min(al0+max_nao_gpu, al1_rank)
            nao_seg = al1 - al0
            ialp_cpu = ialp_buff_cpu[:nocc*nao_seg*naoaux].reshape(nocc, nao_seg, naoaux)
            if use_ialp_file:
                ialp_data.read_direct(ialp_cpu, source_sel=np.s_[:, al0:al1])
            else:
                ialp_cpu[:] = self.ialp_node[:, al0:al1]
            #ialp_gpu = ialp_buff_gpu[:nocc*nao_seg*naoaux].reshape(nocc, nao_seg, naoaux)
            ialp_gpu = cupy.empty((nocc, nao_seg, naoaux))
            ialp_gpu.set(ialp_cpu)

            ialp_gpu = cupy.dot(uo_bar_gpu.T, ialp_gpu.reshape(nocc, -1)).reshape(nocc, nao_seg, naoaux)
            cupy.asnumpy(ialp_gpu, out=ialp_cpu)
            ialp_data[:, al0:al1] = ialp_cpu

            #print_test(ialp_gpu, "(%d %d) ialp"%(al0, al1))
            #print_test(cupy.einsum("ijp,ijp->ip", ialp_gpu, ialp_gpu, optimize="optimal"), "(%d %d) raux"%(al0, al1))
            ialp_gpu *= ialp_gpu
            aux_ratio_gpu += ialp_gpu.sum(axis=1)
            #aux_ratio_gpu += cupy.einsum("ijp,ijp->ip", ialp_gpu, ialp_gpu, optimize="optimal")

        aux_ratio_cpu = cupy.asnumpy(aux_ratio_gpu)

        Accumulate_GA_shm(win_aux_ratio, aux_ratio_node, aux_ratio_cpu)


    if self.int_storage in {0, 3}:
        if nnode > 1:
            # Set ao blocks assigned to other nodes to zeros for cross-node accumulations
            ao_other_nodes = []
            for rank_i, ao_rank in enumerate(ao_slices):
                if (rank_i // nnode != inode) and (ao_rank is not None):
                    ao_other_nodes.extend(ao_rank)
            shm_ao_slice = get_slice(range(nrank_shm), job_list=ao_other_nodes)[irank_shm]
            if shm_ao_slice is not None:
                self.ialp_node[:, shm_ao_slice] = 0.0
            comm.Barrier()

            Acc_and_get_GA(self.ialp_node)

        

        if self.int_storage == 3:
            mo_slice = get_slice(range(nrank), job_size=nmo)[irank]
            if mo_slice is not None:
                i0 = mo_slice[0]
                i1 = mo_slice[-1] + 1
                self.ialp_gpu = cupy.asarray(self.ialp_node[i0:i1])
            comm_shm.Barrier()
            free_win(self.win_ialp)
            self.ialp_node = None
        
        #print_test(self.ialp_node, "ialp_node after")
        #print_test(aux_ratio_node, "aux_ratio_node")
    else:
        file_ialp.close()
    
    comm.Barrier()
    
    close_ipc_handle(uo_bar_ptr)

    return win_aux_ratio, aux_ratio_node

def get_ialp_cuda(self, df_obj, qc_method, log, zvec=True, ijp_shape=True, loc_trans=False, 
                  unsort_ao=True):
    self.t_cal = create_timer()
    self.t_feri = create_timer()
    self.t_ialp = create_timer()
    self.t_read = create_timer()
    self.t_write = create_timer()
    self.t_data = create_timer()

    mol = self.mol
    auxmol = df_obj.auxmol

    tt = get_current_time()
    if loc_trans:
        win_aux_ratio, aux_ratio_node = get_loc_ialp_cuda(self, df_obj, qc_method)
    else:
        if (self.int_storage == 2) or (not self.cal_grad):
            win_aux_ratio, aux_ratio_node = get_ialp_direct_cuda(self, df_obj, qc_method, log, unsort_ao=unsort_ao)
            
        elif self.int_storage in {0, 3}:# Incore/gpu_incore
            win_aux_ratio, aux_ratio_node = get_ialp_incore_cuda(self, df_obj, qc_method, ijp_shape)
                
        elif self.int_storage == 1:# Outcore
            win_aux_ratio, aux_ratio_node = get_ialp_outcore_cuda(self, df_obj, qc_method, ijp_shape)

        else:
            raise NotImplementedError
        
    if self.loc_fit:
        if self.chkfile_fitratio is None:
            comm.Barrier()
            Acc_and_get_GA(aux_ratio_node)
            comm.Barrier()
            '''if irank == 0:
                with h5py.File(f"fit_ratio_{qc_method}.tmp", "w") as ffit:
                    ffit.create_dataset("aux_ratio", data=aux_ratio_node)'''
        self.fit_list, self.fit_seg, self.nfit = get_fit_domain(aux_ratio_node, self.fit_tol)
        self.atom_close, self.bfit_seg, self.nbfit = get_bfit_domain(mol, auxmol, aux_ratio_node, self.fit_tol)
        if qc_method == 'hf' and zvec:
            self.atom_close_z, self.bfit_seg_z, self.nbfit_z = get_bfit_domain(mol, auxmol, aux_ratio_node, self.bfit_tol)
        comm_shm.Barrier()
        win_aux_ratio.Free()


    if self.chkfile_ialp is None:
        time_list = [['feri', self.t_feri], ['ialp', self.t_ialp],
                     ["calculation", self.t_cal], ['reading', self.t_read], ['writing', self.t_write], 
                     ["GPU-CPU data", self.t_data]]
        time_list = get_max_rank_time_list(time_list)
        print_time(time_list, log)

        #sys.exit()

    else:
        self.file_ialp = self.chkfile_ialp
        log.info(f"Use ialp check file:{self.chkfile_ialp}")
