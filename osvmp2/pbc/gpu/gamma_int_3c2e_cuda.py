import os
import math
import ctypes
import itertools
import numpy as np
import h5py
import cupy
import cupyx

from pyscf import lib, gto
from pyscf.lib.parameters import ANGULAR
from pyscf.gto import (ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF,
                       PTR_COORD, conc_env)
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools import k2gamma

from cupyx.scipy.linalg import solve_triangular
from gpu4pyscf.lib.cupy_helper import cart2sph
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray #, get_avail_mem
from gpu4pyscf.gto.mole import (cart2sph_by_l, group_basis, basis_seg_contraction, 
                                PTR_BAS_COORD, extract_pgto_params)
from gpu4pyscf.scf.jk import _scale_sp_ctr_coeff, _nearest_power2, init_constant
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh

#from gpu4pyscf.pbc.df.ft_ao import libpbc, init_constant, most_diffused_pgto, ft_ao_scheme
from gpu4pyscf.pbc.df.ft_ao import libpbc, most_diffused_pgto, ft_ao_scheme
from gpu4pyscf.pbc.df.int3c2e import (estimate_rcut, _conc_locs, 
                                      int3c2e_scheme, to_primitive_bas)
from gpu4pyscf.pbc.df.rsdf_builder import decompose_j2c, estimate_ke_cutoff_for_omega, _get_2c2e

from gpu4pyscf.pbc.df.rsdf_builder import _get_2c2e as get_gpu_2c2e
from osvmp2.__config__ import ngpu
from osvmp2.gpu.cuda_utils import avail_gpu_mem, ave_gpu_memory, equisum_partition, get_seg_gpu, dgemm_cupy
from osvmp2.mpi_addons import *
from osvmp2.osvutil import *
from mpi4py import MPI
from osvmp2.int_2c2e import get_j2c_low

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
#inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm= comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//comm_shm.size
inode = irank // nrank_shm

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)
ngpu_shm = ngpu // nnode
nrank_per_gpu = nrank_shm // ngpu_shm
igpu = irank // nrank_per_gpu
igpu_shm = irank_shm // nrank_per_gpu
irank_gpu = irank % nrank_per_gpu
cupy.cuda.runtime.setDevice(igpu_shm)

LMAX_ON_GPU = 8
LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 45
THREADS = 256
BVK_CELL_SHELLS = 400

OMEGA_MIN = 0.25

# In the ED of the j2c2e metric, the default LINEAR_DEP_THR setting in pyscf-2.8
# is too loose. The linear dependency truncation often leads to serious errors.
# PBC GDF very differs to the molecular GDF approximation where diffused
# functions typically have insignificant contributions. The diffused auxiliary
# crystal orbitals have large impacts on the accuracy of Coulomb integrals. A
# tight linear dependency threshold have to be applied to control the error,
# even this may cause more numerical stability issues.
LINEAR_DEP_THR = 1e-11
# Use eigenvalue decomposition in decompose_j2c
PREFER_ED = False

THREADS = 256

import osvmp2
lib_dir = f"{os.path.dirname(osvmp2.__file__)}/lib"
libpbcint = ctypes.CDLL(f"{lib_dir}/pbcIntCuda.so")


def prim_to_ctr_l_counts(prim_l_counts, prim_to_ctr_mapping):
    ctr_l_counts = []

    pri_sidx0 = 0
    for nl_prim in prim_l_counts:
        pri_sidx1 = pri_sidx0 + nl_prim
        ctr_l_counts.append(len(np.unique(prim_to_ctr_mapping[pri_sidx0:pri_sidx1])))
        pri_sidx0 = pri_sidx1

    return ctr_l_counts

def ctr_to_prim_l_counts(ctr_l_counts, map_l_counts):
    prim_l_counts = []

    ctr_sidx0 = 0
    for nl_ctr in ctr_l_counts:
        ctr_sidx1 = ctr_sidx0 + nl_ctr
        prim_l_counts.append(np.sum(map_l_counts[ctr_sidx0:ctr_sidx1]))
        ctr_sidx0 = ctr_sidx1

    return prim_l_counts

def _split_prim_l_groups(ori_uniq_l, ori_prim_l_counts, ori_ctr_l_counts, prim_ao_loc, map_l_counts, group_size):

    uniq_l = []
    prim_l_counts = []
    ctr_l_counts = []

    pri_sidx0 = 0
    ctr_sidx0 = 0
    for l, nl_pri, nl_ctr in zip(ori_uniq_l, ori_prim_l_counts, ori_ctr_l_counts):
        pri_sidx1 = pri_sidx0 + nl_pri
        ctr_sidx1 = ctr_sidx0 + nl_ctr
        
        if (prim_ao_loc[pri_sidx1] - prim_ao_loc[pri_sidx0]) > group_size:
            map_l_counts_l = map_l_counts[ctr_sidx0:ctr_sidx1]
            nl_ctr_now = 1
            nl_pri_now = map_l_counts_l[0]
            pri_sidx1 = pri_sidx0 + nl_pri_now
            nao_now = prim_ao_loc[pri_sidx1] - prim_ao_loc[pri_sidx0]
            assert nao_now <= group_size

            for nl in map_l_counts_l[1:]:
                pri_seg_sidx0 = pri_sidx1
                pri_seg_sidx1 = pri_seg_sidx0 + nl
                nao_seg = prim_ao_loc[pri_seg_sidx1] - prim_ao_loc[pri_seg_sidx0]
                assert nao_seg <= group_size

                if (nao_now + nao_seg) > group_size:
                    uniq_l.append(l)
                    prim_l_counts.append(nl_pri_now)
                    ctr_l_counts.append(nl_ctr_now)
                    nl_pri_now = nl
                    nl_ctr_now = 1
                    nao_now = nao_seg
                else:
                    nl_pri_now += nl
                    nl_ctr_now += 1
                    nao_now += nao_seg
                
                pri_sidx1 += nl
            
            uniq_l.append(l)
            prim_l_counts.append(nl_pri_now)
            ctr_l_counts.append(nl_ctr_now)
        else:
            uniq_l.append(l)
            prim_l_counts.append(nl_pri)
            ctr_l_counts.append(nl_ctr)

        pri_sidx0 = pri_sidx1
        ctr_sidx0 = ctr_sidx1
    
    uniq_l = np.asarray(uniq_l)
    prim_l_counts = np.asarray(prim_l_counts)
    ctr_l_counts = np.asarray(ctr_l_counts)

    return uniq_l, prim_l_counts, ctr_l_counts


def get_uniq_l_from_counts(ori_ctr_uniq_l, ori_ctr_l_counts, new_ctr_l_counts):
    new_ctr_uniq_l = []

    old_sidx0 = 0
    new_count_idx = 0
    new_sidx = 0
    for l, old_nl in zip(ori_ctr_uniq_l, ori_ctr_l_counts):
        old_sidx1 = old_sidx0 + old_nl
        for new_nl in new_ctr_l_counts[new_count_idx:]:
            new_ctr_uniq_l.append(l)
            new_count_idx += 1
            new_sidx += new_nl
            if new_sidx == old_sidx1:
                break
        old_sidx0 = old_sidx1

    return np.asarray(new_ctr_uniq_l)

class Int3c2eEnvVars(ctypes.Structure):
    _fields_ = [
        ('cell0_natm', ctypes.c_uint16),
        ('cell0_nbas', ctypes.c_uint16),
        ('bvk_ncells', ctypes.c_uint16),
        ('nimgs', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('img_coords', ctypes.c_void_p),
    ]

def get_shells_ranks(uniq_l_ctr, l_ctr_counts, ranks=nrank):
    full_l_ctr = np.repeat(uniq_l_ctr, l_ctr_counts)
    nfs_ls = 2 * full_l_ctr + 1

    assert len(nfs_ls) >= ranks, f"Too many cores: there are {len(nfs_ls)} shells and {ranks} cores"

    nfs_gpus = equisum_partition(nfs_ls, ranks)

    job_offsets = cum_offset([len(nfs) for nfs in nfs_gpus])
    
    uniq_l_ctr_ranks = []
    l_ctr_counts_ranks = []
    for rank_i in range(ranks):
        idx0, idx1 = job_offsets[rank_i:rank_i+2]
        full_ls_rank = full_l_ctr[idx0:idx1]

        uniq_l_ctr_rank, l_ctr_counts_rank = np.unique(full_ls_rank, return_counts=True)
        uniq_l_ctr_ranks.append(uniq_l_ctr_rank)
        l_ctr_counts_ranks.append(l_ctr_counts_rank)


    return np.concatenate(uniq_l_ctr_ranks), np.concatenate(l_ctr_counts_ranks)

def prim_to_ctr_l_counts(prim_l_counts, prim_to_ctr_mapping):
    ctr_l_counts = []
    ctr_idx_last = 0
    for prim_idx in np.cumsum(prim_l_counts):
        ctr_idx = prim_to_ctr_mapping[prim_idx]
        ctr_l_counts.append(ctr_idx - ctr_idx_last)
        ctr_idx_last = ctr_idx
    return ctr_l_counts


class SRInt3c2eOpt:
    def __init__(self, cell, auxcell, omega, bvk_kmesh=None, fitting=False, get_nuc=False):
        assert omega < 0
        self.omega = -omega
        assert cell._bas[:,ANG_OF].max() <= LMAX

        self.cell = cell
        prim_cell, sorted_cell, self.prim_to_ctr_mapping, self.ao_idx = \
                to_primitive_bas(cell)
        self.prim_cell = prim_cell
        self.prim_cell.omega = omega

        self.prim_to_ctr_mapping = np.append(self.prim_to_ctr_mapping, sorted_cell.nbas)
        # This sorted_cell is a fictitious cell object, to define the
        # p2c_mapping for prim_cell. PTRs in sorted_cell are not initialized.
        # This object should not be used for any integral kernel.
        self.sorted_cell = sorted_cell

        self.map_uniq_l, self.map_l_counts = np.unique(
                    self.prim_to_ctr_mapping, return_counts=True)


        #_, _, uniq_l_ctr, self.ori_cell0_ctr_l_counts = group_basis(cell)
        self.ori_cell0_uniq_l, self.ori_cell0_prim_l_counts = np.unique(
            prim_cell._bas[:,ANG_OF], return_counts=True)
        
        _, self.ori_cell0_ctr_l_counts = np.unique(
            sorted_cell._bas[:,ANG_OF], return_counts=True)
        
        self.ori_cell0_uniq_l_ranks, self.ori_cell0_prim_l_counts_ranks = \
                                                                get_shells_ranks(self.ori_cell0_uniq_l, 
                                                                self.ori_cell0_prim_l_counts, 
                                                                ranks=nrank)
 
        self.ori_cell0_ctr_l_counts_ranks = prim_to_ctr_l_counts(self.ori_cell0_prim_l_counts_ranks, 
                                                                 self.prim_to_ctr_mapping)

        '''self.ori_cell0_uniq_l = uniq_l_ctr[:, 0]
                
        self.ori_cell0_prim_l_counts = ctr_to_prim_l_counts(self.ori_cell0_ctr_l_counts, 
                                                            self.map_l_counts)'''

        self.auxcell = auxcell

        self.get_nuc = get_nuc

        if irank_gpu == 0:
            if fitting:
                j2c = get_gpu_2c2e(auxcell, None, -omega, True)[0].real
                j2c[:] = cupy.linalg.cholesky(j2c).T

            auxcell, aux_coeff, uniq_l_ctr, l_ctr_counts = group_basis(auxcell, tile=1)

            if fitting:

                cupyx.scipy.linalg.solve_triangular(j2c.T, aux_coeff.T, lower=True, overwrite_b=True)

                j2c = None
            
            if get_nuc:
                self.aux_coeff = aux_coeff
            else:
                self.aux_coeff = cupy.asnumpy(aux_coeff)
            aux_coeff = None
        else:
            auxcell, _, _, uniq_l_ctr, l_ctr_counts = group_basis(auxcell, tile=1, sparse_coeff=True)
            self.aux_coeff = None

        self.sorted_auxcell = auxcell
        self.uniq_l_ctr_aux = uniq_l_ctr
        self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.sorted_auxcell.omega = omega

        if bvk_kmesh is None:
            bvk_kmesh = np.ones(3, dtype=int)
        self.bvk_kmesh = bvk_kmesh

        self.rcut = None
        self.int3c2e_envs = None
        self.bvk_cell = None
        self.bvkmesh_Ls = None

    def build(self, group_size_aoi=None, group_size_aoj=None, ij_tasks=None, 
              slice_aoi=False, slice_aoj=False, verbose=None):
        
        if nrank > 1 and not self.get_nuc:
            assert (slice_aoi ^ slice_aoj), "Either i or j AOs must be sliced for multiple GPUs"

        '''integral screening'''
        log = logger.new_logger(self.cell, verbose)
        pcell = self.prim_cell
        auxcell = self.sorted_auxcell

        bvk_kmesh = self.bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        self.bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(pcell, bvk_kmesh, True)
        if np.prod(bvk_kmesh) == 1:
            bvkcell = pcell
        else:
            bvkcell = pbctools.super_cell(pcell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

        self.rcut = rcut = estimate_rcut(pcell, auxcell, self.omega).max()
        Ls = asarray(bvkcell.get_lattice_Ls(rcut=rcut))
        Ls = Ls[cupy.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug('int3c2e_kernel rcut = %g, nimgs = %d', rcut, nimgs)

        # Note: sort_orbitals and unsort_orbitals do not transform the
        # s and p orbitals. _scale_sp_ctr_coeff apply these special
        # normalization coefficients to the _env.
        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            bvkcell._atm, bvkcell._bas, _scale_sp_ctr_coeff(bvkcell),
            auxcell._atm, auxcell._bas, _scale_sp_ctr_coeff(auxcell))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[bvkcell.nbas,PTR_EXP] - auxcell._bas[0,PTR_EXP]
        _bas_cpu[bvkcell.nbas:,PTR_BAS_COORD] += off
        self._atm_cpu = _atm_cpu
        self._bas_cpu = _bas_cpu
        self._env_cpu = _env_cpu

        _atm = cupy.array(_atm_cpu, dtype=np.int32)
        _bas = cupy.array(_bas_cpu, dtype=np.int32)
        _env = cupy.array(_env_cpu, dtype=np.float64)
        bvk_ao_loc = bvkcell.ao_loc
        aux_loc = auxcell.ao_loc
        ao_loc = _conc_locs(bvk_ao_loc, aux_loc)
        int3c2e_envs = Int3c2eEnvVars(
            pcell.natm, pcell.nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc.data.ptr, Ls.data.ptr,
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        int3c2e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)
        self.int3c2e_envs = int3c2e_envs
        init_constant(pcell)
        

        self.cell0_uniq_li = self.ori_cell0_uniq_l_ranks if slice_aoi else self.ori_cell0_uniq_l
        self.cell0_prim_li_counts = self.ori_cell0_prim_l_counts_ranks if slice_aoi else self.ori_cell0_prim_l_counts
        self.cell0_ctr_li_counts = self.ori_cell0_ctr_l_counts_ranks if slice_aoi else self.ori_cell0_ctr_l_counts

        self.cell0_uniq_lj = self.ori_cell0_uniq_l_ranks if slice_aoj else self.ori_cell0_uniq_l
        self.cell0_prim_lj_counts = self.ori_cell0_prim_l_counts_ranks if slice_aoj else self.ori_cell0_prim_l_counts
        self.cell0_ctr_lj_counts = self.ori_cell0_ctr_l_counts_ranks if slice_aoj else self.ori_cell0_ctr_l_counts


        '''self.cell0_uniq_li = self.cell0_uniq_lj = self.ori_cell0_uniq_l
        self.cell0_prim_li_counts = self.cell0_prim_lj_counts = self.ori_cell0_prim_l_counts
        self.cell0_ctr_li_counts = self.cell0_ctr_lj_counts = self.ori_cell0_ctr_l_counts'''

        if group_size_aoi is not None:
            self.cell0_uniq_li, self.cell0_prim_li_counts, self.cell0_ctr_li_counts = \
                    _split_prim_l_groups(self.cell0_uniq_li, self.cell0_prim_li_counts, 
                                     self.cell0_ctr_li_counts, self.prim_cell.ao_loc_nr(), 
                                     self.map_l_counts, group_size_aoi)

        if group_size_aoj is not None:
            self.cell0_uniq_lj, self.cell0_prim_lj_counts, self.cell0_ctr_lj_counts = \
                    _split_prim_l_groups(self.cell0_uniq_lj, self.cell0_prim_lj_counts, 
                                     self.cell0_ctr_lj_counts, self.prim_cell.ao_loc_nr(), 
                                     self.map_l_counts, group_size_aoj)
        
        return self

    def estimate_cutoff_with_penalty(self):
        pcell = self.prim_cell
        auxcell = self.sorted_auxcell
        vol = self.bvkcell.vol
        omega = self.omega
        aux_exp, _, aux_l = most_diffused_pgto(auxcell)
        cell_exp, _, cell_l = most_diffused_pgto(pcell)
        if omega == 0:
            theta = 1./(1./cell_exp*2 + 1./aux_exp)
        else:
            theta = 1./(1./cell_exp*2 + 1./aux_exp + omega**-2)
        lsum = cell_l * 2 + aux_l + 1
        rad = vol**(-1./3) * self.rcut + 1
        surface = 4*np.pi * rad**2
        lattice_sum_factor = 2*np.pi*self.rcut*lsum/(vol*theta) + surface
        cutoff = pcell.precision / lattice_sum_factor
        logger.debug1(pcell, 'int3c_kernel integral omega=%g theta=%g cutoff=%g',
                      omega, theta, cutoff)
        return cutoff

    def generate_img_idx(self, cutoff=None, verbose=None):
        log = logger.new_logger(self.cell, verbose)
        cput0 = log.init_timer()
        int3c2e_envs = self.int3c2e_envs
        pcell = self.prim_cell
        auxcell = self.sorted_auxcell
        bvk_ncells = np.prod(self.bvk_kmesh)
        p_nbas = pcell.nbas

        exps, cs = extract_pgto_params(pcell, 'diffused')
        exps = asarray(exps, dtype=np.float32)
        log_coeff = cupy.log(abs(asarray(cs, dtype=np.float32)))

        # Search the most diffused functions on each atom
        aux_exps, aux_cs = extract_pgto_params(auxcell, 'diffused')
        aux_ls = auxcell._bas[:,ANG_OF]
        r2_aux = np.log(aux_cs**2 / pcell.precision * 10**aux_ls) / aux_exps
        atoms = auxcell._bas[:,ATOM_OF]
        atom_aux_exps = np.full(pcell.natm, 1e8, dtype=np.float32)
        for ia in range(pcell.natm):
            bas_mask = atoms == ia
            es = aux_exps[bas_mask]
            if len(es) > 0:
                atom_aux_exps[ia] = es[r2_aux[bas_mask].argmax()]
        atom_aux_exps = asarray(atom_aux_exps, dtype=np.float32)
        if cutoff is None:
            cutoff = self.estimate_cutoff_with_penalty()
        log_cutoff = math.log(cutoff)

        c_shell_counts_i = self.cell0_ctr_li_counts
        c_shell_counts_j = self.cell0_ctr_lj_counts
        c_shell_offsets_i = np.append(0, np.cumsum(c_shell_counts_i))
        c_shell_offsets_j = np.append(0, np.cumsum(c_shell_counts_j))
        p_shell_li_offsets = np.append(0, np.cumsum(self.cell0_prim_li_counts))
        p_shell_lj_offsets = np.append(0, np.cumsum(self.cell0_prim_lj_counts))

        p2c_mapping = asarray(self.prim_to_ctr_mapping, dtype=np.int32)

        def gen_img_idx(cpi, cpj):
            t0 = log.init_timer()

            ish0, ish1 = p_shell_li_offsets[cpi:cpi+2]
            jsh0, jsh1 = p_shell_lj_offsets[cpj:cpj+2]
            nprimi = ish1 - ish0
            nprimj = jsh1 - jsh0
            nctri = c_shell_counts_i[cpi]
            nctrj = c_shell_counts_j[cpj]

            # Number of images for each pair of (bas_i_in_bvkcell, bas_j_in_bvkcell)
            ovlp_img_counts = cupy.zeros((bvk_ncells*nprimi*bvk_ncells*nprimj), dtype=np.int32)

            err = libpbcint.bvk_overlap_img_counts(
                ctypes.cast(ovlp_img_counts.data.ptr, ctypes.c_void_p),
                ctypes.cast(p2c_mapping.data.ptr, ctypes.c_void_p),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff),
                ctypes.c_int(0))
            if err != 0:
                raise RuntimeError('bvk_ovlp_img_counts failed')

            bas_ij = asarray(cupy.where(ovlp_img_counts > 0)[0], dtype=np.int32)
            ovlp_npairs = len(bas_ij)
            if ovlp_npairs == 0:
                img_idx = offsets = bas_ij = pair_mapping = c_pair_idx = np.zeros(0, dtype=np.int32)
                return img_idx, offsets, bas_ij, pair_mapping, c_pair_idx

            counts_sorting = (-ovlp_img_counts[bas_ij]).argsort()
            bas_ij = bas_ij[counts_sorting]
            ovlp_img_counts = ovlp_img_counts[bas_ij]
            ovlp_img_offsets = cupy.empty(ovlp_npairs+1, dtype=np.int32)
            ovlp_img_offsets[0] = 0
            cupy.cumsum(ovlp_img_counts, out=ovlp_img_offsets[1:])
            tot_imgs = int(ovlp_img_offsets[ovlp_npairs])
            ovlp_img_idx = cupy.empty(tot_imgs, dtype=np.int32)
            err = libpbc.bvk_overlap_img_idx(
                ctypes.cast(ovlp_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ovlp_npairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('bvk_overlap_img_idx failed')
            log.timer_debug1('ovlp_img_idx', *cput0)
            nimgs_J = int(ovlp_img_counts[0])
            ovlp_img_counts = counts_sorting = None

            img_counts = cupy.zeros(ovlp_npairs, dtype=np.int32)
            ovlp_pair_sorting = cupy.arange(len(bas_ij), dtype=np.int32)
            err = libpbc.sr_int3c2e_img_idx(
                lib.c_null_ptr(),
                ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_pair_sorting.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ovlp_npairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('sr_int3c2e_img_counts failed')

            n_pairs = int(cupy.count_nonzero(img_counts))
            if n_pairs == 0:
                img_idx = offsets = bas_ij = pair_mapping = c_pair_idx = np.zeros(0, dtype=np.int32)
                return img_idx, offsets, bas_ij, pair_mapping, c_pair_idx

            # Sorting the bas_ij pairs by image counts. This groups bas_ij into
            # groups with similar workloads in int3c2e kernel.
            counts_sorting = cupy.argsort(-img_counts.ravel())[:n_pairs]
            counts_sorting = asarray(counts_sorting, dtype=np.int32)
            bas_ij = bas_ij[counts_sorting]
            ovlp_pair_sorting = counts_sorting
            img_counts = img_counts[counts_sorting]
            offsets = cupy.empty(n_pairs+1, dtype=np.int32)
            cupy.cumsum(img_counts, out=offsets[1:])
            offsets[0] = 0
            tot_imgs = int(offsets[n_pairs])
            img_idx = cupy.empty(tot_imgs, dtype=np.int32)
            err = libpbc.sr_int3c2e_img_idx(
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_pair_sorting.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_pairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('sr_int3c2e_img_idx failed')
            log.debug1('ovlp nimgs=%d pairs=%d tot_imgs=%d. '
                       'double-lattice-sum: largest=%d, medium=%d',
                       nimgs_J, n_pairs, tot_imgs, img_counts[0], img_counts[n_pairs//2])
            t1 = log.timer_debug1('int3c2e_img_idx', *t0)

            # bas_ij stores the non-negligible primitive-pair indices.
            # p2c_mapping converts the bas_ij to contracted GTO-pair indices.
            I, i, J, j = cupy.unravel_index(
                bas_ij, (bvk_ncells, nprimi, bvk_ncells, nprimj))
            i += ish0
            j += jsh0
            bas_ij = cupy.ravel_multi_index(
                (I, i, J, j), (bvk_ncells, p_nbas, bvk_ncells, p_nbas))
            bas_ij = asarray(bas_ij, dtype=np.int32)

            ic = p2c_mapping[i] - c_shell_offsets_i[cpi]
            jc = p2c_mapping[j] - c_shell_offsets_j[cpj]
            I %= bvk_ncells
            J %= bvk_ncells
            reduced_pair_idx = cupy.ravel_multi_index(
                (I, ic, J, jc), (bvk_ncells, nctri, bvk_ncells, nctrj))
            bvk_nctri = bvk_ncells * nctri
            bvk_nctrj = bvk_ncells * nctrj
            c_pair_mask = cupy.zeros(bvk_nctri*bvk_nctrj, dtype=bool)
            c_pair_mask[reduced_pair_idx] = True

            # c_pair_idx indicates the address of the **contracted** pair GTOS
            # within the (li,lj) sub-block. For each shell-pair, there are
            # nfij elements. Note, the nfij elements are sorted as [nfj,nfi]
            # (in F-order) while the shell indices within the c_pair_idx are
            # composed as i*nbas+j (in C-order). c_pair_idx points to the
            # address of the first element.
            c_pair_idx = cupy.where(c_pair_mask)[0]
            n_ctr_pairs = len(c_pair_idx)

            # pair_mapping maps the primitive pair to the contracted pair
            pair_mapping_lookup = cupy.empty(bvk_nctri*bvk_nctrj, dtype=np.int32)
            pair_mapping_lookup[c_pair_idx] = cupy.arange(n_ctr_pairs)
            pair_mapping = asarray(pair_mapping_lookup[reduced_pair_idx], dtype=np.int32)
            #li, lj = self.cell0_uniq_l[[cpi, cpj]]
            li = self.cell0_uniq_li[cpi]
            lj = self.cell0_uniq_lj[cpj]
            log.timer_debug1(f'pair_mapping [{li},{lj}]', *t1)

            return img_idx, offsets, bas_ij, pair_mapping, c_pair_idx
        return gen_img_idx

    def make_img_idx_cache(self, cutoff=None, ij_tasks=None):
        img_idx_cache = {}
        gen_img_idx = self.generate_img_idx(cutoff)

        li_counts = self.cell0_prim_li_counts
        lj_counts = self.cell0_prim_lj_counts

        if ij_tasks is None:
            ij_tasks = ((i, j) for i in range(len(li_counts)) for j in range(len(lj_counts)))

        for cpi, cpj in ij_tasks:
            #if l_counts[cpi] == 0 or l_counts[cpj] == 0:
            if li_counts[cpi] == 0 or lj_counts[cpj] == 0:
                continue
            img_idx_cache[cpi, cpj] = gen_img_idx(cpi, cpj)

        return img_idx_cache

    def int3c2e_evaluator(self, verbose=None, img_idx_cache=None):
        log = logger.new_logger(self.cell, verbose)
        if self.int3c2e_envs is None:
            self.build(verbose=verbose)
        auxcell = self.sorted_auxcell
        bvkcell = self.bvkcell
        l_ctr_aux_offsets = self.l_ctr_aux_offsets
        aux_loc = auxcell.ao_loc
        naux = aux_loc[auxcell.nbas]
        _atm_cpu = self._atm_cpu
        _bas_cpu = self._bas_cpu
        _env_cpu = self._env_cpu

        li_counts = self.cell0_prim_li_counts
        lj_counts = self.cell0_prim_lj_counts
        p_shell_li_offsets = np.append(0, np.cumsum(li_counts))
        p_shell_lj_offsets = np.append(0, np.cumsum(lj_counts))

        uniq_li = self.cell0_uniq_li
        uniq_lj = self.cell0_uniq_lj
        nfcart_i = (uniq_li + 1) * (uniq_li + 2) // 2
        nfcart_j = (uniq_lj + 1) * (uniq_lj + 2) // 2

        kern = libpbc.fill_int3c2e

        if img_idx_cache is None:
            img_idx_cache = self.make_img_idx_cache()

        def evaluate_j3c(cpi, cpj, buf_j3c=None):

            li = uniq_li[cpi]
            lj = uniq_lj[cpj]
            
            if li_counts[cpi] == 0 or lj_counts[cpj] == 0:
                return cupy.empty(0, dtype=np.int32), cupy.empty((naux, 0))

            ish0, ish1 = p_shell_li_offsets[cpi:cpi+2]
            jsh0, jsh1 = p_shell_lj_offsets[cpj:cpj+2]

            img_idx, img_offsets, bas_ij_idx, pair_mapping, c_pair_idx = img_idx_cache[cpi, cpj]

            img_idx = asarray(img_idx)
            img_offsets = asarray(img_offsets)
            bas_ij_idx = asarray(bas_ij_idx)
            pair_mapping = asarray(pair_mapping)
            #nfij = nfcart[cpi] * nfcart[cpj]
            nfij = nfcart_i[cpi] * nfcart_j[cpj]

            # Note the storage order for ij_pair: i takes the smaller stride.
            n_ctr_pairs = len(c_pair_idx)
            n_prim_pairs = len(bas_ij_idx)
            if n_prim_pairs == 0:
                return cupy.empty(0, dtype=np.int32), cupy.empty((naux, 0))

            # eri3c is sorted as (naux, nfj, nfi, n_ctr_pairs)
            if buf_j3c is None:
                eri3c = cupy.zeros((naux, nfij*n_ctr_pairs))
            else:
                eri3c = buf_j3c[:naux*nfij*n_ctr_pairs].reshape(naux, nfij*n_ctr_pairs) #cupy.zeros((naux, nfij*n_ctr_pairs))
                eri3c.fill(0.0)

            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
                shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
                k0 = aux_loc[ksh0]
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(li, lj, lk)
                log.debug2(f'prim_pairs={n_prim_pairs} int3c2e_scheme for %s: %s', lll, scheme)
                

                # Call the kernel with clean inputs
                err = kern(
                    ctypes.cast(eri3c[k0:].data.ptr, ctypes.c_void_p),
                    ctypes.byref(self.int3c2e_envs),
                    (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(*shls_slice),
                    ctypes.c_int(naux),
                    ctypes.c_int(n_prim_pairs),
                    ctypes.c_int(n_ctr_pairs),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(bvkcell.natm),
                    _bas_cpu.ctypes, ctypes.c_int(bvkcell.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
            return c_pair_idx, eri3c
        return evaluate_j3c

    def int3c2e_generator(self, verbose=None, img_idx_cache=None, buf_j3c=None):

        log = logger.new_logger(self.cell, verbose)
        cput0 = log.init_timer()
        evaluate = self.int3c2e_evaluator(verbose, img_idx_cache)
        t1 = log.timer('initialize int3c2e_kernel', *cput0)
        timing_collection = {}
        kern_counts = 0

        nbatch_i = len(self.cell0_prim_li_counts)
        nbatch_j = len(self.cell0_prim_lj_counts)
        ij_tasks = ((i, j) for i in range(nbatch_i) for j in range(nbatch_j))
        for cpi, cpj in ij_tasks:
            c_pair_idx, eri3c = evaluate(cpi, cpj, buf_j3c=buf_j3c)
            li = self.cell0_uniq_li[cpi]
            lj = self.cell0_uniq_lj[cpj]

            if len(c_pair_idx) == 0:
                continue
            if log.verbose >= logger.DEBUG1:
                ll = f'{ANGULAR[li]}{ANGULAR[lj]}'
                t1, t1p = log.timer_debug1(f'processing {ll}, pairs={len(c_pair_idx)}', *t1), t1
                if ll not in timing_collection:
                    timing_collection[ll] = 0
                timing_collection[ll] += t1[1] - t1p[1]
                kern_counts += 1
            yield cpi, cpj, c_pair_idx, eri3c

        if log.verbose >= logger.DEBUG1:
            log.timer('int3c2e', *cput0)
            for ll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', ll, t)

    def int3c2e_slice_generator(self, verbose=None, img_idx_cache=None, 
                                ij_tasks=None):
        log = logger.new_logger(self.cell, verbose)

        if img_idx_cache is None:
            img_idx_cache = self.make_img_idx_cache(ij_tasks=ij_tasks)

        cput0 = log.init_timer()
        evaluate = self.int3c2e_evaluator(verbose, img_idx_cache)
        t1 = log.timer('initialize int3c2e_kernel', *cput0)
        
        def gen_int3c2e_slice(cpi, cpj, buf_j3c=None):
            return evaluate(cpi, cpj, buf_j3c=buf_j3c)

        return gen_int3c2e_slice

    def int3c2e_kernel(self, verbose=None, img_idx_cache=None):
        raise NotImplementedError(
            'The entire int3c2e tensor evaluated in one kernel is not supported')



def sr_aux_e2(cell, auxcell, omega, kpts=None, bvk_kmesh=None, j_only=False):
    r'''
    Short-range 3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    if bvk_kmesh is None and kpts is not None:
        if j_only:
            # Coulomb integrals requires smaller kmesh to converge finite-size effects
            bvk_kmesh = kpts_to_kmesh(cell, kpts)
        else:
            # The remote images may contribute to certain k-point mesh,
            # contributing to the finite-size effects in exchange matrix.
            rcut = estimate_rcut(cell, auxcell, omega).max()
            bvk_kmesh = kpts_to_kmesh(cell, kpts, rcut=rcut)

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, bvk_kmesh).build()
    nao = cell.nao
    naux = int3c2e_opt.aux_coeff.shape[1]

    gamma_point = kpts is None or (kpts.ndim == 1 and is_zero(kpts))
    if gamma_point:
        out = cupy.zeros((nao, nao, naux))
        nL = nkpts = 1
    else:
        kpts = np.asarray(kpts).reshape(-1, 3)
        expLk = cupy.exp(1j*cupy.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
        nL, nkpts = expLk.shape
        if j_only:
            expLLk = contract('Mk,Lk->MLk', expLk.conj(), expLk)
            expLLk = expLLk.view(np.float64).reshape(nL,nL,nkpts,2)
            out = cupy.zeros((nkpts, nao, nao, naux), dtype=np.complex128)
        else:
            out = cupy.zeros((nkpts, nkpts, nao, nao, naux), dtype=np.complex128)

    c_li_counts = np.asarray(int3c2e_opt.cell0_ctr_li_counts)
    c_lj_counts = np.asarray(int3c2e_opt.cell0_ctr_lj_counts)
    uniq_li = np.asarray(int3c2e_opt.cell0_uniq_li)
    uniq_lj = np.asarray(int3c2e_opt.cell0_uniq_lj)

    if cell.cart:
        nf_i = (uniq_li + 1) * (uniq_li + 2) // 2
        nf_j = (uniq_lj + 1) * (uniq_lj + 2) // 2
    else:
        nf_i = uniq_li * 2 + 1
        nf_j = uniq_lj * 2 + 1
    c_aoi_offsets = np.append(0, np.cumsum(c_li_counts*nf_i))
    c_aoj_offsets = np.append(0, np.cumsum(c_lj_counts*nf_j))

    lmax = cell._bas[:,ANG_OF].max()
    c2s = [cart2sph_by_l(l) for l in range(lmax+1)]
    

    aux_coeff = asarray(int3c2e_opt.aux_coeff)

    for cpi, cpj, c_pair_idx, compressed_eri3c in int3c2e_opt.int3c2e_generator():

        li = uniq_li[cpi]
        lj = uniq_lj[cpj]
        i0, i1 = c_aoi_offsets[cpi:cpi+2]
        j0, j1 = c_aoj_offsets[cpj:cpj+2]
        
        nctri = c_li_counts[cpi]
        nctrj = c_lj_counts[cpj]

        nfi = (li+1)*(li+2)//2
        nfj = (lj+1)*(lj+2)//2
        nfij = nfi * nfj
        n_pairs = len(c_pair_idx)

        compressed_eri3c = compressed_eri3c.reshape(-1,nfij*n_pairs)
        compressed_eri3c = compressed_eri3c.T.dot(aux_coeff)
        if not cell.cart: # cart -> sph
            compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs,naux)
            '''compressed_eri3c = contract('qj,qpmk->jpmk', c2s[lj], compressed_eri3c)
            compressed_eri3c = contract('pi,jpmk->jimk', c2s[li], compressed_eri3c)'''
            if li > 1 or lj > 1:
                c2slji = cupy.einsum('qj, pi -> jiqp', c2s[lj], c2s[li])
                compressed_eri3c = cupy.dot(c2slji.reshape(-1, nfj*nfi), compressed_eri3c.reshape(nfj*nfi, -1))
            nfi = li * 2 + 1
            nfj = lj * 2 + 1

        ni = i1 - i0
        nj = j1 - j0
        ish, jsh = divmod(c_pair_idx, nL*nctrj)
        eri3c = cupy.zeros((nL*nctri,nfi, nL*nctrj,nfj, naux))
        compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs,naux)
        eri3c[ish,:,jsh] = compressed_eri3c.transpose(2,1,0,3)
        '''if i0 == j0:
            eri3c[jsh,:,ish] = compressed_eri3c.transpose(2,0,1,3)'''
        eri3c = eri3c.reshape(nL,ni,nL,nj,naux)
        compressed_eri3c = None

        i = int3c2e_opt.ao_idx[i0:i1]
        j = int3c2e_opt.ao_idx[j0:j1]

        if gamma_point:
            eri3c = eri3c.reshape(ni,nj,naux)
            out[i[:,None],j] = eri3c
            #out[i0:i1, j0:j1] = eri3c
            '''if i0 != j0:
                out[j[:,None],i] = eri3c.transpose(1,0,2)'''
        elif j_only:
            eri3c = contract('MLkz,MpLqr->kpqrz', expLLk, eri3c)
            eri3c = eri3c.view(np.complex128)[...,0]
            out[:,i[:,None],j] = eri3c
        else:
            expLkz = expLk.view(np.float64).reshape(nL,nkpts,2)
            eri3c = contract('Lkz,MpLqr->Mkpqrz', expLkz, eri3c)
            eri3c = eri3c.view(np.complex128)[...,0]
            eri3c = contract('Mk,Mlpqr->klpqr', expLk.conj(), eri3c)
            out[:,:,i[:,None],j] = eri3c
        eri3c = None

    return out


def sr_aux_e2_nuc(cell, auxcell, omega, kpts=None, bvk_kmesh=None, j_only=False):
    if bvk_kmesh is None and kpts is not None:
        if j_only:
            # Coulomb integrals requires smaller kmesh to converge finite-size effects
            bvk_kmesh = kpts_to_kmesh(cell, kpts)

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, bvk_kmesh, get_nuc=True).build()

    nao = cell.nao

    gamma_point = kpts is None or (kpts.ndim == 1 and is_zero(kpts))

    if gamma_point:
        out = cupy.zeros((nao, nao))
        nL = nkpts = 1
    else:
        kpts = np.asarray(kpts).reshape(-1, 3)
        expLk = cupy.exp(1j*cupy.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
        nL, nkpts = expLk.shape
        if j_only:
            expLLk = contract('Mk,Lk->MLk', expLk.conj(), expLk)
            expLLk = expLLk.view(np.float64).reshape(nL,nL,nkpts,2)
            #out = cupy.zeros((nkpts, nao, nao), dtype=np.complex128)
            out = cupy.zeros((nkpts, nao, nao), dtype=np.complex128)

    c_li_counts = np.asarray(int3c2e_opt.cell0_ctr_li_counts)
    c_lj_counts = np.asarray(int3c2e_opt.cell0_ctr_lj_counts)
    uniq_li = np.asarray(int3c2e_opt.cell0_uniq_li)
    uniq_lj = np.asarray(int3c2e_opt.cell0_uniq_lj)

    if cell.cart:
        nf_i = (uniq_li + 1) * (uniq_li + 2) // 2
        nf_j = (uniq_lj + 1) * (uniq_lj + 2) // 2
    else:
        nf_i = uniq_li * 2 + 1
        nf_j = uniq_lj * 2 + 1
    c_aoi_offsets = np.append(0, np.cumsum(c_li_counts*nf_i))
    c_aoj_offsets = np.append(0, np.cumsum(c_lj_counts*nf_j))

    lmax = cell._bas[:,ANG_OF].max()
    c2s = [cart2sph_by_l(l) for l in range(lmax+1)]
    
    ncaux = auxcell.nao_nr(cart=True)
    #nsaux = auxcell.nao_nr(cart=False)

    charges_gpu = cupy.asarray(-cell.atom_charges())
    #aux_coeff_pq = asarray(int3c2e_opt.aux_coeff)
    aux_coeff_gpu = cupy.dot(int3c2e_opt.aux_coeff, charges_gpu.reshape(-1, 1))
    int3c2e_opt.aux_coeff = None

    ij_tasks = ((i, j) for i in range(len(c_li_counts)) for j in range(len(c_lj_counts)))

    #sr_int_kernel = int3c2e_opt.int3c2e_slice_generator(ij_tasks=ij_tasks)

    for cpi, cpj, c_pair_idx, compressed_eri3c in int3c2e_opt.int3c2e_generator():
    #for cpi, cpj in ij_tasks:
    #    c_pair_idx, compressed_eri3c = sr_int_kernel(cpi, cpj)

        li = uniq_li[cpi]
        lj = uniq_lj[cpj]
        i0, i1 = c_aoi_offsets[cpi:cpi+2]
        j0, j1 = c_aoj_offsets[cpj:cpj+2]
        
        nctri = c_li_counts[cpi]
        nctrj = c_lj_counts[cpj]

        nfi = (li+1)*(li+2)//2
        nfj = (lj+1)*(lj+2)//2
        nfij = nfi * nfj
        n_pairs = len(c_pair_idx)

        compressed_eri3c = compressed_eri3c.reshape(-1,nfij*n_pairs)
        #compressed_eri3c = compressed_eri3c.T.dot(aux_coeff)
        compressed_eri3c = compressed_eri3c.T.dot(aux_coeff_gpu)
        if not cell.cart: # cart -> sph
            #compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs,naux)
            compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs)
            if li > 1 or lj > 1:
                c2slji = cupy.einsum('qj, pi -> jiqp', c2s[lj], c2s[li])
                compressed_eri3c = cupy.dot(c2slji.reshape(-1, nfj*nfi), compressed_eri3c.reshape(nfj*nfi, -1))
            nfi = li * 2 + 1
            nfj = lj * 2 + 1

        ni = i1 - i0
        nj = j1 - j0
        ish, jsh = divmod(c_pair_idx, nL*nctrj)
        #eri3c = cupy.zeros((nL*nctri,nfi, nL*nctrj,nfj, naux))
        #compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs,naux)
        #eri3c[ish,:,jsh] = compressed_eri3c.transpose(2,1,0,3)

        eri3c = cupy.zeros((nL*nctri, nfi, nL*nctrj, nfj))
        compressed_eri3c = compressed_eri3c.reshape(nfj,nfi,n_pairs)
        eri3c[ish,:,jsh] = compressed_eri3c.transpose(2,1,0)
        
        '''if i0 == j0:
            eri3c[jsh,:,ish] = compressed_eri3c.transpose(2,0,1,3)'''
        #eri3c = eri3c.reshape(nL,ni,nL,nj,naux)
        eri3c = eri3c.reshape(nL,ni,nL,nj)
        compressed_eri3c = None

        i = int3c2e_opt.ao_idx[i0:i1]
        j = int3c2e_opt.ao_idx[j0:j1]

        if gamma_point:
            #eri3c = eri3c.reshape(ni,nj,naux)
            #out[i[:,None],j] = eri3c
            eri3c = eri3c.reshape(ni,nj)
            out[i[:,None],j] = eri3c

            #out[i0:i1, j0:j1] = eri3c
            '''if i0 != j0:
                out[j[:,None],i] = eri3c.transpose(1,0,2)'''
        elif j_only:
            eri3c = contract('MLkz,MpLqr->kpqrz', expLLk, eri3c)
            eri3c = eri3c.view(np.complex128)[...,0]
            out[:,i[:,None],j] = eri3c

        eri3c = None

    return out

def _weighted_coulG_LR(cell, Gv, omega, kws, kpt=np.zeros(3)):
    coulG = pbctools.get_coulG(cell, kpt, exx=False, Gv=Gv, omega=abs(omega))
    coulG *= kws
    if is_zero(kpt):
        assert Gv[0].dot(Gv[0]) == 0
        coulG[0] -= np.pi / omega**2 / cell.vol
    return asarray(coulG)

# The long-range part of the cderi for gamma point. The resultant 3-index tensor
# is compressed.



def _solve_cderi(cd_j2c, j3c, j2ctag):
    if j2ctag == 'ED':
        return contract('rL,pqr->Lpq', cd_j2c, j3c)
    else:
        nao, naux = j3c.shape[1:3]
        j3c = solve_triangular(cd_j2c, j3c.reshape(-1,naux).T, lower=True)
        return j3c.reshape(naux,nao,nao)

def _scale_sp_ctr_coeff(mol):
    # Match normalization factors of s, p functions in libcint
    _env = mol._env.copy()
    ls = mol._bas[:,ANG_OF]
    ptr, idx = np.unique(mol._bas[:,PTR_COEFF], return_index=True)
    ptr = ptr[ls[idx] < 2]
    idx = idx[ls[idx] < 2]
    fac = ((ls[idx]*2+1) / (4*np.pi)) ** .5
    nprim = mol._bas[idx,NPRIM_OF]
    nctr = mol._bas[idx,NCTR_OF]
    for p, n, f in zip(ptr, nprim*nctr, fac):
        _env[p:p+n] *= f
    return _env

def ft_ao_scheme(cell, li, lj, nGv):#, shm_size=SHM_SIZE):
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    gout_size = nfi * nfj
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)

    g_size = (li+1)*(lj+1)
    unit = g_size*3
    device = cupy.cuda.Device()
    shm_size = device.attributes['MaxSharedMemoryPerBlock']
    #shm_size = 0.8*device.attributes['MaxSharedMemoryPerMultiprocessor']
    nGv_nsp_max = shm_size//(unit*16)
    nGv_nsp_max = _nearest_power2(nGv_nsp_max)
    nGv_max = min(nGv_nsp_max, THREADS//gout_stride, 64)

    # gout_stride*nGv_per_block >= 32 is a must due to syncthreads in CUDA kernel
    nGv_per_block = max(32//gout_stride, 1)

    # Test nGv_per_block in 1..nGv_max, find the case of minimal idle threads
    idle_min = nGv_max
    nGv_test = nGv_per_block
    while nGv_test <= nGv_max:
        idle = (-nGv) % nGv_test
        if idle <= idle_min:
            idle_min = idle
            nGv_per_block = nGv_test
        nGv_test *= 2

    sp_blocks = THREADS // (gout_stride * nGv_per_block)
    # the nGv * sp_blocks restrictrions due to shared memory size
    sp_blocks = min(sp_blocks, nGv_nsp_max // nGv_per_block)
    gout_stride = THREADS // (nGv_per_block * sp_blocks)

    return nGv_per_block, gout_stride, sp_blocks


def sort_cell_by_l(cell):
    cell, _ = basis_seg_contraction(cell, sparse_coeff=True)

    c_ls = cell._bas[:,ANG_OF]
    lmax = c_ls.max()
    sorted_cell = cell.copy()
    sorted_idx = np.hstack([np.where(c_ls==l)[0] for l in range(lmax+1)])

    sorted_cell._bas = cell._bas[sorted_idx]
    sorted_cell._bas[:,NCTR_OF] = 1

    ao_loc_sph = np.append(0, (c_ls * 2 + 1).cumsum())
    ao_idx_sph = np.array_split(np.arange(ao_loc_sph[-1]), ao_loc_sph[1:-1])
    sorted_ao_idx_sph = np.hstack([ao_idx_sph[i] for i in sorted_idx])

    ao_loc_cart = np.append(0, ((c_ls + 1) * (c_ls + 2) // 2).cumsum())
    ao_idx_cart = np.array_split(np.arange(ao_loc_cart[-1]), ao_loc_cart[1:-1])
    sorted_ao_idx_cart = np.hstack([ao_idx_cart[i] for i in sorted_idx])

    uniq_l_ctr, l_ctr_counts = np.unique(sorted_cell._bas[:,ANG_OF], 
                                         return_counts=True)
    
    return sorted_cell, sorted_ao_idx_sph, sorted_ao_idx_cart, uniq_l_ctr, l_ctr_counts

class AFTIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
        ('bvk_ncells', ctypes.c_uint16),
        ('nimgs', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('img_coords', ctypes.c_void_p),
    ]

class FTOpt:
    def __init__(self, cell, kpts=None, bvk_kmesh=None):
        self.cell = cell
        
        (self.sorted_cell, self.ao_idx_sph, 
         self.ao_idx_cart, uniq_l_ctr, l_ctr_counts) = sort_cell_by_l(cell)
        
        self.ori_uniq_l_ctr = uniq_l_ctr
        self.ori_ctr_l_counts = l_ctr_counts

        if bvk_kmesh is None:
            if kpts is None or is_zero(kpts):
                bvk_kmesh = np.ones(3, dtype=int)
            else:
                bvk_kmesh = kpts_to_kmesh(self.sorted_cell, kpts)
        self.bvk_kmesh = bvk_kmesh
        self.kpts = kpts

        self.aft_envs = None
        self.bvk_cell = None

    def build(self, ctr_li_counts=None, ctr_lj_counts=None, verbose=None):
        log = logger.new_logger(self.cell, verbose)
        cell = self.sorted_cell
        bvk_kmesh = self.bvk_kmesh

        if np.prod(bvk_kmesh) == 1:
            bvkcell = cell
        else:
            bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

        Ls = cupy.asarray(bvkcell.get_lattice_Ls())
        Ls = Ls[cupy.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)

        bvk_ncells = np.prod(bvk_kmesh)
        nbas = cell.nbas
        log.debug('bvk_ncells=%d, nbas=%d, nimgs=%d', bvk_ncells, nbas, nimgs)

        _atm = cupy.array(bvkcell._atm)
        _bas = cupy.array(bvkcell._bas)
        _env = cupy.array(_scale_sp_ctr_coeff(bvkcell))
        
        ao_loc = cupy.array(bvkcell.ao_loc_nr(cart=True))
        aft_envs = AFTIntEnvVars(
            cell.natm, cell.nbas, bvk_ncells, nimgs, _atm.data.ptr,
            _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr, Ls.data.ptr
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        aft_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)
        self.aft_envs = aft_envs

        init_constant(cell)

        '''_, ctr_li_counts = _split_ctr_l_groups(self.ori_uniq_l_ctr, self.ori_ctr_l_counts, 10)
        _, ctr_lj_counts = _split_ctr_l_groups(self.ori_uniq_l_ctr, self.ori_ctr_l_counts, 10)'''

        if ctr_li_counts is None:
            ctr_li_counts = self.ori_ctr_l_counts
            self.uniq_li_ctr = self.ori_uniq_l_ctr
        else:
            self.uniq_li_ctr = get_uniq_l_from_counts(self.ori_uniq_l_ctr, self.ori_ctr_l_counts, 
                                                      ctr_li_counts)

        if ctr_lj_counts is None:
            ctr_lj_counts = self.ori_ctr_l_counts
            self.uniq_lj_ctr = self.ori_uniq_l_ctr
        else:
            self.uniq_lj_ctr = get_uniq_l_from_counts(self.ori_uniq_l_ctr, self.ori_ctr_l_counts, 
                                                      ctr_lj_counts)

        self.li_ctr_offsets = np.append(0, np.cumsum(ctr_li_counts))
        self.lj_ctr_offsets = np.append(0, np.cumsum(ctr_lj_counts))

        return self

    def make_img_idx_cache(self, permutation_symmetry, verbose=None, ij_tasks=None):
        log = logger.new_logger(self.cell, verbose)
        if self.aft_envs is None:
            self.build(verbose=verbose)

        cell = self.sorted_cell
        nbas = cell.nbas
        '''l_ctr_offsets = self.l_ctr_offsets
        uniq_l = self.uniq_l_ctr[:,0]
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)
        '''
        li_ctr_offsets = self.li_ctr_offsets
        lj_ctr_offsets = self.lj_ctr_offsets
        uniq_li = self.uniq_li_ctr #[:,0]
        uniq_lj = self.uniq_lj_ctr #[:,0]
        li_symb = [lib.param.ANGULAR[l] for l in uniq_li]
        lj_symb = [lib.param.ANGULAR[l] for l in uniq_lj]
        

        bvk_kmesh = self.bvk_kmesh
        if bvk_kmesh is None:
            bvk_ncells = 1
        else:
            bvk_ncells = np.prod(bvk_kmesh)

        rcut = cell.rcut
        vol = cell.vol
        cell_exp, _, cell_l = most_diffused_pgto(cell)
        lsum = cell_l * 2 + 1
        rad = vol**(-1./3) * rcut + 1
        surface = 4*np.pi * rad**2
        lattice_sum_factor = 2*np.pi*rcut*lsum/(vol*cell_exp*2) + surface
        cutoff = cell.precision / lattice_sum_factor
        log_cutoff = math.log(cutoff)
        log.debug1('ft_ao min_exp=%g cutoff=%g', cell_exp, cutoff)

        exps, cs = extract_pgto_params(cell, 'diffused')
        exps = cupy.asarray(exps, dtype=np.float32)
        log_coeff = cupy.log(abs(cupy.asarray(cs, dtype=np.float32)))

        '''if permutation_symmetry:
            # symmetry between ish and jsh can be utilized. The triu part is excluded
            # from computation.
            ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        else:
            ij_tasks = itertools.product(range(n_groups), range(n_groups))'''
        
        if ij_tasks is None:
            n_groups_i = np.count_nonzero(uniq_li <= LMAX)
            n_groups_j = np.count_nonzero(uniq_lj <= LMAX)
            ij_tasks = list(itertools.product(range(n_groups_i), range(n_groups_j)))



        bas_ij_cache = {}
        for i, j in ij_tasks:
            '''ll_pattern = f'{l_symb[i]}{l_symb[j]}'
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]'''

            ll_pattern = f'{li_symb[i]}{lj_symb[j]}'
            ish0, ish1 = li_ctr_offsets[i], li_ctr_offsets[i+1]
            jsh0, jsh1 = lj_ctr_offsets[j], lj_ctr_offsets[j+1]

            nish = ish1 - ish0
            njsh = jsh1 - jsh0
            img_counts = cupy.zeros((nish*bvk_ncells*njsh), dtype=np.int32)
            err = libpbc.overlap_img_counts(
                ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(self.aft_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff),
                ctypes.c_int(int(permutation_symmetry)))
            if err != 0:
                raise RuntimeError(f'{ll_pattern} overlap_img_counts failed')
            bas_ij = cupy.asarray(cupy.where(img_counts > 0)[0], dtype=np.int32)
            n_pairs = len(bas_ij)
            if n_pairs == 0:
                bas_ij_cache[i, j] = (bas_ij, None, None)
                continue

            # Sort according to the number of images. In the CUDA kernel,
            # shell-pairs that have closed number of images are processed on
            # the same SM processor, ensuring the best parallel execution.
            counts_sorting = (-img_counts[bas_ij]).argsort()
            bas_ij = bas_ij[counts_sorting]
            img_counts = img_counts[bas_ij]
            img_offsets = cupy.empty(n_pairs+1, dtype=np.int32)
            img_offsets[0] = 0
            cupy.cumsum(img_counts, out=img_offsets[1:])
            tot_imgs = int(img_offsets[n_pairs])
            img_idx = cupy.empty(tot_imgs, dtype=np.int32)
            err = libpbc.overlap_img_idx(
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_pairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(self.aft_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError(f'{ll_pattern} overlap_img_idx failed')
            img_counts = counts_sorting = None

            # bas_ij stores the non-negligible primitive-pair indices.
            ish, J, jsh = cupy.unravel_index(bas_ij, (nish, bvk_ncells, njsh))
            ish += ish0
            jsh += jsh0
            bas_ij = cupy.ravel_multi_index((ish, J, jsh), (nbas, bvk_ncells, nbas))
            bas_ij = cupy.asarray(bas_ij, dtype=np.int32)
            bas_ij_cache[i, j] = (bas_ij, img_offsets, img_idx)
            log.debug1('task (%d, %d), n_pairs=%d', i, j, n_pairs)
        
        return bas_ij_cache
    
    def gamma_ft_slice_kernel(self, ij_tasks=None, verbose=None):
        r'''
        Generate the analytical fourier transform kernel for AO products

        \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

        By default, the output tensor is saved in the shape [nGv, nao, nao] for
        single k-point case and [nkpts, nGv, nao, nao] for multiple k-points
        '''
        log = logger.new_logger(self.cell, verbose)
        if self.aft_envs is None:
            self.build(verbose=verbose)

        cell = self.sorted_cell
        ao_loc = cell.ao_loc_nr()
        ao_loc_cart = cell.ao_loc_nr(cart=True)
        uniq_li = self.uniq_li_ctr #[:,0]
        uniq_lj = self.uniq_lj_ctr #[:,0]
        li_symb = [lib.param.ANGULAR[l] for l in uniq_li]
        lj_symb = [lib.param.ANGULAR[l] for l in uniq_lj]
        li_ctr_offsets = self.li_ctr_offsets
        lj_ctr_offsets = self.lj_ctr_offsets

        kern = libpbcint.calFtAoPair

        img_idx_cache = self.make_img_idx_cache(False, log, ij_tasks=ij_tasks)

        #bvk_kmesh = self.bvk_kmesh
        #bvk_ncells = np.prod(bvk_kmesh)

        lmax = cell._bas[:,ANG_OF].max()
        c2s = [cart2sph_by_l(l) for l in range(lmax+1)]

        def _ft_sub(cpi, cpj, Gv, q, img_idx_cache, 
                    outR_slice=None, outI_slice=None):
            # Padding zeros, allowing idle threads to access these data
            '''GvT = cupy.asarray(Gv.T) + cupy.asarray(q)[:,None]
            GvT = cupy.append(GvT.ravel(), cupy.zeros(THREADS))'''

            # Only applicable to gamma!
            GvT = cupy.empty(Gv.size + THREADS)
            d0, d1 = Gv.shape
            GvT[:Gv.size].reshape(d1, d0)[:] = Gv.T
            GvT[Gv.size:].fill(0.0)

            nGv = len(Gv)

            #for i, j in img_idx_cache:
            bas_ij, img_offsets, img_idx = img_idx_cache[cpi, cpj]
            npairs = len(bas_ij)
            if npairs == 0:
                return None

            li = uniq_li[cpi]
            lj = uniq_lj[cpj]
            
            ll_pattern = f'{li_symb[cpi]}{lj_symb[cpj]}'
            ish0, ish1 = li_ctr_offsets[cpi], li_ctr_offsets[cpi+1]
            jsh0, jsh1 = lj_ctr_offsets[cpj], lj_ctr_offsets[cpj+1]

            cal0, cal1 = ao_loc_cart[[ish0, ish1]]
            cbe0, cbe1 = ao_loc_cart[[jsh0, jsh1]]
            naoi_cart = cal1 - cal0
            naoj_cart = cbe1 - cbe0
            #out_slice = cupy.zeros((bvk_ncells, naoi, naoj, nGv), dtype=np.complex128)
            #out_slice = cupy.zeros((naoi, naoj, nGv), dtype=np.complex128)
            if outR_slice is None:
                outR_slice = cupy.empty((naoi_cart, naoj_cart, nGv))
            else:
                outR_slice.fill(0.0)

            if outI_slice is None:
                outI_slice = cupy.empty((naoi_cart, naoj_cart, nGv))
            else:
                outI_slice.fill(0.0)

            scheme = ft_ao_scheme(cell, li, lj, nGv)
            #print('ft_ao_scheme for %s: %s'%(ll_pattern, scheme))

            err = kern(
                ctypes.cast(outR_slice.data.ptr, ctypes.c_void_p),
                ctypes.cast(outI_slice.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), 
                ctypes.byref(self.aft_envs), (ctypes.c_int*3)(*scheme),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.c_int(npairs), ctypes.c_int(nGv),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                cell._atm.ctypes, ctypes.c_int(cell.natm),
                cell._bas.ctypes, ctypes.c_int(cell.nbas), cell._env.ctypes)
            if err != 0:
                raise RuntimeError(f'build_ft_aopair kernel for {ll_pattern} failed')
            
            return outR_slice, outI_slice

        def ft_slice_kernel(cpi, cpj, Gv, q=np.zeros(3), 
                            outR_slice=None, outI_slice=None):
            '''
            Analytical FT for orbital products. The output tensor has the shape
            [nao, nao, nGv]
            '''
            assert q.ndim == 1
            nGv = len(Gv)
            assert nGv > 0            

            return _ft_sub(cpi, cpj, Gv, q, img_idx_cache,
                           outR_slice, outI_slice)#.transpose(0,3,1,2)
        return ft_slice_kernel

def print_test(a, info=""):
    print("%s %.6e %.6e %.6e"%(info, a.min(), a.max(), a.mean()))


def generate_ft_ao(cell, Gv, kpt=np.zeros(3), coeff=None):
    '''Analytical Fourier transform basis functions on Gv grids.
    '''
    if coeff is None: # cell is not sorted
        sorted_cell, coeff, _, _ = group_basis(cell, tile=1)
    else:
        sorted_cell = cell

    _atm = cupy.array(sorted_cell._atm, dtype=cupy.int32)
    _bas = cupy.array(sorted_cell._bas, dtype=cupy.int32)
    _env = cupy.array(_scale_sp_ctr_coeff(sorted_cell))
    ao_loc_cpu = sorted_cell.ao_loc
    ao_loc_gpu = cupy.array(ao_loc_cpu, dtype=cupy.int32)

    bas_gpu = [_atm, _bas, _env, ao_loc_gpu]

    envs = AFTIntEnvVars(
        sorted_cell.natm, sorted_cell.nbas, 1, 1, _atm.data.ptr,
        _bas.data.ptr, _env.data.ptr, ao_loc_gpu.data.ptr, 0,
    )

    nao_cart, nao_sph = coeff.shape

    ngrids = len(Gv)

    def cal_ft_ao(p0=0, p1=ngrids, buffers=None):
        Gv_slice = Gv[p0:p1]
        GvT = asarray(np.append((Gv_slice.T + kpt[:,None]).ravel(), np.zeros(THREADS)))
        nGv = p1 - p0
    
        if buffers is None:
            outR_cart = cupy.zeros((nao_cart, nGv))
            outI_cart= cupy.zeros((nao_cart, nGv))
            outR_sph = cupy.empty((nGv, nao_sph))
        else:
            buf_cart_sph, buf_cart1 = buffers

            gaux_cart_size = nao_cart * nGv
            gaux_sph_size = nao_sph * nGv
            outR_cart = buf_cart_sph[:gaux_cart_size].reshape(nao_cart, nGv)
            idx0 = buf_cart_sph.size - gaux_sph_size
            outR_sph = buf_cart_sph[idx0:].reshape(nGv, nao_sph)

            outI_cart = buf_cart1[:gaux_cart_size].reshape(nao_cart, nGv)
        
        libpbcint.calFtAo(
            ctypes.cast(outR_cart.data.ptr, ctypes.c_void_p),
            ctypes.cast(outI_cart.data.ptr, ctypes.c_void_p),
            ctypes.byref(envs), ctypes.c_int(nGv),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
            sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
            sorted_cell._env.ctypes
        )

        dgemm_cupy(1, 0, outR_cart, coeff, outR_sph, 1.0, 0.0) # nao_sph, nGv
        outR_cart = None

        if buffers is None:
            outI_sph = cupy.empty((nGv, nao_sph))
        else:
            outI_sph = buf_cart_sph[:gaux_sph_size].reshape(nGv, nao_sph)
        dgemm_cupy(1, 0, outI_cart, coeff, outI_sph, 1.0, 0.0)

        return outR_sph, outI_sph
 
    return cal_ft_ao, bas_gpu
#def cal_auxG():
    

def gamma_lr_int_generator(cell, auxcell, bvk_kmesh, omega, ctr_li_counts=None, 
                           ctr_lj_counts=None, ij_tasks=None, sorted_auxcell=None,  
                           aux_coeff=None, Gaux_buffers=None, verbose=None):
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    
    mesh = cell.symmetrize_mesh(mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)

    ft_opt = FTOpt(cell, bvk_kmesh=bvk_kmesh).build(ctr_li_counts=ctr_li_counts, 
                   ctr_lj_counts=ctr_lj_counts) #ft_ao.FTOpt(cell, bvk_kmesh=bvk_kmesh)
    ft_kern = ft_opt.gamma_ft_slice_kernel(ij_tasks=ij_tasks, verbose=verbose)
    if bvk_kmesh is None:
        bvk_ncells = 1
    else:
        bvk_ncells = np.prod(bvk_kmesh)

    #sorted_auxcell, aux_coeff = group_basis(auxcell, tile=1)[:2]
    #naux = aux_coeff.shape[1]
    naux = auxcell.nao_nr(cart=False)

    kpt = np.zeros(3)

    Gv_gpu = cupy.asarray(Gv)
    kpt_gpu = cupy.zeros(3)

    if sorted_auxcell is not None:
        assert aux_coeff is not None
    else:
        sorted_auxcell, aux_coeff = group_basis(auxcell, tile=1)[:2]
    
    #coeff = asarray(aux_coeff)

    sorted_cell = ft_opt.sorted_cell
    ao_loc_sph = sorted_cell.ao_loc_nr(cart=False)
    ao_loc_cart = sorted_cell.ao_loc_nr(cart=True)

    uniq_li = ft_opt.uniq_li_ctr
    uniq_lj = ft_opt.uniq_lj_ctr
    li_ctr_offsets = ft_opt.li_ctr_offsets
    lj_ctr_offsets = ft_opt.lj_ctr_offsets

    if irank_gpu == 0:
        ft_ao_kernel, bas_gpu = generate_ft_ao(sorted_auxcell, Gv, coeff=aux_coeff)

        '''avail_mem = avail_gpu_mem(unit="B") * .8
        save_gaux = (ngrids * naux * 16 < avail_mem * .25)'''

        logger.info(cell, 'cache auxG')

        ncaux, nsaux = aux_coeff.shape

        if Gaux_buffers is None:
            Gaux_buffers = [cupy.empty((ncaux+nsaux)*ngrids), cupy.empty(ncaux*ngrids)]
        
        Gaux_real, Gaux_imag = ft_ao_kernel(buffers=Gaux_buffers)
        Gaux_buffers = None
        coulG = asarray(_weighted_coulG_LR(auxcell, Gv, omega, kws, kpt))
        Gaux_real *= coulG.reshape(-1, 1)
        Gaux_imag *= coulG.reshape(-1, 1)
        Gaux_imag *= -1

        
        ft_ao_kernel = bas_gpu = coulG = None
    else:
        Gaux_buffer = Gaux_buffers[0]
        Gaux_size = ngrids*naux
        idx0 = Gaux_buffer.size - Gaux_size
        Gaux_real = Gaux_buffer[idx0:].reshape(ngrids, naux)
        Gaux_imag = Gaux_buffer[:Gaux_size].reshape(ngrids, naux)
    
    comm_shm.Barrier()
    # TODO!!!

    def ft_ao_iter(cpi, cpj, feri_buf=None, Gblksize=None):
        ish0, ish1 = li_ctr_offsets[cpi], li_ctr_offsets[cpi+1]
        jsh0, jsh1 = lj_ctr_offsets[cpj], lj_ctr_offsets[cpj+1]
        cal0, cal1 = ao_loc_cart[[ish0, ish1]]
        cbe0, cbe1 = ao_loc_cart[[jsh0, jsh1]]
        ncaoi = cal1 - cal0
        ncaoj = cbe1 - cbe0
        sal0, sal1 = ao_loc_sph[[ish0, ish1]]
        sbe0, sbe1 = ao_loc_sph[[jsh0, jsh1]]
        nsaoi = sal1 - sal0
        nsaoj = sbe1 - sbe0

        if Gblksize is None:
            avail_mem = avail_gpu_mem(unit="B") * .8
            Gblksize = max(16, int(avail_mem/(16*ncaoi*ncaoj*bvk_ncells))//8*8)
            Gblksize = min(Gblksize, ngrids, 16384)
        
        if feri_buf is None:
            lr_buf = cupy.zeros(ncaoi*ncaoj*naux)
            lr_out = lr_buf.reshape(ncaoi, ncaoj, naux)
            pqg_buf_size = max(max(2 * ncaoi*ncaoj*Gblksize, nsaoi*ncaoj*naux), nsaoj*ncaoi*naux)
            pqg_buf = cupy.empty(pqg_buf_size)

        else:
            lr_out_size = ncaoi*ncaoj*naux
            lr_buf = feri_buf[:lr_out_size]
            lr_out = lr_buf.reshape(ncaoi, ncaoj, naux)
            lr_out.fill(0.0)

            pqg_buf = feri_buf[lr_out_size:]

        max_pqg_size = ncaoi * ncaoj * Gblksize
        pqG_real_buf = pqg_buf[:max_pqg_size]
        pqG_imag_buf = pqg_buf[max_pqg_size:2*max_pqg_size]

        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            gaux_real = Gaux_real[p0:p1]
            gaux_imag = Gaux_imag[p0:p1]

            pqG_real = pqG_real_buf[:ncaoi*ncaoj*nGv].reshape(ncaoi, ncaoj, nGv)
            pqG_imag = pqG_imag_buf[:ncaoi*ncaoj*nGv].reshape(ncaoi, ncaoj, nGv)
            ft_kern(cpi, cpj, Gv_gpu[p0:p1], kpt_gpu, outR_slice=pqG_real, outI_slice=pqG_imag) #(naoi, naoj, nGv)

            dgemm_cupy(0, 0, pqG_real.reshape(-1, nGv), gaux_real, lr_out, 1.0, 1.0)
            dgemm_cupy(0, 0, pqG_imag.reshape(-1, nGv), gaux_imag, lr_out, -1.0, 1.0)

        li = uniq_li[cpi]
        lj = uniq_lj[cpj]

        if li > 1:
            out_li_size = nsaoi*ncaoj*naux
            idx0 = pqg_buf.size - out_li_size
            out_li = pqg_buf[idx0:pqg_buf.size].reshape(nsaoi, ncaoj, naux)
            if lj > 1:
                out_lj = lr_buf[:nsaoi*nsaoj*naux].reshape(nsaoi, nsaoj, naux)

        elif lj > 1:
            out_lj_size = nsaoi*nsaoj*naux
            idx0 = pqg_buf.size - out_lj_size
            out_lj = pqg_buf[idx0:pqg_buf.size].reshape(nsaoi, nsaoj, naux)

        if li > 1:
            lr_out = cart2sph(lr_out, axis=0, ang=li, out=out_li)
        
        if lj > 1:
            lr_out = cart2sph(lr_out, axis=1, ang=lj, out=out_lj)

        is_cart_li = li > 1
        is_cart_lj = lj > 1

        if feri_buf is None:
            if not (is_cart_li == is_cart_lj):
                lr_final = lr_buf[:nsaoi*nsaoj*naux].reshape(nsaoi, nsaoj, naux)
                cupy.copyto(lr_final, lr_out)
                lr_out = lr_final

            return lr_out
        else:
            
            if (is_cart_li == is_cart_lj):
                avail_buffer = pqg_buf
            else:
                idx1 = feri_buf.size - nsaoi*nsaoj*naux
                avail_buffer = feri_buf[:idx1]

            return lr_out, avail_buffer


    return ft_ao_iter

def gamma_aux_2e_cuda(int3c2e_opt, with_long_range=True,
                       linear_dep_threshold=LINEAR_DEP_THR, 
                       solve_j3c=True, ij_tasks=None, 
                       feri_mem_f8=None, return_buf=False):

    kmesh = None

    t0 = get_current_time()

    #int3c2e_opt = SRInt3c2eOpt(cell, auxcell, -omega).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    omega = int3c2e_opt.omega

    ncaux = auxcell.nao_nr(cart=True)
    nsaux = auxcell.nao_nr(cart=False)
    nL = nkpts = 1

    c_li_counts = np.asarray(int3c2e_opt.cell0_ctr_li_counts)
    c_lj_counts = np.asarray(int3c2e_opt.cell0_ctr_lj_counts)
    uniq_li = np.asarray(int3c2e_opt.cell0_uniq_li)
    uniq_lj = np.asarray(int3c2e_opt.cell0_uniq_lj)
    if cell.cart:
        nf_i = (uniq_li + 1) * (uniq_li + 2) // 2
        nf_j = (uniq_lj + 1) * (uniq_lj + 2) // 2
    else:
        nf_i = uniq_li * 2 + 1
        nf_j = uniq_lj * 2 + 1
    c_aoi_offsets = np.append(0, np.cumsum(c_li_counts*nf_i))
    c_aoj_offsets = np.append(0, np.cumsum(c_lj_counts*nf_j))

    lmax = cell._bas[:,ANG_OF].max()
    c2s_gpu = [cupy.asarray(cart2sph_by_l(l)) for l in range(lmax+1)]

    sorted_auxcell = int3c2e_opt.sorted_auxcell

    #aux_coeff = asarray(int3c2e_opt.aux_coeff)

    shared_ptrs = []
    #t1 = get_current_time()
    aux_coeff_gpu, aux_coeff_ptr = get_shared_cupy((ncaux, nsaux), numpy_array=int3c2e_opt.aux_coeff)
    shared_ptrs.append(aux_coeff_ptr)
    #accumulate_time(self.t_data, t1)

    
    comm_shm.Barrier()

    ncaux, nsaux = aux_coeff_gpu.shape

    if ij_tasks is None:
        ij_tasks = list((i, j) for i in range(len(c_li_counts)) for j in range(len(c_lj_counts)))

    sr_int_kernel = int3c2e_opt.int3c2e_slice_generator(ij_tasks=ij_tasks)

    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    ngrid = np.prod(cell.symmetrize_mesh(cell.cutoff_to_mesh(ke_cutoff)))

    if feri_mem_f8 is None:
        shared_buf_size = nsaux*ncaux + ngrid*(nsaux+ncaux)
        gpu_memory_f8 = ave_gpu_memory() * 1e6 // 8 - shared_buf_size
        gpu_memory_f8 /= nrank_per_gpu
        feri_mem_f8 = int(0.4 * gpu_memory_f8)
        #raise NotImplementedError("feri_mem_f8 must be specified!")

    #feri_mem_f8 - 
    max_nGv = nsaux


    feri_buf = cupy.empty(feri_mem_f8)

    if with_long_range:
        # Prepare buffers for Gaux
        gaux_buf_gpu, gaux_buf_ptr = get_shared_cupy(((ncaux+nsaux)*ngrid))
        
        #gaux_cart_buf_gpu = cupy.empty((ncaux*ngrid))
        shared_ptrs.append(gaux_buf_ptr)
        lr_int_kernel = gamma_lr_int_generator(cell, auxcell, kmesh, omega, 
                                ctr_li_counts=c_li_counts, ctr_lj_counts=c_lj_counts, 
                                ij_tasks=ij_tasks, sorted_auxcell=sorted_auxcell, 
                                aux_coeff=aux_coeff_gpu, 
                                Gaux_buffers=[gaux_buf_gpu, feri_buf])
    
    
    
    print("prep, %.4f s"%get_elapsed_time(t0)[-1])


    def cal_cderi_slice(cpi, cpj):

        li = uniq_li[cpi]
        lj = uniq_lj[cpj]
        iao0, iao1 = c_aoi_offsets[cpi:cpi+2]
        jao0, jao1 = c_aoj_offsets[cpj:cpj+2]
        
        nctri = c_li_counts[cpi]
        nctrj = c_lj_counts[cpj]

        ncfi = (li+1)*(li+2)//2
        ncfj = (lj+1)*(lj+2)//2
        ncfij = ncfi * ncfj

        ncaoi = nctri * ncfi
        ncaoj = nctrj * ncfj

        nsfi = li * 2 + 1
        nsfj = lj * 2 + 1
        nsfij = nsfi * nsfj

        nsaoi = nctri * nsfi
        nsaoj = nctrj * nsfj


        if with_long_range:
            t0 = get_current_time()
            eri3c, avail_buffer = lr_int_kernel(cpi, cpj, feri_buf=feri_buf, Gblksize=max_nGv)
            #eri3c = lr_int_kernel(cpi, cpj); avail_buffer = feri_buf
            print(cpi, cpj, nsaoi, nsaoj, "LR, %.4f s"%get_elapsed_time(t0)[-1])
        else:
            avail_buffer = feri_buf
            raise NotImplementedError

        t0 = get_current_time()

        c_pair_idx, compressed_eri3c0 = sr_int_kernel(cpi, cpj, buf_j3c=avail_buffer)
        print(cpi, cpj, nsaoi, nsaoj, "SR, %.4f s"%get_elapsed_time(t0)[-1])
        
        n_pairs = len(c_pair_idx)

        compressed_eri3c0 = compressed_eri3c0.reshape(-1,ncfij*n_pairs)
        compressed_eri3c1_size = ncfj * ncfi * n_pairs * nsaux
        idx0 = avail_buffer.size - compressed_eri3c1_size
        compressed_eri3c1 = avail_buffer[idx0:avail_buffer.size].reshape(ncfj,ncfi,n_pairs,nsaux)
        #compressed_eri3c = compressed_eri3c.T.dot(aux_coeff_gpu)
        dgemm_cupy(1, 0, compressed_eri3c0, aux_coeff_gpu, compressed_eri3c1, 1.0, 0.0)

        eri3c_size = nsaoi * nsaoj * nsaux
        if not cell.cart: # cart -> sph
            '''compressed_eri3c = contract('qj,qpmk->jpmk', c2s[lj], compressed_eri3c)
            compressed_eri3c = contract('pi,jpmk->jimk', c2s[li], compressed_eri3c)'''
            if li > 1 or lj > 1:
                c2slji = cupy.einsum('qj, pi -> jiqp', c2s_gpu[lj], c2s_gpu[li])
                #compressed_eri3c = cupy.dot(c2slji.reshape(-1, ncfj*ncfi), compressed_eri3c.reshape(ncfj*ncfi, -1))
                compressed_eri3c0 = avail_buffer[:nsfij*n_pairs*nsaux].reshape(nsfj,nsfi,n_pairs,nsaux)
                dgemm_cupy(0, 0, c2slji.reshape(-1, ncfij), compressed_eri3c1.reshape(ncfij, -1), compressed_eri3c0, 1.0, 0.0)
                compressed_eri3c = compressed_eri3c0

                '''idx0 = avail_buffer.size - eri3c_size
                eri3c = avail_buffer[idx0:avail_buffer.size].reshape()'''
            else:
                compressed_eri3c = compressed_eri3c1

                #eri3c = avail_buffer[:eri3c_size]
        eri3c = eri3c.reshape(nL*nctri, nsfi, nL*nctrj, nsfj, nsaux)
        #eri3c.fill(0.0)

        naoi = iao1 - iao0
        naoj = jao1 - jao0
        ish, jsh = divmod(c_pair_idx, nL*nctrj)
        #eri3c = cupy.zeros((nL*nctri, nsfi, nL*nctrj, nsfj, nsaux))
        compressed_eri3c = compressed_eri3c.reshape(nsfj,nsfi,n_pairs,nsaux)
        eri3c[ish,:,jsh] += compressed_eri3c.transpose(2,1,0,3)
        eri3c = eri3c.reshape(naoi, naoj, nsaux)
        compressed_eri3c = None

        

        
        return eri3c
    
    if return_buf:
        return cal_cderi_slice, shared_ptrs, feri_buf
    else:
        return cal_cderi_slice, shared_ptrs

    


def build_cderi_gamma_point(cell, auxcell, omega=OMEGA_MIN, with_long_range=True,
                            linear_dep_threshold=LINEAR_DEP_THR, solve_j3c=True):
    
    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, -omega).build()
    int_kernel, shared_ptrs = gamma_aux_2e_cuda(int3c2e_opt, with_long_range=with_long_range,
                                    linear_dep_threshold=linear_dep_threshold, 
                                    solve_j3c=False)


    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    out = cupy.zeros((nao, nao, naux))

    njobs_i = len(int3c2e_opt.cell0_ctr_li_counts)
    njobs_j = len(int3c2e_opt.cell0_ctr_lj_counts)
    ij_tasks = list((i, j) for i in range(njobs_i) for j in range(njobs_j))

    c_li_counts = np.asarray(int3c2e_opt.cell0_ctr_li_counts)
    c_lj_counts = np.asarray(int3c2e_opt.cell0_ctr_lj_counts)
    uniq_li = np.asarray(int3c2e_opt.cell0_uniq_li)
    uniq_lj = np.asarray(int3c2e_opt.cell0_uniq_lj)
    if cell.cart:
        nf_i = (uniq_li + 1) * (uniq_li + 2) // 2
        nf_j = (uniq_lj + 1) * (uniq_lj + 2) // 2
    else:
        nf_i = uniq_li * 2 + 1
        nf_j = uniq_lj * 2 + 1
    c_aoi_offsets = np.append(0, np.cumsum(c_li_counts*nf_i))
    c_aoj_offsets = np.append(0, np.cumsum(c_lj_counts*nf_j))

    ao_idx = int3c2e_opt.ao_idx

    for cpi, cpj in ij_tasks:
        iao0, iao1 = c_aoi_offsets[cpi:cpi+2]
        jao0, jao1 = c_aoj_offsets[cpj:cpj+2]
        iaos = ao_idx[iao0:iao1]
        jaos = ao_idx[jao0:jao1]
        out[iaos[:,None], jaos] = int_kernel(cpi, cpj)


    if solve_j3c:
        j2c = _get_2c2e(auxcell, None, omega, with_long_range)
        j2c = j2c[0].real

        prefer_ed = PREFER_ED
        if cell.dimension == 2:
            prefer_ed = True

        cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(
                j2c, prefer_ed, linear_dep_threshold)
        cupyx.scipy.linalg.solve_triangular(cd_j2c, out.reshape(-1,naux).T, lower=True, overwrite_b=True)

        if cd_j2c_negative is not None:
            assert cell.dimension == 2
            #eri3c = _solve_cderi(cd_j2c_negative, eri3c, j2ctag)
            cupyx.scipy.linalg.solve_triangular(cd_j2c_negative, out.reshape(-1,naux).T, 
                                                lower=True, overwrite_b=True)

    cderi = {(0,0): out.transpose(2, 0, 1)}

    for ptr in shared_ptrs:
        close_ipc_handle(ptr)

    return cderi, None

def get_pbc_nao_range(intopt):
    ls = intopt.ori_cell0_uniq_l
    l_ctr_counts = intopt.ori_cell0_prim_l_counts

    nfs = (ls + 1) * (ls + 2) // 2
    min_nao = 2 * np.max(nfs)
    max_nao = np.max(nfs * l_ctr_counts)
    return min_nao, max_nao

def get_cum_offsets(num_list):
    offsets = np.empty(len(num_list) + 1, dtype=int)
    offsets[0] = 0
    np.cumsum(num_list, out=offsets[1:])
    return offsets

    
def get_pbc_slice_ranks(intopt, ncore=1, group_i=True):
    nao = intopt.cell.nao
    uniq_li = intopt.cell0_uniq_li
    uniq_lj = intopt.cell0_uniq_lj
    
    ctr_li_counts = intopt.cell0_ctr_li_counts
    ctr_lj_counts = intopt.cell0_ctr_lj_counts

    #ctr_nfj = (uniq_lj + 1) * (uniq_lj + 2) // 2
    '''ctr_nfi = uniq_li * 2 + 1 # sph
    ctr_iao_loc = get_cum_offsets(ctr_li_counts * ctr_nfi)
    '''

    if group_i:
        ctr_nfi = uniq_li * 2 + 1 # sph
        ctr_ao_loc = get_cum_offsets(ctr_li_counts * ctr_nfi)
    else:
        ctr_nfj = uniq_lj * 2 + 1 # sph
        ctr_ao_loc = get_cum_offsets(ctr_lj_counts * ctr_nfj)

    ij_tasks = ((i, j) for i in range(len(ctr_li_counts)) for j in range(len(ctr_lj_counts)))

    al_segs = []
    aop_segs = []
    aop_index = np.zeros((nao+1)**2, dtype=int)
    aop_index[:] = -1
    idx_aop = 0
    for cpi, cpj in ij_tasks:
        if group_i:
            al0, al1 = ctr_ao_loc[cpi:cpi+2]
        else:
            al0, al1 = ctr_ao_loc[cpj:cpj+2]
        idx_al = al0 * nao + al1
        if aop_index[idx_al] == -1:
            aop_index[idx_al] = idx_aop
            aop_segs.append([[al0, al1], [[cpi, cpj]]])
            al_segs.append([al0, al1])
            idx_aop += 1
        else:
            aop_segs[aop_index[idx_al]][1].append([cpi, cpj])

    if ncore > 1:
        al_segs = np.asarray(al_segs)
        nao_slice = al_segs[:, 1] - al_segs[:, 0]
        nao_gpus = equisum_partition(nao_slice, ncore)

        ao_slice = []
        slice_ranks = []

        sidx0 = 0
        for inao_gpu in nao_gpus:
            sidx1 = sidx0 + len(inao_gpu)
            al_gpu = al_segs[sidx0:sidx1]
            ao_slice.append([al_gpu[0][0], al_gpu[-1][1]])
            slice_ranks.append(aop_segs[sidx0:sidx1])
            sidx0 = sidx1
    else:
        ao_slice = [[al_segs[0][0], al_segs[-1][1]]]
        slice_ranks = [aop_segs]

    return ao_slice, slice_ranks

def get_gamma_feri_cuda(self, df_obj, qc_method, log, ijp_shape=True, ovlp=None, 
                        fitting=True, with_long_range=True):
    tt = get_current_time()
    t_cal = create_timer()
    t_data = create_timer()
    t_write = create_timer()
    self.debug = False

    if ijp_shape:
        shape_feri = (self.nao, self.nao, self.naoaux)
    else:
        shape_feri = (self.naoaux, self.nao, self.nao)

    if self.int_storage in {0, 3}:# 0: CPU Incore; 3: GPU incore 
        self.win_feri, self.feri_node = get_shared(shape_feri, set_zeros=True)
        feri_data = self.feri_node
    elif self.int_storage == 1:# Outcore
        self.file_feri = f"feri_{qc_method}.tmp"
        file_feri = h5py.File(self.file_feri, 'w', driver='mpio', comm=comm)
        feri_data = file_feri.create_dataset("feri", shape_feri, dtype=np.float64)

    max_memory = get_mem_spare(self.mol, 0.9)
    max_memory_f8 = max_memory * 1e6 // 8
    gpu_memory_f8 = ave_gpu_memory(self.gpu_memory) * 1e6 // 8
    nao = self.nao
    naoaux = self.naoaux
    cell = self.mol
    auxcell = self.auxmol
    omega = self.with_df.df_builder.omega
    
    int3c2e_opt = self.intopt #SRInt3c2eOpt(cell, auxcell, -omega, fitting=True)
    print("Memory now: %.4f MB !!!"%(avail_gpu_mem()))

    ncaux = auxcell.nao_nr(cart=True)
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    ngrid = (np.prod(cell.symmetrize_mesh(cell.cutoff_to_mesh(ke_cutoff))))
    shared_buf_size = naoaux*ncaux + ngrid*(naoaux+ncaux)

    feri_mem_f8 = 0.9 * gpu_memory_f8 - shared_buf_size
    feri_mem_f8 /= nrank_per_gpu
    min_nao, max_nao = get_pbc_nao_range(int3c2e_opt)

    max_l = max(cell._bas[:, gto.ANG_OF])
    max_c2s = ((max_l+1)*(max_l+2)//2) / (2*max_l + 1)

    max_nao0 = min(max_nao, max(min_nao, int(0.2*feri_mem_f8 / (nao*naoaux))))
    max_nao1 = max(min_nao, int(0.7*feri_mem_f8 / (3*max_c2s*max_nao0*naoaux)))
    if max_nao0 > max_nao:
        max_nao0 = max_nao
        max_nao1 = min(max_nao, int(0.7*feri_mem_f8 / (3*max_c2s*max_nao0*naoaux)))
    feri_chunk_f8 = max_nao0 * nao * naoaux
    feri_slice_f8 = int(feri_mem_f8 - feri_chunk_f8)

    int3c2e_opt.build(group_size_aoi=max_nao0, group_size_aoj=max_nao1, slice_aoi=True)

    ao_slice, shell_slice_ranks = get_pbc_slice_ranks(int3c2e_opt, ncore=nrank)

    if igpu is None:
        shell_slice_rank = None
    else:
        shell_slice_rank = shell_slice_ranks[igpu]

    if shell_slice_rank is not None:

        max_nao_gpu = np.max([al1-al0 for (al0, al1), _ in shell_slice_rank])

        '''nao_cpu = min(int((max_memory_f8 - max_nao_gpu*nao*naoaux) / (nao*naoaux)), nao)
        shell_seg, nao_cpu = get_seg_gpu(shell_slice_rank, nao_cpu)'''

        nao_cpu = max_nao_gpu
        buf_feri_cpu = np.empty((nao_cpu*nao*naoaux))
        
        unsort_ao_idx = np.empty(nao, dtype=np.int32)
        unsort_ao_idx[int3c2e_opt.ao_idx] = np.arange(nao, dtype=np.int32)
        unsort_ao_idx_gpu = cupy.asarray(unsort_ao_idx, dtype=cupy.int32)

        sorted_ao_indices = np.array(int3c2e_opt.ao_idx, dtype=np.int32, copy=True)
        if self.int_storage == 1:
            sorted_indices = np.empty_like(sorted_ao_indices)
            for (al0, al1), _ in shell_slice_rank:
                sorted_ao_idx_seg = sorted_ao_indices[al0:al1]
                sorted_idx = np.argsort(sorted_ao_idx_seg)
                sorted_ao_indices[al0:al1] = sorted_ao_idx_seg[sorted_idx]
                sorted_indices[al0:al1] = sorted_idx
            sorted_indices_gpu = cupy.asarray(sorted_indices)

        c_li_counts = np.asarray(int3c2e_opt.cell0_ctr_li_counts)
        c_lj_counts = np.asarray(int3c2e_opt.cell0_ctr_lj_counts)
        uniq_li = np.asarray(int3c2e_opt.cell0_uniq_li)
        uniq_lj = np.asarray(int3c2e_opt.cell0_uniq_lj)
        if cell.cart:
            nf_i = (uniq_li + 1) * (uniq_li + 2) // 2
            nf_j = (uniq_lj + 1) * (uniq_lj + 2) // 2
        else:
            nf_i = uniq_li * 2 + 1
            nf_j = uniq_lj * 2 + 1
        c_aoi_offsets = get_cum_offsets(c_li_counts*nf_i)
        c_aoj_offsets = get_cum_offsets(c_lj_counts*nf_j)

        int_kernel, shared_ptrs, feri_slice_buf = gamma_aux_2e_cuda(int3c2e_opt, solve_j3c=True, 
                                                    with_long_range=with_long_range, 
                                                    feri_mem_f8=feri_slice_f8, 
                                                    return_buf=True)
        
        print(irank, feri_slice_f8*8*1e-9, feri_chunk_f8*8*1e-9)

        feri_chunk_buf = cupy.empty(feri_chunk_f8)


        for (al0, al1), cpidx_list in shell_slice_rank:
            nao0 = al1 - al0
            
            feri_gpu = feri_chunk_buf[:nao0*nao*naoaux].reshape(nao0, nao, naoaux)#cupy.empty((nao0, nao, naoaux))
            t1 = get_current_time()
            for cpi, cpj in cpidx_list:
                be0, be1 = c_aoj_offsets[cpj:cpj+2]
                #sorted_bes = sorted_ao_indices[be0:be1]
                #feri_gpu[:, sorted_bes] = int_kernel(cpi, cpj)
                feri_gpu[:, be0:be1] = int_kernel(cpi, cpj)
            accumulate_time(t_cal, t1)

            t1 = get_current_time()
            feri_gpu1 = feri_slice_buf[:feri_gpu.size].reshape(feri_gpu.shape)
            cupy.take(feri_gpu, unsort_ao_idx_gpu, axis=1, out=feri_gpu1)


            if self.int_storage == 1:
                sorted_idx_seg_gpu = sorted_indices_gpu[al0:al1]
                cupy.take(feri_gpu1, sorted_idx_seg_gpu, axis=0, out=feri_gpu)

                feri0_gpu = feri_gpu
                feri1_gpu = feri_gpu1
            else:
                feri0_gpu = feri_gpu1
                feri1_gpu = feri_gpu

            if ijp_shape:
                feri_cpu = buf_feri_cpu[:feri0_gpu.size].reshape(feri0_gpu.shape)
                cupy.asnumpy(feri0_gpu, out=feri_cpu)
                accumulate_time(t_data, t1)

                t1 = get_current_time()
                feri_data[sorted_ao_indices[al0:al1]] = feri_cpu
                accumulate_time(t_write, t1)
            else:
                feri1_gpu = feri1_gpu.reshape(naoaux, nao0, nao)
                feri1_gpu[:] = feri0_gpu.transpose(2, 0, 1)
                feri_cpu = buf_feri_cpu[:feri1_gpu.size].reshape(feri1_gpu.shape)
                cupy.asnumpy(feri1_gpu, out=feri_cpu)
                accumulate_time(t_data, t1)

                t1 = get_current_time()
                feri_data[:, sorted_ao_indices[al0:al1]] = feri_cpu
                accumulate_time(t_write, t1)

    comm.Barrier()

    int_kernel = feri_slice_buf = None

    for ptr in shared_ptrs:
        close_ipc_handle(ptr)
    
    if self.int_storage in {0, 3}:
        Acc_and_get_GA(self.feri_node)
        comm.Barrier()

        if self.int_storage == 3:
            aux_slice = get_slice(range(ngpu), job_size=naoaux)[igpu]
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
    
    '''if fitting:
        free_win(win_low)'''
    
    time_list = [["writing", t_write], ["GPU-CPU data", t_data], 
               ['computing', t_cal], [f"{qc_method} feri", get_elapsed_time(tt)]]
    time_list = get_max_rank_time_list(time_list)
    print_time(time_list, log)