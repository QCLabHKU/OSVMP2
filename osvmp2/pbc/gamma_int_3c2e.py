import sys
import time
import ctypes
import h5py
import numpy as np
import scipy
from scipy.linalg import blas
from pyscf.pbc import tools
from pyscf.pbc.df.aft import _check_kpts
from pyscf import lib
from pyscf import gto
from pyscf.gto.mole import conc_env
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.gto import _pbcintor
from pyscf.pbc.df import ft_ao, aft
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.gto.moleintor import libcgto
from pyscf.pbc.lib.kpts_helper import is_zero

from osvmp2.int_prescreen import even_partition
from osvmp2.mpi_addons import *
from osvmp2.osvutil import *
from mpi4py import MPI

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

libpbc = _pbcintor.libpbc

def gen_ft_ao_kernel(mol, Gv, b=np.eye(3),
                    gxyz=None, Gvbase=None, by_grid=True):
    r'''Analytical FT transform AO
    \int mu(r) exp(-ikr) dr^3

    The output tensor has the shape [nao, nGv]
    '''

    fn = libcgto.GTO_ft_fill_drv
    if mol.cart:
        intor = getattr(libcgto, 'GTO_ft_ovlp_cart')
    else:
        intor = getattr(libcgto, 'GTO_ft_ovlp_sph')
    
    fill = getattr(libcgto, 'GTO_ft_zfill_s1')

    ghost_atm = np.array([[0,0,0,0,0,0]], dtype=np.int32)
    ghost_bas = np.array([[0,0,1,1,0,0,3,0]], dtype=np.int32)
    ghost_env = np.zeros(4)
    ghost_env[3] = np.sqrt(4*np.pi)  # s function spherical norm
    atm, bas, env = pbc_gto.conc_env(mol._atm, mol._bas, mol._env,
                                 ghost_atm, ghost_bas, ghost_env)
    ao_loc = mol.ao_loc_nr()
    nao = int(ao_loc[mol.nbas])
    ao_loc = np.asarray(np.hstack((ao_loc, [nao+1])), dtype=np.int32)

    lack_inputs = (gxyz is None or b is None or Gvbase is None
        # backward compatibility for pyscf-1.2, in which the argument Gvbase is gs
        or (Gvbase is not None and isinstance(Gvbase[0], (int, np.integer))))

    #GvT = np.asarray(Gv.T, order='C')
    if lack_inputs:
        p_gxyzT = lib.c_null_ptr()
        p_gs = (ctypes.c_int*3)(0,0,0)
        p_b = (ctypes.c_double*1)(0)
        eval_gz = 'GTO_Gv_general'
    else:
        if abs(b-np.diag(b.diagonal())).sum() < 1e-8:
            eval_gz = 'GTO_Gv_orth'
        else:
            eval_gz = 'GTO_Gv_nonorth'
        #gxyzT = np.asarray(gxyz.T, order='C', dtype=np.int32)
        #p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        b = np.hstack((b.ravel(), np.zeros(3)) + Gvbase)
        p_b = b.ctypes.data_as(ctypes.c_void_p)
        p_gs = (ctypes.c_int*3)(*[len(x) for x in Gvbase])
    eval_gz = getattr(libcgto, eval_gz)

    phase = 0
    p_phase = ctypes.c_double(phase)
    p_ao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)
    p_atm = atm.ctypes.data_as(ctypes.c_void_p)
    p_natm = ctypes.c_int(len(atm))
    p_bas = bas.ctypes.data_as(ctypes.c_void_p)
    p_nbas = ctypes.c_int(len(bas))
    p_env = env.ctypes.data_as(ctypes.c_void_p)

    if by_grid:
        shls_slice = (0, mol.nbas, mol.nbas, mol.nbas+1)
        p_shls_slice = (ctypes.c_int*4)(*shls_slice)
        nish = shls_slice[1] - shls_slice[0]
        ovlp_mask = np.ones(nish, dtype=np.int8)
        p_ovlp_mask = ovlp_mask.ctypes.data_as(ctypes.c_void_p)
        ni = int(ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]])


        def ft_ao_by_grid(p0, p1, out=None):
            nGv = p1 - p0
            shape = (ni, nGv)
            if out is None:
                out = np.zeros(shape, order='C', dtype=np.complex128)
            if not lack_inputs:
                gxyzT = np.asarray(gxyz[p0:p1].T, order='C', dtype=np.int32)
                p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
                GvT = np.asarray(Gv[p0:p1].T, order='C')
                p_GvT = GvT.ctypes.data_as(ctypes.c_void_p)
            
            fn(intor, eval_gz, fill,
                out.ctypes.data_as(ctypes.c_void_p),
                p_ovlp_mask,
                ctypes.c_int(1), p_shls_slice,
                p_ao_loc, p_phase,
                p_GvT,
                p_b, p_gxyzT, p_gs, ctypes.c_int(nGv),
                p_atm, p_natm,
                p_bas, p_nbas,
                p_env)
            return out
        
        return ft_ao_by_grid
    
    else:
        nGv = Gv.shape[0]
        p_nGv = ctypes.c_int(nGv)
        if not lack_inputs:
            gxyzT = np.asarray(gxyz.T, order='C', dtype=np.int32)
            p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
            GvT = np.asarray(Gv.T, order='C')
            p_GvT = GvT.ctypes.data_as(ctypes.c_void_p)

        shls_slice = np.asarray([0, mol.nbas, mol.nbas, mol.nbas+1], dtype=np.int32)
        p_shls_slice = (ctypes.c_int*4)(*shls_slice)

        def ft_ao_by_shell(a0, a1, out=None):
            shls_slice[:2] = [a0, a1]

            p_shls_slice = (ctypes.c_int*4)(*shls_slice)
            nish = shls_slice[1] - shls_slice[0]
            ovlp_mask = np.ones(nish, dtype=np.int8)
            p_ovlp_mask = ovlp_mask.ctypes.data_as(ctypes.c_void_p)
            ni = int(ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]])

            shape = (ni, nGv)
            if out is None:
                out = np.zeros(shape, order='C', dtype=np.complex128)

            fn(intor, eval_gz, fill,
                out.ctypes.data_as(ctypes.c_void_p),
                p_ovlp_mask,
                ctypes.c_int(1), p_shls_slice,
                p_ao_loc, p_phase,
                p_GvT,
                p_b, p_gxyzT, p_gs, p_nGv,
                p_atm, p_natm,
                p_bas, p_nbas,
                p_env)
            return out
        
        return ft_ao_by_shell


def get_shell_batches(mol, nchunks, aosym='s1', shell_limits=None, for_mpi_jobs=True):

    if shell_limits is None:
        shell_limits = [0, mol.nbas]
    a0, a1 = shell_limits

    ao_loc = mol.ao_loc

    shells = np.arange(a0, a1)
    if aosym == 's1':
        weight_list = ao_loc[shells+1] - ao_loc[shells]
    else:
        al0_list = ao_loc[shells]
        al1_list = ao_loc[shells+1]
        weight_list = al1_list * (al1_list+1) // 2 - al0_list * (al0_list+1) // 2 #[(a+1)*(a+2)//2 - a*(a+1)//2 for a in range(a0, a1)]

    slice_offsets = even_partition(weight_list, min(nchunks, len(weight_list)))

    if for_mpi_jobs and len(slice_offsets) < nchunks:
        ranks = np.arange(nchunks).reshape(-1, nrank_shm).T.ravel()
        slice_offsets_all = [None] * nchunks
        for jidx in range(len(slice_offsets)):
            slice_offsets_all[ranks[jidx]] = slice_offsets[jidx]
        slice_offsets = slice_offsets_all
    
    return slice_offsets

def get_auxG(rs_auxcell, Gv, b, gxyz, Gvbase):
    ao_loc = rs_auxcell.ao_loc
    nao_list = [ao_loc[a+1] - ao_loc[a] for a in range(rs_auxcell.nbas)]
    slice_offsets = even_partition(nao_list, min(nrank, len(nao_list)))

    if len(slice_offsets) < nrank:
        ranks = np.arange(nrank).reshape(-1, nrank_shm).T.ravel()
        slice_offsets_all = [None] * nrank
        for jidx in range(len(slice_offsets)):
            slice_offsets_all[ranks[jidx]] = slice_offsets[jidx]
        slice_offsets_rank = slice_offsets_all[irank]
    else:
        slice_offsets_rank = slice_offsets[irank]

    nGv = Gv.shape[0]
    win_auxG, auxG = get_shared((rs_auxcell.nao, nGv), dtype=np.complex128)

    if slice_offsets_rank is not None:
        #print(irank, slice_offsets_rank)
        a0, a1 = slice_offsets_rank
        ft_ao_kernel = gen_ft_ao_kernel(rs_auxcell, Gv, b, gxyz, Gvbase, by_grid=False)
        
        p0, p1 = ao_loc[a0], ao_loc[a1]
        
        ft_ao_kernel(a0, a1, out=auxG[p0:p1])


    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(auxG)
        comm.Barrier()

    return win_auxG, auxG

def get_Gaux(rs_auxcell, Gv, b, gxyz, Gvbase):

    nGv = Gv.shape[0]
    win_Gaux, Gaux = get_shared((nGv, rs_auxcell.nao), dtype=np.complex128)

    grid_slice = get_slice(range(nrank), job_size=nGv)[irank]

    if grid_slice is not None:
        g0, g1 = grid_slice[0], grid_slice[-1]+1
        ft_ao_kernel = gen_ft_ao_kernel(rs_auxcell, Gv, b, gxyz, Gvbase, by_grid=True)
        Gaux[g0:g1] = ft_ao_kernel(g0, g1).T

    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(Gaux)
        comm.Barrier()

    return win_Gaux, Gaux


def get_vbar(self):
    def _gaussian_int(cell):
        r'''Regular gaussian integral \int g(r) dr^3'''
        return ft_ao.ft_ao(cell, np.zeros((1,3)))[0].real
    cell = self.cell
    if self.exclude_d_aux and cell.dimension > 0:
        rs_auxcell = self.rs_auxcell
        aux_chg = _gaussian_int(rs_auxcell)
        smooth_ao_idx = rs_auxcell.get_ao_type() == ft_ao.SMOOTH_BASIS
        aux_chg[smooth_ao_idx] = 0
        aux_chg = rs_auxcell.recontract_1d(aux_chg[:,None]).ravel()
    else:
        aux_chg = _gaussian_int(self.auxcell)
    
    vbar = np.pi / self.omega**2 / cell.vol * aux_chg
    vbar_idx = np.where(vbar != 0)[0]
    if len(vbar_idx) == 0:
        vbar = None
    else:
        vbar = vbar[vbar_idx]
    return vbar, vbar_idx

def get_ovlp(self, aosym):
    if self.exclude_dd_block:
        rs_cell = self.rs_cell
        ovlp = rs_cell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts)
        smooth_ao_idx = rs_cell.get_ao_type() == ft_ao.SMOOTH_BASIS
        for s in ovlp:
            s[smooth_ao_idx[:,None] & smooth_ao_idx] = 0
        recontract_2d = rs_cell.recontract(dim=2)
        ovlp = [recontract_2d(s) for s in ovlp]
    else:
        ovlp = self.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts)

    if aosym == 's2':
        ovlp = [lib.pack_tril(s) for s in ovlp]
    else:
        ovlp = [s.ravel() for s in ovlp]
    
    return ovlp

    

def get_lr_gaux(self, kpt, parallel=True):
    cell = self.cell
    rs_cell = self.rs_cell
    rs_auxcell = self.rs_auxcell

    Gv, Gvbase, _ = rs_cell.get_Gv_weights(self.mesh)
    b = rs_cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    coulG = self.weighted_coulG(kpt, False, self.mesh)
    if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
        with lib.temporary_env(cell, dimension=3):
            coulG_SR = self.weighted_coulG_SR(kpt, False, self.mesh)
    else:
        coulG_SR = self.weighted_coulG_SR(kpt, False, self.mesh)
    coulG_LR = coulG - coulG_SR

    coulG = coulG.reshape(-1, 1)
    coulG_LR = coulG_LR.reshape(-1, 1)

    to_smooth = False
    if self.exclude_d_aux and rs_cell.dimension > 0:
        smooth_aux_mask = rs_auxcell.get_ao_type() == ft_ao.SMOOTH_BASIS
        smooth_indices = np.nonzero(smooth_aux_mask)[0]
        if len(smooth_indices) > 0:
            to_smooth = True
            nonsmooth_indices = np.nonzero(~smooth_aux_mask)[0]        


    if parallel:
        win_Gaux, Gaux = get_Gaux(rs_auxcell, Gv, b, gxyz, Gvbase)
        if irank_shm == 0:
            to_scale = True
        else:
            to_scale = False
    else:
        win_Gaux = None
        Gaux = ft_ao.ft_ao(rs_auxcell, Gv, None, b, gxyz, Gvbase, kpt)
        to_scale = True

    if to_scale:
        if to_smooth:
            Gaux[:, smooth_indices] *= coulG
            Gaux[:, nonsmooth_indices] *= coulG_LR
            Gaux[:] = rs_auxcell.recontract_1d(Gaux.T).T
        else:
            Gaux *= coulG_LR
    if parallel:
        comm_shm.Barrier()
    #GauxR = np.asarray(auxg_slice.real, order='C')
    #GauxI = np.asarray(auxg_slice.imag, order='C')
    return win_Gaux, Gaux

    #return gen_gaux_slice


def get_j2c(self, kpts):
    j2c = self.get_2c2e(kpts)[0]
    cd_j2c = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
    return cd_j2c

from pyscf.scf import _vhf

LOG_ADJUST = 32

def _conc_locs(ao_loc1, ao_loc2):
    '''auxiliary basis was appended to regular AO basis when calling int3c2e
    integrals. Composite loc combines locs from regular AO basis and auxiliary
    basis accordingly.'''
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return np.asarray(comp_loc, dtype=np.int32)

libpbc.GTOmax_cache_size.restype = ctypes.c_int
def _get_cache_size(cell, intor):
    '''Cache size for libcint integrals. Cache size cannot be accurately
    estimated in function PBC_ft_bvk_drv
    '''
    cache_size = libpbc.GTOmax_cache_size(
        getattr(libpbc, intor), (ctypes.c_int*2)(0, cell.nbas), ctypes.c_int(1),
        cell._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
        cell._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
        cell._env.ctypes.data_as(ctypes.c_void_p))
    return cache_size

def gen_int3c_kernel(self, intor='int3c2e', aosym='s2', comp=None,
                        j_only=False, reindex_k=None, rs_auxcell=None,
                        auxcell=None, supmol=None, return_complex=False):
    '''Generate function to compute int3c2e with double lattice-sum

    rs_auxcell: range-separated auxcell for gdf/rsdf module

    reindex_k: an index array to sort the order of k-points in output
    '''
    log = lib.logger.new_logger(self)
    cput0 = lib.logger.process_clock(), lib.logger.perf_counter()
    if self.rs_cell is None:
        self.build()
    if auxcell is None:
        auxcell = self.auxcell
    if rs_auxcell is None:
        rs_auxcell = ft_ao._RangeSeparatedCell.from_cell(auxcell)
    elif not isinstance(rs_auxcell, ft_ao._RangeSeparatedCell):
        rs_auxcell = ft_ao._RangeSeparatedCell.from_cell(rs_auxcell)
    if supmol is None:
        supmol = self.supmol
    cell = self.cell
    kpts = self.kpts
    nkpts = len(kpts)
    bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)
    nbasp = cell.nbas  # The number of shells in the primitive cell

    if self.direct_scf_tol is None:
        omega = supmol.omega
        aux_exp = np.hstack(auxcell.bas_exps()).min()
        cell_exp = np.hstack(cell.bas_exps()).min()
        if omega == 0:
            theta = 1./(1./cell_exp + 1./aux_exp)
        else:
            theta = 1./(1./cell_exp + 1./aux_exp + omega**-2)
        lattice_sum_factor = max(2*np.pi*cell.rcut/(cell.vol*theta), 1)
        cutoff = cell.precision / lattice_sum_factor**2 * .1
        log.debug1('int3c_kernel integral omega=%g theta=%g cutoff=%g',
                    omega, theta, cutoff)
    else:
        cutoff = self.direct_scf_tol
    log_cutoff = int(np.log(cutoff) * LOG_ADJUST)

    atm, bas, env = gto.conc_env(supmol._atm, supmol._bas, supmol._env,
                                    rs_auxcell._atm, rs_auxcell._bas, rs_auxcell._env)
    cell0_ao_loc = _conc_locs(cell.ao_loc, auxcell.ao_loc)
    seg_loc = _conc_locs(supmol.seg_loc, rs_auxcell.sh_loc)
    # mimic the lattice sum with range 1
    aux_seg2sh = np.arange(rs_auxcell.nbas + 1)
    seg2sh = _conc_locs(supmol.seg2sh, aux_seg2sh)

    if 'ECP' in intor:
        # rs_auxcell is a placeholder only to represent the ecpbas.
        # Ensure the ECPBAS_OFFSET be consistent with the treatment in pbc.gto.ecp
        env[gto.AS_ECPBAS_OFFSET] = len(bas)
        bas = np.asarray(np.vstack([bas, cell._ecpbas]), dtype=np.int32)
        cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        # sindex may not be accurate enough to screen ECP integral.
        # Add penalty 1e-2 to reduce the screening error
        log_cutoff = int(np.log(cutoff*1e-2) * LOG_ADJUST)
    else:
        cintopt = _vhf.make_cintopt(atm, bas, env, intor)

    sindex = self.get_q_cond(supmol)
    ovlp_mask = sindex > log_cutoff
    bvk_ovlp_mask = lib.condense('np.any', ovlp_mask, supmol.sh_loc)
    cell0_ovlp_mask = bvk_ovlp_mask.reshape(
        bvk_ncells, nbasp, bvk_ncells, nbasp).any(axis=2).any(axis=0)
    cell0_ovlp_mask = cell0_ovlp_mask.astype(np.int8)
    ovlp_mask = None

    # Estimate the buffer size required by PBCfill_nr3c functions
    cache_size = max(_get_cache_size(cell, intor),
                        _get_cache_size(rs_auxcell, intor))
    cell0_dims = cell0_ao_loc[1:] - cell0_ao_loc[:-1]
    dijk = int(cell0_dims[:nbasp].max())**2 * int(cell0_dims[nbasp:].max()) * comp

    aosym = aosym[:2]
    gamma_point_only = is_zero(kpts)
    if gamma_point_only:
        assert nkpts == 1
        fill = f'PBCfill_nr3c_g{aosym}'
        nkpts_ij = 1
        cache_size += dijk
    elif nkpts == 1:
        fill = f'PBCfill_nr3c_nk1{aosym}'
        nkpts_ij = 1
        cache_size += dijk * 3
    elif j_only:
        fill = f'PBCfill_nr3c_k{aosym}'
        # sort kpts then reindex_k in sort_k can be skipped
        if reindex_k is not None:
            kpts = kpts[reindex_k]
            nkpts = len(kpts)
        nkpts_ij = nkpts
        cache_size = (dijk * bvk_ncells + dijk * nkpts * 2 +
                        max(dijk * nkpts * 2, cache_size))
    else:
        assert nkpts < 45000
        fill = f'PBCfill_nr3c_kk{aosym}'
        nkpts_ij = nkpts * nkpts
        cache_size = (max(dijk * bvk_ncells**2 + cache_size, dijk * nkpts**2 * 2) +
                        dijk * bvk_ncells * nkpts * 2)
        if aosym == 's2' and reindex_k is not None:
            kk_mask = np.zeros((nkpts*nkpts), dtype=bool)
            kk_mask[reindex_k] = True
            kk_mask = kk_mask.reshape(nkpts, nkpts)
            if not np.all(kk_mask == kk_mask.T):
                log.warn('aosym=s2 not found in required kpts pairs')

    expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts.T))
    expLkR = np.asarray(expLk.real, order='C')
    expLkI = np.asarray(expLk.imag, order='C')
    expLk = None

    if reindex_k is None:
        reindex_k = np.arange(nkpts_ij, dtype=np.int32)
    else:
        reindex_k = np.asarray(reindex_k, dtype=np.int32)
        nkpts_ij = reindex_k.size

    drv = libpbc.PBCfill_nr3c_drv

    # is_pbcintor controls whether to use memory efficient functions
    # Only supports int3c2e_sph, int3c2e_cart in current C library
    is_pbcintor = intor in ('int3c2e_sph', 'int3c2e_cart') or intor[:3] == 'ECP'
    if is_pbcintor and not intor.startswith('PBC'):
        intor = 'PBC' + intor
    log.debug1('is_pbcintor = %d, intor = %s', is_pbcintor, intor)


    intor_func = getattr(libpbc, intor)
    fill_func = getattr(libpbc, fill)
    p_is_pbcintor =ctypes.c_int(is_pbcintor)
    p_expLkR = expLkR.ctypes.data_as(ctypes.c_void_p)
    p_expLkI = expLkI.ctypes.data_as(ctypes.c_void_p)
    p_reindex_k = reindex_k.ctypes.data_as(ctypes.c_void_p)
    p_nkpts_ij = ctypes.c_int(nkpts_ij)
    p_bvk_ncells = ctypes.c_int(bvk_ncells)
    p_nimgs = ctypes.c_int(nimgs)


    #log.timer('int3c kernel initialization', *cput0)
    #print_time(['int3c kernel initialization', get_elapsed_time(cput0)])

    if sindex is None:
        sindex_ptr = lib.c_null_ptr()
    else:
        sindex_ptr = sindex.ctypes.data_as(ctypes.c_void_p)

    def int3c(shls_slice=None, outR=None, outI=None, zero_buffers=False):
        if shls_slice is None:
            shls_slice = [0, nbasp, 0, nbasp, nbasp, len(cell0_dims)]
        else:
            ksh0 = nbasp + shls_slice[4]
            ksh1 = nbasp + shls_slice[5]
            shls_slice = list(shls_slice[:4]) + [ksh0, ksh1]
        i0, i1, j0, j1, k0, k1 = cell0_ao_loc[shls_slice]
        if aosym == 's1':
            nrow = (i1-i0)*(j1-j0)
        else:
            nrow = i1*(i1+1)//2 - i0*(i0+1)//2
        if comp == 1:
            shape = (nkpts_ij, nrow, k1-k0)
        else:
            shape = (nkpts_ij, comp, nrow, k1-k0)
            
        # output has to be filled with zero first because certain integrals
        # may be skipped by fill_ints driver
        if outR is None:
            outR = np.zeros(shape)
        else:
            assert zero_buffers, "Buffer must be filled with zeros! Doulbe check and set zero_buffers=True"
            outR = np.ndarray(shape, buffer=outR)
            #outR[:] = 0

        if gamma_point_only:
            outI = np.zeros(0)
        else:
            if outR is None:
                outI = np.zeros(shape)
            else:
                assert zero_buffers, "Buffer must be filled with zeros! Doulbe check and set zero_buffers=True"
                outI = np.ndarray(shape, buffer=outI)
                #outR[:] = 0

        drv(intor_func, fill_func,
            p_is_pbcintor,
            outR.ctypes.data_as(ctypes.c_void_p),
            outI.ctypes.data_as(ctypes.c_void_p),
            p_expLkR,
            p_expLkI,
            p_reindex_k, p_nkpts_ij,
            p_bvk_ncells, p_nimgs,
            ctypes.c_int(nkpts), ctypes.c_int(nbasp), ctypes.c_int(comp),
            seg_loc.ctypes.data_as(ctypes.c_void_p),
            seg2sh.ctypes.data_as(ctypes.c_void_p),
            cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*6)(*shls_slice),
            cell0_ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            sindex_ptr, ctypes.c_int(log_cutoff),
            cintopt, ctypes.c_int(cache_size),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            env.ctypes.data_as(ctypes.c_void_p))

        log.timer_debug1(f'pbc integral {intor}', *cput0)
        if return_complex:
            if gamma_point_only:
                return outR
            else:
                return outR + outI * 1j
        else:
            if gamma_point_only:
                return outR, None
            else:
                return outR, outI
    return int3c

def gamma_feri_kernel(self, intor='int3c2e', aosym='s2', comp=None, 
                      cd_j2c=None, ovlp=None, Gaux=None, fitting=True, 
                      with_long_range=True):
    if self.rs_cell is None:
        self.build()

    cell = self.cell
    nao = cell.nao_nr()
    ao_loc = cell.ao_loc
    kpts = self.kpts
    nkpts = len(kpts)

    intor, comp = pbc_gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if self.exclude_d_aux and cell.dimension > 0:
        rs_auxcell = self.rs_auxcell.compact_basis_cell()
    else:
        rs_auxcell = self.rs_auxcell


    ks = np.arange(nkpts, dtype=np.int32)
    kikj_idx = ks * nkpts + ks

    reindex_k = kikj_idx // nkpts

    #with_long_range = self.with_long_range()
    
    # Generate j2c
    if fitting and cd_j2c is None:
        cd_j2c = get_j2c(self, kpts)

    kpt = kpts[0]
    ki = np.arange(nkpts, dtype=np.int32)
    kpt_ij_idx = ki * nkpts + ki


    kptjs = kpts[kpt_ij_idx % nkpts]

    '''int3c_kernel = self.gen_int3c_kernel(intor, aosym, comp, j_only=True,
                                    reindex_k=reindex_k, rs_auxcell=rs_auxcell)'''
    int3c_kernel = gen_int3c_kernel(self, intor, aosym, comp, j_only=True,
                                        reindex_k=reindex_k, rs_auxcell=rs_auxcell)
    
    if with_long_range:
        ft_kernel = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False)
        #GauxR, GauxI = get_lr_gaux(self, kpt)
        if Gaux is None:
            _, Gaux = get_lr_gaux(self, kpt, parallel=False)
        GauxR = Gaux.real
        GauxI = Gaux.imag

        Gv, Gvbase, _ = cell.get_Gv_weights(self.mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]

    vbar, vbar_idx = get_vbar(self)
    if ovlp is None:
        ovlp = get_ovlp(self, aosym)[0].reshape(nao, nao)

    def get_int_slice(shls_slice, Gblksize=None, out=None):
        
        a0, a1, b0, b1 = shls_slice[:4]
        al0, al1, be0, be1 = [ao_loc[a] for a in shls_slice[:4]]

        if out is None:
            j3cR, _ = int3c_kernel(shls_slice)
        else:
            j3cR, _ = int3c_kernel(shls_slice, outR=out, zero_buffers=True)
        
        j3cR = j3cR[0]
        j3cR[:, vbar_idx] -= np.outer(ovlp[al0:al1, be0:be1].ravel(), vbar)

        if with_long_range:
            if Gblksize is None:
                Gblksize = ngrids
            else:
                Gblksize = min(Gblksize, ngrids)
                
            t_gpq = 0.0
            t_acc = 0.0
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                ngrid_slice = p1 - p0

                # shape of Gpq (nkpts, nGv, ni, nj)
                t1 = time.time()
                GpqR, GpqI = ft_kernel(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt,
                                kptjs, (a0, a1, b0, b1)) #, out=buf)
                t_gpq += time.time() - t1
                

                t1 = time.time()
                blas.dgemm(1.0, GauxR[p0:p1].T, GpqR.reshape(ngrid_slice, -1).T, 
                           beta=1.0, c=j3cR.T, trans_b=True, overwrite_c=True)
                
                blas.dgemm(1.0, GauxI[p0:p1].T, GpqI.reshape(ngrid_slice, -1).T, 
                           beta=1.0, c=j3cR.T, trans_b=True, overwrite_c=True)
                t_acc += time.time() - t1
                
        if fitting:
            scipy.linalg.solve_triangular(cd_j2c, j3cR.T, lower=True, overwrite_b=True)

        return j3cR

    return get_int_slice

def get_j3c_full_gamma(self, intor='int3c2e', aosym="s1", comp=1):
    cell = self.cell
    auxcell = self.auxcell
    ao_loc = cell.ao_loc
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)

    shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)
    ish0, ish1, jsh0, jsh1, ksh0, ksh1 = shls_slice
    i0, i1, j0, j1 = ao_loc[list(shls_slice[:4])].astype(np.int64)
    k0, k1 = aux_loc[[ksh0, ksh1]].astype(np.int64)
    if aosym == 's1':
        nao_pair = (i1 - i0) * (j1 - j0)
    else:
        nao_pair = i1*(i1+1)//2 - i0*(i0+1)//2
    naux = k1 - k0

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, self.max_memory-mem_now)

    # split the 3-center tensor (nkpts_ij, i, j, aux) along shell i.
    # plus 1 to ensure the intermediates in libpbc do not overflow
    buflen = min(max(int(max_memory*.9e6/16/naux/(1+1)), 1), nao_pair)
    # lower triangle part
    sh_ranges = _guess_shell_ranges(cell, buflen, aosym, start=ish0, stop=ish1)
    max_buflen = max([x[2] for x in sh_ranges])


    feri_real = np.empty((nao_pair, naux))

    #bufR = np.empty((1, comp, max_buflen, naux))
    #cpu0 = lib.logger.process_clock(), lib.logger.perf_counter()

    int_kernel = gamma_feri_kernel(self, intor='int3c2e', aosym='s1', fitting=True)


    naop_all = [i[2] for i in sh_ranges]
    naop_offsets = np.append([0], np.cumsum(naop_all))

    for idx, (sh_start, sh_end, nrow) in enumerate(sh_ranges):
        if aosym == 's2':
            shls_slice = (sh_start, sh_end, jsh0, sh_end, ksh0, ksh1)
        else:
            shls_slice = (sh_start, sh_end, jsh0, jsh1, ksh0, ksh1)

        aop0, aop1 = naop_offsets[idx:idx+2]

        int_kernel(shls_slice, out=feri_real[aop0:aop1])
    
    return feri_real





