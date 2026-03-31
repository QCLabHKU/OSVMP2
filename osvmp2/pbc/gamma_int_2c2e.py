import os
import time
import ctypes
import h5py
import numpy as np
import scipy
from scipy.linalg import blas
from pyscf import lib
from pyscf import gto
from pyscf.gto.mole import conc_env
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.gto import _pbcintor
from pyscf.pbc.df import ft_ao, aft
from pyscf.pbc.lib.kpts_helper import is_zero
from osvmp2.__config__ import ngpu
from osvmp2.pbc.gamma_int_3c2e import get_shell_batches, get_auxG
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

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)

if ngpu:
    import cupy
    from gpu4pyscf.pbc.df.rsdf_builder import _get_2c2e as get_gpu_2c2e

    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)


libpbc = _pbcintor.libpbc


def get_sr_2c2e_gamma(cell, intor="int2c2e", comp=None, kpts=None, kpt=None,
                **kwargs):
    r'''1-electron integrals from two cells like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''

    cell1 = cell2 = cell

    intor, comp = gto.moleintor._get_intor_and_comp(cell1._add_suffix(intor), comp)

    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    pcell = cell1.copy(deep=False)
    pcell.precision = min(cell1.precision, cell2.precision)
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = conc_env(cell1._atm, cell1._bas, cell1._env,
                                     cell2._atm, cell2._bas, cell2._env)
    
    ao_loc = gto.moleintor.make_loc(bas, intor)
    '''if hermi == 0:
        aosym = 's1'
    else:
        aosym = 's2' '''
    aosym = 's1'
    fill = getattr(libpbc, 'PBCnr2c_fill_k'+aosym)
    fintor = getattr(gto.moleintor.libcgto, intor)
    cintopt = lib.c_null_ptr()

    rcut = max(cell1.rcut, cell2.rcut)
    Ls = cell1.get_lattice_Ls(rcut=rcut)
    expkL = np.asarray(np.exp(1j*np.dot(kpts_lst, Ls.T)), order='C')
    drv = libpbc.PBCnr2c_drv


    i0, i1, j0, j1 = (0, cell1.nbas, 0, cell2.nbas)
    j0 += cell1.nbas
    j1 += cell1.nbas
    
    ni = ao_loc[i1] - ao_loc[i0]
    nj = ao_loc[j1] - ao_loc[j0]
    #out = np.empty((nkpts,comp,ni,nj), dtype=np.complex128)
    win_j2c, j2c_node = get_shared((ni,nj), dtype=np.complex128)
    #out = np.empty((ni,nj), dtype=np.complex128)

    slice_offsets_rank = get_shell_batches(cell1, nrank)[irank]
    
    if slice_offsets_rank is not None:
        i0, i1 = slice_offsets_rank
        p0, p1 = ao_loc[i0], ao_loc[i1]
        drv(fintor, fill, j2c_node[p0:p1].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))

    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(j2c_node)
        comm.Barrier()


    return win_j2c, j2c_node.real

def get_total_2c2e_gamma(self, uniq_kpts):
    def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
        '''Energy cutoff for AFTDF to converge attenuated Coulomb in moment space
        '''
        if precision is None:
            precision = cell.precision
        exps, cs = pbc_gto.cell._extract_pgto_params(cell, 'max')
        ls = cell._bas[:,gto.ANG_OF]
        cs = gto.gto_norm(ls, exps)
        Ecut = aft._estimate_ke_cutoff(exps, ls, cs, precision, omega)
        return Ecut.max()
    
    def _estimate_meshz(cell, precision=None):
        '''For 2D with truncated Coulomb, estimate the necessary mesh size
        that can converge the Gaussian function to the required precision.
        '''
        if precision is None:
            precision = cell.precision
        e = np.hstack(cell.bas_exps()).max()
        ke_cut = -np.log(precision) * 2 * e
        meshz = cell.cutoff_to_mesh(ke_cut)[2]
        return max(meshz, cell.mesh[2])
    
    
    # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
    cell = self.cell
    auxcell = self.auxcell

    precision = auxcell.precision**1.5
    omega = self.omega
    rs_auxcell = self.rs_auxcell
    auxcell_c = rs_auxcell.compact_basis_cell()

    aux_exp = np.hstack(auxcell_c.bas_exps()).min()
    if omega == 0:
        theta = aux_exp / 2
    else:
        theta = 1./(2./aux_exp + omega**-2)
    fac = 2*np.pi**3.5/auxcell.vol * aux_exp**-3 * theta**-1.5
    rcut_sr = (np.log(fac / auxcell_c.rcut / precision + 1.) / theta)**.5
    auxcell_c.rcut = rcut_sr

    with auxcell_c.with_short_range_coulomb(omega):
        #sr_j2c = auxcell_c.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
        win_j2c, j2c = get_sr_2c2e_gamma(auxcell_c, 'int2c2e', hermi=0, kpts=uniq_kpts)

    compact_bas_idx = np.where(rs_auxcell.bas_type != ft_ao.SMOOTH_BASIS)[0]
    compact_ao_idx = rs_auxcell.get_ao_indices(compact_bas_idx)

    # Isn't it ideltical to j2c += j2c_cc?
    ao_map = auxcell.get_ao_indices(rs_auxcell.bas_map[compact_bas_idx])
    def recontract_2d(j2c, j2c_cc):
        return lib.takebak_2d(j2c, j2c_cc, ao_map, ao_map, thread_safe=False)


    # 2c2e integrals the metric can easily cause errors in cderi tensor.
    # self.mesh may not be enough to produce required accuracy.
    # mesh = self.mesh
    ke = estimate_ke_cutoff_for_omega(auxcell, omega, precision)
    mesh = auxcell.cutoff_to_mesh(ke)
    if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
        mesh[2] = _estimate_meshz(auxcell)
    elif cell.dimension < 2:
        mesh[cell.dimension:] = cell.mesh[cell.dimension:]
    mesh = cell.symmetrize_mesh(mesh)

    Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
    b = auxcell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    naux = auxcell.nao

    # Compute auxG
    kpt = uniq_kpts[0]

    #Gaux = ft_ao.ft_ao(rs_auxcell, Gv, (0, rs_auxcell.nbas), b, gxyz, Gvbase, kpt)

    win_auxG, auxG = get_auxG(rs_auxcell, Gv, b, gxyz, Gvbase)


    is_zero_kpt = is_zero(kpt)
    coulG = self.weighted_coulG(kpt, False, mesh)


    # coulG_sr here to first remove the FT-SR-2c2e for compact basis
    # from the analytical 2c2e integrals. The FT-SR-2c2e for compact
    # basis is added back in j2c_k.
    if (cell.dimension == 3 or
        (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
        with lib.temporary_env(cell, dimension=3):
            coulG_sr = self.weighted_coulG_SR(kpt, False, mesh)
        if omega != 0 and is_zero_kpt:
            G0_idx = 0  # due to np.fft.fftfreq convention
            coulG_SR_at_G0 = np.pi/omega**2 * kws
            coulG_sr[G0_idx] += coulG_SR_at_G0
    else:
        coulG_sr = self.weighted_coulG_SR(kpt, False, mesh)

    coulG_sr = coulG_sr.reshape(1, -1)
    coulG = coulG.reshape(1, -1)

    naux = auxG.shape[0]

    full_sr = len(compact_ao_idx) == naux

    nGv = Gv.shape[0]
    if not full_sr:
        win_auxG_sr, auxG_sr = get_shared((rs_auxcell.nao, nGv), dtype=np.complex128)
        if irank_shm == 0:
            auxG_sr = auxG[compact_ao_idx]
        comm_shm.Barrier()
    
    # Check if re-contraction of auxG is nessesary
    ref_ao_loc = rs_auxcell.ref_cell.ao_loc
    current_ao_map = rs_auxcell.get_ao_indices(rs_auxcell.bas_map, ref_ao_loc)

    if not len(current_ao_map) == naux:
        win_auxG_recon, auxG_recon = get_shared((rs_auxcell.nao, nGv), dtype=np.complex128, set_zeros=True)
        if irank_shm == 0:
            lib.takebak_2d(auxG_recon, auxG, current_ao_map, np.arange(nGv), thread_safe=False)
        comm_shm.Barrier()

        free_win(win_auxG); auxG = None

        win_auxG = win_auxG_recon
        auxG = auxG_recon


    aux_slice = get_slice(range(nrank), job_size=naux)[irank]
    #for p0, p1 in lib.prange(0, naux, naux):

    if aux_slice is not None:
        p0, p1 = aux_slice[0], aux_slice[-1] + 1

        auxG_slice = auxG[p0:p1]

        if full_sr:
            coulg = coulG - coulG_sr
            if is_zero_kpt:
                j2c[p0:p1] += np.dot((auxG_slice.conj() * coulg), auxG.T).real
            else:
                j2c[p0:p1] += np.dot((auxG_slice.conj() * coulg), auxG.T)

        else:
            auxG_sr_slice = auxG_sr[p0:p1]

            #print(Gaux_sr.shape, Gcoul_sr.shape)
            if is_zero_kpt:
                j2c_sr = np.dot((auxG_sr_slice.conj() * -coulG_sr), auxG_sr.T).real
            else:
                j2c_sr = np.dot((auxG_sr_slice.conj() * -coulG_sr), auxG_sr.T)
            #auxG = recontract_1d(auxG)
            if is_zero_kpt:  # kpti == kptj
                j2c[p0:p1] += np.dot((auxG_slice.conj() * coulG), auxG.T).real
            else:
                j2c[p0:p1] += np.dot((auxG_slice.conj() * coulG), auxG.T)
            
            recontract_2d(j2c[p0:p1], j2c_sr)

    comm_shm.Barrier()
    free_win(win_auxG)
    if not full_sr:
        free_win(win_auxG_sr)

    return win_j2c, j2c


def get_2c2e_gamma(self, kpts):
    if self.use_gpu:
        if irank_shm == 0:
            j2c = get_gpu_2c2e(self.auxcell, None, self.omega, True).real.get()
            nk, naux, naux = j2c.shape
            j2c = j2c.reshape(naux, naux)
        else:
            j2c = None
        return None, j2c
    else:
        return get_total_2c2e_gamma(self, kpts)