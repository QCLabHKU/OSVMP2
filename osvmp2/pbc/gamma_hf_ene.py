import os
import h5py
import numpy as np
from pyscf import lib
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.df.rsdf_builder import _RSNucBuilder
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df import ft_ao, aft
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.tools import pbc as pbctools
from osvmp2.__config__ import ngpu
from osvmp2.pbc.gamma_int_3c2e import (gen_int3c_kernel, gamma_feri_kernel, 
                                       get_shell_batches, get_lr_gaux)
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
    #from gpu4pyscf.pbc.df.rsdf_builder import get_nuc as gpu_get_nuc
    from osvmp2.pbc.gpu.gamma_hf_ene_cuda import get_nuc as gpu_get_nuc
    from gpu4pyscf.pbc.gto import int1e
    from gpu4pyscf.pbc.df.fft_jk import _ewald_exxdiv_for_G0 as gpu_ewald_exxdiv_for_G0
    
    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)

def gamma_get_ialp_bp(self, dm):
    nocc = self.occ_coeff.shape[1]
    cell = self.mol
    auxcell = self.with_df.auxcell
    nao = cell.nao_nr()
    naoaux = auxcell.nao_nr()
    ao_loc = cell.ao_loc
    df_builder = self.with_df.df_builder

    #shell_slice = get_slice(range(nrank), job_size=cell.nbas)[irank]
    slice_offsets_rank = get_shell_batches(cell, nrank)[irank]

    t0 = get_current_time()
    win_gaux, gaux = get_lr_gaux(df_builder, self.with_df.kpts[0])
    accumulate_time(self.t_feri, t0)

    win_bp, bp_node = get_shared(naoaux, set_zeros=True)

    if nocc != self.nocc_pre:
        file_ialp = h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm)
        ialp_dataset = file_ialp.create_dataset('ialp', (nocc, nao, naoaux), dtype='f8')
    else:
        file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
        ialp_dataset = file_ialp['ialp']

    if slice_offsets_rank is not None:

        '''if irank_shm== 0:
            bp = bp_node
            #size_sub = nocc*naoaux
        else:'''
        bp = np.zeros(naoaux)

        max_nao = get_ncols_for_memory(0.8*get_mem_spare(cell), 3*nao*naoaux, nao)
        a0, a1 = slice_offsets_rank
        shell_segs = [[a0, a0+1]]
        
        for ia in np.arange(a0+1, a1):
            a0_last, _ = shell_segs[-1]
            nao_to_be = ao_loc[ia] - ao_loc[a0_last]

            if nao_to_be > max_nao:
                shell_segs.append([ia, ia+1])
            else:
                shell_segs[-1][1] = ia+1
        
        t0 = get_current_time()
        int_kernel = gamma_feri_kernel(df_builder, intor='int3c2e', aosym='s1', 
                                    fitting=False, Gaux=gaux, ovlp=self.ovlp)
        accumulate_time(self.t_feri, t0)
        
        shls_slice = np.asarray([0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas])
        for a0, a1 in shell_segs:
            shls_slice[2:4] = [a0, a1]
            al0, al1 = ao_loc[a0], ao_loc[a1]
            nao0 = al1 - al0

            t0 = get_current_time()
            feri_slice = int_kernel(shls_slice) # (naopair, naux)
            accumulate_time(self.t_feri, t0)

            #bp += np.dot(dm[:, al0:al1].ravel(), feri_slice)
            t0 = get_current_time()
            ialp_slice = np.dot(self.occ_coeff.T, feri_slice.reshape(nao, -1)).reshape(nocc, nao0, naoaux)
            accumulate_time(self.t_k, t0)
            
            t0 = get_current_time()
            bp += np.dot(self.occ_coeff[al0:al1].T.ravel(), ialp_slice.reshape(-1, naoaux))
            accumulate_time(self.t_j, t0)

            t0 = get_current_time()
            ialp_dataset.write_direct(ialp_slice, dest_sel=np.s_[:, al0:al1])
            accumulate_time(self.t_write, t0)
        #print("Time for int_kernel: %.4f"%(time.time()-t0)); sys.exit()

        #if irank_shm != 0:
        Accumulate_GA_shm(win_bp, bp_node, bp)

    comm_shm.Barrier()
    free_win(win_gaux)
    file_ialp.close()

    return win_bp, bp_node

def gamma_get_j_step2(self, win_vj, vj_node, bp_node):
 
    nocc = self.occ_coeff.shape[1]
    cell = self.mol
    auxcell = self.with_df.auxcell
    nao = cell.nao_nr()
    naoaux = auxcell.nao_nr()
    ao_loc = cell.ao_loc
    df_builder = self.with_df.df_builder

    #shell_slice = get_slice(range(nrank), job_size=cell.nbas)[irank]
    slice_offsets_rank = get_shell_batches(cell, nrank)[irank]

    t0 = get_current_time()
    win_gaux, gaux = get_lr_gaux(df_builder, self.with_df.kpts[0])
    accumulate_time(self.t_feri, t0)

    if slice_offsets_rank is not None:

        max_nao = get_ncols_for_memory(0.8*get_mem_spare(cell), 3*nao*naoaux, nao)
        a0, a1 = slice_offsets_rank
        shell_segs = [[a0, a0+1]]
        
        for ia in np.arange(a0+1, a1):
            a0_last, _ = shell_segs[-1]
            nao_to_be = ao_loc[ia] - ao_loc[a0_last]

            if nao_to_be > max_nao:
                shell_segs.append([ia, ia+1])
            else:
                shell_segs[-1][1] = ia+1
        
        t0 = get_current_time()
        int_kernel = gamma_feri_kernel(df_builder, intor='int3c2e', aosym='s1', 
                                    fitting=False, Gaux=gaux, ovlp=self.ovlp)
        accumulate_time(self.t_feri, t0)
        
        shls_slice = np.asarray([0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas])
        for a0, a1 in shell_segs:
            shls_slice[:2] = [a0, a1]
            al0, al1 = ao_loc[a0], ao_loc[a1]

            t0 = get_current_time()
            feri_slice = int_kernel(shls_slice) # (naopair, naux)
            accumulate_time(self.t_feri, t0)
            #print_time([f"{irank} {a0} {a1} {al1 - al0}, feri", get_current_time() - t0])

            t0 = get_current_time()
            np.dot(feri_slice, bp_node, out=vj_node[al0:al1].ravel())
            accumulate_time(self.t_j, t0)
    comm_shm.Barrier()
    free_win(win_gaux)

def _int_nuc_vloc(self, fakenuc, intor='int3c2e', aosym='s2', comp=None, ovlp=None):
    '''SR-Vnuc
    '''

    cell = self.cell
    kpts = self.kpts
    nkpts = len(kpts)

    is_gamma_point = is_zero(kpts)

    t0 = get_current_time()
    '''int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                    auxcell=fakenuc)'''
    int3c = gen_int3c_kernel(self, intor, aosym, comp=comp, j_only=True,
                                    auxcell=fakenuc)
    #bufRef, bufI = int3c()

    nao = self.cell.nao_nr()
    ao_loc = self.cell.ao_loc
    nao_pair = nao*(nao+1) // 2
    naux = fakenuc.nao_nr()

    #bufR = np.zeros((nkpts, nao_pair, naux))
    win_bufR, bufR = get_shared((nkpts, nao_pair, naux), set_zeros=True)
    if not is_gamma_point:
        win_bufI, bufI = get_shared((nkpts, nao_pair, naux), set_zeros=True)
    else:
        win_bufI = None
    
    slice_offsets = get_shell_batches(self.cell, nrank, aosym=aosym)[irank]

    if slice_offsets is not None:
        a0, a1 = slice_offsets
        shls_slice = (a0, a1, 0, self.cell.nbas, 0, fakenuc.nbas)
        al0, al1 = ao_loc[[a0, a1]]
        aop0 = al0 * (al0+1) // 2
        aop1 = al1 * (al1+1) // 2
        if is_gamma_point:
            int3c(shls_slice, outR=bufR[0, aop0:aop1], zero_buffers=True)
        else:
            raise NotImplementedError
    comm.Barrier()
    if nnode > 1:
        Accumulate_GA(array=bufR, target_rank=0)
        if not is_gamma_point:
            Accumulate_GA(array=bufI, target_rank=0)
        comm.Barrier()

    if inode == 0:
        set_zeros = False
    else:
        set_zeros = True
    win_mat, mat = get_shared((nkpts, nao_pair), set_zeros=set_zeros)

    if irank == 0:
        charge = -cell.atom_charges()

        #print(nao_pair, bufR.shape, charge.shape); sys.exit()
        if is_gamma_point:
            np.einsum('k...z,z->k...', bufR, charge, out=mat)
        else:
            mat[:] = (np.einsum('k...z,z->k...', bufR, charge) +
                    np.einsum('k...z,z->k...', bufI, charge) * 1j)
    comm_shm.Barrier()    
    
    free_win(win_bufR)
    if not is_gamma_point:
        free_win(win_bufI)
    bufR = bufI = None

    if irank == 0:
        # G = 0 contributions to SR integrals
        if (self.omega != 0 and
            (intor in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart')) and
            (cell.dimension == 3 or
                (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            nucbar = np.pi / self.omega**2 / cell.vol * charge.sum()
            if self.exclude_dd_block:
                rs_cell = self.rs_cell
                ovlp = rs_cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
                smooth_ao_idx = rs_cell.get_ao_type() == ft_ao.SMOOTH_BASIS
                for s in ovlp:
                    s[smooth_ao_idx[:,None] & smooth_ao_idx] = 0
                recontract_2d = rs_cell.recontract(dim=2)
                ovlp = [recontract_2d(s) for s in ovlp]
            else:
                ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)

            for k in range(nkpts):
                if aosym == 's1':
                    mat[k] -= nucbar * ovlp[k].ravel()
                else:
                    mat[k] -= nucbar * lib.pack_tril(ovlp[k])
    comm_shm.Barrier() 

    return win_mat, mat

def get_nuc_kenel(self, mesh=None, with_pseudo=True):
        
    if self.rs_cell is None:
        self.build()
    cell = self.cell
    kpts = self.kpts
    nkpts = len(kpts)
    nao = cell.nao_nr()
    aosym = 's2'
    nao_pair = nao * (nao+1) // 2
    mesh = self.mesh

    fakenuc = aft._fake_nuc(cell, with_pseudo=with_pseudo)
    t0 = get_current_time()
    win_vj, vj = _int_nuc_vloc(self, fakenuc) #self._int_nuc_vloc(fakenuc)
    #print_time(['vnuc pass1: analytic int', get_elapsed_time(t0)]); t0 = get_current_time()
    if cell.dimension == 0:
        nuc = lib.unpack_tril(vj)
        free_win(win_vj)
        return nuc

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    b = cell.reciprocal_vectors()
    ngrids = Gv.shape[0]
    kpt_allow = np.zeros(3)

    win_vG, vG = get_shared(ngrids, dtype=np.complex128)

    if irank_shm == 0:
        if self.exclude_dd_block and irank == 0:
            cell_d = self.rs_cell.smooth_basis_cell()
            if cell_d.nao > 0 and fakenuc.natm > 0:
                merge_dd = self.rs_cell.merge_diffused_block(aosym)
                if is_zero(kpts):
                    vj_dd = self._int_dd_block(fakenuc)
                    merge_dd(vj, vj_dd)
                else:
                    vj_ddR, vj_ddI = self._int_dd_block(fakenuc)
                    for k in range(nkpts):
                        outR = vj[k].real.copy()
                        outI = vj[k].imag.copy()
                        merge_dd(outR, vj_ddR[k])
                        merge_dd(outI, vj_ddI[k])
                        vj[k] = outR + outI * 1j

        

        aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
        charges = -cell.atom_charges()

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            with lib.temporary_env(cell, dimension=3):
                coulG_SR = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
                                                omega=-self.omega)
            coulG_LR = coulG - coulG_SR
        else:
            coulG_LR = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
                                            omega=self.omega)
        wcoulG = coulG_LR * kws
        np.einsum('i,xi,x->x', charges, aoaux, wcoulG, out=vG)
        #print_time(['vnuc pass2', get_elapsed_time(t0)]); t0 = get_current_time()

        # contributions due to pseudo.pp_int.get_gth_vlocG_part1
        if (cell.dimension == 3 or
            (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
            G0_idx = 0
            exps = np.hstack(fakenuc.bas_exps())
            vG[G0_idx] -= charges.dot(np.pi/exps) * kws
    comm_shm.Barrier()

    grid_slice = get_slice(range(nrank), job_size=ngrids)[irank]

    if grid_slice is not None:
        '''if irank_shm == 0:
            vj_data = vj
        else:'''
        vj_data = np.zeros_like(vj)
        g0_rank, g1_rank = grid_slice[0], grid_slice[-1]+1
        ngrid_rank = g1_rank - g0_rank

        ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False,
                                            kpts=kpts)
        max_memory = max(2000, (self.max_memory-lib.current_memory()[0]) // nrank_shm)
        Gblksize = max(16, int(max_memory*.8e6/16/(nao_pair*nkpts))//8*8)
        Gblksize = min(Gblksize, ngrid_rank, 200000)
        vGR = vG.real
        vGI = vG.imag

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        #for p0, p1 in lib.prange(0, ngrids, Gblksize):
        for g0, g1 in lib.prange(g0_rank, g1_rank, Gblksize):
            # shape of Gpq (nkpts, nGv, nao_pair)
            Gpq = ft_kern(Gv[g0:g1], gxyz[g0:g1], Gvbase, kpt_allow, out=buf)
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                # rho_ij(G) nuc(-G) / G^2
                # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
                vR = np.einsum('k,kx->x', vGR[g0:g1], GpqR)
                vR += np.einsum('k,kx->x', vGI[g0:g1], GpqI)
                vj_data[k] += vR
                if not is_zero(kpts[k]):
                    vI = np.einsum('k,kx->x', vGR[g0:g1], GpqI)
                    vI -= np.einsum('k,kx->x', vGI[g0:g1], GpqR)
                    vj_data[k].imag += vI
        #if irank_shm != 0:
        Accumulate_GA_shm(win_vj, vj, vj_data)
    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(vj)
        comm.Barrier()

    #print_time(['contracting Vnuc', get_elapsed_time(t0)]); t0 = get_current_time()


    if irank_shm == 0:
        vj_kpts = []
        for k, kpt in enumerate(kpts):
            if is_zero(kpt):
                vj_kpts.append(lib.unpack_tril(vj[k].real))
            else:
                vj_kpts.append(lib.unpack_tril(vj[k]))
        vj_kpts = np.asarray(vj_kpts)
    else:
        vj_kpts = [None]
    comm_shm.Barrier()
    for win in [win_vG, win_vj]:
        free_win(win)

    return vj_kpts

def get_nuc(self, kpts=None):
    kpts, is_single_kpt = _check_kpts(self, kpts)
    dfbuilder = _RSNucBuilder(self.cell, kpts).build()
    #dfbuilder = self
    nuc = get_nuc_kenel(dfbuilder, with_pseudo=False)

    if is_single_kpt:
        nuc = nuc[0]
    return nuc

def get_hcore(self, kpts=None):

    if self.use_gpu:
        if irank == 0:
            int_nuc_gpu = gpu_get_nuc(self.cell)
            int_nuc_gpu += int1e.int1e_kin(self.cell, kpts)
            hcore = cupy.asnumpy(int_nuc_gpu)
        else:
            hcore = None
    else:
        hcore = get_nuc(self)
        if irank == 0:
            hcore += self.cell.pbc_intor('int1e_kin', hermi=1)
        else:
            hcore = None

    return hcore


def correct_vk(cell, kpt, dm, vk, use_gpu=False):
    if use_gpu:
        vk_gpu = cupy.asarray(vk)
        gpu_ewald_exxdiv_for_G0(cell, kpt, cupy.asarray(dm), vk_gpu)
        cupy.asnumpy(vk_gpu, out=vk)
    else:
        _ewald_exxdiv_for_G0(cell, kpt, dm, vk)