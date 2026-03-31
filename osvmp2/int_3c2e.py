import os
import sys
import shutil
import ctypes
import numbers
import h5py
import scipy
#import scipy.linalg.interpolative as sli
import numpy as np
from pyscf.lib import logger
from pyscf.gto.moleintor import make_loc
from osvmp2.__config__ import ngpu
from osvmp2.loc.loc_addons import slice_fit, get_fit_domain, get_bfit_domain
from osvmp2 import int_prescreen
from osvmp2.mpi_addons import *
from osvmp2.osvutil import *
from osvmp2.int_2c2e import get_j2c_low
from osvmp2.pbc.gamma_int_3c2e import get_lr_gaux, gamma_feri_kernel, get_shell_batches
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
if ngpu:
    #set up gpu parameters
    import cupy
    import cupyx
    import cupyx.scipy.linalg
    from osvmp2.gpu.int_3c2e_cuda import get_feri_cuda, get_ialp_cuda
    from osvmp2.gpu.cuda_utils import avail_gpu_mem

    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)

    THREADS_PER_AXIS = 16


def get_feri(self, df_obj, qc_method, log, ijp_shape=True, 
             ovlp=None, fitting=True, with_long_range=True):
    tt = get_current_time()
    t_cal = create_timer()
    t_write = create_timer()
    self.debug = False


    t0 = get_current_time()
    file_j2c = f"j2c_{qc_method}.tmp"
    win_low, low_node = get_j2c_low(self, file_j2c)
    print_time(["j2c", get_elapsed_time(t0)], log)

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
    else:
        raise NotImplementedError
    
    max_memory = get_mem_spare(self.mol, 0.9)
    max_memory_f8 = max_memory * 1e6 // 8
    

    if self.mol.pbc:
        cell = self.mol
        auxcell = df_obj.auxcell
        nao = cell.nao_nr()
        naoaux = auxcell.nao_nr()
        ao_loc = cell.ao_loc
        df_builder = self.with_df.df_builder

        #shell_slice = get_slice(range(nrank), job_size=cell.nbas)[irank]
        slice_offsets_rank = get_shell_batches(cell, nrank)[irank]

        win_gaux, gaux = get_lr_gaux(df_builder, self.with_df.kpts[0])

        if slice_offsets_rank is not None:
            t0 = time.time()
            max_nao = get_ncols_for_memory(0.8*max_memory, 3*nao*naoaux, nao)
            a0, a1 = slice_offsets_rank
            shell_segs = [[a0, a0+1]]
            
            for ia in np.arange(a0+1, a1):
                a0_last, _ = shell_segs[-1]
                nao_to_be = ao_loc[ia] - ao_loc[a0_last]

                if nao_to_be > max_nao:
                    shell_segs.append([ia, ia+1])
                else:
                    shell_segs[-1][1] = ia+1
            
            int_kernel = gamma_feri_kernel(df_builder, intor='int3c2e', aosym='s1', 
                                            cd_j2c=low_node, Gaux=gaux, ovlp=ovlp, 
                                            fitting=fitting, with_long_range=with_long_range)
            
            shls_slice = np.asarray([0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas])
            for a0, a1 in shell_segs:
                shls_slice[:2] = [a0, a1]
                al0, al1 = ao_loc[a0], ao_loc[a1]
                nao0 = al1 - al0

                if ijp_shape:
                    if self.int_storage == 0:
                        int_kernel(shls_slice, out=self.feri_node[al0:al1])
                    else:
                        feri_data[al0:al1] = int_kernel(shls_slice)
                else:
                    feri_slice = int_kernel(shls_slice)
                    feri_data[:, al0:al1] = feri_slice.reshape(nao0, nao, self.naoaux).transpose(2, 0, 1)
            #print("Time for int_kernel: %.4f"%(time.time()-t0)); sys.exit()
        comm_shm.Barrier()
        free_win(win_gaux)

    else:
        auxmol = df_obj.auxmol
        self.shell_slice = int_prescreen.shell_prescreen(self.mol, auxmol, log, shell_slice=self.shell_slice, shell_tol=self.shell_tol)
        shell_slice_rank = int_prescreen.get_slice_rank(self.shell_slice, aslice=False)
        if shell_slice_rank is not None:
            size_feri, shell_slice_rank = int_prescreen.mem_control(self.mol, None, self.naoaux, 
                                                                    shell_slice_rank, "feri", 
                                                                    max_memory)
            feri_buf = np.empty(size_feri)
            c_feri_buf = feri_buf.ctypes.data_as(ctypes.c_void_p)
            s_slice = np.empty(6, dtype=np.int32)
            c_s_slice = s_slice.ctypes.data_as(ctypes.c_void_p)
            
            shell_segs = int_prescreen.get_shell_segs(self.mol, shell_slice_rank)

            ao_loc = make_loc(self.mol._bas, 'sph')
            for a0, a1, b_segs in shell_segs:
                al0, al1 = ao_loc[a0], ao_loc[a1]
                nao0 = al1 - al0

                for b0, b1 in b_segs:
                    be0, be1 = ao_loc[b0], ao_loc[b1]
                    nao1 = be1 - be0

                    #s_slice[:] = (a0, a1, b0, b1, self.mol.nbas, self.mol.nbas+self.auxmol.nbas)
                    s_slice[:] = (b0, b1, a0, a1, self.mol.nbas, self.mol.nbas+auxmol.nbas)

                    t1 = get_current_time()
                    aux_3c2e(auxmol, c_s_slice, c_feri_buf)
                    feri_size_seg = nao0 * nao1 * self.naoaux
                    feri_tmp = feri_buf[:feri_size_seg].reshape(self.naoaux, -1) #

                    feri_tmp = scipy.linalg.solve_triangular(low_node, feri_tmp, lower=True, overwrite_b=True, 
                                                            check_finite=False)
                    t_cal += get_elapsed_time(t1)

                    t1 = get_current_time()
                    if ijp_shape:
                        feri_data[al0:al1, be0:be1] = feri_tmp.T.reshape(nao0, nao1, self.naoaux)
                    else:
                        feri_data[:, al0:al1, be0:be1] = feri_tmp.reshape(self.naoaux, nao0, nao1)
                    t_write += get_elapsed_time(t1)
    
    comm.Barrier()
    
    if self.int_storage in {0, 3}:
        if nnode > 1:
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

    elif self.int_storage == 1:
        file_feri.close()
    
    free_win(win_low)
    
    time_list = [["writing", t_write], 
               ['computing', t_cal],
               [f"{qc_method} feri", get_elapsed_time(tt)]]
    time_list = get_max_rank_time_list(time_list)

    print_time(time_list, log)

def get_df_int3c2e(self, df_obj, qc_method="hf", ijp_shape=True, ovlp=None, 
                   fitting=True, with_long_range=True, log=None):
    if log is None:
        log = logger.Logger(sys.stdout, self.verbose)
    if self.use_gpu:
        get_feri_cuda(self, df_obj, qc_method, log, ijp_shape=ijp_shape, ovlp=ovlp, 
                 fitting=fitting, with_long_range=with_long_range)
    else:
        get_feri(self, df_obj, qc_method, log, ijp_shape=ijp_shape, ovlp=ovlp, 
                 fitting=fitting, with_long_range=with_long_range)

def get_ialp_direct(self, df_obj, qc_method, log):
     
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
    #comm.Barrier()

    
    file_j2c = f"j2c_{qc_method}.tmp"
    t1 = get_current_time()
    win_low, low_node = get_j2c_low(self, file_j2c)
    print_time(['Fitting int', get_elapsed_time(t1)], log)

    _, shell_slice_rank = int_prescreen.get_slice_rank(self.shell_slice, aslice=True)

    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=True)
    if self.int_storage == 0:
        self.win_ialp, self.ialp_node = get_shared(shape_ialp, set_zeros=True)
        ialp_data = self.ialp_node
    else:
        self.file_ialp = f"ialp_{qc_method}.tmp"
        file_ialp = h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm)
        ialp_data = file_ialp.create_dataset("ialp", shape_ialp, dtype=np.float64)
        

    if shell_slice_rank is not None:
        max_memory = get_mem_spare(mol, 0.9)

        if irank_shm == 0:
            aux_ratio = aux_ratio_node
        else:
            aux_ratio = np.zeros((nocc, naoaux))

        
        size_ialp, size_feri, shell_slice_rank = int_prescreen.mem_control(mol, nmo, naoaux, shell_slice_rank, 
                                                            "half_trans", max_memory)
        #buf_ialp = np.empty(size_ialp)
        buf_feri = np.empty(size_feri)
        c_buf_feri = buf_feri.ctypes.data_as(ctypes.c_void_p)
        s_slice = np.empty(6, dtype=np.int32)
        c_s_slice = s_slice.ctypes.data_as(ctypes.c_void_p)
        
        SHELL_SEG = slice2seg(mol, shell_slice_rank, max_nao=size_ialp//(nmo*naoaux))

        for seg_i in SHELL_SEG:
            A0, A1 = seg_i[0][0], seg_i[-1][1]
            AL0, AL1 = ao_loc[A0], ao_loc[A1]
            nao_seg = AL1 - AL0
            #ialp_tmp = buf_ialp[:nmo*nao_seg*naoaux].reshape(nmo, nao_seg, naoaux)
            #ialp_tmp[:] = 0
            ialp_tmp = np.zeros((nmo, nao_seg, naoaux))
            buf_idx0 = 0
            for a0, a1, b_list in seg_i:
                al0, al1 = ao_loc[a0], ao_loc[a1]
                nao0 = al1 - al0
                buf_idx1 = buf_idx0 + nao0
                for b0, b1 in b_list:
                    be0, be1 = ao_loc[b0], ao_loc[b1]
                    nao1 = be1 - be0
                    #s_slice = (a0, a1, b0, b1, mol.nbas, mol.nbas+auxmol.nbas)
                    #s_slice = (b0, b1, a0, a1, mol.nbas, mol.nbas+auxmol.nbas)
                    s_slice[:] = (a0, a1, b0, b1, mol.nbas, mol.nbas+auxmol.nbas)
                    t1 = get_current_time()
                    #feri_tmp = aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buf_feri)#.transpose(1,0,2) #(nao1, nao0, naoaux)
                    aux_3c2e(auxmol, c_s_slice, c_buf_feri)
                    feri_size_seg = nao0 * nao1 * naoaux
                    feri_tmp = buf_feri[:feri_size_seg].reshape(naoaux, -1).T #(naoaux, nao1, nao0)
                    accumulate_time(self.t_feri, t1)

                    t1 = get_current_time()
                    ialp_tmp[:, buf_idx0:buf_idx1] += np.dot(occ_coeff[be0:be1].T, feri_tmp.reshape(nao1, -1)).reshape(nmo, nao0, naoaux)
                    accumulate_time(self.t_ialp, t1)
                buf_idx0 = buf_idx1

            
            scipy.linalg.solve_triangular(low_node, ialp_tmp.reshape(-1, naoaux).T, 
                                            lower=True, overwrite_b=True, check_finite=False)
            
            t1 = get_current_time()
            #ialp_data.write_direct(ialp_tmp, dest_sel=np.s_[:, AL0:AL1])
            ialp_data[:, AL0:AL1] = ialp_tmp
            accumulate_time(self.t_write, t1)

            np.square(ialp_tmp, out=ialp_tmp)  # Perform squaring in-place
            aux_ratio[nocc_core:] += np.sum(ialp_tmp, axis=1)
        
        if irank_shm != 0:
            Accumulate_GA_shm(win_aux_ratio, aux_ratio_node, aux_ratio)

        buf_ialp, ialp_tmp = None, None
    
    comm.Barrier()
    free_win(win_low)

    if self.int_storage == 0:
        if nnode > 1:
            Acc_and_get_GA(self.ialp_node)
            comm.Barrier()
    else:
        file_ialp.close()

    if (qc_method == 'hf') and (self.chkfile_ialp is None):
        free_win(win_j2c)
    
    return win_aux_ratio, aux_ratio_node

def get_ialp_incore(self, df_obj, qc_method, ijp_shape):
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

    self.win_ialp, self.ialp_node = get_shared(shape_ialp, set_zeros=True)
    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=True)

    mo_indices = np.arange(nmo)

    moidx_slice = get_slice(range(nrank), job_list=mo_indices)[irank]

    if moidx_slice is not None:
        mo_idx0 = moidx_slice[0]
        mo_idx1 = moidx_slice[-1] + 1
        nmo_slice = len(moidx_slice)
        ialp_slice = self.ialp_node[mo_idx0:mo_idx1]

        if ijp_shape:
            assert self.feri_node.shape == (nao, nao, naoaux)
            np.dot(occ_coeff[:, mo_idx0:mo_idx1].T, self.feri_node.reshape(nao, -1), out=ialp_slice.reshape(nmo_slice, -1))
            
        else:
            assert self.feri_node.shape == (naoaux, nao, nao)
            ialp_slice[:] = np.dot(self.feri_node.reshape(-1, nao), occ_coeff[:, mo_idx0:mo_idx1]).reshape(naoaux, nao, nmo_slice).transpose(2, 1, 0)
        #print(ijp_shape, ialp_slice);sys.exit()
        i0, i1 = mo_idx0 + nocc_core, mo_idx1 + nocc_core
        np.einsum("ijp,ijp->ip", ialp_slice, ialp_slice, out=aux_ratio_node[i0:i1], optimize="optimal")
    
    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(self.ialp_node)
        comm.Barrier()
    return win_aux_ratio, aux_ratio_node

def get_ialp_outcore(self, df_obj, qc_method, log, zvec=True, ijp_shape=True):
    #assert not ijp_shape
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
    ialp_data = file_ialp.create_dataset("ialp", shape_ialp, dtype=np.float64)
    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=True)

    aux_slice = get_slice(range(nrank), job_size=naoaux)[irank]
    if aux_slice is not None:
        max_memory = get_mem_spare(mol, 0.9)
        naux_slice = len(aux_slice)
        max_naux = get_ncols_for_memory(0.8*max_memory, nao*(nao + nmo), naux_slice)

        with h5py.File(self.file_feri, 'r') as file_feri:
            feri_data = file_feri["feri"]
            for pidx0 in np.arange(naux_slice, step=max_naux):
                pidx1 = min(pidx0+max_naux, naux_slice)
                naux_seg = pidx1 - pidx0
                p0 = aux_slice[pidx0]
                p1 = aux_slice[pidx1-1] + 1

                if ijp_shape:
                    ialp_slice = np.dot(occ_coeff.T, feri_data[:, :, p0:p1].reshape(nao, -1)).reshape(nmo, nao, naux_seg)
                    ialp_data[:, :, p0:p1] = ialp_slice
                    aux_ratio_node[nocc_core:, p0:p1] = np.einsum("ijp,ijp->ip", ialp_slice, ialp_slice, optimize="optimal")
                    
                else:
                    #ialp_slice = np.dot(feri_data[p0:p1].reshape(-1, nao), occ_coeff).reshape(naux_seg, nao, nmo)
                    ialp_slice = np.tensordot(feri_data[p0:p1], occ_coeff, axes=([1], [0]))
                    ialp_data[:, :, p0:p1] = ialp_slice.transpose(2, 1, 0)
                    aux_ratio_node[nocc_core:, p0:p1] = np.einsum("pji,pji->ip", ialp_slice, ialp_slice, optimize="optimal")
        

    file_ialp.close()

    return win_aux_ratio, aux_ratio_node

def get_loc_ialp(self, df_obj, qc_method):
    nao = self.mol.nao_nr()
    nocc = self.mol.nelectron//2
    nmo = len(self.mo_list)
    nocc_core = nocc - nmo
    occ_coeff = self.o[:, nocc_core:]
    naoaux = self.naoaux
    mol = self.mol
    auxmol = df_obj.auxmol
    # in hf, (ial|P) is computed with occupation number weighted MO coefficients. 
    # The weights must be taken away for MP2 (ial|P)
    uo_bar = self.uo / self.mo_occ[self.mo_occ>0]**0.5 

    shape_ialp = (nmo, self.nao, self.naoaux)

    win_aux_ratio, aux_ratio_node = get_shared((nocc, naoaux), set_zeros=True)

    aux_slices = get_slice(range(nrank), job_size=naoaux)
    aux_slice = aux_slices[irank]
    if self.int_storage == 0:
        if aux_slice is not None:
            p0, p1 = aux_slice[0], aux_slice[-1]+1
            naux_slice = len(aux_slice)
            ialp_loc_slice = self.ialp_node[:, :, p0:p1]
            self.ialp_node[:, :, p0:p1] = np.einsum("ji,jkl->ikl", uo_bar, ialp_loc_slice, optimize="optimal")
            aux_ratio_node[:, p0:p1] = np.einsum("ijp,ijp->ip", ialp_loc_slice, ialp_loc_slice, optimize="optimal")

        if nnode > 1:
            # Set aux blocks assigned to other nodes to zeros for cross-node accumulations
            aux_other_nodes = []
            for rank_i, aux_rank in enumerate(aux_slices):
                if (rank_i // nnode != inode) and (aux_rank is not None):
                    aux_other_nodes.extend(aux_rank)
            shm_aux_slice = get_slice(range(nrank_shm), job_list=aux_other_nodes)[irank_shm]
            if shm_aux_slice is not None:
                self.ialp_node[:, :, shm_aux_slice] = 0.0
            comm.Barrier()

            Acc_and_get_GA(self.ialp_node)

        

    elif self.int_storage in {1, 2}:
        file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
        ialp_data = file_ialp["ialp"]
        if aux_slice is not None:
            p0_rank, p1_rank = aux_slice[0], aux_slice[-1]+1
            naux_slice = len(aux_slice)
            max_memory = get_mem_spare(mol)
            max_naux = get_ncols_for_memory(0.8 * max_memory, nocc*nao, naux_slice)

            ialp_buff = np.empty((max_naux*nocc*nao))
            for p0 in np.arange(p0_rank, p1_rank, step=max_naux):
                p1 = min(p1_rank, p0 + max_naux)
                naux_seg = p1 - p0
                ialp_seg = ialp_buff[:naux_seg*nocc*nao].reshape(nocc, nao, naux_seg)
                ialp_data.read_direct(ialp_seg, source_sel=np.s_[:, :, p0:p1])

                ialp_seg_ix = ialp_seg.reshape(nocc, -1)
                np.dot(uo_bar.T, ialp_seg_ix, out=ialp_seg_ix)
                aux_ratio_node[:, p0:p1] = np.einsum("ijp,ijp->ip", ialp_seg, ialp_seg, optimize="optimal")

                ialp_data.write_direct(ialp_seg, dest_sel=np.s_[:, :, p0:p1])
        file_ialp.close()
    comm.Barrier()

    return win_aux_ratio, aux_ratio_node

def get_ialp(self, df_obj, qc_method, log, zvec=True, ijp_shape=True, loc_trans=False):
    self.t_gpu = create_timer()
    self.t_feri = create_timer()
    self.t_ialp = create_timer()
    self.t_read = create_timer()
    self.t_write = create_timer()
    self.t_data = create_timer()

    mol = self.mol
    auxmol = df_obj.auxmol

    tt=get_current_time()

    max_memory = get_mem_spare(mol, 0.9)

    if loc_trans:
        win_aux_ratio, aux_ratio_node = get_loc_ialp(self, df_obj, qc_method)
    else:
        if (self.int_storage == 2) or (not self.cal_grad):
            win_aux_ratio, aux_ratio_node = get_ialp_direct(self, df_obj, qc_method, log)
            
        elif self.int_storage == 0:# Incore
            win_aux_ratio, aux_ratio_node = get_ialp_incore(self, df_obj, qc_method, ijp_shape)
                
        elif self.int_storage == 1:# Outcore
            win_aux_ratio, aux_ratio_node = get_ialp_outcore(self, df_obj, qc_method, ijp_shape)

        elif self.int_storage == 3:# gpu_incore
            pass
        else:
            raise NotImplementedError
        
    
    def print_shells(mol):
        for ia in range(mol.natm):
            aoslice = mol.aoslice_by_atom()
            symbol = mol.atom_symbol(ia)
            charge = mol.atom_charge(ia)
            sh0, sh1, ao0, ao1 = aoslice[ia]
            
            print(f"Atom {ia:2d}  {symbol:<3}  Z={charge:2d}    "
                f"AOs {ao0:4d} - {ao1-1:<4d} ({ao1-ao0:3d} functions)    "
                f"shells {sh0:3d} - {sh1-1}")
            
            for sh in range(sh0, sh1):
                l = mol.bas_angular(sh)
                nprim = mol.bas_nprim(sh)
                nctr  = mol.bas_nctr(sh)
                label = "spdfgh"[l] if l <= 5 else f"l={l}"
                
                print(f"  shell {sh:3d}   l = {l} ({label})    "
                    f"{nprim:2d} primitives → {nctr:2d} contracted    "
                    f"AOs {mol.ao_loc[sh]:4d} - {mol.ao_loc[sh+1]-1}")
            
            print()   # blank line between atoms

    if self.loc_fit:
        comm.Barrier()
        if nnode > 1:
            Acc_and_get_GA(aux_ratio_node)
            comm.Barrier()

        if irank ==0 :
            with h5py.File(f"fit_ratio_{qc_method}.tmp", "w") as ffit:
                ffit.create_dataset("aux_ratio", data=aux_ratio_node)
            ao_slice = self.mol.aoslice_by_atom()
            aux_slice = self.auxmol_mp2.aoslice_by_atom()
            #print_shells(self.mol)
            #print_shells(self.auxmol_mp2)
            max_l = np.max(self.mol._bas[:,1])
            #max_l = max(, np.max(self.auxmol_mp2._bas[:,1]))+1
            '''for i in self.mo_list:
                for ia in range(self.mol.natm):
                    ao_s0, ao_s1 = ao_slice[ia][:2]
                    max_l = np.max(self.mol._bas[ao_s0:ao_s1,1])
                    shell_slices = [[] for i in range(max_l+1)]
                    for s in range(ao_s0, ao_s1):
                        shell_slices[mol.bas_angular(s)].append(s)

                    aux_s0, aux_s1 = aux_slice[ia][:2]
                    aux_shell_slices = [[] for i in range(max_l+1)]
                    for s in range(aux_s0, aux_s1):
                        l = self.auxmol_mp2.bas_angular(s)
                        idx = l if l <= max_l else max_l
                        aux_shell_slices[idx].append(s)

                    for l, (shs, aux_shs) in enumerate(zip(shell_slices, aux_shell_slices)):
                        if len(shs) == 0 and len(aux_shs) == 0: continue
                        if len(shs) == 0:
                            ao_ratio = 0.0
                        else:
                            s0, s1 = shs[0], shs[-1]+1
                            al0, al1 = self.mol.ao_loc[[s0, s1]]
                            ao_ratio = np.max(self.o[al0:al1, i]**2)

                        aux_s0, aux_s1 = aux_shs[0], aux_shs[-1]+1
                        p0, p1 = self.auxmol_mp2.ao_loc[[aux_s0, aux_s1]]

                        if i == self.mo_list[0]:
                            print(i, ia, l, shs, aux_shs, "%.2e"%ao_ratio, "%.2e"%np.max(aux_ratio_node[i, p0:p1]))'''

        self.fit_list, self.fit_seg, self.nfit = get_fit_domain(aux_ratio_node, self.fit_tol)
        self.atom_close, self.bfit_seg, self.nbfit = get_bfit_domain(mol, auxmol, aux_ratio_node, self.fit_tol)
        if qc_method == 'hf' and zvec:
            self.atom_close_z, self.bfit_seg_z, self.nbfit_z = get_bfit_domain(mol, auxmol, aux_ratio_node, self.bfit_tol)
        comm_shm.Barrier()
        win_aux_ratio.Free()
    


    if self.chkfile_ialp is None:
        print_time(['computing ialp', get_elapsed_time(tt)], log)
    else:
        self.file_ialp = self.chkfile_ialp
        log.info(f"Use ialp check file:{self.chkfile_ialp}")


    time_list = [['feri', self.t_feri], ['ialp', self.t_ialp],
                 ['reading', self.t_read], ['writing', self.t_write]]
    time_list = get_max_rank_time_list(time_list)

    if irank == 0:
        print_time(time_list, log)

        
def get_df_ialp(self, df_obj, qc_method, log, zvec=True, ijp_shape=True, 
                loc_trans=False, unsort_ao=True):
    if self.use_gpu:
        get_ialp_cuda(self, df_obj, qc_method, log, zvec, ijp_shape, loc_trans, unsort_ao)
    else:
        get_ialp(self, df_obj, qc_method, log, zvec, ijp_shape, loc_trans)