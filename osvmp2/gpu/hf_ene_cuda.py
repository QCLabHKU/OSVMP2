import os
import sys
import h5py
import numpy as np
import cupy
import cupyx
from pyscf import gto
from pyscf.lib import logger
import gpu4pyscf.lib as gpu_lib
from gpu4pyscf.lib.cupy_helper import (
    contract, eigh, sandwich_dot, pack_tril, unpack_tril, get_avail_mem,
    asarray)
from osvmp2.__config__ import ngpu
from osvmp2.osvutil import *
from osvmp2.mpi_addons import *
from osvmp2.gpu.int_3c2e_cuda import get_slice_gpu, aux_e2_gpu, get_nao_range
from osvmp2.gpu.cuda_utils import (avail_gpu_mem, ave_gpu_memory, dgemm_cupy, dgemv_cupy, 
                                   free_cupy_mem, get_cupy_buffers, get_loc_buf)
from osvmp2.pbc.gpu.gamma_int_3c2e_cuda import SRInt3c2eOpt
from osvmp2.pbc.gpu.gamma_hf_ene_cuda import get_gamma_ialp_bp_cuda, get_gamma_j_step2_cuda
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
ngpu_shm = ngpu // nnode
nrank_per_gpu = nrank_shm // ngpu_shm
igpu = irank // nrank_per_gpu
igpu_shm = irank_shm // nrank_per_gpu
irank_gpu = irank % nrank_per_gpu
cupy.cuda.runtime.setDevice(igpu_shm)
comm_gpu = comm_shm.Split(color=igpu_shm, key=irank_shm)

# Set up process pairs for double buffering
ipair_rank = irank // 2
irank_pair = irank % 2
comm_pair = comm.Split(color=ipair_rank, key=irank)

def eigh_cuda(F, S):
    """
    Solves generalized eigenvalue problem F X = S X E for Hermitian matrices F and S (S positive definite).
    
    Parameters:
        F (np.ndarray): Hermitian matrix (symmetric if real).
        S (np.ndarray): Hermitian positive definite matrix (symmetric positive definite if real).
    
    Returns:
        eigenvalues (np.ndarray): Generalized eigenvalues.
        X (np.ndarray): Generalized eigenvectors (columns correspond to eigenvalues).
    """
    # Cholesky decomposition: S = L L^H (L is lower triangular)
    L = cupy.linalg.cholesky(S.T)
    
    # Solve L B = F for B (B = L^{-1} F) using forward substitution
    B = cupyx.scipy.linalg.solve_triangular(L, F.T, lower=True)
    #B = cupyx.scipy.linalg.solve_triangular(L.T, F.T, lower=False)
    
    # Solve L C = B^H for C (C = L^{-1} B^H = L^{-1} F (L^{-1})^H)
    C = cupyx.scipy.linalg.solve_triangular(L, B.conj().T, lower=True)
    #C = cupyx.scipy.linalg.solve_triangular(L.T, B.T, lower=False)
    
    # Solve standard Hermitian eigenvalue problem
    eigenvalues, Y = cupy.linalg.eigh(C)
    
    # Solve L^H X = Y for X (X = (L^{-1})^H Y) using backward substitution
    X = cupyx.scipy.linalg.solve_triangular(L.T, Y, lower=False)
    
    return eigenvalues, X


class CDIIS(gpu_lib.diis.DIIS):
    incore = None

    def __init__(self, mf=None, filename=None):
        gpu_lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.Corth = None
        self.space = 8

    def update(self, s, d, f, *args, **kwargs):
        errvec = self._sdf_err_vec(s, d, f)
        if self.incore is None:
            mem_avail = get_avail_mem()
            self.incore = errvec.nbytes*2 * (20+self.space) < mem_avail
            if not self.incore:
                logger.debug(self, 'Large system detected. DIIS intermediates '
                             'are saved in the host memory')
        nao = self.Corth.shape[1]
        errvec = pack_tril(errvec.reshape(-1,nao,nao))
        f_tril = pack_tril(f.reshape(-1,nao,nao))
        xnew = gpu_lib.diis.DIIS.update(self, f_tril, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return unpack_tril(xnew).reshape(f.shape)

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

    def _sdf_err_vec(self, s, d, f):
        '''error vector = SDF - FDS'''
        if f.ndim == s.ndim+1: # UHF
            assert len(f) == 2
            if s.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh(f[0], s)[1]
                sdf = cupy.empty_like(f)
                s.dot(d[0]).dot(f[0], out=sdf[0])
                s.dot(d[1]).dot(f[1], out=sdf[1])
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().transpose(0,2,1)
            else: # k-points
                if self.Corth is None:
                    self.Corth = cupy.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f[0], s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                Corth = asarray(self.Corth)
                sdf = cupy.empty_like(f)
                tmp = None
                tmp = contract('Kij,Kjk->Kik', d[0], f[0], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[0])
                tmp = contract('Kpq,Kqj->Kpj', sdf[0], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[0])

                tmp = contract('Kij,Kjk->Kik', d[1], f[1], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[1])
                tmp = contract('Kpq,Kqj->Kpj', sdf[1], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[1])
                errvec = sdf - sdf.conj().transpose(0,1,3,2)
        else: # RHF
            assert f.ndim == s.ndim
            if f.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh_cuda(f, s)[1]
                sdf = s.dot(d).dot(f)
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().T
            else: # k-points
                if self.Corth is None:
                    self.Corth = cupy.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f, s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                sd = contract('Kij,Kjk->Kik', s, d)
                sdf = contract('Kij,Kjk->Kik', sd, f)
                Corth = asarray(self.Corth)
                sdf = contract('Kpq,Kqj->Kpj', sdf, Corth)
                sdf = contract('Kpj,Kpi->Kij', sdf, Corth.conj())
                errvec = sdf - sdf.conj().transpose(0,2,1)
        return errvec.ravel()


def get_occ_cuda(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = cupy.argsort(mo_energy)
    nmo = mo_energy.size
    mo_occ = cupy.zeros(nmo)
    nocc = mf.mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    if mf.verbose >= logger.INFO and nocc < nmo:
        homo = float(mo_energy[e_idx[nocc-1]])
        lumo = float(mo_energy[e_idx[nocc]])
        if homo+1e-3 > lumo:
            logger.warn(mf, 'HOMO %.15g == LUMO %.15g', homo, lumo)
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', homo, lumo)
    return mo_occ

def get_jk_direct_cuda(self, dm, loc_df=False, log=None):
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

    def get_ialp_bp_cuda(shm_buf_gpu=None, loc_buf_gpu=None):
        nocc = self.occ_coeff.shape[1]
        mol = self.mol
        auxmol = self.with_df.auxmol
        nao = mol.nao_nr()
        naoaux = auxmol.nao_nr()
        
        max_l = max(mol._bas[:, gto.ANG_OF])
        max_c2s = ((max_l+1)*(max_l+2)//2) / (2*max_l + 1)

        if loc_buf_gpu is None:
            gpu_memory_f8 = ave_gpu_memory(max_mem=self.gpu_memory) * 1e6 // 8 - nao*nocc
            gpu_memory_f8 /= nrank_per_gpu
            if self.double_buffer:
                gpu_memory_f8 *= 2              
        else:
            gpu_memory_f8 = loc_buf_gpu.size

        if self.double_buffer:
            n_ialp_buff = 3
        else:
            n_ialp_buff = 2
        
        max_naux_slice = np.max(self.intopt.init_aux_ncaos)
        min_nao, max_nao = get_nao_range(self.intopt)
        max_nao1 = min_nao #min(max_nao, max(min_nao, int(0.3*gpu_memory_f8 / (2*nocc*naoaux))))
        mem_left = 0.9*gpu_memory_f8 - n_ialp_buff*nocc*max_nao1*naoaux
        max_nao0 = int(mem_left / ((max_c2s**2)*max_nao1*(naoaux + 2*max_naux_slice)))
        if max_nao0 < min_nao:
            max_nao0 = min_nao
        elif max_nao0 > max_nao:
            max_nao0 = max_nao
        max_nao1 = min(max_nao, int(0.9 * gpu_memory_f8 / ((max_c2s**2)*max_nao0*(naoaux + 2*max_naux_slice) + 
                                                           n_ialp_buff*nocc*naoaux)))
        if max_nao1 < min_nao:
            print(gpu_memory_f8, max_nao0, max_nao1)
            raise MemoryError("Insufficient GPU memory!")
        
        if irank_gpu == 0:
            occ_coeff_sorted = self.intopt.sort_orbitals(self.occ_coeff, axis=[0])
        else:
            occ_coeff_sorted = None
        
        t1 = get_current_time()
        if shm_buf_gpu is None:
            occ_coeff_gpu, occ_coeff_ptr = get_shared_cupy((nao, nocc), numpy_array=occ_coeff_sorted)
        else:
            occ_coeff_gpu = shm_buf_gpu[:nao*nocc].reshape(nao, nocc)
            if irank_gpu == 0:
                occ_coeff_gpu.set(occ_coeff_sorted)
            comm_shm.Barrier()
        accumulate_time(self.t_data, t1)

        self.intopt.build(1e-9, diag_block_with_triu=True, group_size_aoi=max_nao0, group_size_aoj=max_nao1, slice_aoj=True)

        if self.double_buffer:
            shell_slice_rank = get_slice_gpu(self.intopt, nrank//2)[ipair_rank]
        else:
            shell_slice_rank = get_slice_gpu(self.intopt, nrank)[irank]

        nao = self.nao
        naoaux = self.naoaux
        
        win_bp, bp_node = get_shared(naoaux, set_zeros=True)

        
        if nocc != self.nocc_pre:
            t0 = get_current_time()
            file_ialp = h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm)
            nao_chunk = min(536870912 // naoaux, nao) # chunks must be smaller than 4GB
            '''ialp_data = file_ialp.create_dataset('ialp', (nocc, nao, naoaux), 
                                                 dtype='f8', chunks=(1, nao_chunk, naoaux))'''
            ialp_data = create_h5py_dataset(file_ialp, "ialp", (nocc, nao, naoaux), 
                                            dtype=np.float64, chunks=(1, nao_chunk, naoaux))
            #print_time(["creating ialp dataset", get_elapsed_time(t0)], log=log)
        else:
            file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
            ialp_data = file_ialp['ialp']

        max_memory = get_mem_spare(mol, 0.9)

        t1 = get_current_time()

        accumulate_time(self.t_data, t1)

        if shell_slice_rank is not None:
            
            max_ncaux_slice = np.max(self.intopt.init_aux_ncaos)
            max_ncal_gpu = np.max(self.intopt.cart_ao_loc_si[1:] - self.intopt.cart_ao_loc_si[:-1])
            max_ncbe_gpu = np.max(self.intopt.cart_ao_loc_sj[1:] - self.intopt.cart_ao_loc_sj[:-1])
            max_nal_gpu = np.max(self.intopt.ao_loc_si[1:] - self.intopt.ao_loc_si[:-1])
            max_nbe_gpu = np.max(self.intopt.ao_loc_sj[1:] - self.intopt.ao_loc_sj[:-1])
            max_nao_gpu = np.max([al1-al0 for (al0, al1), _ in shell_slice_rank])

            recorder_feri = []
            recorder_k = []
            recorder_j = []
            recorder_data = []
            recorder_write = []

            if is_cal_rank:

                sizes = [naoaux, max_nal_gpu*max_nbe_gpu*naoaux, 
                        2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice, 
                        n_ialp_buff*nocc*max_nao_gpu*naoaux]
                                
                (bp_gpu, feri_buf_gpu, feri_aux_buf_gpu, 
                ialp_buf_gpu) = get_cupy_buffers(sizes, buf=loc_buf_gpu)
                ialp_buf_gpu = ialp_buf_gpu.reshape(n_ialp_buff, -1)

                bp_gpu.fill(0.0)
  
                unsorted_aux_ids = np.empty(naoaux, dtype=np.int32)
                unsorted_aux_ids[self.intopt._aux_ao_idx] = np.arange(naoaux, dtype=np.int32)
                unsorted_aux_ids_gpu = cupy.asarray(unsorted_aux_ids)

            if self.double_buffer:
                win_ialp_buf, double_ialp_buf_cpu = get_shared((2, nocc*max_nao_gpu*naoaux), rank_shm_buf=irank_cal, 
                                                               comm_shared=comm_pair, use_pin=True)
                comp_events = [cupy.cuda.Event(), cupy.cuda.Event()]
                data_events = [cupy.cuda.Event(), cupy.cuda.Event()]
            else:
                ialp_buf_cpu = cupyx.empty_pinned((nocc*max_nao_gpu*naoaux,))

            for bidx, ((al0, al1), cpidx_list) in enumerate(shell_slice_rank):

                if self.double_buffer:
                    io_idx = 0 if bidx % 2 else 1
                    cal_idx = 1 if bidx % 2 else 0
                    ialp_buf_cpu = double_ialp_buf_cpu[cal_idx]

                nao0 = al1 - al0

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

                        buf_idx0 = 0 if bidx % 2 else 1
                        buf_idx1 = 1 if bidx % 2 else 0
                        ialp0_gpu = ialp_comp_gpu[buf_idx0][:naoaux*nao0*nocc].reshape(naoaux, nao0, nocc)
                        ialp1_gpu = ialp_comp_gpu[buf_idx1][:naoaux*nao0*nocc].reshape(naoaux, nao0, nocc)

                        ialp0_gpu.fill(0.0)
                        for cp_ij_id in cpidx_list:
                            cpi = self.intopt.cp_idx[cp_ij_id]
                            be0, be1 = self.intopt.ao_loc_si[cpi], self.intopt.ao_loc_si[cpi+1]
                            nao1 = be1 - be0
                            t1 = get_current_time()
                            int3c_slice = feri_buf_gpu[:(nao0*nao1*naoaux)]
                            aux_e2_gpu(self.intopt, cp_ij_id, feri_shape="pji", out=int3c_slice, buf_aux=feri_aux_buf_gpu)
                            #accumulate_time(self.t_feri, t1)
                            recorder_feri.append(record_elapsed_time(t1))

                            t1 = get_current_time()
                            #dgemm_cupy(0, 0, int3c_slice.reshape(-1, nao1), occ_coeff_gpu[be0:be1], ialp_gpu, 1.0, 1.0)
                            dgemm_cupy(0, 0, int3c_slice.reshape(-1, nao1), occ_coeff_gpu[be0:be1], ialp0_gpu, 1.0, 1.0)
                            #accumulate_time(self.t_k, t1)
                            recorder_k.append(record_elapsed_time(t1))

                            #istream_comp.synchronize()

                        #ialp_gpu = ialp_gpu[unsorted_aux_ids_gpu]
                        cupy.take(ialp0_gpu, unsorted_aux_ids_gpu, axis=0, out=ialp1_gpu)


                        t1 = get_current_time()
                        #dgemv_cupy(1, ialp_gpu.reshape(naoaux, -1), occ_coeff_gpu[al0:al1].ravel(), bp_gpu, 1.0, 1.0)
                        dgemv_cupy(1, ialp1_gpu.reshape(naoaux, -1), occ_coeff_gpu[al0:al1].ravel(), bp_gpu, 1.0, 1.0)
                        recorder_j.append(record_elapsed_time(t1))
                    
                        t1 = get_current_time()
                        #ialp_gpu = cupy.ascontiguousarray(ialp_gpu.transpose(2, 1, 0))
                        ialp0_gpu = ialp0_gpu.reshape(nocc, nao0, naoaux)
                        ialp0_gpu[:] = ialp1_gpu.transpose(2, 1, 0)
                        recorder_k.append(record_elapsed_time(t1))

                        if self.double_buffer:
                            comp_events[comp_event_idx].record()

                    with istream_data:
                        if self.double_buffer:
                            if bidx > 0:
                                istream_data.wait_event(comp_events[data_event_idx])

                                t1 = get_current_time()
                                (prev_al0, prev_al1), _ = shell_slice_rank[bidx - 1]
                                prev_nao0 = prev_al1 - prev_al0
                                ialp_cpu = ialp_buf_cpu[:prev_nao0*nocc*naoaux].reshape(nocc, prev_nao0, naoaux)
                                ialp_gpu = ialp_data_gpu[:prev_nao0*nocc*naoaux].reshape(nocc, prev_nao0, naoaux)
                                cupy.asnumpy(ialp_gpu, out=ialp_cpu)
                                recorder_data.append(record_elapsed_time(t1))

                                data_events[data_event_idx].record()
                                istream_data.synchronize()
                        else:
                            t1 = get_current_time()
                            ialp_cpu = ialp_buf_cpu[:nao0*nocc*naoaux].reshape(nocc, nao0, naoaux)
                            cupy.asnumpy(ialp0_gpu, out=ialp_cpu)
                            recorder_data.append(record_elapsed_time(t1))

                            istream_data.synchronize()

                    '''if self.double_buffer:
                        istream_comp.synchronize()
                        istream_data.synchronize()'''
                
                if self.double_buffer:
                    if is_io_rank and bidx > 1:
                        t1 = get_current_time()
                        (pprev_al0, pprev_al1), _ = shell_slice_rank[bidx - 2]
                        pprev_nao0 = pprev_al1 - pprev_al0
                        ialp_write_cpu = double_ialp_buf_cpu[io_idx][:nocc*pprev_nao0*naoaux].reshape(nocc, pprev_nao0, naoaux)
                        ialp_data.write_direct(ialp_write_cpu, dest_sel=np.s_[:, pprev_al0:pprev_al1])
                        recorder_write.append(record_elapsed_time(t1))

                    comm_pair.Barrier()
                else:
                    t1 = get_current_time()
                    ialp_data.write_direct(ialp_cpu, dest_sel=np.s_[:, al0:al1])
                    recorder_write.append(record_elapsed_time(t1))

            if is_cal_rank:
                with istream_comp:
                    t1 = get_current_time()
                    bp = cupy.asnumpy(bp_gpu)
                    #accumulate_time(self.t_data, t1)
                    recorder_data.append(record_elapsed_time(t1))
                
                    #if irank_shm != 0:
                    Accumulate_GA_shm(win_bp, bp_node, bp) 

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
                            ialp_cpu = ialp_buf_cpu[:prev_nao0*nocc*naoaux].reshape(nocc, prev_nao0, naoaux)
                            ialp_gpu = ialp_buf_gpu[data_idx][:prev_nao0*nocc*naoaux].reshape(nocc, prev_nao0, naoaux)
                            cupy.asnumpy(ialp_gpu, out=ialp_cpu)
                            recorder_data.append(record_elapsed_time(t1))

                            data_events[data_event_idx].record()

                            istream_data.synchronize()

                    if is_io_rank:
                        t1 = get_current_time()
                        (pprev_al0, pprev_al1), _ = shell_slice_rank[bidx - 2]
                        pprev_nao0 = pprev_al1 - pprev_al0
                        ialp_write_cpu = double_ialp_buf_cpu[io_idx][:nocc*pprev_nao0*naoaux].reshape(nocc, pprev_nao0, naoaux)
                        ialp_data.write_direct(ialp_write_cpu, dest_sel=np.s_[:, pprev_al0:pprev_al1])
                        recorder_write.append(record_elapsed_time(t1))

                    comm_pair.Barrier()

                istream_comp.synchronize()

                if is_cal_rank:
                    unregister_pinned_memory(double_ialp_buf_cpu)
                win_ialp_buf.Free()


        batch_accumulate_time(self.t_feri, recorder_feri)
        batch_accumulate_time(self.t_k, recorder_k)
        batch_accumulate_time(self.t_j, recorder_j)
        batch_accumulate_time(self.t_data, recorder_data)
        batch_accumulate_time(self.t_write, recorder_write)

        comm.Barrier()
        file_ialp.close()

        if shm_buf_gpu is None:
            close_ipc_handle(occ_coeff_ptr)
        
        self.intopt.free_bpcache()

        return win_bp, bp_node
    

    def fit_bp(bp_node, j2c_type="j2c", buffer=None):
        j2c_type = "low"
        if nnode > 1:
            bp_to_col = bp_node if irank_shm == 0 else None
            win_bp_col = get_win_col(bp_to_col)
            Accumulate_GA(win_bp_col, bp_node)
            win_bp_col.Fence()
        
        if irank == 0:
            t1 = get_current_time()
            #j2c = read_file(self.file_j2c, j2c_type)#, buffer=low_node)#buf_pq)
            with h5py.File(self.file_j2c, 'r') as f:
                j2c = np.asarray(f[j2c_type])
            accumulate_time(self.t_read, t1)

            '''j2c = np.ascontiguousarray(j2c.T)

            t1 = get_current_time()
            j2c_gpu = cupy.asarray(j2c)
            bp_gpu = cupy.asarray(bp_node)
            accumulate_time(self.t_data, t1)'''

            if buffer is None:
                j2c_gpu = cupy.empty((naoaux, naoaux))
                j2c_load = cupy.empty((naoaux, naoaux))
                bp_gpu = cupy.asarray(bp_node)
            else:
                j2c_size = naoaux*naoaux
                j2c_gpu = buffer[:j2c_size].reshape(naoaux, naoaux)
                j2c_load = buffer[j2c_size:2*j2c_size].reshape(naoaux, naoaux)
                bp_gpu = buffer[2*j2c_size:2*j2c_size+naoaux] 
                bp_gpu.set(bp_node)
                

            j2c_load.set(j2c)
            j2c_gpu[:] = j2c_load.T

            t1 = get_current_time()
            if j2c_type == "j2c":
                j2c_load = None
                bp_gpu = cupy.linalg.solve(j2c_gpu.T, bp_gpu)#, overwrite_b=True)
            elif j2c_type == "low":
                cupyx.scipy.linalg.solve_triangular(j2c_gpu.T, bp_gpu.reshape(-1, naoaux).T, 
                                                    lower=True, overwrite_b=True, check_finite=False)
                cupyx.scipy.linalg.solve_triangular(j2c_load.T, bp_gpu.reshape(-1, naoaux).T, 
                                                    lower=False, overwrite_b=True, check_finite=False)
                j2c_load = None
                
            accumulate_time(self.t_j, t1)

            t1 = get_current_time()
            cupy.asnumpy(bp_gpu, out=bp_node)
            accumulate_time(self.t_data, t1)

            j2c = None
        comm.Barrier()
        
        #buf_pq = None
        if nnode > 1: # and irank_shm== 0:
            if irank_shm== 0 and irank != 0:
                Get_GA(win_bp_col, bp_node)
            win_bp_col.Fence()
            free_win(win_bp_col)
        
        return bp_node

    #Step 2 for J matrix
    def get_j_step2_cuda(win_vj, vj_node, bp_node, shm_buf_gpu=None, loc_buf_gpu=None):

        nao = self.nao
        naoaux = self.intopt.auxmol.nao

        if loc_buf_gpu is None:
            gpu_memory_f8 = ave_gpu_memory(self.gpu_memory) * 1e6 // 8
            feri_mem_f8 = gpu_memory_f8 - nao * nao
            feri_mem_f8 /= nrank_per_gpu
        else:
            feri_mem_f8 = loc_buf_gpu.size

        max_naux_slice = np.max(self.intopt.init_aux_ncaos)
        min_nao, max_nao = get_nao_range(self.intopt)
        max_l = max(mol._bas[:, gto.ANG_OF])
        max_s2c = (2*max_l + 1) / ((max_l+1)*(max_l+2)//2) 
        fraction = 0.9 * (max_s2c**2)
        max_nao1 = min(max_nao, max(min_nao, int(fraction*feri_mem_f8 / (max_nao*(naoaux + 2*max_naux_slice)))))
        max_nao0 = max(min_nao, int(fraction*feri_mem_f8 / (max_nao1*(naoaux + 2*max_naux_slice))))
        if max_nao0 > max_nao:
            max_nao0 = max_nao
            max_nao1 = min(max_nao, int(fraction*feri_mem_f8 / (max_nao0*(naoaux + 2*max_naux_slice))))

        if irank_gpu == 0:
            bp_sorted = self.intopt.sort_orbitals(bp_node, aux_axis=[0])
        else:
            bp_sorted = None

        if shm_buf_gpu is None:
            '''vj_gpu, vj_ptr = get_shared_cupy((nao, nao), set_zeros=True)
            bp_gpu, bp_ptr = get_shared_cupy((naoaux), numpy_array=bp_sorted)'''
            shm_buf_gpu, shm_ptr = get_shared_cupy((nao*nao+naoaux))
        else:
            shm_ptr = None

        vj_gpu = shm_buf_gpu[:nao*nao].reshape(nao, nao)
        bp_gpu = shm_buf_gpu[nao*nao:nao*nao+naoaux]

        if irank_gpu == 0:
            vj_gpu.fill(0.0)
            bp_gpu.set(bp_sorted)
        comm_shm.Barrier()

        self.intopt.build(1e-9, diag_block_with_triu=True, group_size_aoj=max_nao1, group_size_aoi=max_nao0, slice_aoj=True)
        shell_slice_ranks = get_slice_gpu(self.intopt, nrank)
        shell_slice_rank = shell_slice_ranks[irank]

        if shell_slice_rank is not None:
            max_ncal_gpu = np.max(self.intopt.cart_ao_loc_si[1:] - self.intopt.cart_ao_loc_si[:-1])
            max_ncbe_gpu = np.max(self.intopt.cart_ao_loc_sj[1:] - self.intopt.cart_ao_loc_sj[:-1])
            max_nsal_gpu = np.max(self.intopt.ao_loc_si[1:] - self.intopt.ao_loc_si[:-1])
            max_nsbe_gpu = np.max(self.intopt.ao_loc_sj[1:] - self.intopt.ao_loc_sj[:-1])
            max_ncaux_slice = np.max(self.intopt.init_aux_ncaos)

            '''feri_buf_gpu = cupy.empty(max_nao0*max_nao1*naoaux)
            feri_aux_buf_gpu = cupy.empty(2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice)'''

            sizes = [max_nao0*max_nao1*naoaux, 
                     2*max_ncal_gpu*max_ncbe_gpu*max_ncaux_slice, 
                     max_nsal_gpu*max_nsbe_gpu]
            feri_buf_gpu, feri_aux_buf_gpu, vj_buf_gpu = get_cupy_buffers(sizes, buf=loc_buf_gpu)

            recorder_feri = []
            recorder_j = []

            for (be0, be1), ij_ids in shell_slice_rank:
                nao1 = be1 - be0
                for cp_ij_id in ij_ids:
                    cpi = self.intopt.cp_idx[cp_ij_id]
                    cpj = self.intopt.cp_jdx[cp_ij_id]
                    al0, al1 = self.intopt.ao_loc_si[cpi], self.intopt.ao_loc_si[cpi+1]
                    #be0, be1 = self.intopt.ao_loc_sj[cpj], self.intopt.ao_loc_sj[cpj+1]
                    nao0 = al1 - al0

                    t1 = get_current_time()
                    #int3c_slice = aux_e2_gpu(self.intopt, cp_ij_id)#, omega=None, out=int3c_slice)
                    int3c_slice = feri_buf_gpu[:nao0*nao1*naoaux]
                    aux_e2_gpu(self.intopt, cp_ij_id, out=int3c_slice, buf_aux=feri_aux_buf_gpu)
                    #accumulate_time(self.t_feri, t1)
                    recorder_feri.append(record_elapsed_time(t1))

                    t1 = get_current_time()
                    #vj_gpu[be0:be1, al0:al1] = cupy.dot(bp_gpu, int3c_slice.reshape(self.naoaux, -1)).reshape(nao1, nao0)
                    vj_slice = vj_buf_gpu[:nao1*nao0]
                    vj_gpu[be0:be1, al0:al1] = cupy.dot(bp_gpu, int3c_slice.reshape(self.naoaux, -1), out=vj_slice).reshape(nao1, nao0)
                    #accumulate_time(self.t_j, t1)
                    recorder_j.append(record_elapsed_time(t1))

                    #cupy.cuda.Stream.null.synchronize()

        batch_accumulate_time(self.t_feri, recorder_feri)
        batch_accumulate_time(self.t_j, recorder_j)

        cupy.cuda.Stream.null.synchronize()
        comm_shm.Barrier()

        if irank_gpu == 0:
            t1 = get_current_time()
            vj = cupy.asnumpy(vj_gpu)
            accumulate_time(self.t_data, t1)
            vj_sorted = self.intopt.unsort_orbitals(vj, axis=[0, 1])
            np.copyto(vj, vj_sorted)

            #if irank_shm != 0:
            Accumulate_GA_shm(win_vj, vj_node, vj)
            
        else:
            vj = None

        comm_shm.Barrier()

        '''if shm_buf_gpu is None:
            close_ipc_handle(vj_ptr)
            close_ipc_handle(bp_ptr)'''
        if shm_ptr is not None:
            close_ipc_handle(shm_ptr)
        
        self.intopt.free_bpcache()
        
        return vj
    

    #Step 2 for K matrix
    def get_k_step2_cuda(win_vk, vk_node, shm_buf_gpu=None, loc_buf_gpu=None):

        if self.double_buffer:
            mo_slice = get_slice(job_size=nocc, rank_list=range(nrank//2))[ipair_rank]
        else:
            mo_slice = get_slice(job_size=nocc, rank_list=range(nrank))[irank]

        mem_avail = get_mem_spare(mol)
        max_memory = mem_avail - (8*naoaux*naoaux*1e-6)
        max_gpu_mem_f8 = avail_gpu_mem(max_mem=self.gpu_memory) * 0.9 *1e6 / 8

        if max_memory < 0:
            raise MemoryError("No sufficient memory")

        if not self.mol.pbc:
            win_low, low_node = get_shared((naoaux, naoaux))
            if irank_shm == 0:
                t1 = get_current_time()
                #np.copyto(low_node, read_file(self.file_j2c, 'low').T) #transpose
                read_file(self.file_j2c, 'low', buffer=low_node)
                accumulate_time(self.t_read, t1)
            comm_shm.Barrier()

            t1 = get_current_time()
            if shm_buf_gpu is None:
                low_gpu, low_ptr = get_shared_cupy((naoaux, naoaux))
                low_load = cupy.empty_like(low_gpu)
            else:
                low_gpu = shm_buf_gpu[:naoaux*naoaux].reshape(naoaux, naoaux)
                low_load = shm_buf_gpu[naoaux*naoaux:2*naoaux*naoaux].reshape(naoaux, naoaux)

            if irank_gpu == 0:
                low_load.set(low_node)
                low_gpu[:] = low_load.T
                low_load = None
            
            comm_shm.Barrier()
            accumulate_time(self.t_data, t1)

        if self.shared_int:
            file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
        else:
            file_ialp = h5py.File(self.file_ialp, 'r')#, driver='mpio', comm=comm)
        ialp_data = file_ialp["ialp"]

        if mo_slice is not None:
            if loc_buf_gpu is None:
                if self.mol.pbc:
                    max_gpu_mem_f8 = max_gpu_mem_f8 / nrank_per_gpu
                else:
                    max_gpu_mem_f8 = (max_gpu_mem_f8 - naoaux * naoaux) / nrank_per_gpu
            else:
                max_gpu_mem_f8 = loc_buf_gpu.size
            
            ialp_size = nao*naoaux
            if self.double_buffer:
                win_ialp_buf, double_ialp_buf_cpu = get_shared((2, nao, naoaux), rank_shm_buf=irank_cal, 
                                                               comm_shared=comm_pair, use_pin=True)
                ialp_size *= 2
            else:
                ialp_buf_cpu = cupyx.empty_pinned((nao, naoaux), dtype=cupy.float64)
            #ialp_buf_cpu = np.empty((ialp_size,))

            if is_cal_rank:
                vk = cupyx.empty_pinned((nao, nao), dtype=cupy.float64)
                ialp_buf_gpu, vk_gpu = get_cupy_buffers([ialp_size, nao*nao], buf=loc_buf_gpu)

                vk_gpu = vk_gpu.reshape(nao, nao)
                vk_gpu.fill(0.0)

                if self.double_buffer:
                    ialp_buf_gpu = ialp_buf_gpu.reshape(2, nao, naoaux)
                else:
                    ialp_buf_gpu = ialp_buf_gpu.reshape(nao, naoaux)

                unsorted_ao_ids = np.empty(nao, dtype=np.int32)
                if self.mol.pbc:
                    '''cell = self.mol
                    auxcell = self.auxmol
                    omega = self.with_df.df_builder.omega
                    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, -omega, fitting=True)
                    unsorted_ao_ids[int3c2e_opt.ao_idx] = np.arange(nao, dtype=np.int32)'''
                    unsorted_ao_ids[self.ao_idx] = np.arange(nao, dtype=np.int32)
                else:
                    unsorted_ao_ids[self.intopt._ao_idx] = np.arange(nao, dtype=np.int32)

                unsorted_ao_ids_gpu = cupy.asarray(unsorted_ao_ids)

            recorder_k = []
            recorder_data = []
            recorder_read = []
            recorder_write = []

            if self.double_buffer: 
                comp_events = [cupy.cuda.Event(), cupy.cuda.Event()]
                data_events = [cupy.cuda.Event(), cupy.cuda.Event()]

                for iidx in range(2):
                    io_idx = 0 if iidx % 2 else 1
                    cal_idx = 1 if iidx % 2 else 0
                    if is_io_rank:
                        ialp_io_cpu = double_ialp_buf_cpu[io_idx]
                        t1 = get_current_time()
                        ialp_data.read_direct(ialp_io_cpu, np.s_[mo_slice[iidx]])
                        recorder_read.append(record_elapsed_time(t1))
                    
                    if is_cal_rank and iidx > 0:
                        data_idx = 1 if iidx % 2 else 0
                        with istream_data:
                            t1 = get_current_time()
                            ialp_cpu = double_ialp_buf_cpu[cal_idx]
                            ialp_gpu = ialp_buf_gpu[data_idx]
                            ialp_gpu.set(ialp_cpu)
                            recorder_data.append(record_elapsed_time(t1))

                            data_events[data_idx].record()

                            istream_data.synchronize()

                    comm_pair.Barrier()

            for iidx, i in enumerate(mo_slice):
                if self.double_buffer:
                    io_idx = 0 if iidx % 2 else 1
                    cal_idx = 1 if iidx % 2 else 0
                    data_idx = 1 if iidx % 2 else 0
                    comp_idx = 0 if iidx % 2 else 1
                    if self.shared_int: NotImplementedError
                    
                    if is_io_rank:
                        nnext_iidx = iidx + 2
                        if nnext_iidx < len(mo_slice):
                            ialp_read_cpu = double_ialp_buf_cpu[io_idx]
                            nnext_i = mo_slice[nnext_iidx]
                            t1 = get_current_time()
                            ialp_data.read_direct(ialp_read_cpu, np.s_[nnext_i])
                            recorder_read.append(record_elapsed_time(t1)) 
                    
                    if is_cal_rank:
                        ialp_cpu = double_ialp_buf_cpu[cal_idx].reshape(nao, naoaux)
                        ialp_comp_gpu = ialp_buf_gpu[comp_idx]
                        ialp_data_gpu = ialp_buf_gpu[data_idx]

                        if iidx + 1 < len(mo_slice):
                            with istream_data:
                                if iidx > 0:
                                    istream_data.wait_event(comp_events[data_idx])

                                t1 = get_current_time()
                                ialp_data_gpu.set(ialp_cpu)
                                #accumulate_time(self.t_data, t1)
                                recorder_data.append(record_elapsed_time(t1))

                                data_events[data_idx].record()

                                istream_data.synchronize()
                            
                else:
                    t1 = get_current_time()
                    ialp_cpu = ialp_buf_cpu.reshape(nao, naoaux)
                    ialp_data.read_direct(ialp_cpu, np.s_[i])
                    recorder_read.append(record_elapsed_time(t1))   

                    t1 = get_current_time()
                    ialp_buf_gpu.set(ialp_cpu)
                    #accumulate_time(self.t_data, t1)
                    recorder_data.append(record_elapsed_time(t1))  
                    ialp_comp_gpu = ialp_buf_gpu

                    cupy.cuda.Stream.null.synchronize()

                if is_cal_rank:
                    with istream_comp:
                        if self.double_buffer:
                            istream_comp.wait_event(data_events[comp_idx])

                        t1 = get_current_time()
                        if not self.mol.pbc:
                            cupyx.scipy.linalg.solve_triangular(low_gpu.T, ialp_comp_gpu.T, 
                                                                lower=True, overwrite_b=True, check_finite=False)
                        dgemm_cupy(0, 1, ialp_comp_gpu, ialp_comp_gpu, vk_gpu, 1.0, 1.0)
                        recorder_k.append(record_elapsed_time(t1))                   
                    
                        #cupy.cuda.Stream.null.synchronize()

                        if self.double_buffer:
                            comp_events[comp_idx].record()
            
                if self.shared_int:
                    raise NotImplementedError
                    t1 = get_current_time()
                    cupy.asnumpy(ialp_comp_gpu[unsorted_ao_ids_gpu], out=ialp_cpu)
                    recorder_data.append(record_elapsed_time(t1))

                    if not self.double_buffer:
                        t1 = get_current_time()
                        file_ialp["ialp"].write_direct(ialp_cpu, dest_sel=np.s_[i0:i1])
                        recorder_write.append(record_elapsed_time(t1))
                
                if self.double_buffer:
                    comm_pair.Barrier()
            
            if is_cal_rank:
                if self.double_buffer:
                    istream_comp.synchronize()
                t1 = get_current_time()
                cupy.asnumpy(vk_gpu, out=vk)
                recorder_k.append(record_elapsed_time(t1))

                Accumulate_GA_shm(win_vk, vk_node, vk)  
            else:
                vk = None

            batch_accumulate_time(self.t_k, recorder_k)
            batch_accumulate_time(self.t_data, recorder_data)
            batch_accumulate_time(self.t_read, recorder_read)
            batch_accumulate_time(self.t_write, recorder_write)                

            if self.double_buffer:
                if is_cal_rank:
                    unregister_pinned_memory(double_ialp_buf_cpu)
                win_ialp_buf.Free()

        else:
            vk = None   

        comm_shm.Barrier()
        if irank_shm == 0:
            vk_node[:] = self.intopt.unsort_orbitals(vk_node, axis=[0, 1])

        comm.Barrier()

        

        file_ialp.close()
        
        if not self.mol.pbc:
            free_win(win_low); low_node=None
            if shm_buf_gpu is None:
                close_ipc_handle(low_ptr)
        return vk
 
    ###########################################################################
    if log is None:
        log = logger.Logger(sys.stdout, self.verbose)

    nao = self.mol.nao_nr()
    naoaux = self.with_df.auxmol.nao_nr()
    nocc = self.occ_coeff.shape[-1]
    ao_loc = make_loc(self.mol._bas, 'sph')
    mol = self.mol
    auxmol = self.with_df.auxmol

    win_vj, vj_node = get_shared((nao, nao), set_zeros=True)
    win_vk, vk_node = get_shared((nao, nao), set_zeros=True)
    comm.Barrier()

    t0 = get_current_time()
    t_io_init = self.t_read + self.t_write
    t_cal_init = self.t_j + self.t_k + self.t_feri
    t_data_init = np.copy(self.t_data)
    t_feri_init = np.copy(self.t_feri)
    t_j_init = np.copy(self.t_j)
    t_k_init = np.copy(self.t_k)

    if self.mol.pbc:
        win_bp, bp_node = get_gamma_ialp_bp_cuda(self, dm)
    else:
        gpu_memory_f8 = ave_gpu_memory(max_mem=self.gpu_memory) * 1e6 // 8
        buf_size_f8 = int(0.8 * gpu_memory_f8)
        
        use_buffer = False

        if use_buffer:
            shm_buf_gpu, shm_buf_ptr = get_shared_cupy(buf_size_f8)
            loc_buf_gpu = get_loc_buf(shm_buf_gpu, nao * nocc)
        else:
            shm_buf_gpu = loc_buf_gpu = None

        win_bp, bp_node = get_ialp_bp_cuda(shm_buf_gpu, loc_buf_gpu)
        bp_node = fit_bp(bp_node, buffer=shm_buf_gpu)

    t_cart2sph = sum_elapsed_time(self.intopt.recorder_cart2sph)
    t_ferikern = sum_elapsed_time(self.intopt.recorder_ferikern)
    t_prep = sum_elapsed_time(self.intopt.recorder_prep)
    t_feritot = sum_elapsed_time(self.intopt.recorder_feritot)

    print_time([['feri cart2sph', t_cart2sph], ['feri kernel', t_ferikern],
                ['feri prep', t_prep], ['feri total', t_feritot]], log=log)


    tcal, tio, tdata = get_max_rank_times([(self.t_j+self.t_k+self.t_feri)-t_cal_init, 
                                           self.t_read+self.t_write-t_io_init, 
                                           self.t_data-t_data_init])

    if irank == 0:
        print_mem('Half trans (%.2f secs)'%(get_elapsed_time(t0))[1], self.pid_list, log)
        print_time([["calculation", tcal],
                    ["IO", tio], 
                    ["GPU-CPU data", tdata]], log)
    
    t0 = get_current_time()
    t_io_init = self.t_read + self.t_write
    t_cal_init = self.t_j + self.t_k + self.t_feri
    t_data_init = np.copy(self.t_data)

    #free_cupy_mem()

    if self.mol.pbc:
        get_gamma_j_step2_cuda(self, win_vj, vj_node, bp_node)
    else:
        if use_buffer:
            loc_buf_gpu = get_loc_buf(shm_buf_gpu, nao*nao+naoaux)

        get_j_step2_cuda(win_vj, vj_node, bp_node, shm_buf_gpu, loc_buf_gpu)
    
    free_win(win_bp)
                    

    tcal, tio, tdata = get_max_rank_times([(self.t_j+self.t_k+self.t_feri)-t_cal_init, 
                                           self.t_read+self.t_write-t_io_init, 
                                           self.t_data-t_data_init])
    if irank == 0:
        print_mem('J matrix (%.2f secs)'%(get_elapsed_time(t0))[1], self.pid_list, log)
        print_time([["calculation", tcal],
                    ["IO", tio], 
                    ["GPU-CPU data", tdata]], log)

    t0 = get_current_time()
    t_io_init = self.t_read + self.t_write
    t_cal_init = self.t_j + self.t_k + self.t_feri
    t_data_init = np.copy(self.t_data)

    #free_cupy_mem()
    if use_buffer:
        loc_buf_gpu = get_loc_buf(shm_buf_gpu, naoaux*naoaux)
    get_k_step2_cuda(win_vk, vk_node, shm_buf_gpu, loc_buf_gpu)
    
    if nnode > 1:
        Acc_and_get_GA(vj_node)
        Acc_and_get_GA(vk_node)
        comm.Barrier()

    if irank_shm == 0:
        vj = np.copy(vj_node)
        vk = np.copy(vk_node)
    else:
        vj, vk = None, None
    #sys.exit()
    comm_shm.Barrier()
    for win in [win_vj, win_vk]:
        free_win(win)

    tcal, tio, tdata = get_max_rank_times([(self.t_j+self.t_k+self.t_feri)-t_cal_init, 
                                           self.t_read+self.t_write-t_io_init, 
                                           self.t_data-t_data_init])

    if irank == 0:
        print_mem('K matrix (%.2f secs)'%(get_elapsed_time(t0))[1], self.pid_list, log)
        print_time([["calculation", tcal],
                    ["IO", tio], 
                    ["GPU-CPU data", tdata]], log)
    self.nocc_pre = nocc

    #free_cupy_mem()
    if use_buffer:
        close_ipc_handle(shm_buf_ptr)
        shm_buf_gpu = loc_buf_gpu = None

    free_cupy_mem()
    return vj, vk


def get_jk_with_int_cuda(self, dm, loc_df=False, log=None):

    tt = t1 = get_current_time()
    if log is None:
        log = logger.Logger(sys.stdout, self.verbose)
    nao = self.mol.nao_nr()
    naoaux = self.with_df.auxmol.nao_nr()
    nocc = self.occ_coeff.shape[-1]

    win_vj, vj_node = get_shared((nao, nao), set_zeros=True)
    win_vk, vk_node = get_shared((nao, nao), set_zeros=True)

    aux_slice = get_slice(range(nrank), job_size=naoaux)[irank]

    if self.shared_int:
        if self.int_storage in {0, 3}:
            ialp_data = self.ialp_node
        else:
            file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
            ialp_data = file_ialp["ialp"]
    
    if not (self.unsort_alpha and self.unsort_beta):
        occ_coeff_buff_gpu, occ_coeff_ptr = get_shared_cupy((2, nao, nocc))
        occ_coeff_gpu = occ_coeff_buff_gpu[0]
        sorted_occ_coeff_gpu = occ_coeff_buff_gpu[1]
        if irank_gpu == 0:
            occ_coeff_gpu.set(self.occ_coeff)
            sorted_occ_coeff_gpu.set(self.intopt.sort_orbitals(self.occ_coeff, axis=[0]))
    else:
        occ_coeff_gpu, occ_coeff_ptr = get_shared_cupy((nao, nocc), numpy_array=self.occ_coeff)

    comm_shm.Barrier()

    if aux_slice is not None:
        vj_gpu = cupy.zeros((nao, nao), dtype='f8')
        vk_gpu = cupy.zeros((nao, nao), dtype='f8')
        #occ_coeff_gpu = cupy.asarray(self.occ_coeff)


        naux_slice = len(aux_slice)
        if self.int_storage == 3:
            max_naux = naux_slice
            feri_gpu = self.feri_gpu
        else:
            gpu_mem_left = (self.gpu_memory - (nao*nocc * 8 * 1e-6)) / nrank_per_gpu 
            gpu_mem_left -= 2 * nao*nao * 8 * 1e-6
            max_naux = get_ncols_for_memory(0.8*gpu_mem_left, nao * (nao + nocc), naux_slice)
            feri_buf_gpu = cupy.empty(nao*nao*max_naux)

            if self.int_storage == 0:
                feri_save = self.feri_node
            else:
                file_feri = h5py.File(self.file_feri, 'r')
                feri_save = file_feri["feri"]

        for pidx0 in np.arange(naux_slice, step=max_naux):
            pidx1 = min(pidx0 + max_naux, naux_slice)
            p0 = aux_slice[pidx0]
            p1 = aux_slice[pidx1-1] + 1
            naux_seg = pidx1 - pidx0

            if self.int_storage != 3:
                t1 = get_current_time()
                if self.ijp_shape:
                    feri_cpu = np.asarray(feri_save[:, :, p0:p1]) #(nao, nao, naux_seg)
                else:
                    feri_cpu = np.asarray(feri_save[p0:p1])
                accumulate_time(self.t_read, t1)

                t1 = get_current_time()
                feri_gpu = feri_buf_gpu[:naux_seg*nao*nao].reshape(feri_cpu.shape)
                feri_gpu.set(feri_cpu)
                accumulate_time(self.t_data, t1)

            if self.ijp_shape:
                t1 = get_current_time()
                ialp_gpu = cupy.empty((nocc, nao, naux_seg))
                if self.unsort_alpha:
                    dgemm_cupy(1, 0, occ_coeff_gpu, feri_gpu.reshape(nao, -1), ialp_gpu, 1.0, 0.0)
                else:
                    dgemm_cupy(1, 0, sorted_occ_coeff_gpu, feri_gpu.reshape(nao, -1), ialp_gpu, 1.0, 0.0)
                alip_gpu = ialp_gpu.transpose(1, 0, 2).reshape(nao, -1)
                dgemm_cupy(0, 1, alip_gpu, alip_gpu, vk_gpu, 1.0, 1.0)
                accumulate_time(self.t_k, t1)

                t1 = get_current_time()
                #gammq_gpu = cupy.dot(occ_coeff_gpu.ravel(), alip_gpu.reshape(-1, naoaux))
                gammq_gpu = cupy.empty(naux_seg)
                if self.unsort_beta:
                    dgemv_cupy(0, alip_gpu.reshape(-1, naux_seg), occ_coeff_gpu, gammq_gpu, 1.0, 0.0)
                else:
                    dgemv_cupy(0, alip_gpu.reshape(-1, naux_seg), sorted_occ_coeff_gpu, gammq_gpu, 1.0, 0.0)
                feri_gpu = feri_gpu.reshape(-1, naux_seg)

                dgemv_cupy(1, feri_gpu, gammq_gpu, vj_gpu, 1.0, 1.0)
                accumulate_time(self.t_j, t1)

                if self.shared_int:
                    ialp_data[:, :, p0:p1] = cupy.asnumpy(ialp_gpu)
            else:

                t1 = get_current_time()
                #pali_gpu = cupy.dot(feri_gpu.reshape(-1, nao), occ_coeff_gpu).reshape(naux_seg, nao, nocc)
                pali_gpu = cupy.empty((naux_seg, nao, nocc))
                if self.unsort_beta:
                    dgemm_cupy(0, 0, feri_gpu.reshape(-1, nao), occ_coeff_gpu, pali_gpu, 1.0, 0.0)
                else:
                    dgemm_cupy(0, 0, feri_gpu.reshape(-1, nao), sorted_occ_coeff_gpu, pali_gpu, 1.0, 0.0)
                accumulate_time(self.t_k, t1)

                t1 = get_current_time()
                #gammq_gpu = cupy.dot(pali_gpu.reshape(naux_seg, -1), occ_coeff_gpu.ravel())  # Shape:(naux_seg,)
                #vj_gpu += cupy.einsum('p,pij->ij', gammq_gpu, feri_gpu)  # Shape:(nao, nao)
                gammq_gpu = cupy.empty(naux_seg)

                if self.unsort_alpha:
                    dgemv_cupy(1, pali_gpu.reshape(naux_seg, -1), occ_coeff_gpu.ravel(), gammq_gpu, 1.0, 0.0)
                else:
                    dgemv_cupy(1, pali_gpu.reshape(naux_seg, -1), sorted_occ_coeff_gpu.ravel(), gammq_gpu, 1.0, 0.0)
                dgemv_cupy(0, feri_gpu.reshape(naux_seg, -1), gammq_gpu, vj_gpu, 1.0, 1.0)
                accumulate_time(self.t_j, t1)

                t1 = get_current_time()
                #vk_gpu += cupy.einsum("pji,pli->jl", pali, pali, optimize="optimal")
                pial_gpu = pali_gpu.transpose(0, 2, 1).reshape(-1, nao)
                #vk_gpu += cupy.dot(pial_gpu.T, pial_gpu)
                dgemm_cupy(1, 0, pial_gpu, pial_gpu, vk_gpu, 1.0, 1.0)
                accumulate_time(self.t_k, t1)

                #coef_pali = occ_coeff_gpu if self.unsort_beta else sorted_occ_coeff_gpu
                
                #print_test(pali_gpu, f"{p0} pali_gpu")

                if self.shared_int:
                    ialp_data[:, :, p0:p1] = cupy.asnumpy(pali_gpu).transpose(2, 1, 0)


        vj_cpu = cupy.asnumpy(vj_gpu)
        vk_cpu = cupy.asnumpy(vk_gpu)
        Accumulate_GA_shm(win_vj, vj_node, vj_cpu)
        Accumulate_GA_shm(win_vk, vk_node, vk_cpu)

    comm.Barrier()

    if self.int_storage == 1:
        file_feri.close()
        if self.shared_int:
            file_ialp.close()
    
    close_ipc_handle(occ_coeff_ptr)

    if nnode > 1:
        Acc_and_get_GA(vj_node)
        Acc_and_get_GA(vk_node)
        '''if self.int_storage == 1:
            Acc_and_get_GA(self.ialp_node)'''
        comm.Barrier()

    if irank_shm == 0:

        if self.unsort_alpha and self.unsort_beta:
            vj = np.copy(vj_node)
        else:
            if not self.unsort_alpha and not self.unsort_beta:
                vj = self.intopt.unsort_orbitals(vj_node, axis=[0, 1])
            elif not self.unsort_alpha:
                vj = self.intopt.unsort_orbitals(vj_node, axis=[0])
            elif not self.unsort_beta:
                vj = self.intopt.unsort_orbitals(vj_node, axis=[1])

        if (self.ijp_shape and self.unsort_beta or
            not self.ijp_shape and self.unsort_alpha):
            vk = np.copy(vk_node)
        else:
            vk = self.intopt.unsort_orbitals(vk_node, axis=[0, 1])
    else:
        vj, vk = None, None
    
    comm_shm.Barrier()
    for win in [win_vj, win_vk]:
        free_win(win)
    
    return vj, vk
