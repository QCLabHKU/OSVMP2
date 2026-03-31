import os
import sys
import gc
import h5py
import numpy as np
import cupy
import cupyx
from pyscf import lib
from pyscf.lib import logger
from osvmp2.__config__ import ngpu
from osvmp2.lib import osvMp2Cuda, randSvdCuda
from osvmp2.gpu.rand_svd_cuda import randSVD, rand_svd_Cupy
from osvmp2.gpu.cuda_utils import (avail_gpu_mem, sliceJobsFor2DBlocks, 
                                   numpy_to_cupy, uniform_cum_offset_cupy, cum_offset_cupy, 
                                   dgemm_cupy, dger_cupy, sq_sum_axis0, screen_axis1)
from osvmp2.gpu.sparse_int3c2e_cuda import generate_ialp_cuda, preprocess_half_trans, sparse_half_trans
from osvmp2.loc.loc_addons import get_fit_domain, get_bfit_domain
from osvmp2.osvutil import *
from osvmp2.mpi_addons import *
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
comm_gpu = comm_shm.Split(color=igpu_shm, key=irank_shm)

# Set up process pairs for double buffering
ipair_rank = irank // 2
irank_pair = irank % 2
comm_pair = comm.Split(color=ipair_rank, key=irank)

THREADS_PER_AXIS = 16

def compute_tii_cuda(mo_vir, ialp_i, eo_i, ene_vir):
    iap = cupy.dot(mo_vir.T, ialp_i)
    iiab = cupy.dot(iap, iap.T)
    iap = None

    #fiiab = cupy.outer(ene_vir, ene_vir)
    fiiab = ene_vir[:, None] + ene_vir[None, :]
    fiiab -= 2 * eo_i
    iiab /= fiiab
    return iiab 

def get_rng_offset(idx0, n, r, maxIter):
    return idx0 * (n * r + maxIter * n)

def get_osv_kernel_cuda(self, ialp_buf_gpu=None, use_double_buffer=False, 
                        streams=None, mo_slice=None, r=10):

    #cupy.get_default_memory_pool().free_all_blocks()

    if use_double_buffer:
        assert mo_slice is not None
        irank_cal = irank_pair - 1 if irank_pair % 2 else irank_pair
        irank_io = irank_pair if irank_pair % 2 else irank_pair + 1
        is_cal_rank = irank_pair == irank_cal
        is_io_rank = irank_pair == irank_io

        if is_cal_rank:
            if streams is None:
                istream_data = cupy.cuda.Stream(non_blocking=True)
                istream_comp = cupy.cuda.Stream(non_blocking=True)
            else:
                istream_data, istream_comp = streams

    else:
        is_cal_rank = is_io_rank = True
        istream_data = istream_comp = cupy.cuda.Stream.null
    
    if is_cal_rank:
        #ev_gpu, self.ev_ptr = get_shared_cupy(self.ev.shape, numpy_array=self.ev)
        ev_gpu = cupy.asarray(self.ev)
        #print_test(ev_gpu, f"{irank}");sys.exit()
        if use_double_buffer:
            double_ialp_slice_gpu = ialp_buf_gpu[self.nao*self.naoaux:].reshape(2, -1)
    
        max_rsvd_iter = min(1000, self.nv) #May have a better estimates later

        rng = cupy.random.RandomState(seed=42, method=cupy.cuda.curand.CURAND_RNG_PSEUDO_XORWOW)
        cublasH = cupy.cuda.device.get_cublas_handle()
        cusolverH = cupy.cuda.device.get_cusolver_handle()
        svd_buf_size = randSvdCuda.randomizedSvdBufferSizeCupy(cusolverH, self.nv, self.nv, 
                                                                    max_rsvd_iter, r, False)
        svd_workspace = cupy.empty(svd_buf_size, dtype=cupy.float64)
        S_buf = cupy.empty(max_rsvd_iter, dtype=cupy.float64)
        U_buf = cupy.empty(self.nv*max_rsvd_iter, dtype=cupy.float64)

    def get_osv_cuda(iidx, i, ialp_cpu, eo_i):
        if use_double_buffer:
            data_idx = 1 if iidx % 2 else 0
            comp_idx = 0 if iidx % 2 else 1

            ialp_data_gpu = double_ialp_slice_gpu[data_idx]
            ialp_comp_gpu = double_ialp_slice_gpu[comp_idx]

            if iidx + 1 < len(mo_slice):
                with istream_data:
                    if not self.loc_fit: 
                        raise NotImplementedError
                    
                    if iidx > 0:
                        istream_data.wait_event(self.comp_events[data_idx])

                    t1 = get_current_time()
                    ialp_read_gpu = ialp_buf_gpu[:self.nao*self.naoaux].reshape(self.nao, self.naoaux)
                    if self.unsort_ialp_ao:
                        ialp_read_gpu.set(ialp_cpu)
                    else:
                        ialp_load_gpu = ialp_data_gpu.reshape(self.nao, self.naoaux)
                        ialp_load_gpu.set(ialp_cpu)
                        cupy.take(ialp_load_gpu, self.intopt.unsorted_ao_ids, axis=0, out=ialp_read_gpu)

                    if use_double_buffer:
                        next_i = mo_slice[iidx+1]
                        next_nfit = len(self.fit_list[next_i])
                        ialp_data_gpu = ialp_data_gpu[:self.nao*next_nfit].reshape(self.nao, next_nfit)
                        cupy.take(ialp_read_gpu, cupy.asarray(self.fit_list[next_i]), axis=1, out=ialp_data_gpu)
                    else:
                        nfit_i = len(self.fit_list[i])
                        ialp_gpu = ialp_comp_gpu[:self.nao*nfit_i].reshape(self.nao, nfit_i)
                        cupy.take(ialp_read_gpu, cupy.asarray(self.fit_list[i]), axis=1, out=ialp_gpu)
                    self.recorder_data.append(record_elapsed_time(t1))

                    self.data_events[data_idx].record()

                    istream_data.synchronize()
            
            with istream_comp:
                nfit_i = len(self.fit_list[i])
                ialp_gpu = ialp_comp_gpu[:self.nao*nfit_i].reshape(self.nao, nfit_i)
        else:
            t1 = get_current_time()
            ialp_gpu = cupy.asarray(ialp_cpu)
            #print_test(ialp_gpu, f"ialp {i}")
            
            if not self.unsort_ialp_ao:
                ialp_gpu = ialp_gpu[self.intopt.unsorted_ao_ids]
            if self.loc_fit:
                ialp_gpu = ialp_gpu[:, cupy.asarray(self.fit_list[i])]

            self.recorder_data.append(record_elapsed_time(t1))

        with istream_comp:
            if self.double_buffer:
                istream_comp.wait_event(self.data_events[comp_idx])

            t1 = get_current_time()
            tii_gpu = compute_tii_cuda(self.vir_mo_gpu, ialp_gpu, eo_i, ev_gpu)
            self.recorder_tii.append(record_elapsed_time(t1))

            t1 = get_current_time()
            if self.svd_method == 0:
                qcp_gpu, s_gpu, _ = cupy.linalg.svd(tii_gpu)
            elif self.svd_method == 1:
                qcp_gpu, s_gpu, _ = rand_svd_Cupy(tii_gpu, self.cposv_tol, max_iter=max_rsvd_iter, get_Vt=False, 
                                                    workspace=svd_workspace, U_buf=U_buf, S_buf=S_buf, 
                                                    curandG=rng._generator, 
                                                    cublasH=cublasH, cusolverH=cusolverH)
                #qcp_gpu, s_gpu, _ = oldRandSvdCuda.randomizedSvdCupy(tii_gpu, self.cposv_tol, max_rsvd_iter, getVt=False)
                #qcp_gpu, s_gpu, _ = randSVD(tii_gpu, self.cposv_tol, max_iter=max_rsvd_iter)
            else:
                raise NotImplementedError(f"SVD method {self.svd_method} is not implemented.")
            self.recorder_svd.append(record_elapsed_time(t1))

            nosv_i = cupy.count_nonzero(s_gpu >= self.osv_tol)
            self.nosv_cp[i] = len(s_gpu)
            self.nosv[i] = nosv_i
            qmat_gpu = qcp_gpu[:, :nosv_i]
            qao_gpu = cupy.dot(self.vir_mo_gpu, qmat_gpu)

            if self.double_buffer:
                self.comp_events[comp_idx].record()

            t1 = get_current_time()
            qmat_cpu = cupy.asnumpy(qmat_gpu)
            qao_cpu = cupy.asnumpy(qao_gpu)
            if self.cal_grad:
                qcp_cpu = cupy.asnumpy(qcp_gpu)
                s_cpu = cupy.asnumpy(s_gpu)
                tii_cpu = cupy.asnumpy(tii_gpu)
            else:
                qcp_cpu = None
                s_cpu = None
                tii_cpu = None
            self.recorder_data.append(record_elapsed_time(t1))
            
        
        '''if use_double_buffer:
            istream_comp.synchronize()
            istream_data.synchronize()'''

        return qmat_cpu, qao_cpu, qcp_cpu, s_cpu, tii_cpu
    return get_osv_cuda

def get_direct_osv_cuda(self, mo_slices, qmat_save, qao_save, log=None, r=10):
    if log is None:
        log = logger.Logger(sys.stdout, self.verbose)

    cupy.get_default_memory_pool().free_all_blocks()
    #old_allocator = cupy.cuda.get_allocator() 
    #cupy.cuda.set_allocator(None)

    mo_slice = mo_slices[irank]
    if self.fully_direct == 2:
        comp_events = [cupy.cuda.Event(), cupy.cuda.Event()]
        data_events = [cupy.cuda.Event(), cupy.cuda.Event()]
        events = [comp_events, data_events]
    else:
        events = None
    
    recorders = [[], [], self.recorder_data]
    recorder_feri, recorder_ialp, _ = recorders

    gpu_mem_f8 = avail_gpu_mem(self.gpu_memory, unit="B") / 8

    nocc = self.mol.nelectron//2
    nmo = len(self.mo_list)
    #c2s_coeff_gpu = cupy.asarray(self.intopt.coeff)
    nao_cart = self.intopt.coeff.shape[0]
    naux_cart = self.intopt.sorted_auxmol.nao
    #c2s_aux_coeff_cpu = self.intopt.aux_coeff
    #occ_coeff_cpu = np.dot(self.o[:, mo_slice].T, self.intopt.coeff.T)
    nocc_rank = len(mo_slice)
    vir_mo_gpu = cupy.asarray(np.dot(self.intopt.coeff, self.v))
    c2s_aux_coeff_gpu = cupy.asarray(self.intopt.aux_coeff)
    sorted_occ_coeff = self.intopt.coeff.dot(self.o[:, mo_slice])
    occ_coeff_gpu = cupy.asarray(sorted_occ_coeff.T, order='C')
    ene_vir_gpu = cupy.asarray(self.ev)
    
    nocc_rank = len(mo_slice)
    if self.loc_fit:
        win_aux_ratio, aux_ratio_node = get_shared((nocc, self.naoaux))#, set_zeros=True)
        fit_ratio_gpu = cupy.empty((nocc_rank, self.naoaux), dtype=cupy.float64)
    
    max_rsvd_iter = min(1000, self.nv) #May have a better estimates later
    rng = cupy.random.RandomState(seed=42, method=cupy.cuda.curand.CURAND_RNG_PSEUDO_XORWOW)
    cublasH = cupy.cuda.device.get_cublas_handle()
    cusolverH = cupy.cuda.device.get_cusolver_handle()
    svd_buf_size = randSvdCuda.randomizedSvdBufferSizeCupy(cusolverH, self.nv, self.nv, 
                                                                 max_rsvd_iter, r, False)
    svd_workspace = cupy.empty(svd_buf_size, dtype=cupy.float64)
    S_buf = cupy.empty(max_rsvd_iter, dtype=cupy.float64)
    U_buf = cupy.empty(self.nv*max_rsvd_iter, dtype=cupy.float64)
    ialp_sph_buf_gpu = cupy.empty((nao_cart*self.naoaux), dtype=cupy.float64)

    #buf_size = int(0.8 * avail_gpu_mem(self.gpu_memory, unit="B") / 8)
    #buf_size -= nao_cart * self.naoaux - 2* self.nv**2
    int_buf_size = int(0.8 * (gpu_mem_f8 - vir_mo_gpu.size - c2s_aux_coeff_gpu.size -
                              occ_coeff_gpu.size - ene_vir_gpu.size - fit_ratio_gpu.size -
                              svd_workspace.size - S_buf.size - U_buf.size - ialp_sph_buf_gpu.size))
    ialp_generator = generate_ialp_cuda(self, self.intopt, occ_coeff_gpu, 
                                        #buf_gpu=buf_gpu[total_array_size:], 
                                        buf_size=int_buf_size,
                                        events=events, 
                                        recorders=recorders, 
                                        log=log)
    
    
    
    fit_indices = cupy.zeros(self.naoaux, dtype=cupy.int32)
    fit_count = cupy.zeros(1, dtype=cupy.int32)

    for indices, ialp in ialp_generator:
        if self.fully_direct == 2:
            comp_idx, i_idx0, i_idx1 = indices
        else:
            i_idx0, i_idx1 = indices
        #ialp = cupy.einsum('ibp,ba->iap', ialp, c2s_coeff_gpu, optimize=True)
        #ialp = cupy.einsum('iap,pq->iaq', ialp, c2s_aux_coeff_gpu, optimize=True)
        #ialp_cpy = cupy.copy(ialp)
        #cupy.cuda.Stream.null.synchronize()
        ialp_gen_buf = ialp.ravel()
        buf_idx0 = 0
        for i_idx in range(i_idx0, i_idx1):
            i = mo_slice[i_idx]
            eo_i = self.eo[i]
            t1 = get_current_time()
            ialp_cart_i = ialp[i_idx - i_idx0]
            ialp_cart_buf_gpu = ialp_cart_i.ravel()
            #ialp_i = cupy.dot(ialp_cart_i, c2s_aux_coeff_gpu, out=ialp_sph_buf_gpu)
            ialp_i = ialp_sph_buf_gpu.reshape(nao_cart, self.naoaux)
            dgemm_cupy(0, 0, ialp_cart_i, c2s_aux_coeff_gpu, ialp_i, 1.0, 0.0, cublasH=cublasH)
            recorder_ialp.append(record_elapsed_time(t1))

            '''fit_u, fit_s, _ = rand_svd_Cupy(cupy.dot(ialp_cart_i.T, ialp_cart_i), self.cposv_tol, max_iter=1000,
                                        get_U=True, get_Vt=False)
            fit_s = fit_s[fit_s>self.cposv_tol]
            print(fit_s)#fit_u.shape, fit_s.shape)
            #ialp_i = cupy.dot(ialp_i, fit_u[:, :len(fit_s)])'''
            
            if self.loc_fit:
                t1 = get_current_time()
                sq_sum_axis0(ialp_i, out=fit_ratio_gpu[i_idx])
                ialp_i = screen_axis1(ialp_i, fit_ratio_gpu[i_idx], self.fit_tol, 
                                      buffer=ialp_cart_buf_gpu, 
                                      kept_indices=fit_indices, kept_count=fit_count)

                self.recorder_io.append(record_elapsed_time(t1))
            else:
                raise NotImplementedError
            

            
            t1 = get_current_time()
            nsaux = ialp_i.shape[1]
            iap = ialp_sph_buf_gpu[:self.nv*nsaux].reshape(self.nv, nsaux)
            dgemm_cupy(1, 0, vir_mo_gpu, ialp_i, iap, 1.0, 0.0, cublasH=cublasH)
            iiab = ialp_cart_buf_gpu[:self.nv**2].reshape(self.nv, self.nv)
            dgemm_cupy(0, 1, iap, iap, iiab, 1.0, 0.0, cublasH=cublasH)

            fiiab = ialp_sph_buf_gpu[:self.nv**2].reshape(self.nv, self.nv)
            #fiiab.fill(-2*eo_i)
            
            #dger_cupy(ene_vir_gpu, ene_vir_gpu, fiiab, 1.0, cublasH=cublasH)
            cupy.add.outer(ene_vir_gpu, ene_vir_gpu, out=fiiab)
            fiiab -= 2*eo_i

            iiab /= fiiab
            tii_gpu = iiab

            self.recorder_tii.append(record_elapsed_time(t1))

            t1 = get_current_time()
            if self.svd_method == 0:
                qcp_gpu, s_gpu, _ = cupy.linalg.svd(tii_gpu)
            if self.svd_method == 1:
                #with cupy.cuda.profile():
                qcp_gpu, s_gpu, _ = rand_svd_Cupy(tii_gpu, self.cposv_tol, max_iter=max_rsvd_iter, get_Vt=False, 
                                                    workspace=svd_workspace, U_buf=U_buf, S_buf=S_buf, 
                                                    curandG=rng._generator, 
                                                    cublasH=cublasH, cusolverH=cusolverH)

                #qcp_gpu, s_gpu, _ = randSvdCuda.randomizedSvdCupy(tii_gpu, self.cposv_tol, max_rsvd_iter, getVt=False)
                #qcp_gpu, s_gpu, _ = randSVD(tii_gpu, self.cposv_tol, max_iter=max_rsvd_iter)
            
            else:
                raise NotImplementedError(f"SVD method {self.svd_method} is not implemented.")
            self.recorder_svd.append(record_elapsed_time(t1))

            t1 = get_current_time()
            self.nosv_cp[i] = len(s_gpu)
            s_gpu = s_gpu[s_gpu >= self.osv_tol]
            nosv_i = len(s_gpu)
            self.nosv[i] = nosv_i
            #qmat_gpu = qcp_gpu[:, :nosv_i]
            #qao_gpu = cupy.dot(vir_mo_gpu, qmat_gpu)

            buf_idx1 = buf_idx0 + self.nv*nosv_i
            qmat_gpu = ialp_gen_buf[buf_idx0:buf_idx1].reshape(self.nv, nosv_i)
            buf_idx0 = buf_idx1

            buf_idx1 = buf_idx0 + nao_cart*nosv_i
            qao_gpu = ialp_gen_buf[buf_idx0:buf_idx1].reshape(nao_cart, nosv_i)
            buf_idx0 = buf_idx1

            cupy.take(qcp_gpu, cupy.arange(nosv_i), axis=1, out=qmat_gpu)
            dgemm_cupy(0, 0, vir_mo_gpu, qmat_gpu, qao_gpu, 1.0, 0.0, cublasH=cublasH)
            self.recorder_io.append(record_elapsed_time(t1))

        t1 = get_current_time()
        buf_cpu = cupyx.empty_pinned((buf_idx1,), dtype=cupy.float64)
        cupy.asnumpy(ialp_gen_buf[:buf_idx1], out=buf_cpu)
        self.recorder_data.append(record_elapsed_time(t1))
        
        buf_idx0 = 0
        for i_idx in range(i_idx0, i_idx1):
            i = mo_slice[i_idx]
            nosv_i = self.nosv[i]

            buf_idx1 = buf_idx0 + self.nv*nosv_i
            qmat_save[i] = buf_cpu[buf_idx0:buf_idx1].reshape(self.nv, nosv_i)
            buf_idx0 = buf_idx1

            buf_idx1 = buf_idx0 + nao_cart*nosv_i
            qao_save[i] = buf_cpu[buf_idx0:buf_idx1].reshape(nao_cart, nosv_i)
            buf_idx0 = buf_idx1

        if self.fully_direct == 2:
            comp_events[comp_idx].record()
    #cupy.cuda.Stream.null.synchronize()
    #aux_ratio_node[mo_slice] = fit_ratio_gpu.get()
    i0, i1 = mo_slice[0], mo_slice[-1]+1
    cupy.asnumpy(fit_ratio_gpu, out=aux_ratio_node[i0:i1])

    if self.loc_fit:
        comm.Barrier()
        if nnode > 1:
            #Acc_and_get_GA(aux_ratio_node)
            mo_offsets = [[i, i+1] for i in range(self.no)]
            mo_node_offsets = get_node_offsets(mo_slices, mo_offsets)
            Get_from_other_nodes_GA(aux_ratio_node, mo_node_offsets)
            comm.Barrier()

        self.fit_list, self.fit_seg, self.nfit = get_fit_domain(aux_ratio_node, self.fit_tol)
        if self.cal_grad:
            self.atom_close, self.bfit_seg, self.nbfit = get_bfit_domain(self.mol, self.with_df.auxmol, 
                                                                        aux_ratio_node, self.fit_tol)
        
        comm_shm.Barrier()
        win_aux_ratio.Free()
    
    #cupy.cuda.set_allocator(old_allocator)

    if self.fully_direct:
        tferi = sum_elapsed_time(recorder_feri)
        tialp = sum_elapsed_time(recorder_ialp)

        self.t_feri_mp2 += tferi + tialp
        print_time([["feri", tferi], ["ialp", tialp]], log)


def get_sf_cuda(self, pair_slice):

    #cupy.get_default_memory_pool().free_all_blocks()

    max_nosv = np.max(self.nosv)
    max_memory = avail_gpu_mem(self.gpu_memory) / nrank_per_gpu
    nocc_batch = get_ncols_for_memory(0.8 * max_memory, self.nv * max_nosv, len(self.mo_list))

    t0 = get_current_time()
    nosv_gpu = numpy_to_cupy(self.nosv, dtype=cupy.int32)
    ev_gpu = cupy.asarray(self.ev)
    
    if self.int_storage == 3:
        nocc_core = self.nocc - len(self.mo_list)
        #qmat_gpu = cupy.asarray(self.qmat_node)
        qmat_gpu, qmat_ptr = get_shared_cupy(self.qmat_node.shape, numpy_array=self.qmat_node)
        qmat_offsets_gpu = cum_offset_cupy(self.nv * nosv_gpu[nocc_core:])
    else:
        qmat_offsets_gpu = cupy.empty(self.no, dtype=cupy.int64)

    self.recorder_data.append(record_elapsed_time(t0))
    batch_slice = get_mo_batches(pair_slice, nocc_batch, self.no)

    size_sf_rank = 0
    pairs_kept_rank = []
    smat_save = {}
    fmat_save = {}
    '''smat_save = []
    fmat_save = []'''
    for mos_batch, pairs_batch in batch_slice:
        npair_batch = len(pairs_batch)

        if self.int_storage == 3:
            max_npair = npair_batch
        else:
            qmat_sizes = self.nv * self.nosv[mos_batch]
            qmat_offsets = cum_offset(qmat_sizes)
            qmat_gpu = cupy.empty(qmat_offsets[-1], dtype=cupy.float64)

            read_indices = merge_intervals([self.qmat_offsets_node[i] for i in mos_batch])

            t0 = get_current_time()
            buf_idx0 = 0
            for qmat_idx0, qmat_idx1 in read_indices:
                buf_idx1 = buf_idx0 + qmat_idx1 - qmat_idx0
                qmat_gpu[buf_idx0:buf_idx1].set(self.qmat_node[qmat_idx0:qmat_idx1])
                buf_idx0 = buf_idx1

            qmat_offsets_gpu[mos_batch] = cupy.asarray(qmat_offsets[:-1], dtype=cupy.int64)
            self.recorder_data.append(record_elapsed_time(t0))
        
            max_memory = avail_gpu_mem(self.gpu_memory) / nrank_per_gpu
            max_npair = get_ncols_for_memory(0.8 * max_memory, 2 * max_nosv**2, npair_batch)

        for pidx0 in np.arange(npair_batch, step=max_npair):
            pidx1 = min(npair_batch, pidx0 + max_npair)
            pairs_now = pairs_batch[pidx0:pidx1]
            npair_now = len(pairs_now)

            ilist = pairs_now // self.no
            jlist = pairs_now % self.no

            sf_sizes = self.nosv[ilist] * self.nosv[jlist]
            sf_offsets = cum_offset(sf_sizes)

            smat_gpu = cupy.empty(sf_offsets[-1], dtype=cupy.float64)
            fmat_gpu = cupy.empty_like(smat_gpu)
            sratio_gpu = cupy.empty(npair_now, dtype=cupy.float64)

            t0 = get_current_time()
            pairs_now_gpu = numpy_to_cupy(pairs_now, dtype=cupy.int32)
            sf_offsets_gpu = cupy.asarray(sf_offsets, dtype=cupy.int64)
            self.recorder_data.append(record_elapsed_time(t0))

            t0 = get_current_time()
            osvMp2Cuda.osvSFCupy(qmat_gpu, smat_gpu, fmat_gpu, sratio_gpu, ev_gpu, 
                                        pairs_now_gpu, nosv_gpu, qmat_offsets_gpu, 
                                        sf_offsets_gpu, np.int32(self.nv), 
                                        np.int32(self.no), np.int32(npair_now))
            self.recorder_cal.append(record_elapsed_time(t0))

            t0 = get_current_time()
            smat_cpu = cupy.asnumpy(smat_gpu) 
            fmat_cpu = cupy.asnumpy(fmat_gpu) 
            sratio_cpu = cupy.asnumpy(sratio_gpu)
            self.recorder_data.append(record_elapsed_time(t0))

            for pidx, ipair in enumerate(pairs_now):
                
                if sratio_cpu[pidx] < self.disc_tol:
                    continue
                pairs_kept_rank.append(ipair)
                size_sf_rank += sf_sizes[pidx]

                sf_idx0, sf_idx1 = sf_offsets[pidx:pidx+2]
                smat_save[ipair] = smat_cpu[sf_idx0:sf_idx1]
                fmat_save[ipair] = fmat_cpu[sf_idx0:sf_idx1]

            pairs_ji = (pairs_now%self.no) * self.no + (pairs_now//self.no)
            self.s_ratio[pairs_ji] = self.s_ratio[pairs_now] = sratio_cpu

            
    if self.int_storage == 3:
        close_ipc_handle(qmat_ptr)

    return size_sf_rank, pairs_kept_rank, smat_save, fmat_save

def make_imujp_mo_batch(mo_slice, imujp_sizes_occ, max_memory):
    max_imujp_size = int(max_memory*1e6 / 8)
    first_mo = mo_slice[0]
    mo_seg = [[imujp_sizes_occ[first_mo], [first_mo]]]

    for i in mo_slice[1:]:
        size_i = imujp_sizes_occ[i]
        size_last = mo_seg[-1][0]
        if size_last + size_i > max_imujp_size:
            mo_seg.append([size_i, [i]])
        else:
            mo_seg[-1][0] += size_i
            mo_seg[-1][1].append(i)
    
    return mo_seg


def get_imujp_batches(jlist_close_full, mo_slice, max_nj, nocc):
    nmo_slice = len(mo_slice)
    n_common = np.zeros((nmo_slice, nmo_slice), dtype=np.int32)
    js_close_set = {}
    for i in mo_slice:
        js_close_set[i] = set(jlist_close_full[i])
    
    js_common_dic = {}
    for i0_idx, i0 in enumerate(mo_slice[:-1]):
        i1_idx = i0_idx + 1
        for i1 in mo_slice[i1_idx:]:
            js_common = js_close_set[i0] & js_close_set[i1]
            js_common_dic[(i0, i1)] = js_common
            n_common[i1_idx, i0_idx] = n_common[i0_idx, i1_idx] = len(js_common)
            i1_idx += 1
    np.fill_diagonal(n_common, -10)
    idx_most_com = np.argmax(n_common.sum(axis=1))
    full_indices = set(range(nmo_slice))
    sorted_indices = [idx_most_com]
    while len(sorted_indices) < nmo_slice:
        idx_last = sorted_indices[-1]
        max_ncom_idx = np.argmax(n_common[idx_last])
        if n_common[idx_last, max_ncom_idx] == 0:
            indices_rest = list(full_indices - set(sorted_indices))
            max_ncom_idx = indices_rest[np.argmax(n_common[indices_rest].sum(axis=1))]
        sorted_indices.append(max_ncom_idx)
        n_common[idx_last] = -10
        n_common[:, idx_last] = -10
    mo_slice_sorted = mo_slice[sorted_indices]
    
    if len(mo_slice_sorted) == 1:
        sorted_js = [list(js_close_set[mo_slice_sorted[0]])]
    elif len(mo_slice_sorted) == 2:
        i, j = mo_slice_sorted
        js_common = js_common_dic[tuple(sorted([i, j]))]
        sorted_js_i = list(js_close_set[i] - js_common) +\
                      sorted(list(js_common), reverse=True)
        sorted_js_j = list(js_close_set[j] - js_common) +\
                      sorted(list(js_common))
        sorted_js = [sorted_js_i, sorted_js_j]
    else:

        # Get uniq_common_js
        uniq_common_js = []
        for iidx, i in list(enumerate(mo_slice_sorted))[1:-1]:
            i_last = mo_slice_sorted[iidx-1]
            js_common_last = js_common_dic[tuple(sorted([i_last, i]))]

            i_next = mo_slice_sorted[iidx+1]
            js_common_next = js_common_dic[tuple(sorted([i_next, i]))]

            uniq_common_last = js_common_last - js_common_next
            uniq_common_next = js_common_next - js_common_last

            uniq_common_js.append([uniq_common_last, uniq_common_next])

        uniq_common_js.append([set([]), set([])])

        for idx, (uniq_common_last, uniq_common_next) in enumerate(uniq_common_js[:-1]):
            uniq_common_union = uniq_common_next | uniq_common_js[idx+1][0]
            uniq_common_js[idx][1] = uniq_common_union
            uniq_common_js[idx+1][0] = uniq_common_union

        for idx, (uniq_common_last, uniq_common_next) in enumerate(uniq_common_js[:-1]):
            common_last_next = uniq_common_last & uniq_common_next
            
            new_uniq_common_next = uniq_common_next - common_last_next
            uniq_common_js[idx][1] = new_uniq_common_next
            uniq_common_js[idx+1][0] = new_uniq_common_next
        
        sorted_js = []
        for iidx, i in list(enumerate(mo_slice_sorted))[1:-1]:

            i_last = mo_slice_sorted[iidx-1]
            js_common_last = js_common_dic[tuple(sorted([i_last, i]))]

            i_next = mo_slice_sorted[iidx+1]
            js_common_next = js_common_dic[tuple(sorted([i_next, i]))]

            uniq_common_last, uniq_common_next = uniq_common_js[iidx - 1]
            common_last_and_next = (js_common_last - uniq_common_last) & ( 
                                    js_common_next - uniq_common_next)
            
            current_common = common_last_and_next | uniq_common_last | uniq_common_next

            uniq_common_last_supp = js_common_last - current_common
            uniq_common_next_supp = js_common_next - current_common

            uniq_common_last = list(uniq_common_last | uniq_common_last_supp)
            uniq_common_next = list(uniq_common_next | uniq_common_next_supp)
            common_last_and_next = list(common_last_and_next)

            nj_count = 0

            if iidx % 2 == 0:
                common_last_and_next.sort()
            else:
                common_last_and_next.sort(reverse=True)

            js_now = js_close_set[i]

            sorted_js_now = np.empty(len(js_now), dtype=np.int32)
            idx_first = 0
            idx_last = len(js_now)
            if len(uniq_common_last) > 0:
                idx_first = len(uniq_common_last)
                sorted_js_now[:idx_first] = sorted(uniq_common_last)
                nj_count += idx_first

            if len(uniq_common_next) > 0:
                idx_last = len(js_now) - len(uniq_common_next)
                sorted_js_now[idx_last:] = sorted(uniq_common_next, reverse=True)
                nj_count += len(uniq_common_next)

            n_common_total = len(common_last_and_next)
            n_common_last = n_common_total // 2
            n_common_next = n_common_total - n_common_last

            if n_common_last > 0:
                sorted_js_now[idx_first:(idx_first+n_common_last)] = sorted(common_last_and_next[:n_common_last])
                nj_count += n_common_last
            
            if n_common_next > 0:
                sorted_js_now[(idx_last-n_common_next):idx_last] = sorted(common_last_and_next[n_common_last:], reverse=True)
                nj_count += n_common_next

            uniq_js_now = list(js_now - js_common_last - js_common_next)
            if len(uniq_js_now) > 0:
                idx0 = idx_first + n_common_last
                idx1 = idx0 + len(uniq_js_now)
                sorted_js_now[idx0:idx1] = uniq_js_now
                nj_count += len(uniq_js_now)

            assert nj_count == len(js_now), "%d %d %d"%(iidx, nj_count, len(js_now))


            if iidx == 1:
                js_start = js_close_set[i_last]
                sorted_js_start = np.empty(len(js_start), dtype=np.int32)
                idx_last = len(js_start) - len(uniq_common_last)
                sorted_js_start[idx_last:] = sorted(uniq_common_last, reverse=True)
                idx0 = idx_last - n_common_last
                sorted_js_start[idx0:idx_last] = sorted(common_last_and_next[:n_common_last], reverse=True)
                idx1 = idx0
                sorted_js_start[:idx1] = list(js_start - set(uniq_common_last) - set(common_last_and_next[:n_common_last]))
                sorted_js.append(sorted_js_start)

            sorted_js.append(sorted_js_now)

            if iidx == nmo_slice - 2:
                js_final = js_close_set[i_next]
                sorted_js_final = np.empty(len(js_final), dtype=np.int32)
                idx_first = len(uniq_common_next)
                sorted_js_final[:idx_first] = sorted(uniq_common_next)
                idx1 = idx_first + n_common_next
                sorted_js_final[idx_first:idx1] = sorted(common_last_and_next[n_common_last:])
                sorted_js_final[idx1:] = list(js_final - set(uniq_common_next) - set(common_last_and_next[n_common_last:]))
                sorted_js.append(sorted_js_final)

    batches = []

    is_now = []
    js_now = []
    check_j = np.zeros(nocc, dtype=bool)

    for i, js_i in zip(mo_slice_sorted, sorted_js):
        js_i_batch = []
        for j in js_i:
            js_i_batch.append(j)
            if not check_j[j]:
                js_now.append(j)
                check_j[j] = True
            if len(js_now) == max_nj:
                js_i_batch.sort()
                js_now.sort()
                is_now.append([i, js_i_batch])
                batches.append([js_now, is_now])
                js_i_batch = []
                is_now = []
                js_now = []
                check_j = np.zeros(nocc, dtype=bool)
        if len(js_now) > 0:
            js_i_batch.sort()
            is_now.append([i, js_i_batch])

    if len(js_now) > 0:
        js_now.sort()
        batches.append([js_now, is_now])

    return batches


def get_imujp_cuda(self, mo_slices, ialp_data, imujp_data, log=None):

    if log is None:
        log = lib.logger.Logger(self.stdout, self.verbose)
    
    use_ialp_file = isinstance(ialp_data, h5py.Dataset)
    use_imujp_file = isinstance(imujp_data, h5py.Dataset)

    use_double_buffer = self.double_buffer and use_ialp_file

    if use_double_buffer:
        irank_cal = irank_pair - 1 if irank_pair % 2 else irank_pair
        irank_io = irank_pair if irank_pair % 2 else irank_pair + 1
        is_cal_rank = irank_pair == irank_cal
        is_io_rank = irank_pair == irank_io
        istream_data = cupy.cuda.Stream(non_blocking=True)
        istream_comp = cupy.cuda.Stream(non_blocking=True)
        comp_events = [cupy.cuda.Event(), cupy.cuda.Event()]
        data_events = [cupy.cuda.Event(), cupy.cuda.Event()]

        nslice = nrank_per_gpu // 2
        nbuf_ialp = 3

        mo_slice = np.asarray([], dtype=int)
        other_rank_pair = irank-1 if irank % 2 else irank+1
        for rank in [irank, other_rank_pair]:
            if mo_slices[rank] is not None:
                mo_slice = np.append(mo_slice, mo_slices[rank])
        mo_slice.sort()
    else:
        is_cal_rank = is_io_rank = True
        istream_data = istream_comp = cupy.cuda.Stream.null

        nslice = nrank_per_gpu
        if self.int_storage == 3:
            nbuf_ialp = 1
        else:
            nbuf_ialp = 2
        
        mo_slice = mo_slices[irank]

    nocc_core = self.no - len(self.mo_list)

    t0 = get_current_time()
    nosv_gpu = numpy_to_cupy(self.nosv, dtype=np.int32)
    self.recorder_data.append(record_elapsed_time(t0))

    nmo = len(self.mo_list)
    max_nosv = np.max(self.nosv)
    max_nfit = np.max(self.nfit_pair)

    max_mem_f8 = avail_gpu_mem(self.gpu_memory, unit="B") / nslice / 8
    if self.fully_direct:
        mem_j_f8 = 0.2 * max_mem_f8
    else:
        mem_j_f8 = 0.9 * (max_mem_f8 - nbuf_ialp * self.nao * self.naoaux)
    
    try:
        full_js = np.unique(np.concatenate([self.jlist_close_full[i] for i in mo_slice]))
    except ValueError:
        print(mo_slices, flush=True)
        raise ValueError
    njs_full = len(full_js)
    max_nj = min(int(mem_j_f8) // ((self.nao + max_nfit) * max_nosv), njs_full)

    if is_cal_rank:
        qao_offsets_gpu = cupy.empty(self.no, dtype=cupy.int64)
    
    if self.int_storage == 3:
        full_js_gpu = []
        for rank_i in ranks_gpu_shm:
            for i in mo_slice[rank_i]:
                njs_full.append(self.jlist_close_full[i])
        full_js_gpu = np.unique(np.concatenate(full_js_gpu))
        qao_size = self.nao * self.nosv[full_js]
        qao_offsets = cum_offset(qao_size)
        #qao_gpu = cupy.empty(qao_offsets[-1], dtype=cupy.float64)
        qao_gpu, qao_gpu_ptr = get_shared_cupy(qao_offsets[-1])
        if irank_gpu == 0:
            qao_offsets_node = [self.qao_offsets_node[j] for j in full_js_gpu]
            qao_offsets_node = merge_intervals(qao_offsets_node)         
            buf_idx0 = 0
            for qao_idx0, qao_idx1 in qao_offsets_node:
                buf_idx1 = buf_idx0 + qao_idx1 - qao_idx0
                qao_gpu[buf_idx0:buf_idx1].set(self.qao_node[qao_idx0:qao_idx1])
                buf_idx0 = buf_idx1
        cupy.cuda.Stream.null.synchronize()
        comm_shm.Barrier()
        qao_offsets_gpu[full_js_gpu] = cupy.asarray(qao_offsets[:-1])
        
        full_is = [[i, self.jlist_close_full[i]] for i in mo_slice]
        imujp_batches = [[full_js_gpu, full_is]]

    else:
        mo_slice = np.sort(mo_slice)
        #max_cpu_memory = get_mem_spare(self.mol, 0.8)
        #mo_segs = make_imujp_mo_batch(mo_slice, self.imujp_sizes_occ, max_cpu_memory)

        t0 = get_current_time()
        imujp_batches = get_imujp_batches(self.jlist_close_full, mo_slice, max_nj, self.no)
        print_time(["imujp batches", get_elapsed_time(t0)], log)


        if is_cal_rank:
            if self.fully_direct:
                nao = self.intopt.sorted_mol.nao
            else:
                nao = self.nao
            qao_buf_size = max_nj * max_nosv * nao
            imujp_buf_size = (max_nj - 1) * max_nosv * max_nfit + max_nosv * self.naoaux
            qao_gpu = cupy.empty(qao_buf_size)
            imujp_buf_gpu = cupy.empty(imujp_buf_size)
    
        last_jlist_set = set([])

    all_mos_batches = []
    for _, is_batch in imujp_batches:
        for i, _ in is_batch:
            all_mos_batches.append(i)
    all_mos_batches = np.asarray(all_mos_batches, dtype=int)
    indexes = np.unique(all_mos_batches, return_index=True)[1]
    uniq_mos_batches = all_mos_batches[sorted(indexes)]


    if self.fully_direct:
        nocc = self.mol.nelectron//2
        nmo = len(self.mo_list)
        nocc_core = nocc - nmo
        #c2s_coeff_gpu = cupy.asarray(self.intopt.coeff)
        nao_cart = self.intopt.coeff.shape[0]
        c2s_aux_coeff_gpu = cupy.asarray(self.intopt.aux_coeff)
        sorted_occ_coeff = self.intopt.coeff.dot(self.o[:, uniq_mos_batches])
        occ_coeff_gpu = cupy.asarray(sorted_occ_coeff.T, order='C')
        ialp_buf_gpu = cupy.empty((nao_cart*self.naoaux+2))
        ialp_gpu = ialp_buf_gpu[:nao_cart*self.naoaux].reshape(nao_cart, self.naoaux)

        buf_size = int(0.8 * (max_mem_f8 - qao_buf_size - imujp_buf_size - 
                         c2s_aux_coeff_gpu.size - occ_coeff_gpu.size - 
                         ialp_buf_gpu.size))

        recorders = [[], [], self.recorder_data]
        recorder_feri, recorder_ialp, _ = recorders
        if self.fully_direct == 2:
            comp_events = [cupy.cuda.Event(), cupy.cuda.Event()]
            data_events = [cupy.cuda.Event(), cupy.cuda.Event()]
            events = [comp_events, data_events]
        else:
            events = None

        #buffer_size = int(0.9 * avail_gpu_mem(self.gpu_memory, unit="B") / 8)
        #buffer_size -= nao_cart * self.naoaux - 2* self.nv**2
        #buf_gpu = cupy.empty(buffer_size, dtype=cupy.float64)
        ialp_generator = generate_ialp_cuda(self, self.intopt, occ_coeff_gpu, 
                                            #buf_gpu=buf_gpu, 
                                            buf_size=buf_size,
                                            events=events, 
                                            recorders=recorders)
    else:
        if self.int_storage == 3:
            ialp_gpu = cupy.empty((self.naoaux, self.nao))
        else:
            nbuf = 3 if use_double_buffer else 2
            ialp_buffer_gpu = cupy.empty((nbuf, self.nao*self.naoaux))


    if use_ialp_file:
        if use_double_buffer:
            win_ialp_buf, double_ialp_buf_cpu = get_shared((2, self.nao, self.naoaux), rank_shm_buf=irank_cal, 
                                                               comm_shared=comm_pair, use_pin=True)
            
            for bidx in range(2):
                io_idx = 0 if bidx % 2 else 1
                cal_idx = 1 if bidx % 2 else 0
                if is_io_rank:
                    ialp_io_cpu = double_ialp_buf_cpu[io_idx]
                    t1 = get_current_time()
                    ialp_data.read_direct(ialp_io_cpu, np.s_[uniq_mos_batches[bidx]-nocc_core])
                    self.recorder_io.append(record_elapsed_time(t1))
                
                if is_cal_rank and bidx > 0:
                    data_idx = 2 if bidx % 2 else 1
                    with istream_data:
                        t1 = get_current_time()
                        ialp_cpu = double_ialp_buf_cpu[cal_idx]
                        
                        ialp_load_gpu = ialp_buffer_gpu[0].reshape(self.nao, self.naoaux)
                        ialp_data_gpu = ialp_buffer_gpu[data_idx]
                        
                        if self.unsort_ialp_ao:
                            ialp_load_gpu.set(ialp_cpu)
                            ialp_data_gpu = ialp_data_gpu.reshape(self.naoaux, self.nao)
                            ialp_data_gpu[:] = ialp_load_gpu.T
                        else:
                            #ialp_gpu = cupy.ascontiguousarray(cupy.asarray(ialp_cpu)[self.intopt.unsorted_ao_ids].T)
                            ialp_data_gpu = ialp_data_gpu.reshape(self.nao, self.naoaux)
                            ialp_data_gpu.set(ialp_cpu)
                            cupy.take(ialp_data_gpu, self.intopt.unsorted_ao_ids, axis=0, out=ialp_load_gpu)
                            ialp_data_gpu = ialp_data_gpu.reshape(self.naoaux, self.nao)
                            ialp_data_gpu[:] = ialp_load_gpu.T
                        self.recorder_data.append(record_elapsed_time(t1))

                        self.data_events[data_idx-1].record()
                        istream_data.synchronize()

                comm_pair.Barrier()
                
        else:
            ialp_cpu = cupyx.empty_pinned((self.nao, self.naoaux), dtype=cupy.float64)
    
    
    if use_imujp_file:
        # initlize imujp cpu buffer
        max_imujp_occ_size = np.max(self.imujp_sizes_occ[mo_slice])
        imujp_buf_cpu = cupyx.empty_pinned((max_imujp_occ_size,), dtype=cupy.float64)

    last_i = -1
    bidx = 0
    iidx_all = 0
    i_idx = 0
    i_idx1 = 0
    cpu_buf_idx0 = 0
    

    for jlist_batch, is_batch in imujp_batches:

        #jlist_batch_gpu = cupy.asarray(jlist_batch)
        if self.int_storage != 3 and is_cal_rank:
            if self.fully_direct:
                nao = self.intopt.sorted_mol.nao
            else:
                nao = self.nao
            t0 = get_current_time()
            if len(last_jlist_set) == 0:
                for jidx, j in enumerate(jlist_batch):
                    buf_idx0 = jidx * max_nosv * nao
                    buf_idx1 = buf_idx0 + nao * self.nosv[j]
                    qao_offsets_gpu[j] = buf_idx0

                    src_idx0, src_idx1 = self.qao_offsets_node[j]
                    qao_gpu[buf_idx0:buf_idx1].set(self.qao_node[src_idx0:src_idx1])

            else:
                jlist_batch_set = set(jlist_batch)
                uniq_jlist = list(jlist_batch_set - last_jlist_set)
                obsolete_jlist = list(last_jlist_set - jlist_batch_set)

                assert len(uniq_jlist) == len(obsolete_jlist)

                for new_j, old_j in zip(uniq_jlist, obsolete_jlist):
                    buf_idx0 = int(qao_offsets_gpu[old_j])
                    buf_idx1 = buf_idx0 + nao * self.nosv[new_j]
                    qao_offsets_gpu[new_j] = buf_idx0

                    src_idx0, src_idx1 = self.qao_offsets_node[new_j]
                    qao_gpu[buf_idx0:buf_idx1].set(self.qao_node[src_idx0:src_idx1])

                last_jlist_set = jlist_batch_set

            self.recorder_data.append(record_elapsed_time(t0))

        for iidx, (i, jlist_i) in enumerate(is_batch):
            if i != last_i:
                if use_double_buffer:
                    io_idx = 0 if bidx % 2 else 1
                    cal_idx = 1 if bidx % 2 else 0
                    data_idx = 2 if bidx % 2 else 1
                    comp_idx = 1 if bidx % 2 else 2

                    data_event_idx = data_idx - 1
                    comp_event_idx = comp_idx - 1

                    if is_io_rank:
                        nnext_bidx = bidx + 2
                        if nnext_bidx < len(uniq_mos_batches):
                            ialp_read_cpu = double_ialp_buf_cpu[io_idx]
                            nnext_i = uniq_mos_batches[nnext_bidx]
                            t1 = get_current_time()
                            ialp_data.read_direct(ialp_read_cpu, np.s_[nnext_i-nocc_core])
                            self.recorder_io.append(record_elapsed_time(t1))
                    
                    if is_cal_rank:
                        ialp_load_gpu = ialp_buffer_gpu[0].reshape(self.nao, self.naoaux)
                        ialp_data_gpu = ialp_buffer_gpu[data_idx]
                        ialp_gpu = ialp_buffer_gpu[comp_idx].reshape(self.naoaux, self.nao)

                        if bidx + 1 < len(uniq_mos_batches):
                            with istream_data:
                                if iidx > 0:
                                    istream_data.wait_event(comp_events[data_event_idx])

                                t1 = get_current_time()
                                ialp_load_cpu = double_ialp_buf_cpu[cal_idx]
                                if self.unsort_ialp_ao:
                                    ialp_load_gpu.set(ialp_load_cpu)
                                    ialp_data_gpu = ialp_data_gpu.reshape(self.naoaux, self.nao)
                                    ialp_data_gpu[:] = ialp_load_gpu.T
                                else:
                                    #ialp_gpu = cupy.ascontiguousarray(cupy.asarray(ialp_cpu)[self.intopt.unsorted_ao_ids].T)
                                    ialp_data_gpu = ialp_data_gpu.reshape(self.nao, self.naoaux)
                                    ialp_data_gpu.set(ialp_load_cpu)
                                    cupy.take(ialp_data_gpu, self.intopt.unsorted_ao_ids, axis=0, out=ialp_load_gpu)
                                    ialp_data_gpu = ialp_data_gpu.reshape(self.naoaux, self.nao)
                                    ialp_data_gpu[:] = ialp_load_gpu.T
                                self.recorder_data.append(record_elapsed_time(t1))

                                data_events[data_event_idx].record()
                                istream_data.synchronize()
                    bidx += 1
                else:
                    if self.fully_direct:
                        
                        if i_idx == 0 or i_idx == i_idx1:
                            if i_idx > 0 and self.fully_direct == 2:
                                comp_events[comp_idx].record()

                            indices, ialp_seg_gpu = next(ialp_generator)
                            if self.fully_direct == 2:
                                comp_idx, i_idx0, i_idx1 = indices
                            else:
                                i_idx0, i_idx1 = indices

                        #ialp_gpu = ialp_data
                        

                        ialp_cart_i = ialp_seg_gpu[i_idx - i_idx0]
                        #ialp_gpu = cupy.dot(c2s_aux_coeff_gpu.T, ialp_cart_i.T, out=ialp_gpu)
                        t0 = get_current_time()
                        dgemm_cupy(1, 1, c2s_aux_coeff_gpu, ialp_cart_i, ialp_gpu, 1.0, 0.0)
                        recorder_ialp.append(record_elapsed_time(t0))
                        
                    elif self.int_storage == 3:
                        #ialp_gpu = cupy.ascontiguousarray(ialp_data[iidx].T)
                        ialp_gpu[:] = ialp_data[iidx].T
                    else:
                        t0 = get_current_time()
                        if use_ialp_file:
                            ialp_data.read_direct(ialp_cpu, source_sel=np.s_[i-nocc_core])
                        else:
                            ialp_cpu = ialp_data[i-nocc_core] #(naux, nao)
                        self.recorder_io.append(record_elapsed_time(t0))

                        t0 = get_current_time()
                        ialp_load_gpu = ialp_buffer_gpu[0].reshape(self.nao, self.naoaux)
                        ialp_load_gpu.set(ialp_cpu)
                        if self.unsort_ialp_ao:
                            #ialp_gpu = cupy.ascontiguousarray(cupy.asarray(ialp_cpu).T)
                            ialp_gpu = ialp_buffer_gpu[1].reshape(self.naoaux, self.nao)
                            ialp_gpu[:] = ialp_load_gpu.T
                        else:
                            #ialp_gpu = cupy.ascontiguousarray(cupy.asarray(ialp_cpu)[self.intopt.unsorted_ao_ids].T)
                            ialp_sorted_gpu = ialp_buffer_gpu[1].reshape(self.nao, self.naoaux)
                            cupy.take(ialp_load_gpu, self.intopt.unsorted_ao_ids, axis=0, out=ialp_sorted_gpu)
                            ialp_gpu = ialp_load_gpu.reshape(self.naoaux, self.nao)
                            ialp_gpu[:] = ialp_sorted_gpu.T
                        self.recorder_data.append(record_elapsed_time(t0))
                i_idx += 1

            if is_cal_rank:
                with istream_comp:
                    nfit = self.nfit_pair[i * self.no + jlist_i]

                    jidx_diag = np.where(jlist_i == i)[0]
                    if len(jidx_diag) > 0:
                        nfit[jidx_diag[0]] = self.naoaux

                    imujp_size = nfit * self.nosv[jlist_i]

                    fit_local_offsets = cum_offset(nfit)
                    aux_full = np.arange(self.naoaux, dtype=np.int32)
                    '''fit_batch_cpu = np.concatenate([self.fit_pair[i*self.no+j] if i != j 
                                                    else aux_full for j in jlist_i])'''
                    fit_batch_cpu = []
                    for j in jlist_i:
                        if j == i:
                            fit_pair = aux_full
                        else:
                            fit_pair = np.union1d(self.fit_list[i], self.fit_list[j])
                        fit_batch_cpu.append(fit_pair)
                    fit_batch_cpu = np.concatenate(fit_batch_cpu)
                    
                    threadsY = 32
                    threadsX = 32
                    
                    js_block, j_indices, fitLocalIndices, osvIndices = sliceJobsFor2DBlocks(jlist_i, nfit,
                                                                                            self.nosv[jlist_i], 
                                                                                            threadsX, threadsY)
                    t0 = get_current_time()
                    imujp_offsets = cum_offset(imujp_size)
                    total_imujp_size = imujp_offsets[-1]
                    #imujp_gpu = cupy.empty(total_imujp_size, dtype=cupy.float64)
                    assert imujp_buf_gpu.size >= total_imujp_size
                    imujp_gpu = imujp_buf_gpu[:total_imujp_size]
                    
                    imujp_offsets_gpu = cupy.asarray(imujp_offsets)
                    fit_local_offsets_gpu = numpy_to_cupy(fit_local_offsets, dtype=np.int32)
                    nfit_gpu = numpy_to_cupy(nfit, dtype=np.int32)
                    fit_batch_gpu = numpy_to_cupy(fit_batch_cpu, dtype=np.int32)
                    js_block_gpu = numpy_to_cupy(js_block, dtype=np.int32)
                    j_indices_gpu = numpy_to_cupy(j_indices, dtype=np.int32)
                    osvIndices_gpu = numpy_to_cupy(osvIndices, dtype=np.int32)
                    fitLocalIndices_gpu = numpy_to_cupy(fitLocalIndices, dtype=np.int32)
                    self.recorder_data.append(record_elapsed_time(t0))

                    if self.double_buffer:
                        istream_comp.wait_event(data_events[comp_event_idx])
                    
                    if self.fully_direct:
                        nao = self.intopt.sorted_mol.nao
                    else:
                        nao = self.nao

                    t0 = get_current_time()
                    osvMp2Cuda.imujpCupy(ialp_gpu, qao_gpu, imujp_gpu, fit_batch_gpu, 
                                        nfit_gpu, nosv_gpu, qao_offsets_gpu, 
                                        imujp_offsets_gpu, fit_local_offsets_gpu, 
                                        js_block_gpu, j_indices_gpu, osvIndices_gpu, 
                                        fitLocalIndices_gpu, np.int32(nao), 
                                        np.int32(self.naoaux), np.int32(len(js_block_gpu)),
                                        np.int32(threadsX), np.int32(threadsY))
                    
                    self.recorder_cal.append(record_elapsed_time(t0))
                    if self.double_buffer:
                        comp_events[comp_event_idx].record()

                    #t0 = get_current_time()
                    #save_idx1 = save_idx0 + total_imujp_size
                    #cupy.asnumpy(imujp_gpu, out=imujp_cpu[save_idx0:save_idx1])
                    #print_test(imujp_gpu, f"{i}")
                    #istream_comp.synchronize()
                    #imujp_cpu = cupy.asnumpy(imujp_gpu)
                    #imujp_cpu = imujp_gpu.get(stream=istream_comp)
                    #self.recorder_data.append(record_elapsed_time(t0))

                    
            
                if use_double_buffer:
                    istream_comp.synchronize()

                t0 = get_current_time()
                

                '''if use_imujp_file:
                    cpu_buf_idx1 = cpu_buf_idx0 + imujp_gpu.size
                    imujp_cpu = imujp_buf_cpu[cpu_buf_idx0:cpu_buf_idx1]
                    cpu_buf_idx0 = cpu_buf_idx1
                else:
                    imujp_idx0 = self.imujp_offsets_pair[i*self.no+jlist_i[0]][0]
                    imujp_idx1 = imujp_idx0 + imujp_gpu.size
                    imujp_cpu = imujp_data[imujp_idx0:imujp_idx1]
                cupy.asnumpy(imujp_gpu, out=imujp_cpu)'''

                pairs_to_save = i * self.no + jlist_i
                imujp_mo_idx0 = self.imujp_offsets_mo[i]
                imujp_offsets = np.asarray([self.imujp_offsets_pair[ipair]
                                            for ipair in pairs_to_save])
                
                if use_imujp_file:
                    imujp_offsets -= imujp_mo_idx0
                    imujp_cpu = imujp_buf_cpu
                else:
                    imujp_cpu = imujp_data
                #print(irank, imujp_offsets.tolist())
                imujp_offsets = merge_intervals(imujp_offsets)
                #print(irank, imujp_offsets);sys.exit()

                gpu_idx0 = 0
                for cpu_idx0, cpu_idx1 in imujp_offsets:
                    gpu_idx1 = gpu_idx0 + (cpu_idx1-cpu_idx0)
                    try:
                        #print(irank, gpu_idx0, gpu_idx1, cpu_idx0, cpu_idx1, imujp_gpu.size, imujp_cpu.size, flush=True)
                        cupy.asnumpy(imujp_gpu[gpu_idx0:gpu_idx1], out=imujp_cpu[cpu_idx0:cpu_idx1])
                        #imujp_cpu[cpu_idx0:cpu_idx1] = imujp_gpu[gpu_idx0:gpu_idx1].get()
                    except ValueError:
                        print(gpu_idx0, gpu_idx1, cpu_idx0, cpu_idx1, imujp_gpu.size, imujp_cpu.size, flush=True)
                        raise ValueError
                    gpu_idx0 = gpu_idx1

                self.recorder_data.append(record_elapsed_time(t0))
                
                
                if use_imujp_file:
                    next_i = -1
                    if iidx_all + 1 < len(all_mos_batches):
                        next_i = all_mos_batches[iidx_all + 1]

                    if i != next_i:
                        cupy.cuda.Stream.null.synchronize()
                        t0 = get_current_time()
                        imujp_idx0, imujp_idx1 = self.imujp_offsets_mo[i:i+2]
                        imujp_cpu = imujp_buf_cpu[:imujp_idx1-imujp_idx0]
                        imujp_data[imujp_idx0:imujp_idx1] = imujp_cpu
                        self.recorder_io.append(record_elapsed_time(t0))

                        cpu_buf_idx0 = 0

            if use_double_buffer:
                comm_pair.Barrier()

            last_i = i
            iidx_all += 1


    if self.int_storage == 3:
        close_ipc_handle(qao_gpu_ptr)

    if use_double_buffer:
        if is_cal_rank:
            unregister_pinned_memory(double_ialp_buf_cpu)
        win_ialp_buf.Free()
    
    if self.fully_direct:
        tferi = sum_elapsed_time(recorder_feri)
        tialp = sum_elapsed_time(recorder_ialp)

        if not self.cal_grad:
            self.intopt = None


        self.t_feri_mp2 += tferi + tialp
        print_time([["feri", tferi], ["ialp", tialp]], log)
        


def imujp_to_kmat_cuda(self, pair_slice, imujp_data, imuip_node=None, avail_cpu_mem=None):

    #cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()

    '''if self.int_storage in {1, 2}:
        use_imujp_file = True
    else:
        use_imujp_file = False'''
    #use_imujp_file = not isinstance(imujp_data, np.ndarray)
    use_imujp_file = isinstance(imujp_data, h5py.Dataset)

    max_nosv = np.max(self.nosv)
    max_nfit = np.max(self.nfit_pair)
    '''mask_mo = np.zeros(self.no, dtype=bool)
    mask_mo[pair_slice//self.no] = True
    mask_mo[pair_slice%self.no] = True
    mos_slice = np.arange(self.no)[mask_mo]'''
    ilist, jlist = np.divmod(pair_slice, self.no)
    mos_slice = np.union1d(ilist, jlist)
    max_nocc = get_ncols_for_memory(0.6*avail_gpu_mem(self.gpu_memory)/nrank_per_gpu, 
                                    self.naoaux * max_nosv, len(mos_slice))
    batch_slice = get_mo_batches(pair_slice, max_nocc, self.no)
    max_nocc = np.max([len(mos) for mos, _ in batch_slice])

    t0 = get_current_time()
    nosv_gpu = cupy.asarray(self.nosv, dtype=cupy.int32)
    self.recorder_data.append(record_elapsed_time(t0))

    imujp_offsets_gpu = cupy.empty(self.no**2, dtype=cupy.int64)
    if self.int_storage == 3:
        close_pairs_slice = pair_slice[self.is_close[pair_slice]]
        ilist_close, jlist_close = np.divmod(close_pairs_slice, self.no)
        mask_nondiag = ilist_close != jlist_close
        ilist_nondiag = ilist_close[mask_nondiag]
        jlist_nondiag = jlist_close[mask_nondiag]
        close_nondiag_pairs = close_pairs_slice[mask_nondiag]
        diag_pairs = mos_slice * (self.no + 1)
        imujp_pairs = np.concatenate((diag_pairs, close_nondiag_pairs, jlist_nondiag*self.no+ilist_nondiag))
        imujp_pairs.sort()
        imujp_size = [np.prod(self.imujp_dim[ipair]) for ipair in imujp_pairs]
        imujp_offsets = cum_offset(imujp_size)
        imujp_offsets_gpu[imujp_pairs] = imujp_offsets[:-1]
        
        read_offsets = [self.imujp_offsets_pair[ipair] for ipair in imujp_pairs]
        read_offsets = merge_intervals(read_offsets)

        #imujp_cpu = np.empty(imujp_offsets[-1])
        #imujp_gpu = cupy.empty(imujp_offsets[-1])
        imujp_gpu, imujp_ptr = get_shared_cupy(imujp_offsets[-1])
        #read_offsets_slice = get_slice(cores_gpu_shm, job_list=read_offsets)[irank_shm]
        if irank_gpu == 0:
            buf_idx0 = 0
            for read_idx0, read_idx1 in read_offsets:
                buf_idx1 = buf_idx0 + (read_idx1 - read_idx0)
                #np.copyto(imujp_cpu[buf_idx0:buf_idx1], self.imujp_node[read_idx0:read_idx1])
                imujp_gpu[buf_idx0:buf_idx1].set(self.imujp_node[read_idx0:read_idx1])
                buf_idx0 = buf_idx1
        cupy.cuda.Stream.null.synchronize()
        comm_shm.Barrier()
        #imujp_gpu = cupy.asarray(imujp_cpu)
    
    npair_close_batch = []
    npair_remote_batch = []
    for mos_batch, pairs_batch_all in batch_slice:
        npair_close = self.is_close[pairs_batch_all].sum()
        npair_remote = len(pairs_batch_all) - npair_close
        npair_close_batch.append(npair_close)
        npair_remote_batch.append(npair_remote)
    max_pairs_close_batch = np.max(npair_close_batch)
    max_pairs_remote_batch = np.max(npair_remote_batch)
    
    max_imuip_size = max_nocc * max_nosv * self.naoaux
    memory_left = avail_gpu_mem(self.gpu_memory)/nrank_per_gpu - max_imuip_size*8*1e-6
    
    rows_close = 2*max_nosv*max_nfit + (2*max_nosv)**2
    max_npair_close = get_ncols_for_memory(0.8*memory_left, rows_close, 
                                           max_pairs_close_batch)
    rows_remote = max_nosv**2 + max_nfit
    max_npair_remote = get_ncols_for_memory(0.8*memory_left, rows_remote, max_pairs_remote_batch)

    if use_imujp_file:
        if avail_cpu_mem is None:
            avail_cpu_mem = get_mem_spare(self.mol, per_core=True)
        rows_close_cpu = (2*max_nosv+1)*max_nfit
        max_npair_close_cpu = get_ncols_for_memory(0.8*avail_cpu_mem, rows_close_cpu, 
                                                max_pairs_close_batch)
        max_npair_close = min(max_npair_close, max_npair_close_cpu)

        max_npair_remote_cpu = get_ncols_for_memory(0.8*avail_cpu_mem, max_nfit, 
                                                    max_pairs_remote_batch, dtype=np.int32)
        max_npair_remote = min(max_npair_remote, max_npair_remote_cpu)
    
    #max_imujp_size = max_imuip_size + max_npair_close * 2 * max_nosv * max_nfit
    #imujp_buf_gpu = cupy.empty(max_imujp_size, dtype=cupy.float64)
    pair_buf_size = max(max_npair_close*rows_close, max_npair_remote*max_nosv**2)
    buf_gpu = cupy.empty(max_imuip_size+pair_buf_size, dtype=cupy.float64)

    t0 = get_current_time()
    if use_imujp_file:
        #imujp_buf_cpu = cupyx.empty_pinned((2*max_npair_close*max_nosv*max_nfit,), 
        buf_cpu = np.empty(((2*max_npair_close*max_nosv+1)*max_nfit,),
                                    dtype=np.float64)
        register_pinned_memory(buf_cpu)
    else:
        max_pair = max(max_npair_close, max_npair_remote)
        fit_pair_buf = np.empty((max_pair*max_nfit,), dtype=np.int32)
    self.recorder_io.append(record_elapsed_time(t0))

    for mos_batch, pairs_batch_all in batch_slice:

        check_close = self.is_close[pairs_batch_all]
        pairs_close_batch = pairs_batch_all[check_close]
        pairs_remote_batch = pairs_batch_all[np.invert(check_close)]

        if self.int_storage == 3:
            max_npair_close = len(pairs_close_batch)
            max_npair_remote = len(pairs_remote_batch)
        else:
            pairs_diag_batch = mos_batch * (self.no + 1)

            imuip_sizes = self.nosv[mos_batch] * self.naoaux
            total_imuip_size = np.sum(imuip_sizes)
            total_imujp_size_offdiag = max_npair_close * 2 * max_nosv * max_nfit
            total_imujp_size = total_imuip_size + total_imujp_size_offdiag

            imujp_gpu = buf_gpu[:total_imujp_size]
            close_kmat_buf_gpu = buf_gpu[total_imujp_size:]
            remote_kmat_buf_gpu = buf_gpu[total_imuip_size:]
            # Load imuip
            #t0 = get_current_time()
            #imuip_cpu = np.empty(total_imuip_size)
            
            if use_imujp_file:
                read_offsets = [self.imuip_offsets_node[i] for i in mos_batch]
            else:
                read_offsets = [self.imujp_offsets_pair[ipair] for ipair in pairs_diag_batch]
            
            read_offsets = merge_intervals(read_offsets)
            buf_idx0 = 0
            for read_idx0, read_idx1 in read_offsets:
                buf_idx1 = buf_idx0 + (read_idx1 - read_idx0)
                '''if use_imujp_file:
                    np.copyto(imuip_cpu[buf_idx0:buf_idx1], imuip_node[read_idx0:read_idx1])
                else:
                    np.copyto(imuip_cpu[buf_idx0:buf_idx1], self.imujp_node[read_idx0:read_idx1])'''
                if use_imujp_file:
                    imujp_gpu[buf_idx0:buf_idx1].set(imuip_node[read_idx0:read_idx1])
                else:
                    imujp_gpu[buf_idx0:buf_idx1].set(self.imujp_node[read_idx0:read_idx1])

                buf_idx0 = buf_idx1
            self.recorder_data.append(record_elapsed_time(t0))

            t0 = get_current_time()
            #imujp_gpu[:total_imuip_size].set(imuip_cpu)
        
            last_imujp_idx = buf_idx1
            imuip_offsets = cum_offset(imuip_sizes)
            imujp_offsets_gpu[pairs_diag_batch] = cupy.asarray(imuip_offsets[:-1])
            self.recorder_data.append(record_elapsed_time(t0))

        for type_idx, (max_npair, pairs_batch) in enumerate([[max_npair_close, pairs_close_batch], 
                                                            [max_npair_remote, pairs_remote_batch]]):
            npairs_batch = len(pairs_batch)
            if type_idx == 0:
                cal_close = True
            else:
                cal_close = False
            for idx0 in np.arange(npairs_batch, step=max_npair):
                idx1 = min(npairs_batch, idx0 + max_npair)
                pairs_now = pairs_batch[idx0:idx1]
                ilist_now, jlist_now = np.divmod(pairs_now, self.no)
                nosv_is = self.nosv[ilist_now]
                nosv_js = self.nosv[jlist_now]

                if cal_close:# Close pairs
                    if self.int_storage != 3:
                        # Load imujp
                        pairs_offdiag = pairs_now[ilist_now != jlist_now]
                        pairs_ji_offdiag = (pairs_offdiag%self.no) * self.no + (pairs_offdiag//self.no)
                        imujp_pairs_supp = np.append(pairs_offdiag, pairs_ji_offdiag)

                        if len(imujp_pairs_supp) > 0:
                            imujp_pairs_supp.sort()
                            
                            jlist_supp = imujp_pairs_supp % self.no
                            imujp_size_supp = self.nosv[jlist_supp] * self.nfit_pair[imujp_pairs_supp]
                            imujp_offsets_supp = cum_offset(imujp_size_supp)
                            total_imujp_size_supp = imujp_offsets_supp[-1]

                            t0 = get_current_time()
                            imujp_offsets_gpu[imujp_pairs_supp] = cupy.asarray((imujp_offsets_supp[:-1] + imuip_offsets[-1]))
                            self.recorder_data.append(record_elapsed_time(t0))
                            
                            
                            #imujp_cpu = np.empty(total_imuip_size_supp)
                            #imujp_cpu = cupyx.empty_pinned((total_imuip_size_supp,))
                            if use_imujp_file:
                                imujp_cpu = buf_cpu[:total_imujp_size_supp]

                            read_offsets = [self.imujp_offsets_pair[ipair] for ipair in imujp_pairs_supp]
                            read_offsets = merge_intervals(read_offsets)

                            buf_idx0 = 0
                            for read_idx0, read_idx1 in read_offsets:
                                buf_idx1 = buf_idx0 + read_idx1 - read_idx0
                                if use_imujp_file:
                                    t0 = get_current_time()
                                    imujp_data.read_direct(imujp_cpu, source_sel=np.s_[read_idx0:read_idx1], 
                                                            dest_sel=np.s_[buf_idx0:buf_idx1])
                                    self.recorder_io.append(record_elapsed_time(t0))
                                else:
                                    #np.copyto(imujp_cpu[buf_idx0:buf_idx1], self.imujp_node[read_idx0:read_idx1])
                                    t0 = get_current_time()
                                    imujp_gpu_idx0 = buf_idx0 + last_imujp_idx
                                    imujp_gpu_idx1 = imujp_gpu_idx0 + (buf_idx1 - buf_idx0)
                                    imujp_gpu[imujp_gpu_idx0:imujp_gpu_idx1].set(self.imujp_node[read_idx0:read_idx1])
                                    self.recorder_data.append(record_elapsed_time(t0))
                                buf_idx0 = buf_idx1
                            
                            if use_imujp_file:
                                t0 = get_current_time()
                                imujp_gpu_idx0 = last_imujp_idx
                                imujp_gpu_idx1 = imujp_gpu_idx0 + total_imujp_size_supp
                                imujp_gpu[imujp_gpu_idx0:imujp_gpu_idx1].set(imujp_cpu)
                                self.recorder_data.append(record_elapsed_time(t0))

                                fit_pair_buf = buf_cpu[total_imujp_size_supp:].view(np.int32)

                    kmat_size = (nosv_is + nosv_js)**2
                else:
                    kmat_size = nosv_is * nosv_js
                    if use_imujp_file:
                        fit_pair_buf = buf_cpu.view(np.int32)

                kmat_size = [np.prod(self.kmat_dim[ipair]) for ipair in pairs_now]

                kmat_offsets = cum_offset(kmat_size)
                #kmat_gpu = cupy.empty(kmat_offsets[-1], dtype=cupy.float64)
                if cal_close:
                    kmat_gpu = close_kmat_buf_gpu[:kmat_offsets[-1]]
                else:
                    kmat_gpu = remote_kmat_buf_gpu[:kmat_offsets[-1]]

                t0 = get_current_time()
                pairs_now_gpu = cupy.asarray(pairs_now, dtype=cupy.int32)
                kmat_offsets_gpu = cupy.asarray(kmat_offsets)
                #nfit_pair_batch_gpu = cupy.asarray(self.nfit_pair[pairs_now], dtype=cupy.int32)
                
                #fit_pair_batch = np.concatenate([self.fit_pair[ipair] for ipair in pairs_now])
                nfit_pair_batch = []
                #fit_pair_batch = []
                fit_idx0 = 0
                for ipair in pairs_now:
                    fit_pair = np.union1d(self.fit_list[ipair//self.no], 
                                          self.fit_list[ipair%self.no])
                    nfit_pair = len(fit_pair)
                    fit_idx1 = fit_idx0 + nfit_pair
                    #fit_pair_batch.append(fit_pair)
                    fit_pair_buf[fit_idx0:fit_idx1] = fit_pair
                    fit_idx0 = fit_idx1

                    nfit_pair_batch.append(nfit_pair)
                fit_pair_batch = fit_pair_buf[:fit_idx1]
                #fit_pair_batch = np.concatenate(fit_pair_batch)
                nfit_pair_batch_gpu = cupy.asarray(nfit_pair_batch)
                fit_local_offsets_gpu = cum_offset_cupy(nfit_pair_batch_gpu, dtype=cupy.int64)
                fit_pair_batch_gpu = cupy.asarray(fit_pair_batch, dtype=cupy.int32)
                self.recorder_data.append(record_elapsed_time(t0))

                #print_mem(f'close:{cal_close}, {len(pairs_now)} pairs', self.pid_list)
                
                t0 = get_current_time()
                if cal_close:
                    osvMp2Cuda.closeOsvKmatCupy(imujp_gpu, kmat_gpu, pairs_now_gpu,
                                            fit_pair_batch_gpu, nosv_gpu, fit_local_offsets_gpu,
                                            imujp_offsets_gpu, kmat_offsets_gpu, 
                                            np.int32(self.no), np.int32(self.naoaux), 
                                            np.int32(len(pairs_now)))
                else:
                    osvMp2Cuda.remoteOsvKmatCupy(imujp_gpu, kmat_gpu, pairs_now_gpu,
                                            fit_pair_batch_gpu, nosv_gpu, fit_local_offsets_gpu,
                                            imujp_offsets_gpu, kmat_offsets_gpu, 
                                            np.int32(self.no), np.int32(self.naoaux), 
                                            np.int32(len(pairs_now)))
                self.recorder_cal.append(record_elapsed_time(t0))

                t0 = get_current_time()
                #kmat_cpu = kmat_gpu.get()
 
                save_offsets = [self.kmat_offsets_node[ipair] for ipair in pairs_now]
                save_offsets = merge_intervals(save_offsets)

                buf_idx0 = 0
                #for ipair in pairs_now:
                #    kidx0, kidx1 = self.kmat_offsets_node[ipair]
                kmat_node = self.close_kmat_node if cal_close else self.remote_kmat_node
                for kidx0, kidx1 in save_offsets:
                    buf_idx1 = buf_idx0 + (kidx1 - kidx0)
                    #self.kmat_node[kidx0:kidx1] = kmat_cpu[buf_idx0:buf_idx1]
                    #cupy.asnumpy(kmat_gpu[buf_idx0:buf_idx1], out=self.kmat_node[kidx0:kidx1])
                    cupy.asnumpy(kmat_gpu[buf_idx0:buf_idx1], out=kmat_node[kidx0:kidx1])
                    buf_idx0 = buf_idx1
                self.recorder_data.append(record_elapsed_time(t0))

                
        if self.int_storage != 3:
            imujp_gpu = None
        kmat_gpu = None
    if use_imujp_file:
        unregister_pinned_memory(buf_cpu)



def get_precon_remote_cuda(self, mo_slice):

    #cupy.get_default_memory_pool().free_all_blocks()

    nmo_slice = len(mo_slice)
    #nstream = min(16, nmo_slice)
    #streams = [cupy.cuda.Stream() for _ in range(nstream)]

    max_nosv = np.max(self.nosv)
    memory_left = avail_gpu_mem(self.gpu_memory)/nrank_per_gpu - 4 * max_nosv**2 * 8 * 1e-6
    max_nmo = get_ncols_for_memory(0.6 * memory_left, 2 * max_nosv**2, nmo_slice)
    
    for batch_midx0 in np.arange(nmo_slice, step=max_nmo):
        batch_midx1 = min(batch_midx0 + max_nmo, nmo_slice)
        nmo_batch = batch_midx1 - batch_midx0
        mos_batch = mo_slice[batch_midx0:batch_midx1]

        nosv_batch = self.nosv[mos_batch]

        fmat_size = nosv_batch ** 2
        #fmat_size_sum = np.cumsum(fmat_size)
        #emui_size_sum = np.cumsum(nosv_batch)
        fmat_offsets = cum_offset(fmat_size)
        emui_offsets = cum_offset(nosv_batch)
        #fmat_cpu = np.empty(fmat_offsets[-1])
        fmat_cpu = cupyx.empty_pinned((fmat_offsets[-1],))

        sf_offsets_node = [self.sf_offsets_node[i*self.no+i] for i in mos_batch]
        sf_offsets_node = merge_intervals(sf_offsets_node)

        save_idx0 = 0
        for read_idx0, read_idx1 in sf_offsets_node:
            save_idx1 = save_idx0 + (read_idx1 - read_idx0)
            fmat_cpu[save_idx0:save_idx1] = self.fmat_node[read_idx0:read_idx1]
            save_idx0 = save_idx1
        
        t0 = get_current_time()
        fmat_gpu = cupy.asarray(fmat_cpu)
        xii_gpu = cupy.empty_like(fmat_gpu)
        emui_gpu = cupy.empty(emui_offsets[-1])
        temp_gpu = cupy.empty_like(fmat_gpu)
        fmat_offsets_gpu = cupy.asarray(fmat_offsets)
        emui_offsets_gpu = cupy.asarray(emui_offsets)
        nosv_batch_gpu = numpy_to_cupy(nosv_batch, dtype=cupy.int32)
        self.recorder_data.append(record_elapsed_time(t0))

        t0 = get_current_time()
        osvMp2Cuda.remoteOsvPreconCupy(fmat_gpu, xii_gpu, emui_gpu,
                                        temp_gpu, nosv_batch_gpu,
                                        fmat_offsets_gpu, emui_offsets_gpu,
                                        nmo_batch, 256, np.max(nosv_batch))


        self.recorder_cal.append(record_elapsed_time(t0))
        
        xii_cpu = xii_gpu.get()
        emui_cpu = emui_gpu.get()
        xii_gpu = emui_gpu = None

        for midx, i in enumerate(mos_batch):

            xidx0, xidx1 = self.xii_offsets_node[i]
            #local_xidx1 = fmat_size_sum[midx]
            #local_xidx0 = local_xidx1 - (xidx1 - xidx0)
            src_xidx0, src_xidx1 = fmat_offsets[midx:midx+2]
            self.xii_node[xidx0:xidx1] = xii_cpu[src_xidx0:src_xidx1]

            eidx0, eidx1 = self.emui_offsets_node[i]
            #local_eidx1 = emui_size_sum[midx]
            #local_eidx0 = local_eidx1 - (eidx1 - eidx0)
            src_eidx0, src_eidx1 = emui_offsets[midx:midx+2]
            self.emui_node[eidx0:eidx1] = emui_cpu[src_eidx0:src_eidx1]

def get_precon_close_cuda(self, pair_slice, full_xmat_size_sum):

    #cupy.get_default_memory_pool().free_all_blocks()

    npair_slice = len(pair_slice)

    max_nosv = np.max(self.nosv)
    memory_left = avail_gpu_mem(self.gpu_memory)/nrank_per_gpu - 4 * 4 * max_nosv**2 * 8 * 1e-6
    max_npair = get_ncols_for_memory(0.6 * memory_left, 2 * 4 * max_nosv**2, npair_slice)
    pair_batches = get_pair_batches(pair_slice, max_npair, self.no)

    t0 = get_current_time()
    eo_gpu = cupy.asarray(self.eo)
    nosv_gpu = numpy_to_cupy(self.nosv, dtype=cupy.int32)
    self.recorder_data.append(record_elapsed_time(t0))

    for pairs_batch in pair_batches:
        
        ext_pairs = get_pairs_with_diag(pairs_batch, self.no)
        ext_pairs = ext_pairs.astype(np.int32)

        ext_ilist, ext_jlist = np.divmod(ext_pairs, self.no)

        smat_size = self.nosv[ext_ilist] * self.nosv[ext_jlist]
        #smat_size_sum = np.cumsum(smat_size)
        #total_smat_size = smat_size_sum[-1]
        sf_offsets = cum_offset(smat_size)
        total_smat_size = sf_offsets[-1]
        '''smat_cpu = np.empty(total_smat_size)
        fmat_cpu = np.empty(total_smat_size)'''
        smat_cpu = cupyx.empty_pinned((total_smat_size,))
        fmat_cpu = cupyx.empty_pinned((total_smat_size,))

        sf_offsets_node = [self.sf_offsets_node[ipair] for ipair in ext_pairs]
        sf_offsets_node = merge_intervals(sf_offsets_node)
        
        save_idx0 = 0
        for read_idx0, read_idx1 in sf_offsets_node:
            save_idx1 = save_idx0 + (read_idx1 - read_idx0)

            smat_cpu[save_idx0:save_idx1] = self.smat_node[read_idx0:read_idx1]
            fmat_cpu[save_idx0:save_idx1] = self.fmat_node[read_idx0:read_idx1]

            save_idx0 = save_idx1

        
        ilist, jlist = np.divmod(pairs_batch, self.no)
        nosv_ij_batch = self.nosv[ilist] + self.nosv[jlist]
        full_xmat_size_batch = nosv_ij_batch**2
        full_xmat_offsets_batch = cum_offset(full_xmat_size_batch)

        xmat_gpu = cupy.empty(full_xmat_offsets_batch[-1])
        emuij_gpu = cupy.empty_like(xmat_gpu)
        tempA_gpu = cupy.empty_like(xmat_gpu)
        tempB_gpu = cupy.empty_like(xmat_gpu)
        tempC_gpu = cupy.empty_like(xmat_gpu)

        npairs_batch = len(pairs_batch)
        ncol_xmat_gpu = cupy.empty(npairs_batch, dtype=cupy.int32)

        t0 = get_current_time()
        ext_pair_indices_gpu = cupy.empty(self.no**2, dtype=cupy.int32)
        ext_pair_indices_gpu[ext_pairs] = cupy.arange(len(ext_pairs), dtype=cupy.int32)
        pairs_batch_gpu = numpy_to_cupy(pairs_batch, dtype=cupy.int32)
        sf_offsets_gpu = cupy.asarray(sf_offsets)
        smat_gpu = cupy.asarray(smat_cpu)
        fmat_gpu = cupy.asarray(fmat_cpu)
        xmat_offsets_gpu = numpy_to_cupy(full_xmat_offsets_batch)
        self.recorder_data.append(record_elapsed_time(t0))

        max_nosv_batch = np.max(nosv_ij_batch)

        t0 = get_current_time()
        osvMp2Cuda.closeOsvPreconCupy(smat_gpu, fmat_gpu, eo_gpu, xmat_gpu, 
                                      emuij_gpu, tempA_gpu, tempB_gpu, tempC_gpu, 
                                      pairs_batch_gpu, ext_pair_indices_gpu, nosv_gpu, 
                                      ncol_xmat_gpu, sf_offsets_gpu, xmat_offsets_gpu, 
                                      self.no, npairs_batch, 256, max_nosv_batch)
        self.recorder_cal.append(record_elapsed_time(t0))

        t0 = get_current_time()
        xmat_cpu = cupy.asnumpy(xmat_gpu)
        emuij_cpu = cupy.asnumpy(emuij_gpu)
        self.recorder_data.append(record_elapsed_time(t0))

        ncol_xmat = ncol_xmat_gpu.get()
        for pidx, ipair in enumerate(pairs_batch):
            ncol = ncol_xmat[pidx]
            nosv_ij = nosv_ij_batch[pidx]
            self.xmat_dim[ipair] = [nosv_ij, ncol]
            self.emuij_dim[ipair] = [ncol, ncol]

            xmatij_size = nosv_ij * ncol
            emuij_size = ncol * ncol

            src_eidx0 = src_xidx0 = full_xmat_offsets_batch[pidx]
            src_xidx1 = src_xidx0 + xmatij_size
            src_eidx1 = src_eidx0 + emuij_size

            dst_eidx1 = dst_xidx1 = full_xmat_size_sum[ipair]
            dst_xidx0 = dst_xidx1 - xmatij_size
            dst_eidx0 = dst_eidx1 - emuij_size

            self.xmat_node[dst_xidx0:dst_xidx1] = xmat_cpu[src_xidx0:src_xidx1]
            self.emuij_node[dst_eidx0:dst_eidx1] = emuij_cpu[src_eidx0:src_eidx1]


def local_to_node(pairlist, node_mat, local_mat, node_address, local_offsets_all, local_pair_indices):
    node_offsets = [np.copy(node_address[pairlist[0]])]
    pidx0 = local_pair_indices[0]
    local_offsets = [[local_offsets_all[pidx0], local_offsets_all[pidx0+1]]]

    for pidx, ipair in zip(local_pair_indices[1:], pairlist[1:]):
        node_idx0, node_idx1 = node_address[ipair]
        local_idx0, local_idx1 = local_offsets_all[pidx: pidx+2]
        
        if node_idx0 == node_offsets[-1][1] and \
           local_idx0 == local_offsets[-1][1]:
            node_offsets[-1][1] = node_idx1
            local_offsets[-1][1] = local_idx1
        else:
            node_offsets.append([node_idx0, node_idx1])
            local_offsets.append([local_idx0, local_idx1])

    for (node_idx0, node_idx1), (local_idx0, local_idx1) in zip(node_offsets, local_offsets):
        node_mat[node_idx0: node_idx1] = local_mat[local_idx0: local_idx1]


def local_to_node_new(pairlist, node_mat, local_mat, node_address, local_offsets_all):
    pair0 = pairlist[0]
    node_offsets = [np.copy(node_address[pair0])]
    
    loc_idx0 = local_offsets_all[pair0]
    mat_size_pair0 = node_address[pair0][1] - node_address[pair0][0]
    loc_idx1 = loc_idx0 + mat_size_pair0
    local_offsets = [[loc_idx0, loc_idx1]]

    for ipair in pairlist[1:]:
        node_idx0, node_idx1 = node_address[ipair]
        local_idx0 = local_offsets_all[ipair]
        local_idx1 = local_idx0 + (node_idx1 - node_idx0)
        
        if node_idx0 == node_offsets[-1][1] and \
           local_idx0 == local_offsets[-1][1]:
            node_offsets[-1][1] = node_idx1
            local_offsets[-1][1] = local_idx1
        else:
            node_offsets.append([node_idx0, node_idx1])
            local_offsets.append([local_idx0, local_idx1])


    for (node_idx0, node_idx1), (local_idx0, local_idx1) in zip(node_offsets, local_offsets):
        node_mat[node_idx0: node_idx1] = local_mat[local_idx0: local_idx1]


def load_gpu_mat(indices, offsets_node, mat_node, mat_offsets_gpu=None, get_full_offsets=False,
                 use_pin=True, sync=False, mat_gpu=None, gpu_buf_idx0=0):
    read_offsets = np.asarray([offsets_node[i] for i in indices])
    mat_sizes = read_offsets[:, 1] - read_offsets[:, 0]
    mat_offsets = cum_offset(mat_sizes) + gpu_buf_idx0

    if mat_offsets_gpu is None:
        mat_offsets_gpu = cupy.asarray(mat_offsets)
    else:
        if isinstance(mat_offsets_gpu, int):
            if get_full_offsets:
                mat_offsets_gpu = cupy.empty((mat_offsets_gpu, 2), dtype=int)
            else:
                mat_offsets_gpu = cupy.empty(mat_offsets_gpu, dtype=int)
        if get_full_offsets:
            offsets_gpu = cupy.asarray(mat_offsets)
            mat_offsets_gpu[indices, 0] = offsets_gpu[:-1]
            mat_offsets_gpu[indices, 1] = offsets_gpu[1:]
        else:
            mat_offsets_gpu[indices] = cupy.asarray(mat_offsets[:-1])

    mat_size = mat_offsets[-1] - gpu_buf_idx0

    if mat_gpu is None:
        mat_gpu = cupy.empty(mat_size, dtype=cupy.float64)
    if use_pin:
        mat_cpu = cupyx.empty_pinned((mat_size,), dtype=cupy.float64)
        buf_idx0 = 0
        for read_idx0, read_idx1 in merge_intervals(read_offsets):
            buf_idx1 = buf_idx0 + (read_idx1 - read_idx0)
            np.copyto(mat_cpu[buf_idx0:buf_idx1], mat_node[read_idx0:read_idx1])
            #mat_cpu[buf_idx0:buf_idx1] = mat_node[read_idx0:read_idx1]
            buf_idx0 = buf_idx1
        gpu_buf_idx1 = gpu_buf_idx0 + (mat_cpu.size)
        mat_gpu[gpu_buf_idx0:gpu_buf_idx1].set(mat_cpu)
        cupy.cuda.Stream.null.synchronize()
    else:
        #buf_idx0 = 0
        buf_idx0 = gpu_buf_idx0
        for read_idx0, read_idx1 in merge_intervals(read_offsets):
            buf_idx1 = buf_idx0 + (read_idx1 - read_idx0)
            #np.copyto(mat_cpu[buf_idx0:buf_idx1], mat_node[read_idx0:read_idx1])
            mat_gpu[buf_idx0:buf_idx1].set(mat_node[read_idx0:read_idx1])
            buf_idx0 = buf_idx1

    return mat_offsets_gpu, mat_gpu


def save_mbe_sub(self, nmo, pairs_clus, tmat_sub, pair_ene_sub, tmat_offsets_sub,
                 tmat_batch=None, pair_ene_rank=None, tmat_sizesum_batch=None):
    # For cumulative amplitudes
    npair_clus = nmo * (nmo + 1) // 2
    ncluster = pairs_clus.size // npair_clus

    if self.method == 3:
        if nmo == 1:
            pairs = pairs_clus
            pidx_sub = np.arange(npair_clus)
        elif nmo == 2:
            pairs = pairs_clus.reshape(-1, npair_clus)[:, 1]
            pidx_sub = np.arange(ncluster) * npair_clus + 1
        
        for pidx, ipair in zip(pidx_sub, pairs):
            save_tidx0, save_tidx1 = self.tmat_offsets_node[ipair]
            read_tidx0, read_tidx1 = tmat_offsets_sub[pidx: pidx+2]
            #self.tmat_node[save_tidx0: save_tidx1] = tmat_sub[read_tidx0: read_tidx1]
            self.close_tmat_node[save_tidx0: save_tidx1] = tmat_sub[read_tidx0: read_tidx1]
    else:
        acc_tnode = tmat_sub is not None
        if nmo == 1:
            for pidx, ipair in enumerate(pairs_clus):
                save_tidx0, save_tidx1 = self.tmat_offsets_node[ipair]
                read_tidx0, read_tidx1 = tmat_offsets_sub[pidx: pidx+2]
                self.tmat_save[save_tidx0: save_tidx1] = tmat_sub[read_tidx0: read_tidx1]
                if acc_tnode:
                    #self.tmat_node[save_tidx0: save_tidx1] = self.oneb_counts[ipair] * tmat_sub[read_tidx0: read_tidx1]
                    self.close_tmat_node[save_tidx0: save_tidx1] = self.oneb_counts[ipair] * tmat_sub[read_tidx0: read_tidx1]
                self.pairene_node[ipair] = self.oneb_counts[ipair] * pair_ene_sub[pidx]
        elif nmo == 2:
            pairs_clus = pairs_clus.reshape(-1, npair_clus)
            pairs_ij = pairs_clus[:, 1]
            count_offdiag = self.twob_counts_offdiag[pairs_ij]
            pidx_sub = np.arange(len(pairs_ij)) * npair_clus + 1
            for pidx, count, ipair in zip(pidx_sub, count_offdiag, pairs_ij):
                save_tidx0, save_tidx1 = self.tmat_offsets_node[ipair]
                read_tidx0, read_tidx1 = tmat_offsets_sub[pidx: pidx+2]
                self.tmat_save[save_tidx0: save_tidx1] = tmat_sub[read_tidx0: read_tidx1]
                if acc_tnode:
                    #self.tmat_node[save_tidx0: save_tidx1] = count * tmat_sub[read_tidx0: read_tidx1]
                    self.close_tmat_node[save_tidx0: save_tidx1] = count * tmat_sub[read_tidx0: read_tidx1]
                self.pairene_node[ipair] = count * pair_ene_sub[pidx]


            count_diag = self.twob_counts_diag[pairs_ij]
            for clus_idx in range(ncluster):
                count = count_diag[clus_idx]
                pairs = pairs_clus[clus_idx]
                for local_pidx in [0, 2]:
                    pidx = clus_idx * npair_clus + local_pidx
                    ipair = pairs[local_pidx]

                    read_tidx0, read_tidx1 = tmat_offsets_sub[pidx: pidx+2]

                    save_tidx1 = tmat_sizesum_batch[ipair]
                    save_tidx0 = save_tidx1 - (read_tidx1 - read_tidx0)
                    
                    if acc_tnode:
                        tmat_batch[save_tidx0: save_tidx1] += count * tmat_sub[read_tidx0: read_tidx1]
                    pair_ene_rank[ipair] += count * pair_ene_sub[pidx]
        else:

            for pidx, ipair in enumerate(pairs_clus):

                read_tidx0, read_tidx1 = tmat_offsets_sub[pidx: pidx+2]

                save_tidx1 = tmat_sizesum_batch[ipair]
                save_tidx0 = save_tidx1 - (read_tidx1 - read_tidx0)

                if acc_tnode:
                    tmat_batch[save_tidx0: save_tidx1] += tmat_sub[read_tidx0: read_tidx1]
                pair_ene_rank[ipair] += pair_ene_sub[pidx]





def close_residual_iter_cuda(self, pairlist):

    #cupy.get_default_memory_pool().free_all_blocks()

    mp2 = self.mp2

    max_nosv = np.max(mp2.nosv)
    max_tmat_size = 4 * max_nosv**2

    max_nk = np.max([len(self.klist_pairs[ipair]) for ipair in pairlist])

    max_npair_gpu = get_ncols_for_memory(0.6*mp2.gpu_memory, max_nk * max_tmat_size, len(pairlist))
    temp_offsets_gpu = uniform_cum_offset_cupy(max_tmat_size, max_npair_gpu)
    temp_a_gpu = cupy.empty(int(temp_offsets_gpu[-1]))

    temp_b_gpu = cupy.empty_like(temp_a_gpu)

    t0 = get_current_time()
    loc_fock_gpu = cupy.asarray(mp2.loc_fock).ravel()
    #nosv_gpu = cupy.asarray(mp2.nosv)
    nosv_gpu = numpy_to_cupy(mp2.nosv, dtype=cupy.int32)
    accumulate_time(mp2.t_data, t0)
    #ext_pair_indices_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)
    
    def get_ene(tinit_node=None, tinit_offsets=None):

        if mp2.int_storage == 3:
            sf_offsets_gpu, smat_gpu = self.sf_offsets_gpu, self.smat_gpu
            fmat_gpu = self.fmat_gpu
            kmat_offsets_gpu, kmat_gpu = self.kmat_offsets_gpu, self.kmat_gpu            
            xmat_offsets_full_gpu, xmat_gpu = self.xmat_offsets_full_gpu, self.xmat_gpu
            emuij_offsets_gpu, emuij_gpu = self.emuij_offsets_gpu, self.emuij_gpu
        else:
            sf_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)
            xmat_offsets_full_gpu = cupy.empty((mp2.no**2, 2), dtype=cupy.int64)
            emuij_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)
            kmat_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)
        
        tinit_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)
        if tinit_node is None:
            tinit_node = mp2.tmat_node
            tinit_offsets = mp2.tmat_offsets_node
        else:
            tnew_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)

        ene_new = 0.0
        pair_batches = get_pair_batches(pairlist, max_npair_gpu, mp2.no)
        for pairs_batch in pair_batches:
            npairs_batch = len(pairs_batch)
            extended_pairs_batch = get_extended_pairs(pairs_batch, mp2.no, self.klist_pairs)
            #ext_pair_indices_gpu[extended_pairs_batch] = cupy.arange(len(extended_pairs_batch))

            nk_pairs_batch = np.asarray([len(self.klist_pairs[ipair]) for ipair in pairs_batch])
            klist_offsets = cum_offset(nk_pairs_batch)
            klist_pairs_batch = np.concatenate([self.klist_pairs[ipair] for ipair in pairs_batch])

            t0 = get_current_time()
            if mp2.int_storage != 3:
                sf_offsets_gpu, smat_gpu = load_gpu_mat(extended_pairs_batch, mp2.sf_offsets_node, 
                                                        mp2.smat_node, sf_offsets_gpu)
                _, fmat_gpu = load_gpu_mat(extended_pairs_batch, mp2.sf_offsets_node, mp2.fmat_node, 
                                           sf_offsets_gpu)
                #kmat_offsets_gpu, kmat_gpu = load_gpu_mat(pairs_batch, mp2.kmat_offsets_node, mp2.kmat_node, kmat_offsets_gpu)
                kmat_offsets_gpu, kmat_gpu = load_gpu_mat(pairs_batch, mp2.kmat_offsets_node, 
                                                          mp2.close_kmat_node, kmat_offsets_gpu)
                xmat_offsets_full_gpu, xmat_gpu = load_gpu_mat(pairs_batch, mp2.xmat_offsets_node, 
                                                               mp2.xmat_node, xmat_offsets_full_gpu, 
                                                               get_full_offsets=True)
                emuij_offsets_gpu, emuij_gpu = load_gpu_mat(pairs_batch, mp2.emuij_offsets_node, 
                                                            mp2.emuij_node, emuij_offsets_gpu)

            tinit_offsets_gpu, tinit_gpu = load_gpu_mat(extended_pairs_batch, tinit_offsets, tinit_node, 
                                                                    tinit_offsets_gpu)
            if tinit_node is None:
                tnew_offsets_gpu, tnew_gpu = tinit_offsets_gpu, tinit_gpu
            else:
                '''tnew_offsets_gpu, tnew_gpu = load_gpu_mat(pairs_batch, mp2.tmat_offsets_node, mp2.tmat_node, 
                                                                    tnew_offsets_gpu)'''
                tnew_offsets_gpu, tnew_gpu = load_gpu_mat(pairs_batch, mp2.tmat_offsets_node, 
                                                          mp2.close_tmat_node, tnew_offsets_gpu)

            #cupy.cuda.Stream.null.synchronize()

            ilist_batch, jlist_batch = np.divmod(pairs_batch, mp2.no)
            rmat_sizes = (mp2.nosv[ilist_batch] + mp2.nosv[jlist_batch])**2
            rmat_offsets = cum_offset(rmat_sizes)
            rmat_offsets_gpu = cupy.asarray(rmat_offsets)
            rmat_gpu = cupy.empty(rmat_offsets[-1], dtype=cupy.float64)

            pair_ene_gpu = cupy.empty(npairs_batch)

            #pairs_batch_gpu = cupy.asarray(pairs_batch)
            #klist_pairs_batch_gpu = cupy.asarray(klist_pairs_batch)
            pairs_batch_gpu = numpy_to_cupy(pairs_batch, dtype=cupy.int32)
            klist_pairs_batch_gpu = numpy_to_cupy(klist_pairs_batch, dtype=cupy.int32)
            klist_offsets_gpu = cupy.asarray(klist_offsets)

            accumulate_time(mp2.t_data, t0)

            # pairs_batch_gpu, nosv_gpu, klist_pairs_batch_gpu, nOcc
            t0 = get_current_time()
            osvMp2Cuda.closeResidualCupy(smat_gpu, fmat_gpu, tinit_gpu, tnew_gpu, kmat_gpu, 
                                         rmat_gpu, xmat_gpu, emuij_gpu, temp_a_gpu,
                                        temp_b_gpu, loc_fock_gpu, pair_ene_gpu, pairs_batch_gpu, 
                                        nosv_gpu, klist_pairs_batch_gpu, rmat_offsets_gpu, 
                                        sf_offsets_gpu, tinit_offsets_gpu, tnew_offsets_gpu, 
                                        kmat_offsets_gpu, xmat_offsets_full_gpu, emuij_offsets_gpu, 
                                        temp_offsets_gpu, klist_offsets_gpu, mp2.no, 1e-5)
            accumulate_time(mp2.t_cal, t0)
            
            t0 = get_current_time()
            tnew_cpu = cupy.asnumpy(tnew_gpu)
            pair_ene_cpu = cupy.asnumpy(pair_ene_gpu)
            tnew_offsets = tnew_offsets_gpu.get()
            accumulate_time(mp2.t_data, t0)

            #self.pair_ene[pairs_batch] = pair_ene_cpu
            if self.check_ene:
                self.pair_ene[pairs_batch] = pair_ene_cpu
            ene_new += np.sum(pair_ene_cpu)
            '''local_to_node_new(pairs_batch, mp2.tmat_node, tnew_cpu, mp2.tmat_offsets_node, 
                              tnew_offsets)'''
            local_to_node_new(pairs_batch, mp2.close_tmat_node, tnew_cpu, mp2.tmat_offsets_node, 
                              tnew_offsets)

        return ene_new
    return get_ene


def remote_residual_iter_cuda(self, pairlist):

    #cupy.get_default_memory_pool().free_all_blocks()

    mp2 = self.mp2

    max_nosv = np.max(mp2.nosv)
    max_tmat_size = max_nosv**2 

    max_nk = 9 # 2 diagonal 4-block and 1 off-diagonal 1-block

    max_npair_gpu = get_ncols_for_memory(0.6*mp2.gpu_memory, max_nk * max_tmat_size, len(pairlist))
    temp_offsets_gpu = uniform_cum_offset_cupy(max_tmat_size, max_npair_gpu)
    temp_a_gpu = cupy.empty(int(temp_offsets_gpu[-1]))

    t0 = get_current_time()
    eo_gpu = cupy.asarray(mp2.eo)
    loc_fock_gpu = cupy.asarray(mp2.loc_fock).ravel()
    #nosv_gpu = cupy.asarray(mp2.nosv)
    nosv_gpu = numpy_to_cupy(mp2.nosv, dtype=cupy.int32)
    accumulate_time(mp2.t_data, t0)
    
    def get_ene(tinit_node=None, tinit_offsets=None):
        if mp2.int_storage == 3:
            sf_offsets_gpu, smat_gpu = self.sf_offsets_gpu, self.smat_gpu
            fmat_gpu = self.fmat_gpu
            kmat_offsets_gpu, kmat_gpu = self.kmat_offsets_gpu, self.kmat_gpu  
            xii_offsets_gpu, xii_gpu = self.xii_offsets_gpu, self.xii_gpu
            emui_offsets_gpu, emui_gpu = self.emui_offsets_gpu, self.emui_gpu
        else:
            sf_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)
            kmat_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)
            xii_offsets_gpu = cupy.empty(mp2.no, dtype=cupy.int64)
            emui_offsets_gpu = cupy.empty(mp2.no, dtype=cupy.int64)
        tmat_offsets_gpu = cupy.empty(mp2.no**2, dtype=cupy.int64)

        ene_new = 0.0
        pair_batches = get_pair_batches(pairlist, max_npair_gpu, mp2.no)
        for pairs_batch in pair_batches:
            npairs_batch = len(pairs_batch)
            ilist_batch, jlist_batch = np.divmod(pairs_batch, mp2.no)
            mos_batch = np.union1d(ilist_batch, jlist_batch)
            diag_pairs = mos_batch * mp2.no + mos_batch
            extended_pairs_batch = get_pairs_with_diag(pairs_batch, mp2.no)
 
            t0 = get_current_time()
            if mp2.int_storage != 3:
                sf_offsets_gpu, smat_gpu = load_gpu_mat(extended_pairs_batch, mp2.sf_offsets_node, 
                                                        mp2.smat_node, sf_offsets_gpu)
                _, fmat_gpu = load_gpu_mat(extended_pairs_batch, mp2.sf_offsets_node, mp2.fmat_node, 
                                           sf_offsets_gpu)
                #kmat_offsets_gpu, kmat_gpu = load_gpu_mat(pairs_batch, mp2.kmat_offsets_node, mp2.kmat_node, kmat_offsets_gpu)
                kmat_offsets_gpu, kmat_gpu = load_gpu_mat(pairs_batch, mp2.kmat_offsets_node, 
                                                          mp2.remote_kmat_node, kmat_offsets_gpu)
                xii_offsets_gpu, xii_gpu = load_gpu_mat(mos_batch, mp2.xii_offsets_node, mp2.xii_node, 
                                                        xii_offsets_gpu)
                emui_offsets_gpu, emui_gpu = load_gpu_mat(mos_batch, mp2.emui_offsets_node, mp2.emui_node, 
                                                          emui_offsets_gpu)

            '''tmat_offsets_gpu, tmat_gpu = load_gpu_mat(self.extended_pairs, mp2.tmat_offsets_node, mp2.tmat_node, 
                                                      tmat_offsets_gpu)'''
            diag_tmat_size = np.sum(np.prod(mp2.tmat_dim[diag_pairs], axis=1))
            
            remote_tmat_sizes = np.prod(mp2.tmat_dim[pairs_batch], axis=1)
            remote_tmat_offsets = cum_offset(remote_tmat_sizes) + diag_tmat_size
            tmat_offsets_gpu[pairs_batch] = cupy.asarray(remote_tmat_offsets[:-1])
            
            tmat_gpu = cupy.zeros(diag_tmat_size + remote_tmat_offsets[-1])

            _, _ = load_gpu_mat(diag_pairs, mp2.tmat_offsets_node, mp2.close_tmat_node, 
                                tmat_offsets_gpu, mat_gpu=tmat_gpu)

            rmat_sizes = mp2.nosv[ilist_batch] * mp2.nosv[jlist_batch]
            rmat_offsets = cum_offset(rmat_sizes)
            rmat_offsets_gpu = cupy.asarray(rmat_offsets)
            rmat_gpu = cupy.empty(rmat_offsets[-1], dtype=cupy.float64)
            pair_ene_gpu = cupy.empty(npairs_batch)
            #pairs_batch_gpu = cupy.asarray(pairs_batch)
            pairs_batch_gpu = numpy_to_cupy(pairs_batch, dtype=cupy.int32)
            accumulate_time(mp2.t_data, t0)

            t0 = get_current_time()
            # pairs, nOsvOcc, nOcc
            osvMp2Cuda.remoteResidualCupy(smat_gpu, fmat_gpu, tmat_gpu, kmat_gpu,
                                            rmat_gpu, xii_gpu, emui_gpu, eo_gpu, temp_a_gpu,
                                            loc_fock_gpu, pair_ene_gpu,
                                            pairs_batch_gpu, nosv_gpu, rmat_offsets_gpu,
                                            sf_offsets_gpu, tmat_offsets_gpu, kmat_offsets_gpu,
                                            xii_offsets_gpu, emui_offsets_gpu, temp_offsets_gpu,
                                            #ext_pair_indices_gpu, mp2.no)
                                            mp2.no)
            accumulate_time(mp2.t_cal, t0)
            
            t0 = get_current_time()
            tmat_cpu = cupy.asnumpy(tmat_gpu)
            pair_ene_cpu = cupy.asnumpy(pair_ene_gpu)
            accumulate_time(mp2.t_data, t0)


            if self.check_ene:
                self.pair_ene[pairs_batch] = pair_ene_cpu
            ene_new += np.sum(pair_ene_cpu)

            '''local_to_node_new(pairs_batch, mp2.tmat_node, tmat_cpu, mp2.tmat_offsets_node, 
                          tmat_offsets_gpu.get())'''
            if mp2.cal_grad:
                local_to_node_new(pairs_batch, mp2.remote_tmat_node, tmat_cpu, mp2.tmat_offsets_node, 
                            tmat_offsets_gpu.get())
        return ene_new
            
    return get_ene

def save_mat_mbe(self, nmo, pairlist, T_matrix, pair_ene, tmat_batch=None, 
                 pair_ene_rank=None, tmat_sizesum_batch=None):
    
    if self.method == 3:
        if nmo == 1:
            ipair = pairlist[0]
        elif nmo == 2:
            ipair = pairlist[1]
        tidx0, tidx1 = self.tmat_offsets_node[ipair]
        #self.tmat_node[tidx0: tidx1] = T_matrix[ipair].ravel()
        self.close_tmat_node[tidx0: tidx1] = T_matrix[ipair].ravel()
    else:
        # For cumulative amplitudes
        if nmo == 1:
            ipair = pairlist[0]
            tidx0, tidx1 = self.tmat_offsets_node[ipair]
            self.tmat_save[tidx0: tidx1] = T_matrix[ipair].ravel()
            #self.tmat_node[tidx0: tidx1] = self.oneb_counts[ipair] * T_matrix[ipair].ravel()
            self.close_tmat_node[tidx0: tidx1] = self.oneb_counts[ipair] * T_matrix[ipair].ravel()
            self.pairene_node[ipair] = self.oneb_counts[ipair] * pair_ene[ipair]
        elif nmo == 2:
            ij = pairlist[1]
            tidx0, tidx1 = self.tmat_offsets_node[ij]
            count = self.twob_counts_offdiag[ij]
            self.tmat_save[tidx0: tidx1] = T_matrix[ij].ravel()
            #self.tmat_node[tidx0: tidx1] = count * T_matrix[ij].ravel()
            self.close_tmat_node[tidx0: tidx1] = count * T_matrix[ij].ravel()
            self.pairene_node[ij] = count * pair_ene[ij]

            count_ij = self.twob_counts_diag[ij]
            for ipair in pairlist[[0, 2]]:
                tidx1 = tmat_sizesum_batch[ipair]
                tidx0 = tidx1 - T_matrix[ipair].size
                tmat_batch[tidx0: tidx1] += count_ij * T_matrix[ipair].ravel()
                pair_ene_rank[ipair] += count_ij * pair_ene[ipair]
        else:
            for ipair in pairlist:
                tidx1 = tmat_sizesum_batch[ipair]
                tidx0 = tidx1 - T_matrix[ipair].size
                tmat_batch[tidx0: tidx1] += T_matrix[ipair].ravel()
                pair_ene_rank[ipair] += pair_ene[ipair]

def mbe_residual_iter_cuda(self, nmo, cidx_slice, pair_ene_rank, ene_tol, kocc_tol=1e-5, 
                           max_cycles=30, use_dynt=True, log=None):

    #cupy.get_default_memory_pool().free_all_blocks()
    #print(irank, "Memory NOW!!!! %.2f MB"%(cupy.cuda.Device(igpu_shm).mem_info[0]*1e-6))
    if log is None:
        log = lib.logger.Logger(self.stdout, self.verbose)

    npair_cluster = (nmo * (nmo + 1) // 2)
    max_nosv = np.max(self.nosv)
    total_ncluster = len(cidx_slice)
    max_tmat_size = (npair_cluster + 4) * 4 * max_nosv**2

    max_ncluster_gpu = get_ncols_for_memory(0.4*avail_gpu_mem(self.gpu_memory)/nrank_per_gpu, 
                                            max_tmat_size, total_ncluster)

    #temp_offsets_gpu = uniform_cum_offset_cupy(4 * max_nosv**2, max_ncluster_gpu)

    t0 = get_current_time()
    #temp_a_gpu = cupy.empty(int(temp_offsets_gpu[-1]))
    #temp_b_gpu = cupy.empty_like(temp_a_gpu)
    #pair_idx_gpu = cupy.empty(self.no**2, dtype=cupy.int64)
    pair_idx_gpu = cupy.empty(self.no**2, dtype=cupy.int32)
    loc_fock_gpu = cupy.asarray(self.loc_fock).ravel()
    #nosv_gpu = cupy.asarray(self.nosv)
    nosv_gpu = numpy_to_cupy(self.nosv, dtype=cupy.int32)
    self.recorder_data.append(record_elapsed_time(t0))

    acc_tnode = True #(nmo < 3 or self.method != 2 or self.cal_grad)

    def get_ene(clusters_batch, pairs_clus_batch, tmat_batch, tmat_sizesum_batch):
        nclusters_batch = len(clusters_batch)
        

        for clus_idx0 in np.arange(nclusters_batch, step=max_ncluster_gpu):
            clus_idx1 = min(clus_idx0 + max_ncluster_gpu, nclusters_batch)
            nclusters_sub = clus_idx1 - clus_idx0
            clusters_sub = clusters_batch[clus_idx0: clus_idx1]
            pairs_clus_sub = pairs_clus_batch[clus_idx0: clus_idx1].ravel()
            npairs_clus_sub = len(pairs_clus_sub)
            klist_pairs_sub = np.repeat(clusters_sub, npair_cluster, axis=0).ravel()
            

            pairs_sub = np.unique(pairs_clus_sub.ravel()) 
            #pair_idx_gpu[pairs_sub] = cupy.arange(len(pairs_sub))
            pair_idx_gpu[pairs_sub] = cupy.arange(len(pairs_sub), dtype=cupy.int32)

            ilist_clus, jlist_clus = np.divmod(pairs_clus_sub, self.no)
            nosv_i_clus = self.nosv[ilist_clus]
            nosv_j_clus = self.nosv[jlist_clus]
            
        
            t0 = get_current_time()
            sf_offsets_gpu, smat_gpu = load_gpu_mat(pairs_sub, self.sf_offsets_node, self.smat_node)
            _, fmat_gpu = load_gpu_mat(pairs_sub, self.sf_offsets_node, self.fmat_node)
            #kmat_offsets_gpu, kmat_gpu = load_gpu_mat(pairs_sub, self.kmat_offsets_node, self.kmat_node)
            kmat_offsets_gpu, kmat_gpu = load_gpu_mat(pairs_sub, self.kmat_offsets_node, self.close_kmat_node)
            xmat_offsets_full_gpu, xmat_gpu = load_gpu_mat(pairs_sub, self.xmat_offsets_node, self.xmat_node)
            emuij_offsets_gpu, emuij_gpu = load_gpu_mat(pairs_sub, self.emuij_offsets_node, self.emuij_node)
            
            tmat_sizes = (nosv_i_clus + nosv_j_clus)**2
            tmat_offsets = cum_offset(tmat_sizes)

            #large_size = (tmat_offsets[-1] + int(rmat_offsets_gpu[-1]) * 3) * 8 * 1e-6
            #print(irank, "need %.2f MB, left %.2f MB"%(large_size, cupy.cuda.Device(igpu_shm).mem_info[0]*1e-6))

            if nmo < 3:
                tinit_gpu = cupy.zeros(tmat_offsets[-1])
            else:
                _, tinit_gpu = load_gpu_mat(pairs_sub, self.tmat_offsets_node, self.tmat_save)
            
            tmat_offsets_gpu = cupy.asarray(tmat_offsets)
            tmat_gpu = cupy.empty(tmat_offsets[-1])

            temp_sizes = np.max((tmat_sizes).reshape(nclusters_sub, -1), axis=1)
            temp_offsets_gpu = cupy.asarray(cum_offset(temp_sizes))

            temp_a_gpu = cupy.empty(int(temp_offsets_gpu[-1]))
            temp_b_gpu = cupy.empty_like(temp_a_gpu)

            if use_dynt:
                rmat_offsets_gpu = temp_offsets_gpu
                rmat_gpu = cupy.empty_like(temp_a_gpu)
            else:
                rmat_offsets_gpu = tmat_offsets_gpu
                rmat_gpu = cupy.empty_like(tmat_gpu)

            pair_ene_gpu = cupy.empty(npairs_clus_sub)
            pairs_clus_sub_gpu = numpy_to_cupy(pairs_clus_sub, dtype=cupy.int32)
            klist_pairs_sub_gpu = numpy_to_cupy(klist_pairs_sub, dtype=cupy.int32)

            klist_offsets_gpu = uniform_cum_offset_cupy(nmo, npairs_clus_sub)
            pair_offsets_gpu = uniform_cum_offset_cupy(npair_cluster, nclusters_sub, dtype=cupy.int32)
            self.recorder_data.append(record_elapsed_time(t0))

            res_cycles_gpu = cupy.empty(nclusters_sub, dtype=cupy.int32)

            
            t0 = get_current_time()
            # pairs_clus_sub_gpu, nosv_gpu, klist_pairs_sub_gpu, pair_idx_gpu, pair_offsets_gpu, nOcc
            osvMp2Cuda.clusterResidualCupy(smat_gpu, fmat_gpu, tmat_gpu, tinit_gpu, kmat_gpu,
                                            rmat_gpu, xmat_gpu, emuij_gpu, temp_a_gpu,
                                            temp_b_gpu, loc_fock_gpu, pair_ene_gpu, res_cycles_gpu,
                                            pairs_clus_sub_gpu, nosv_gpu, klist_pairs_sub_gpu,
                                            sf_offsets_gpu, kmat_offsets_gpu, tmat_offsets_gpu,
                                            rmat_offsets_gpu, xmat_offsets_full_gpu, 
                                            emuij_offsets_gpu, temp_offsets_gpu, klist_offsets_gpu,
                                            pair_idx_gpu, pair_offsets_gpu,
                                            self.no, kocc_tol, ene_tol, max_cycles, use_dynt)
            self.recorder_cal.append(record_elapsed_time(t0))

            t0 = get_current_time()
            pair_ene_cpu = cupy.asnumpy(pair_ene_gpu)

            if acc_tnode:
                tmat_cpu = cupy.asnumpy(tmat_gpu)
            else:
                tmat_cpu = None
            #tmat_offsets = cupy.asnumpy(tmat_offsets_gpu)
            self.recorder_data.append(record_elapsed_time(t0))

            if nmo > 1:
                log.info("    %d %db clusters cycles: Ave: %d, max: %d"%(nclusters_sub, nmo, np.mean(res_cycles_gpu), np.max(res_cycles_gpu)))

            save_mbe_sub(self, nmo, pairs_clus_sub, tmat_cpu, pair_ene_cpu, tmat_offsets, 
                        tmat_batch, pair_ene_rank, tmat_sizesum_batch)


            smat_gpu, fmat_gpu = None, None
            tmat_cpu, tmat_gpu = None, None
            xmat_gpu, emuij_gpu = None, None
            tinit_gpu, kmat_gpu = None, None
            rmat_gpu, pair_ene_gpu = None, None
            temp_a_gpu, temp_b_gpu = None, None
    return get_ene