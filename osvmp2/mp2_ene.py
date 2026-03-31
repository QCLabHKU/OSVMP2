import sys
import h5py
import os 
import numpy as np
from numpy.linalg import multi_dot
import scipy.linalg
import shutil
from pyscf import lib
from pyscf.df import DF
from osvmp2 import int_prescreen
from osvmp2.lib import randSvd
from osvmp2.loc.loc_addons import *#get_ncore, localization, LMO_domains, AO_domains
from osvmp2.osv_mbe import residualMBE, select_clusters
#from osvmp2.ga_addons import *
from osvmp2.int_3c2e import get_df_int3c2e, get_df_ialp
from osvmp2.__config__ import ngpu
from osvmp2.mpi_addons import *
from osvmp2.osvutil import *
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
    import cupy
    import cupyx
    from osvmp2.gpu.cuda_utils import free_cupy_mem
    from osvmp2.gpu.int_3c2e_cuda import VHFOpt
    from osvmp2.gpu.sparse_int3c2e_cuda import Int3c2eOpt
    from osvmp2.gpu.mp2_ene_cuda import (get_osv_kernel_cuda, get_direct_osv_cuda, get_sf_cuda, 
                                         get_precon_remote_cuda, get_precon_close_cuda, get_imujp_cuda, 
                                         imujp_to_kmat_cuda,
                                         remote_residual_iter_cuda, close_residual_iter_cuda, load_gpu_mat)

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

def get_osv_kernel(self):
    def compute_tii(mo_vir, ialp_i, eo_i, ene_vir):
        iap = np.dot(mo_vir.T, ialp_i)
        iiab = np.dot(iap, iap.T)
        fiiab = -2 * eo_i + (ene_vir[:, None] + ene_vir[None, :])
        return iiab / (fiiab + 1e-10)
    def get_osv(iidx, i, ialp_i, eo_i):
        max_rsvd_iter = min(1000, self.nv)

        if self.loc_fit:
            ialp_i = ialp_i[:, self.fit_list[i]]

        t1 = get_current_time()
        #tii = compute_tii(self.v, ialp_i, eo_i, self.ev)
        iap = np.dot(self.v.T, ialp_i)
        iiab = np.dot(iap, iap.T)
        fiiab = -2 * eo_i + np.add.outer(self.ev, self.ev)#(ene_vir[:, None] + ene_vir[None, :])
        iiab /= fiiab
        tii = iiab
        #accumulate_time(self.t_tii, t1)
        self.recorder_tii.append(record_elapsed_time(t1))

        #qcp, s, v = sli.svd(tii, self.nosv_id)
        #qcp, s, v = sli.svd(tii, self.idsvd_tol)
        t1 = get_current_time()
        if self.svd_method == 0:
            qcp, s, _ = np.linalg.svd(tii)
        elif self.svd_method == 1:
            qcp, s, _ = randSvd.randomizedSvdNumpy(tii, self.cposv_tol, max_rsvd_iter, getVt=False)
        else:
            qcp, s, _ = scipy.linalg.interpolative.svd(tii, self.cposv_tol)
        
        self.nosv_cp[i] = len(s)
        #accumulate_time(self.t_svd, t1)
        self.recorder_svd.append(record_elapsed_time(t1))

        
        self.nosv[i] = np.count_nonzero(s >= self.osv_tol)
        qmat = qcp[:, :self.nosv[i]]
        qao = np.dot(self.v, qmat)

        return qmat, qao, qcp, s, tii
    return get_osv

def generate_osv(self, log):

    if self.int_storage in {1, 2} and not self.fully_direct:
        use_ialp_file = True
    else:
        use_ialp_file = False

    use_double_buffer = (self.double_buffer and self.use_gpu and use_ialp_file)

    nmo = len(self.mo_list)
    nocc_core = self.no - nmo

    if use_double_buffer:
        mo_slices = get_slice(job_list=self.mo_list, rank_list=range(nrank//2))
        mo_slice = mo_slices[ipair_rank]

        if self.use_gpu:
            irank_cal = irank_pair - 1 if irank_pair % 2 else irank_pair
            irank_io = irank_pair if irank_pair % 2 else irank_pair + 1
            is_cal_rank = irank_pair == irank_cal
            is_io_rank = irank_pair == irank_io
            istream_data = cupy.cuda.Stream(non_blocking=True)
            istream_comp = cupy.cuda.Stream(non_blocking=True)

            self.comp_events = [cupy.cuda.Event(), cupy.cuda.Event()]
            self.data_events = [cupy.cuda.Event(), cupy.cuda.Event()]
        
    else:
        mo_slices = get_slice(job_list=self.mo_list, rank_list=range(nrank))
        mo_slice = mo_slices[irank]
        is_cal_rank = is_io_rank = True
        if self.use_gpu:
            istream_data = istream_comp = cupy.cuda.Stream.null
    
    
    self.win_nosv, self.nosv = get_shared(self.no, dtype=np.int64)#, set_zeros=True)
    self.win_nosv_cp, self.nosv_cp = get_shared(self.no, dtype=np.int64)#, set_zeros=True)
    if irank_shm == 0:
        self.nosv_cp[:nocc_core] = 0
        self.nosv[:nocc_core] = 0

    if self.cal_grad:
        self.dir_tii = 'ti_tmp'
        self.dir_qcp = 'qcp_tmp'
        for idir in [self.dir_tii, self.dir_qcp]:
            os.makedirs(idir, exist_ok=True)
    
    comm.Barrier()
    self.t_tii = create_timer()
    self.t_svd = create_timer()
    self.t_io = create_timer()
    self.t_data = create_timer()

    self.recorder_tii = []
    self.recorder_svd = []
    self.recorder_io = []
    self.recorder_data = []

    if mo_slice is not None:
        qmat_save = {}
        qao_save = {}
        
        if self.fully_direct:
            if self.use_gpu:
                get_direct_osv_cuda(self, mo_slices, qmat_save, qao_save)
            else:
                raise NotImplementedError
        else:
            if use_ialp_file:
                file_ialp = h5py.File(self.file_ialp, 'r')
                ialp_data = file_ialp["ialp"]

                if self.use_gpu:
                    if use_double_buffer:
                        win_ialp_buf, double_ialp_buf_cpu = get_shared((2, self.nao, self.naoaux), rank_shm_buf=irank_cal, 
                                                                    comm_shared=comm_pair, use_pin=True)
                    else:
                        ialp_tmp = cupyx.empty_pinned((self.nao, self.naoaux), dtype=cupy.float64)
                else:
                    ialp_tmp = np.empty((self.nao, self.naoaux))

            if self.use_gpu:
                t1 = get_current_time()
                self.vir_mo_gpu, self.vir_mo_ptr = get_shared_cupy((self.nao, self.nv), numpy_array=self.v)
                self.recorder_data.append(record_elapsed_time(t1))

                if is_cal_rank:
                    if use_double_buffer:
                        max_nfit = np.max([len(self.fit_list[i]) for i in mo_slice])

                        if self.unsort_ialp_ao:
                            ialp_slice_size = self.nao*max_nfit
                        else:
                            ialp_slice_size = self.nao*self.naoaux
                        ialp_buf_gpu = cupy.empty(self.nao*self.naoaux+2*ialp_slice_size)
                    else:
                        ialp_buf_gpu = None
                else:
                    ialp_buf_gpu = None

                gen_osv_kern = get_osv_kernel_cuda(self, ialp_buf_gpu, use_double_buffer,
                                                    streams=[istream_data, istream_comp], 
                                                    mo_slice=mo_slice)
            else:
                gen_osv_kern = get_osv_kernel(self)
            
            if use_double_buffer:
                if not self.loc_fit: raise NotImplementedError
                if use_ialp_file:
                    if is_cal_rank:
                        ialp_read_gpu = ialp_buf_gpu[:self.nao*self.naoaux].reshape(self.nao, self.naoaux)
                        double_ialp_buf_gpu = ialp_buf_gpu[self.nao*self.naoaux:].reshape(2, -1)

                    for iidx_rank in range(2):
                        io_idx = 0 if iidx_rank % 2 else 1
                        cal_idx = 1 if iidx_rank % 2 else 0
                        if is_io_rank:
                            ialp_io_cpu = double_ialp_buf_cpu[io_idx]
                            t1 = get_current_time()
                            ialp_data.read_direct(ialp_io_cpu, np.s_[mo_slice[iidx_rank]-nocc_core])
                            #accumulate_time(self.t_io, t1)
                            self.recorder_io.append(record_elapsed_time(t1))
                        
                        if is_cal_rank and iidx_rank > 0:
                            data_idx = 1 if iidx_rank % 2 else 0
                            with istream_data:
                                ialp_data_cpu = double_ialp_buf_cpu[cal_idx]
                                if self.unsort_ialp_ao:
                                    ialp_read_gpu.set(ialp_data_cpu)
                                else:
                                    ialp_load_gpu = double_ialp_buf_gpu[data_idx].reshape(self.nao, self.naoaux)
                                    ialp_load_gpu.set(ialp_data_cpu)
                                    cupy.take(ialp_load_gpu, self.intopt.unsorted_ao_ids, axis=0, out=ialp_read_gpu)
                                
                                pre_i = mo_slice[iidx_rank-1]
                                pre_nfit = len(self.fit_list[pre_i])
                                ialp_data_gpu = double_ialp_buf_gpu[data_idx][:self.nao*pre_nfit].reshape(self.nao, pre_nfit)
                                cupy.take(ialp_read_gpu, cupy.asarray(self.fit_list[pre_i]), axis=1, out=ialp_data_gpu)

                                self.data_events[data_idx].record()

                                istream_data.synchronize()

                        comm_pair.Barrier()
                else:
                    raise NotImplementedError


            for iidx_rank, i in enumerate(mo_slice):
                iidx = i-nocc_core
                
                
                '''if self.loc_fit:
                    if use_ialp_file:
                        ialp_tmp = ialp_data[iidx][:, self.fit_list[i]]
                    elif self.int_storage == 0:
                        ialp_tmp = self.ialp_node[iidx][:, self.fit_list[i]]
                    elif self.int_storage == 3:  
                        ialp_tmp = self.ialp_gpu[iidx_rank][:, cupy.asarray(self.fit_list[i])]
                else:
                    if use_ialp_file:
                        ialp_data.read_direct(ialp_tmp, np.s_[iidx])
                    elif self.int_storage == 0:
                        ialp_tmp = self.ialp_node[iidx]
                    elif self.int_storage == 3:  
                        ialp_tmp = self.ialp_gpu[iidx_rank]'''

                if use_ialp_file:
                    t1 = get_current_time()
                    if use_double_buffer:
                        io_idx = 0 if iidx_rank % 2 else 1
                        cal_idx = 1 if iidx_rank % 2 else 0
                        ialp_tmp = double_ialp_buf_cpu[cal_idx]
                        nnext_iidx_rank = iidx_rank + 2
                        if is_io_rank and nnext_iidx_rank < len(mo_slice):
                            nnext_iidx = mo_slice[nnext_iidx_rank] - nocc_core
                            ialp_data.read_direct(double_ialp_buf_cpu[io_idx], np.s_[nnext_iidx])
                    else:
                        ialp_data.read_direct(ialp_tmp, np.s_[iidx])
                    self.recorder_io.append(record_elapsed_time(t1))

                elif self.int_storage == 0:
                    ialp_tmp = self.ialp_node[iidx]
                elif self.int_storage == 3:  
                    ialp_tmp = self.ialp_gpu[iidx_rank]
                    

                if is_cal_rank:
                    
                    #ijp = self.o.T.dot(ialp_tmp)
                    #ijik = np.dot(ijp, ijp.T)
                    '''ijik = np.dot(self.o.T, self.o)
                    u, s, vt = np.linalg.svd(ijik)
                    print(s)'''
                    qmat, qao, qcp, s, tii = gen_osv_kern(iidx_rank, i, ialp_tmp, self.eo[i])
                    qmat_save[i] = qmat
                    qao_save[i] = qao

                if use_double_buffer:
                    comm_pair.Barrier()
                
                
                if self.cal_grad:
                    t1 = get_current_time()
                    self.nosv_cp[i] = np.count_nonzero(s >= self.cposv_tol)
                    qcp = qcp[:, :self.nosv_cp[i]]
                    s = qcp[:self.nosv_cp[i]]

                    with h5py.File('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'w') as file_qcp:
                        file_qcp.create_dataset('qcp', data=qcp)
                        file_qcp.create_dataset('s', data=s)
                    with h5py.File('%s/ti_%d.tmp'%(self.dir_tii, i), 'w') as file_tii:
                        file_tii.create_dataset('tii', data=tii)
                    self.recorder_io.append(record_elapsed_time(t1))
            
            if use_ialp_file:
                file_ialp.close()

            if use_double_buffer:
                istream_comp.synchronize()
                if is_cal_rank:
                    unregister_pinned_memory(double_ialp_buf_cpu)
                win_ialp_buf.Free()

        for i in mo_slice:
            self.nosv[i] = qmat_save[i].shape[1]
    comm.Barrier()

    
    t1 = get_current_time()
    if nnode > 1:
        '''mo_node_offsets = [None for node_i in range(nnode)]
        for node_i in range(nnode):
            rank_start = node_i * nrank_shm
            rank_last = rank_start + nrank_shm - 1
            if mo_slices[rank_start] is None:
                continue
            for rank_1 in np.arange(rank_last, rank_start-1, step=-1):
                if mo_slices[rank_1] is not None:
                    mo_node_offsets[node_i] = [mo_slices[rank_start][0], 
                                               mo_slices[rank_1][-1]+1]
                    break'''
        
        
        mo_offsets = [[i, i+1] for i in range(self.no)]
        mo_node_offsets = get_node_offsets(mo_slices, mo_offsets)
        
        Get_from_other_nodes_GA(self.nosv_cp, mo_node_offsets)
        Get_from_other_nodes_GA(self.nosv, mo_node_offsets)
        comm.Barrier()

    size_qmat = np.sum([self.nv*self.nosv[i] for i in self.mo_list])
    if self.fully_direct:
        nao = self.intopt.sorted_mol.nao
    else:
        nao = self.nao
    size_qao = np.sum([nao*self.nosv[i] for i in self.mo_list])

    self.win_qmat_node, self.qmat_node = get_shared(size_qmat) #, set_zeros=True)
    self.win_qao_node, self.qao_node = get_shared(size_qao) #, set_zeros=True)
    
    self.qmat_offsets_node = {}
    self.qao_offsets_node = {}
    
    qmat_idx0 = 0
    qao_idx0 = 0
    for i in self.mo_list:
        qmat_idx1 = qmat_idx0 + self.nv*self.nosv[i]
        self.qmat_offsets_node[i] = [qmat_idx0, qmat_idx1]
        qmat_idx0 = qmat_idx1

        qao_idx1 = qao_idx0 + nao*self.nosv[i]
        self.qao_offsets_node[i] = [qao_idx0, qao_idx1]
        qao_idx0 = qao_idx1

    
    if mo_slice is not None and is_cal_rank:
        for i in mo_slice:
            qmat_idx0, qmat_idx1 = self.qmat_offsets_node[i]
            self.qmat_node[qmat_idx0:qmat_idx1] = qmat_save[i].ravel()

            qao_idx0, qao_idx1 = self.qao_offsets_node[i]
            self.qao_node[qao_idx0:qao_idx1] = qao_save[i].ravel()

    self.qcp_dim = {}
    self.qmat_dim = {}
    self.qao_dim = {}

    for i in self.mo_list:
        self.qcp_dim[i] = (self.nv, self.nosv_cp[i])
        self.qmat_dim[i] = (self.nv, self.nosv[i])
        self.qao_dim[i] = (self.nao, self.nosv[i])

    comm.Barrier()
    if nnode > 1:

        qmat_node_offsets = get_node_offsets(mo_slices, self.qmat_offsets_node)
        qao_node_offsets = get_node_offsets(mo_slices, self.qao_offsets_node)

        Get_from_other_nodes_GA(self.qmat_node, qmat_node_offsets)
        Get_from_other_nodes_GA(self.qao_node, qao_node_offsets)

        #Acc_and_get_GA(self.qmat_node)
        #Acc_and_get_GA(self.qao_node)
        comm.Barrier()
    self.recorder_io.append(record_elapsed_time(t1))
    

    batch_accumulate_time(self.t_tii, self.recorder_tii)
    batch_accumulate_time(self.t_svd, self.recorder_svd)
    batch_accumulate_time(self.t_io, self.recorder_io)
    batch_accumulate_time(self.t_data, self.recorder_data)

    time_list = [['T_ii', self.t_tii], ['SVD', self.t_svd],
                 ['IO', self.t_io]]
    
    if self.use_gpu:
        if not self.fully_direct:
            close_ipc_handle(self.vir_mo_ptr)
            self.vir_mo_gpu = None
        #close_ipc_handle(self.ev_ptr)
        time_list.append(['GPU-CPU data', self.t_data])
    time_list = get_max_rank_time_list(time_list)
    print_time(time_list, log)
    

    

def get_sf_GA(self):
    ilist = self.pairlist // self.no
    jlist = self.pairlist % self.no
    
    self.sf_dim = [None]*self.no**2
    for ipair, i, j in zip(self.pairlist, ilist, jlist):
        self.sf_dim[ipair] = (self.nosv[i], self.nosv[j])

    size_sf = self.nosv[ilist] * self.nosv[jlist]
    pair_slices = get_slice(range(nrank), job_list=self.pairlist, 
                            weight_list=size_sf)
    pair_slice = pair_slices[irank]
    #print(irank, len(pair_slice));sys.exit()
    self.t_data = create_timer()
    self.t_cal = create_timer()

    self.recorder_data = []
    self.recorder_cal = []

    self.win_sratio, self.s_ratio = get_shared(self.no**2, dtype=np.float64, set_zeros=True)
    win_size_sf_kept, size_sf_kept = get_shared(nrank_shm, dtype=np.int64, set_zeros=True)

    
    if pair_slice is not None:
        if self.use_gpu:
            size_sf_rank, pairs_kept_rank, smat_save, fmat_save = get_sf_cuda(self, pair_slice)
        else:
            pairs_kept_rank = []
            size_sf_rank = 0
            smat_save = {}
            fmat_save = {}
            for ipair in pair_slice:
                i = ipair // self.no
                j = ipair % self.no
                pair_ji = j * self.no + i

                qi_idx0, qi_idx1 = self.qmat_offsets_node[i]
                qj_idx0, qj_idx1 = self.qmat_offsets_node[j]
                qmat_i = self.qmat_node[qi_idx0:qi_idx1].reshape(self.qmat_dim[i])
                qmat_j = self.qmat_node[qj_idx0:qj_idx1].reshape(self.qmat_dim[j])
                ene_vir = self.ev

                t0 = get_current_time()
                smat_ij = np.dot(qmat_i.T, qmat_j)
                self.s_ratio[pair_ji] = self.s_ratio[ipair] = sum(smat_ij.ravel()**2)/((self.nosv[i]+self.nosv[j])*0.5)
                if self.s_ratio[ipair] > self.disc_tol:
                    size_sf_rank += self.nosv[i] * self.nosv[j]
                    pairs_kept_rank.append(ipair)
                    smat_save[ipair] = smat_ij
                    fmat_save[ipair] = np.dot(np.multiply(qmat_i.T, ene_vir), qmat_j)
                #accumulate_time(self.t_cal, t2)
                self.recorder_cal.append(record_elapsed_time(t0))

        size_sf_kept[irank_shm] = size_sf_rank
    comm.Barrier()

    t0 = get_current_time()
    if nnode > 1:
        Acc_and_get_GA(size_sf_kept)
        Acc_and_get_GA(self.s_ratio)
        comm.Barrier()


    total_sf_size = np.sum(size_sf_kept)
    
    self.win_smat_node, self.smat_node = get_shared(total_sf_size)#, set_zeros=True)
    self.win_fmat_node, self.fmat_node = get_shared(total_sf_size)#, set_zeros=True)

    index_kept = self.s_ratio[self.pairlist] > self.disc_tol
    pairs_kept = self.pairlist[index_kept]
    
    self.sf_offsets_node = {}
    sf_idx0 = 0
    for ipair in pairs_kept:
        d0, d1 = self.sf_dim[ipair]
        sf_idx1 = sf_idx0 + d0 * d1
        self.sf_offsets_node[ipair] = [sf_idx0, sf_idx1]
        sf_idx0 = sf_idx1

    if pair_slice is not None:
        for ipair in pairs_kept_rank:
            sf_idx0, sf_idx1 = self.sf_offsets_node[ipair]
            self.smat_node[sf_idx0:sf_idx1] = smat_save[ipair].ravel()
            self.fmat_node[sf_idx0:sf_idx1] = fmat_save[ipair].ravel()
    comm.Barrier()
    if nnode > 1:
        '''Acc_and_get_GA(self.smat_node)
        Acc_and_get_GA(self.fmat_node)'''
        kept_pair_slices = [None] * nrank
        for rank_i, pair_slice in enumerate(pair_slices):
            if pair_slice is not None:
                mask_kept = self.s_ratio[pair_slice] > self.disc_tol
                kept_pair_slices[rank_i] = pair_slice[mask_kept]

        sf_node_offsets = get_node_offsets(kept_pair_slices, self.sf_offsets_node)

        Get_from_other_nodes_GA(self.smat_node, sf_node_offsets)
        Get_from_other_nodes_GA(self.fmat_node, sf_node_offsets)
        comm.Barrier()
    free_win(win_size_sf_kept)

    self.t_col = get_elapsed_time(t0)

    batch_accumulate_time(self.t_cal, self.recorder_cal)
    batch_accumulate_time(self.t_data, self.recorder_data)

    log = lib.logger.Logger(self.stdout, self.verbose)
    time_list = [['calculation', self.t_cal]]
    if self.use_gpu:
        time_list.append(["GPU-CPU data", self.t_data])
    time_list.append(["accumulation", self.t_col])
    time_list = get_max_rank_time_list(time_list)
    print_time(time_list, log)

    


def get_precon_remote(self):
    self.xii_dim = [None]*self.no
    self.emui_dim = [None]*self.no
    self.xii_offsets_node = [None]*self.no
    self.emui_offsets_node = [None]*self.no

    xidx0, eidx0 = 0, 0
    for i in self.mo_list:
        self.xii_dim[i] = (self.nosv[i], self.nosv[i])
        xidx1 = xidx0 + self.nosv[i]**2
        self.xii_offsets_node[i] = [xidx0, xidx1]

        self.emui_dim[i] = self.nosv[i]
        eidx1 = eidx0 + self.nosv[i]
        self.emui_offsets_node[i] = [eidx0, eidx1]

        xidx0, eidx0 = xidx1, eidx1
    
    self.win_xii_node, self.xii_node = get_shared(xidx1)#, set_zeros=True)
    self.win_emui_node, self.emui_node = get_shared(eidx1)#, set_zeros=True)

    nocc_core = self.no - len(self.mo_list)
    mo_slices = get_slice(range(nrank), job_list=self.mo_list, 
                          weight_list=self.nosv[nocc_core:])
    mo_slice = mo_slices[irank]

    if mo_slice is not None:
        if self.use_gpu:
            get_precon_remote_cuda(self, mo_slice)
        else:
            for i in mo_slice:
                ii = i * self.no + i            
                f_ii = getMatFromNode(ii, self.fmat_node, self.sf_offsets_node, self.sf_dim)
                emui, xii = np.linalg.eigh(f_ii)

                xidx0, xidx1 = self.xii_offsets_node[i]
                self.xii_node[xidx0:xidx1] = xii.ravel()

                eidx0, eidx1 = self.emui_offsets_node[i]
                self.emui_node[eidx0:eidx1] = emui.ravel()
    
    comm.Barrier()
    if nnode > 1:
        #Acc_and_get_GA(self.xii_node)
        #Acc_and_get_GA(self.emui_node)
        xii_node_offsets = get_node_offsets(mo_slices, self.xii_offsets_node)
        emui_node_offsets = get_node_offsets(mo_slices, self.emui_offsets_node)

        Get_from_other_nodes_GA(self.xii_node, xii_node_offsets)
        Get_from_other_nodes_GA(self.emui_node, emui_node_offsets)
        comm.Barrier()

def get_precon_close(self):
    ilist = self.pairlist_close // self.no
    jlist = self.pairlist_close % self.no
    weight_list = (self.nosv[ilist] + self.nosv[jlist])**3
    pair_slices = get_slice(range(nrank), job_list=self.pairlist_close, 
                            weight_list=weight_list)
    pair_slice = pair_slices[irank]

    ilist_close, jlist_close = np.divmod(self.pairlist_close, self.no)
    nosv_ij = self.nosv[ilist_close] + self.nosv[jlist_close]
    full_xmat_size = nosv_ij ** 2
    full_xmat_size_sum = np.empty(self.no**2, dtype=np.int64)
    full_xmat_size_sum_close = np.cumsum(full_xmat_size)
    full_xmat_size_sum[self.pairlist_close] = full_xmat_size_sum_close
    total_full_xmat_size = full_xmat_size_sum[self.pairlist_close[-1]]

    self.win_xmat_node, self.xmat_node = get_shared(total_full_xmat_size)#, set_zeros=True)
    self.win_emuij_node, self.emuij_node = get_shared(total_full_xmat_size)#, set_zeros=True)

    self.win_xmat_dim, self.xmat_dim = get_shared((self.no**2, 2), dtype=np.int64)#, set_zeros=True)
    self.win_emuij_dim, self.emuij_dim = get_shared((self.no**2, 2), dtype=np.int64)#, set_zeros=True)

    if pair_slice is not None:
        
        if self.use_gpu:
            get_precon_close_cuda(self, pair_slice, full_xmat_size_sum)
        else:
            for ipair in pair_slice:
                i = ipair // self.no
                j = ipair % self.no
                S_mat = getSuperMatShared([i,j,i,j], self.smat_node, self.sf_offsets_node, self.sf_dim, self.no)
                F_mat = getSuperMatShared([i,j,i,j], self.fmat_node, self.sf_offsets_node, self.sf_dim, self.no)
                
                '''eigval, eigvec = np.linalg.eigh(S_mat)
                mask_nonred = eigval>1e-5
                newvec = eigvec[:, mask_nonred] / np.sqrt(eigval[mask_nonred])
                
                newh = multi_dot([newvec.T, F_mat, newvec])

                eigval, eigvec = np.linalg.eigh(newh)

                effective_c = np.dot(newvec, eigvec)'''
                #S_mat = S_mat + 1e-10 * np.eye(S_mat.shape[0])
                S_mat[np.diag_indices_from(S_mat)] += 1e-10
                eigval, effective_c = scipy.linalg.eigh(F_mat, S_mat)

                eij = self.eo[i]+self.eo[j] 
                eab = eigval+eigval.reshape(-1, 1)
                effective_d = 1.0/(eij - eab)

                eidx1 = xidx1 = full_xmat_size_sum[ipair]

                xidx0 = xidx1 - effective_c.size
                self.xmat_node[xidx0:xidx1] = effective_c.ravel()
                self.xmat_dim[ipair] = effective_c.shape
                
                eidx0 = eidx1 - effective_d.size
                self.emuij_node[eidx0:eidx1] = effective_d.ravel()
                self.emuij_dim[ipair] = effective_d.shape

    comm.Barrier()
    if nnode > 1:
        '''Acc_and_get_GA(self.xmat_dim)
        Acc_and_get_GA(self.emuij_dim)
        Acc_and_get_GA(self.xmat_node)
        Acc_and_get_GA(self.emuij_node)'''

        pair_offsets = np.arange(self.no**2)[:, np.newaxis] + np.array([0, 1])

        full_xmat_offsets = np.empty((self.no**2, 2), dtype=np.int64)
        full_xmat_offsets[self.pairlist_close] = np.column_stack((np.append(0, full_xmat_size_sum_close[:-1]), 
                                                                  full_xmat_size_sum_close))

        pair_node_offsets = get_node_offsets(pair_slices, pair_offsets)
        xmat_node_offests = get_node_offsets(pair_slices, full_xmat_offsets)

        Get_from_other_nodes_GA(self.xmat_dim, pair_node_offsets)
        Get_from_other_nodes_GA(self.emuij_dim, pair_node_offsets)
        Get_from_other_nodes_GA(self.xmat_node, xmat_node_offests)
        Get_from_other_nodes_GA(self.emuij_node, xmat_node_offests)
        comm.Barrier()

    self.xmat_offsets_node = [None]*self.no**2
    self.emuij_offsets_node = [None]*self.no**2

    for ipair in self.pairlist_close:
        eidx1 = xidx1 = full_xmat_size_sum[ipair]

        xidx0 = xidx1 - np.prod(self.xmat_dim[ipair])
        self.xmat_offsets_node[ipair] = [xidx0, xidx1]

        eidx0 = eidx1 - np.prod(self.emuij_dim[ipair])
        self.emuij_offsets_node[ipair] = [eidx0, eidx1]

def slice_ialp(ialp_ori, fit_pair, axis=1):
    '''if self.use_gpu:
        fit_pair = self.fit_bool_gpu[ipair]#self.fit_pair_gpu
    else:
        fit_pair = self.fit_pair[ipair]'''
    #pidx = self.pair_indices[ipair]
    if axis == 1:
        #return ialp_ori[:, fit_pair[pidx]]
        return ialp_ori[:, fit_pair]
    elif axis == 0:
        #return ialp_ori[fit_pair[pidx]]
        return ialp_ori[fit_pair]
    else:
        raise NotImplementedError

def get_imujp(self, log):
    if self.use_gpu:
        cupy.get_default_memory_pool().free_all_blocks()

    self.t_io = create_timer()
    self.t_cal = create_timer()
    self.t_data = create_timer()

    self.recorder_io = []
    self.recorder_cal = []
    self.recorder_data = []

    tt = get_current_time()
    self.imujp_dim = {}
    self.imujp_offsets_pair = {}
    self.imujp_sizes_occ = np.zeros((self.no), dtype=np.int64)

    idx0 = 0
    for i in self.mo_list:
        size_imujp_i = 0
        for j in self.jlist_close_full[i]:
            ipair = i * self.no + j
            nosv_j = self.nosv[j]
            if self.loc_fit and i != j:
                nfit_ij = self.nfit_pair[ipair] #self.nfit_pair[self.pair_indices[ipair]]
            else:
                nfit_ij = self.naoaux
            size_ij = nfit_ij * nosv_j
            size_imujp_i += size_ij
            idx1 = idx0 + size_ij
            self.imujp_dim[ipair] = [nfit_ij, nosv_j]
            self.imujp_offsets_pair[ipair] = [idx0, idx1]
            idx0 = idx1
        self.imujp_sizes_occ[i] = size_imujp_i
    
    self.imujp_offsets_mo = cum_offset(self.imujp_sizes_occ)
    total_imujp_size = self.imujp_offsets_mo[-1]

    nocc_core = self.no - len(self.mo_list)

    mo_slices = get_slice(range(nrank), job_list=self.mo_list) # To align with ialp_gpu
    
    
    self.imujp_mo_slices = mo_slices
    mo_slice = mo_slices[irank]

    if self.int_storage in {1, 2} and not self.fully_direct:
        use_ialp_file = True
    else:
        use_ialp_file = False

    avail_cpu_memory = ave_mem_spare(self.mol, per_core=False)

    close_is, close_js = np.divmod(self.pairlist_close, self.no)
    remote_is, remote_js = np.divmod(self.pairlist_remote, self.no)
    kmat_size = (np.sum((self.nosv[close_is] + self.nosv[close_js])**2)
                + np.sum((self.nosv[remote_is] + self.nosv[remote_js])**2))
    max_mem_f8 = avail_cpu_memory*1e6/8 - kmat_size

    if 0.8*max_mem_f8 > total_imujp_size:
        use_imujp_file = False
        self.win_imujp, self.imujp_node = get_shared(total_imujp_size, dtype=np.float64)#, set_zeros=True)
        imujp_data = self.imujp_node
        store_imujp = "in shared memory"
    else:
        use_imujp_file = True
        self.file_imujp = "imujp.tmp"
        file_imujp = h5py.File(self.file_imujp, 'w', driver='mpio', comm=comm)
        imujp_data = file_imujp.create_dataset("imujp", (total_imujp_size,), dtype=np.float64)
        store_imujp = "on disk"
    
    log.info(f"    imujp {store_imujp}: %.2f GB"%(total_imujp_size*8*1e-9))
    
    if mo_slice is not None:
        # For close pairs
        if use_ialp_file:
            file_ialp = h5py.File(self.file_ialp, 'r')
            ialp_data = file_ialp["ialp"]
        elif self.int_storage == 0:
            ialp_data = self.ialp_node
        elif self.int_storage == 3:
            ialp_data = self.ialp_gpu
        else:
            ialp_data = None

        if self.use_gpu:
            '''if self.fully_direct:
                get_direct_imujp_cuda(self, mo_slice, log)
            else:'''
            get_imujp_cuda(self, mo_slices, ialp_data, imujp_data)
        else:
            for idx_i, i in enumerate(mo_slice):
                imujp_i = np.empty(self.imujp_sizes_occ[i])

                t0 = get_current_time()
                ialp_i = np.ascontiguousarray(ialp_data[i-nocc_core].T) #(naux, nao)
                #self.t_io += get_elapsed_time(t0)
                self.recorder_io.append(record_elapsed_time(t0))

                t0 = get_current_time()
                imujp_idx0 = 0
                for j in self.jlist_close_full[i]:
                    ipair = i * self.no + j
                    qj_idx0, qj_idx1 = self.qao_offsets_node[j]
                    qao_j = self.qao_node[qj_idx0:qj_idx1].reshape(self.qao_dim[j])

                    if (self.loc_fit) and (i != j):
                        #t1 = get_current_time()
                        fit_pair = np.union1d(self.fit_list[i], self.fit_list[j])
                        ialp_slice = slice_ialp(ialp_i, fit_pair, axis=0)
                        #self.t_cal += get_current_time() - t1
                    else:
                        ialp_slice = ialp_i
                    
                    infit, inosv = self.imujp_dim[ipair]
                    imujp_idx1 = imujp_idx0 + infit * inosv
                    buff_imujp_i = imujp_i[imujp_idx0:imujp_idx1].reshape(self.imujp_dim[ipair])
                    #np.dot(qao_j.T, ialp_slice, out=buff_imujp_i)
                    np.dot(ialp_slice, qao_j, out=buff_imujp_i)
                    imujp_idx0 = imujp_idx1
                #self.t_cal += get_elapsed_time(t0)
                self.recorder_io.append(record_elapsed_time(t0))
                
                idx0, idx1 = self.imujp_offsets_mo[i:i+2]
                if use_imujp_file:
                    t0 = get_current_time()
                    imujp_data.write_direct(imujp_i, dest_sel=np.s_[idx0:idx1])
                    #self.t_io += get_elapsed_time(t0)
                    self.recorder_io.append(record_elapsed_time(t0))
                else:
                    np.copyto(self.imujp_node[idx0:idx1], imujp_i)

        if use_ialp_file:
            file_ialp.close()
    comm.Barrier()

    if use_imujp_file:
        file_imujp.close()
    elif nnode > 1:
        '''t0 = get_current_time()
        Acc_and_get_GA(self.imujp_node)
        comm.Barrier()
        self.t_io += get_elapsed_time(t0)'''

    if not self.cal_grad:
        if use_ialp_file:
            if irank == 0:
                os.remove(self.file_ialp)
        else:
            if self.int_storage == 3:
                self.ialp_gpu = None
            elif self.int_storage == 0:
                free_win(self.win_ialp)
                self.ialp_node = None

    batch_accumulate_time(self.t_cal, self.recorder_cal)
    batch_accumulate_time(self.t_io, self.recorder_io)
    batch_accumulate_time(self.t_data, self.recorder_data)
    
    time_list = [['calculation', self.t_cal], ['IO', self.t_io]]
    if self.use_gpu:
        time_list.append(['GPU-CPU data', self.t_data])
    time_list.append(["imujp", get_elapsed_time(tt)])
    time_list = get_max_rank_time_list(time_list)
    print_time(time_list, log)

def imujp_to_kmat(self, log):
    self.t_io = create_timer()
    self.t_cal = create_timer()
    self.t_data = create_timer()

    self.recorder_io = []
    self.recorder_cal = []
    self.recorder_data = []

    tt = get_current_time()
    #self.kmat_dim = [None]*self.no**2
    #self.kmat_offsets_node = [None]*self.no**2
    #cost_kmat = np.empty(len(self.pairlist), dtype=np.int64)
    
    '''kidx0 = 0
    for pidx, ipair in enumerate(self.pairlist):
        i = ipair // self.no
        j = ipair % self.no
        if self.is_close[ipair]:
            nosv_ij = self.nosv[i] + self.nosv[j]
            self.kmat_dim[ipair] = [nosv_ij, nosv_ij]
        else:
            self.kmat_dim[ipair] = [self.nosv[i], self.nosv[j]]
        
        kidx1 = kidx0 + np.prod(self.kmat_dim[ipair])
        self.kmat_offsets_node[ipair] = [kidx0, kidx1]
        kidx0 = kidx1

        if self.loc_fit:
            cost_kmat[pidx] = np.prod(self.kmat_dim[ipair]) * self.nfit_pair[ipair]
        else:
            cost_kmat[pidx] = np.prod(self.kmat_dim[ipair]) * self.naoaux'''
    
    self.kmat_dim = np.empty((self.no**2, 2), dtype=np.int64)
    self.kmat_offsets_node = np.empty((self.no**2, 2), dtype=np.int64)

    close_ilist, close_jlist = np.divmod(self.pairlist_close, self.no)
    close_nosv_is = self.nosv[close_ilist]
    close_nosv_js = self.nosv[close_jlist]
    close_nosv_ijs = close_nosv_is + close_nosv_js
    close_kmat_dim = np.column_stack((close_nosv_ijs, close_nosv_ijs))
    close_kmat_sizes = close_nosv_ijs**2
    close_k_offsets = cum_offset(close_kmat_sizes, return_2d=True)
    self.kmat_dim[self.pairlist_close] = close_kmat_dim
    self.kmat_offsets_node[self.pairlist_close] = close_k_offsets
    close_k_costs = close_kmat_sizes * self.nfit_pair[self.pairlist_close]
    close_pair_slices = get_slice(range(nrank), job_list=self.pairlist_close, 
                                  weight_list=close_k_costs)
    self.win_close_kmat_node, self.close_kmat_node = get_shared(close_k_offsets[-1][1])
    close_kmat_size = close_k_offsets[-1][1]

    if len(self.pairlist_remote) > 0:
        remote_ilist, remote_jlist = np.divmod(self.pairlist_remote, self.no)
        remote_nosv_is = self.nosv[remote_ilist]
        remote_nosv_js = self.nosv[remote_jlist]
        remote_kmat_dim = np.column_stack((remote_nosv_is, remote_nosv_js))
        remote_kmat_sizes = remote_nosv_is * remote_nosv_js
        remote_k_offsets = cum_offset(remote_kmat_sizes, return_2d=True)
        self.kmat_dim[self.pairlist_remote] = remote_kmat_dim
        self.kmat_offsets_node[self.pairlist_remote] = remote_k_offsets
        remote_k_costs = remote_kmat_sizes * self.nfit_pair[self.pairlist_remote]
        remote_pair_slices = get_slice(range(nrank), job_list=self.pairlist_remote, 
                                       weight_list=remote_k_costs)
        self.win_remote_kmat_node, self.remote_kmat_node = get_shared(remote_k_offsets[-1][1])

        pair_slices = combine_job_slices(close_pair_slices, remote_pair_slices)
        remote_kmat_size = remote_k_offsets[-1][1]
    else:
        pair_slices = close_pair_slices
        remote_kmat_size = 0
        


    ranks_node = inode * nrank_shm + np.arange(nrank_shm)
    check_mos = np.zeros(self.no, dtype=bool)
    for rank_i in ranks_node:
        pairs_i = pair_slices[rank_i]
        if pairs_i is not None:
            ilist, jlist = np.divmod(pairs_i, self.no)
            check_mos[ilist] = True
            check_mos[jlist] = True
    mos_node = np.arange(self.no, dtype=np.int32)[check_mos]

    use_imujp_file = not (hasattr(self, "imujp_node"))

    if use_imujp_file:
        file_imujp = h5py.File(self.file_imujp, 'r')
        imujp_data = file_imujp["imujp"]

        # Read imuip
        nosv_node = self.nosv[mos_node]
        mo_slice_all = get_slice(range(nrank_shm), job_list=mos_node, 
                                 weight_list=nosv_node)
        self.imuip_offsets_node = {}
        idx0 = 0
        for i in mos_node:
            idx1 = idx0 + self.naoaux * self.nosv[i]
            self.imuip_offsets_node[i] = [idx0, idx1]
            idx0 = idx1
        
        size_imuip = self.naoaux * np.sum(nosv_node)
        win_imuip_node, imuip_node = get_shared(size_imuip)
        mo_slice = mo_slice_all[irank_shm]
        if mo_slice is not None:
            for i in mo_slice:
                sidx0, sidx1 = self.imuip_offsets_node[i]
                idx0, idx1 = self.imujp_offsets_pair[i*self.no+i]
                imujp_data.read_direct(imuip_node, np.s_[idx0:idx1], np.s_[sidx0:sidx1])
        comm_shm.Barrier()
    else:
        imujp_data = self.imujp_node
        imuip_node = None

        if nnode > 1:
            node_mos = np.empty(self.no, dtype=np.int32)
            for node_i in range(nnode):
                ranks_node_i = node_i * nrank_shm + np.arange(nrank_shm)
                for rank_i in ranks_node_i:
                    mo_slice = self.imujp_mo_slices[rank_i]
                    if mo_slice is not None:
                        node_mos[mo_slice] = node_i
            close_pairs_node = []
            for rank_i in ranks_node:
                if close_pair_slices[rank_i] is not None:
                    close_pairs_node.append(close_pair_slices[rank_i])
            close_pairs_node = np.concatenate(close_pairs_node)

            ilist_node, jlist_node = np.divmod(close_pairs_node, self.no)
            diag_pairs_node = mos_node*self.no+mos_node
            offdiag_mask = ilist_node != jlist_node
            close_pairs_node = np.concatenate([diag_pairs_node, close_pairs_node[offdiag_mask],
                                               jlist_node[offdiag_mask]*self.no+ilist_node[offdiag_mask]])
            close_pairs_node.sort()

            imujp_offsets_off_node = {}
            for node_i in range(nnode):
                if node_i != inode:
                    imujp_offsets_off_node[node_i] = []
            full_ilist = close_pairs_node // self.no
            node_list = node_mos[full_ilist]
            for ipair in close_pairs_node[node_list != inode]:
                node_i = node_mos[ipair // self.no]
                imujp_offsets_off_node[node_i].append(self.imujp_offsets_pair[ipair])
            
            job_list = []
            for node_i in imujp_offsets_off_node.keys():
                for offsets in merge_intervals(imujp_offsets_off_node[node_i]):
                    job_list.append([node_i, offsets])
            job_list = np.asarray(job_list)

            slice_ids = get_slice(range(nrank_shm), job_size=len(job_list))[irank_shm]

            win_col = create_win(self.imujp_node, comm=comm)
            win_col.Fence()
            itemsize = np.dtype(np.float64).itemsize
            if slice_ids is not None:
                job_slice = [job_list[id] for id in slice_ids]
                for node_i, (idx0, idx1) in job_slice:
                    target_rank = node_i * nrank_shm + (nrank_shm - irank_shm - 1)
                    win_col.Get(self.imujp_node[idx0:idx1], 
                                target=(idx0*itemsize, idx1-idx0, MPI.DOUBLE),
                                target_rank=target_rank)

            win_col.Fence()
            free_win(win_col)



    '''mo_slice = mo_slice_ranks[irank]
    if mo_slice is not None:
        mo_slice.sort()
        pair_slice = []
        for i in mo_slice:
            pair_slice.append(i * self.no + self.jlist_close[i])
            pair_slice.append(i * self.no + self.jlist_remote[i])
        pair_slice = np.hstack(pair_slice)'''
    pair_slice = pair_slices[irank]
    if pair_slice is not None:
        if self.use_gpu:
            avail_cpu_mem = get_mem_spare(self.mol, per_core=False)
            avail_cpu_mem -= (close_kmat_size + remote_kmat_size)*8*1e-6
            avail_cpu_mem /= nrank_shm
            imujp_to_kmat_cuda(self, pair_slice, imujp_data, imuip_node, 
                               avail_cpu_mem=avail_cpu_mem)
        else:
            for ipair in pair_slice:
                i = ipair // self.no
                j = ipair % self.no

                if i == j:
                    if use_imujp_file:
                        idx0, idx1 = self.imuip_offsets_node[i]
                        imuip = imuip_node[idx0:idx1].reshape(self.naoaux, self.nosv[i])
                    else:
                        idx0, idx1 = self.imujp_offsets_pair[ipair]
                        imuip = self.imujp_node[idx0:idx1].reshape(self.naoaux, self.nosv[i])
                    imuip = slice_ialp(imuip, self.fit_list[i], axis=0)
                    imup = np.hstack((imuip, imuip))
                    jmup = imup
                else:
                    pair_ii = i * self.no + i
                    pair_jj = j * self.no + j
                    pair_ji = j * self.no + i

                    if use_imujp_file:
                        t0 = get_current_time()
                        idx0, idx1 = self.imuip_offsets_node[i]
                        imuip = imuip_node[idx0:idx1].reshape(self.naoaux, self.nosv[i])
                        idx0, idx1 = self.imuip_offsets_node[j]
                        jmujp = imuip_node[idx0:idx1].reshape(self.naoaux, self.nosv[j])
                        if self.is_close[ipair]:
                            ij_idx0, ij_idx1 = self.imujp_offsets_pair[ipair]
                            ji_idx0, ji_idx1 = self.imujp_offsets_pair[pair_ji]
                            imujp = np.empty(self.imujp_dim[ipair])
                            jmuip = np.empty(self.imujp_dim[pair_ji])
                            imujp_data.read_direct(imujp.ravel(), np.s_[ij_idx0:ij_idx1])
                            imujp_data.read_direct(jmuip.ravel(), np.s_[ji_idx0:ji_idx1])
                        self.recorder_io.append(record_elapsed_time(t0))
                    else:
                        idx0, idx1 = self.imujp_offsets_pair[pair_ii]
                        imuip = self.imujp_node[idx0:idx1].reshape(self.naoaux, self.nosv[i])
                        idx0, idx1 = self.imujp_offsets_pair[pair_jj]
                        jmujp = self.imujp_node[idx0:idx1].reshape(self.naoaux, self.nosv[j])
                        if self.is_close[ipair]:
                            ij_idx0, ij_idx1 = self.imujp_offsets_pair[ipair]
                            ji_idx0, ji_idx1 = self.imujp_offsets_pair[pair_ji]
                            idx0, idx1 = self.imujp_offsets_pair[ipair]
                            imujp = self.imujp_node[idx0:idx1].reshape(self.imujp_dim[ipair])
                            idx0, idx1 = self.imujp_offsets_pair[pair_ji]
                            jmuip = self.imujp_node[idx0:idx1].reshape(self.imujp_dim[pair_ji])
                    fit_pair = np.union1d(self.fit_list[i], self.fit_list[j])
                    imuip = slice_ialp(imuip, fit_pair, axis=0)
                    jmujp = slice_ialp(jmujp, fit_pair, axis=0)


                    if self.is_close[ipair]:
                        imup = np.hstack((imuip, imujp))
                        jmup = np.hstack((jmuip, jmujp))
                    else:
                        imup = imuip
                        jmup = jmujp

                t0 = get_current_time()
                #idx1 = kidx0 + np.prod(self.kmat_dim[ipair])
                kidx0, kidx1 = self.kmat_offsets_node[ipair]
                #kmat_ij = self.kmat_node[kidx0:kidx1].reshape(self.kmat_dim[ipair])
                kmat_node = self.close_kmat_node if self.is_close[ipair] else self.remote_kmat_node
                kmat_ij = kmat_node[kidx0:kidx1].reshape(self.kmat_dim[ipair])
                np.dot(imup.T, jmup, out=kmat_ij)
                #accumulate_time(self.t_cal, t0)
                self.recorder_cal.append(record_elapsed_time(t0))

    comm.Barrier()
    if nnode > 1:
        #Acc_and_get_GA(self.kmat_node)
        #raise NotImplementedError
        
        close_k_node_offsets = get_node_offsets(close_pair_slices, self.kmat_offsets_node)
        remote_k_node_offsets = get_node_offsets(remote_pair_slices, self.kmat_offsets_node)

        Get_from_other_nodes_GA(self.close_kmat_node, close_k_node_offsets)
        Get_from_other_nodes_GA(self.remote_kmat_node, remote_k_node_offsets)
        comm.Barrier()

    if use_imujp_file:
        free_win(win_imuip_node)
        file_imujp.close()
    elif not self.cal_grad:
        free_win(self.win_imujp)
        self.imujp_node = None
    
    batch_accumulate_time(self.t_cal, self.recorder_cal)
    batch_accumulate_time(self.t_io, self.recorder_io)
    batch_accumulate_time(self.t_data, self.recorder_data)

    time_list = [['calculation', self.t_cal], ['IO', self.t_io]]
    if self.use_gpu:
        time_list.append(['GPU-CPU data', self.t_data])
    time_list.append(["kmat", get_elapsed_time(tt)])
    time_list = get_max_rank_time_list(time_list)
    print_time(time_list, log)

    

def get_kmatrix_GA(self):
    
    tt = get_current_time()
    log = lib.logger.Logger(self.stdout, self.verbose)

    log.info("    Start computing imujp...")
    t0 = get_current_time()
    get_imujp(self, log)
    t_imujp = get_elapsed_time(t0)

    if not self.cal_grad:
        free_win(self.win_qao_node)
        self.qao_node = None
    
    print_mem('imujp computation', self.pid_list, log)

    log.info("\n    Start assembling K matrix...")
    t0 = get_current_time()
    imujp_to_kmat(self, log)
    t_kmat = get_elapsed_time(t0)

    log = lib.logger.Logger(self.stdout, self.verbose)
    log.info("")
    time_list = [["OSV K matrix", get_elapsed_time(tt)]]
    '''if self.loc_fit:
        time_list += [['cholesky decomp.', t_cd], ['solve tri', t_solve]]'''
    time_list = get_max_rank_time_list(time_list)
    print_time(time_list, log)
    print_mem('K assembly', self.pid_list, log)

def get_klist_pairs(pairlist, jlist, is_close, s_ratio, nocc, threeb_tol):
    klist_pairs = {}
    for ipair in pairlist:
        i = ipair//nocc
        j = ipair%nocc
        mos_now = np.union1d(jlist[i], jlist[j])

        mos_offdiag = mos_now[(mos_now!=i) & (mos_now!=j)]
        if len(mos_offdiag) > 0:
            iks = i * nocc + mos_offdiag
            jks = j * nocc + mos_offdiag
            sijk_col = np.empty((len(mos_offdiag), 3))
            sijk_col[:, 0] = s_ratio[i*nocc+j]
            sijk_col[:, 1] = s_ratio[i*nocc+mos_offdiag]
            sijk_col[:, 2] = s_ratio[j*nocc+mos_offdiag]
            sijk_col.sort(axis=1)
            
            '''mask_mos = sijk_col[:, 0] > remo_tol
            mask_mos[np.mean(sijk_col, axis=1) < threeb_tol] = False
            mask_mos[sijk_col[:, 1] > threeb_tol] = True'''
            mask_mos = (is_close[iks]) & (
                        is_close[jks]) & (
                        (np.mean(sijk_col, axis=1) > threeb_tol) | 
                        (sijk_col[:, 1] > threeb_tol))
            selected_k = mos_offdiag[mask_mos]
        else:
            selected_k = np.array([], dtype=int)

        mos_diag = [i] if i == j else [i, j]
        klist_pairs[ipair] = np.array(mos_diag + list(selected_k), dtype=int)
    return klist_pairs

def get_pairs_off_node(pair_slices, ncore_shm, node_now):
    pairs_off_node = []
    for core_i, pairs in enumerate(pair_slices):
        if pairs is None: continue
        node_of_core = core_i // ncore_shm
        if node_of_core != node_now:
            pairs_off_node.extend(pairs)
    return pairs_off_node

class ResidualIterations():
    def __init__(self, mp2, pairlist=None, ene_tol=None, use_tinit=False, 
                 use_dynt=True, k_tol=1e-5, max_cycle=None):
        self.mp2 = mp2

        if pairlist is None:
            self.close_pairs_global = mp2.pairlist_close
            self.remote_pairs_global = mp2.pairlist_remote
        else:
            self.close_pairs_global = pairlist[mp2.is_close[pairlist]]
            self.remote_pairs_global = pairlist[mp2.is_remote[pairlist]]
        use_dynt = False

        if ene_tol is None:
            self.ene_tol = mp2.ene_tol
        else:
            self.ene_tol = ene_tol
        self.use_tinit = use_tinit
        self.k_tol = k_tol
        self.use_dynt = use_dynt

        if mp2.local_type == 0:
            self.max_cycle = 1
        elif max_cycle is None:
            self.max_cycle = mp2.max_cycle
        else:
            self.max_cycle = max_cycle

        self.check_ene = True
        if nnode > 1:
            self.check_ene = False
        if self.check_ene:
            self.win_pair_ene, self.pair_ene = get_shared(mp2.no**2)
        else:
            self.pair_ene = {}
        self.log = lib.logger.Logger(mp2.stdout, mp2.verbose)
    
    def map_pairs_to_ranks(self):
        mp2 = self.mp2

        if len(self.close_pairs_global) > 0:
            ilist_close, jlist_close = np.divmod(self.close_pairs_global, mp2.no)
            weight_list = (mp2.nosv[ilist_close] + mp2.nosv[jlist_close])**3
            self.close_pairs_ranks = get_slice(range(nrank), job_list=self.close_pairs_global, 
                                              weight_list=weight_list)
            
            self.close_pairs_rank = self.close_pairs_ranks[irank]
            if self.close_pairs_rank is None:
                self.close_pairs_rank = [] 
            self.exist_close = True

            if nnode > 1: 
                self.close_tnode_offsets = get_node_offsets(self.close_pairs_ranks, mp2.tmat_offsets_node)
                close_pairs_off_node = get_pairs_off_node(self.close_pairs_ranks, nrank_shm, irank//nrank_shm)
                close_pairs_off_node_rank = get_slice(range(nrank_shm), job_list=close_pairs_off_node)[irank_shm]
                tmat_offsets_off_node = [mp2.tmat_offsets_node[ipair] for ipair in close_pairs_off_node_rank]
                self.tmat_offsets_off_node = merge_intervals(tmat_offsets_off_node)
            else:
                self.tmat_offsets_off_node = None
        else:
            self.tmat_offsets_off_node = None
            self.close_pairs_ranks = []
            self.close_pairs_rank = []
            self.exist_close = False

        if len(self.remote_pairs_global) > 0:
            ilist_remote, jlist_remote = np.divmod(self.remote_pairs_global, mp2.no)
            weight_list = (mp2.nosv[ilist_remote] * mp2.nosv[jlist_remote])**3
            self.remote_pairs_ranks = get_slice(range(nrank), job_list=self.remote_pairs_global, 
                                               weight_list=weight_list)
            self.remote_pairs_rank = self.remote_pairs_ranks[irank]
            if self.remote_pairs_rank is None:
                self.remote_pairs_rank = [] 
            self.exist_remote = True

            if nnode > 1 and mp2.cal_grad:
                #self.remote_tnode_offsets = get_node_offsets(self.remote_pairs_ranks, self.tmat_offsets_node)
                self.remote_tnode_offsets = get_node_offsets(self.remote_pairs_ranks, mp2.tmat_offsets_node)

        else:
            self.remote_pairs_ranks = []
            self.remote_pairs_rank = []
            self.exist_remote = False


        self.close_pairs_rank = np.asarray(self.close_pairs_rank, dtype=int)
        self.remote_pairs_rank = np.asarray(self.remote_pairs_rank, dtype=int)
        self.pairs_rank = np.concatenate((self.close_pairs_rank, self.remote_pairs_rank), dtype=int)

        if len(self.remote_pairs_rank) > 0:
            ilist_remote, jlist_remote = np.divmod(self.remote_pairs_rank, mp2.no)
            self.mos_remote = np.unique(np.concatenate((ilist_remote, jlist_remote)))
        else:
            self.mos_remote = []
        

        if len(self.close_pairs_rank) > 0:
            self.klist_pairs = get_klist_pairs(self.close_pairs_rank, mp2.jlist_close_full, 
                                            mp2.is_close, mp2.s_ratio, mp2.no, mp2.threeb_tol)

            #if not mp2.use_gpu:
            self.extended_pairs = get_extended_pairs(self.close_pairs_rank, mp2.no, 
                                                        self.klist_pairs, self.remote_pairs_rank,
                                                        self.mos_remote)
        elif len(self.remote_pairs_rank) > 0: #and (not mp2.use_gpu):
            self.extended_pairs = get_pairs_with_diag(self.remote_pairs_rank, mp2.no)

        else:
            self.extended_pairs = []

    def load_mat(self):

        mp2 = self.mp2
        if mp2.int_storage == 3:
            npair_full = mp2.no**2
            self.kmat_offsets_gpu, self.kmat_gpu = load_gpu_mat(self.pairs_rank, mp2.kmat_offsets_node, mp2.kmat_node, npair_full)
            self.sf_offsets_gpu, self.smat_gpu = load_gpu_mat(self.extended_pairs, mp2.sf_offsets_node, mp2.smat_node, npair_full)
            _, self.fmat_gpu = load_gpu_mat(self.extended_pairs, mp2.sf_offsets_node, mp2.fmat_node, npair_full)

            if len(self.close_pairs_rank) > 0:
                self.xmat_offsets_full_gpu, self.xmat_gpu = load_gpu_mat(self.close_pairs_rank, mp2.xmat_offsets_node, mp2.xmat_node, 
                                                                    npair_full, get_full_offsets=True)
                self.emuij_offsets_gpu, self.emuij_gpu = load_gpu_mat(self.close_pairs_rank, mp2.emuij_offsets_node, mp2.emuij_node, npair_full)

            if len(self.mos_remote) > 0:
                self.xii_offsets_gpu, self.xii_gpu = load_gpu_mat(self.mos_remote, mp2.xii_offsets_node, mp2.xii_node, mp2.no)
                self.emui_offsets_gpu, self.emui_gpu = load_gpu_mat(self.mos_remote, mp2.emui_offsets_node, mp2.emui_node, mp2.no)
                
        else:
            # Get OSV matrices from shared memory
            self.K_matrix = [None] * mp2.no**2
            self.S_matrix = [None] * mp2.no**2
            self.F_matrix = [None] * mp2.no**2
            self.T_new = [None] * mp2.no**2
            
            for ipair in self.pairs_rank:
                kmat_node = mp2.close_kmat_node if mp2.is_close[ipair] else mp2.remote_kmat_node
                self.K_matrix[ipair] = getMatFromNode(ipair, kmat_node, mp2.kmat_offsets_node, mp2.kmat_dim)
                #self.K_matrix[ipair] = getMatFromNode(ipair, mp2.kmat_node, mp2.kmat_offsets_node, mp2.kmat_dim)
 
            for ipair in self.extended_pairs:
                self.S_matrix[ipair] = getMatFromNode(ipair, mp2.smat_node, mp2.sf_offsets_node, mp2.sf_dim)
                self.F_matrix[ipair] = getMatFromNode(ipair, mp2.fmat_node, mp2.sf_offsets_node, mp2.sf_dim)
                #self.T_new[ipair] = getMatFromNode(ipair, mp2.tmat_node, mp2.tmat_offsets_node, mp2.tmat_dim)
                if mp2.is_close[ipair]:
                    self.T_new[ipair] = getMatFromNode(ipair, mp2.close_tmat_node, 
                                                       mp2.tmat_offsets_node, mp2.tmat_dim)
                else:
                    if mp2.cal_grad:
                        self.T_new[ipair] = getMatFromNode(ipair, mp2.remote_tmat_node, 
                                                       mp2.tmat_offsets_node, mp2.tmat_dim)
                    else:
                        self.T_new[ipair] = np.zeros(mp2.tmat_dim[ipair])
                
            
            if len(self.close_pairs_rank) > 0:
                self.X_matrix = [None] * mp2.no**2
                self.emu_ij = [None] * mp2.no**2
                for ipair in self.close_pairs_rank:
                    self.X_matrix[ipair] = getMatFromNode(ipair, mp2.xmat_node, mp2.xmat_offsets_node, mp2.xmat_dim)
                    self.emu_ij[ipair] = getMatFromNode(ipair, mp2.emuij_node, mp2.emuij_offsets_node, mp2.emuij_dim)

            if len(self.mos_remote) > 0:
                self.xii = [None] * mp2.no
                self.emui = [None] * mp2.no
                for i in self.mos_remote:
                    self.xii[i] = getMatFromNode(i, mp2.xii_node, mp2.xii_offsets_node, mp2.xii_dim)
                    self.emui[i] = getMatFromNode(i, mp2.emui_node, mp2.emui_offsets_node, mp2.emui_dim)


    def get_rmat(self, ipair):
        mp2 = self.mp2
        i = ipair // mp2.no
        j = ipair % mp2.no
        rmat = np.copy(self.K_matrix[ipair])
        if mp2.is_remote[ipair]:
            T_ii = self.T_init[i*mp2.no+i][:mp2.nosv[i], mp2.nosv[i]:]
            T_jj = self.T_init[j*mp2.no+j][:mp2.nosv[j], mp2.nosv[j]:]
            rmat += np.dot(self.T_init[ipair], (self.F_matrix[j*mp2.no+j] - mp2.loc_fock[j,j]))
            rmat += np.dot((self.F_matrix[i*mp2.no+i] - mp2.loc_fock[i,i]), self.T_init[ipair])
            rmat -= mp2.loc_fock[i,j] * (np.dot(self.S_matrix[i*mp2.no+j], T_jj) + 
                                        np.dot(T_ii, self.S_matrix[i*mp2.no+j]))
        else:
            F_ijij = generation_SuperMat([i, j, i, j], self.F_matrix, mp2.nosv, mp2.no)
            for k in self.klist_pairs[ipair]:
                if abs(mp2.loc_fock[k, j]) > self.k_tol:
                    S_ikij = generation_SuperMat([i, k, i, j], self.S_matrix, mp2.nosv, mp2.no)

                    B = -mp2.loc_fock[k, j] * S_ikij
                    if (k==j):
                        B += F_ijij
                    
                    if i > k:
                        T_ik = flip_ij(k, i, self.T_init[k*mp2.no+i], mp2.nosv)
                    else:
                        T_ik = self.T_init[i*mp2.no+k]

                    rmat += multi_dot([S_ikij.T, T_ik, B])
                if abs(mp2.loc_fock[i, k]) > self.k_tol:
                    S_ijkj = generation_SuperMat([i, j, k, j], self.S_matrix, mp2.nosv, mp2.no)
                    C = -mp2.loc_fock[i, k] * S_ijkj
                    if (i==k):
                        C += F_ijij
                    if k > j:
                        T_kj = flip_ij(j, k, self.T_init[j*mp2.no+k], mp2.nosv)
                    else:
                        T_kj = self.T_init[k*mp2.no+j]

                    rmat += multi_dot([C, T_kj, S_ijkj.T])
        return rmat

    def run_iteration(self, run_remote=False):
        mp2 = self.mp2
        if run_remote:
            ptype = "remote"
            pairlist = self.remote_pairs_rank
            max_cycle = 1
        else:
            ptype = "close"
            pairlist = self.close_pairs_rank
            max_cycle = self.max_cycle


        self.log.info(f"\n    Solving residual equation for {ptype} pairs")
        win_ene_node, ene_node = get_shared(nrank_shm, set_zeros=True)

        tinit_is_tnew = (self.use_dynt or run_remote)

        if tinit_is_tnew:
            if not mp2.use_gpu: self.T_init = self.T_new
            close_tmat_offsets = mp2.tmat_offsets_node
            tinit_node = mp2.close_tmat_node
        else:
            read_tmat_offsets = np.asarray([mp2.tmat_offsets_node[ipair] for ipair in self.close_pairs_global])
            close_tmat_offsets_pidx = cum_offset(read_tmat_offsets[:, 1] - read_tmat_offsets[:, 0], return_2d=True)
            win_tinit, tinit_node = get_shared(close_tmat_offsets_pidx[-1][1])
            close_tmat_offsets = {}
            for pidx, ipair in enumerate(self.close_pairs_global):
                close_tmat_offsets[ipair] = close_tmat_offsets_pidx[pidx]

            if not mp2.use_gpu:
                self.T_init = {}
                for ipair in self.extended_pairs:
                    if mp2.is_remote[ipair]: continue
                    tidx0, tidx1 = close_tmat_offsets[ipair]
                    self.T_init[ipair] = tinit_node[tidx0:tidx1].reshape(mp2.tmat_dim[ipair])

        if mp2.use_gpu and len(pairlist) > 0:
            if run_remote:
                res_kernel_gpu = remote_residual_iter_cuda(self, pairlist)
            else:
                res_kernel_gpu = close_residual_iter_cuda(self, pairlist)

        #diis is not needed for clusters
        ene_old = 0.0
        for ite in range(max_cycle):

            if not tinit_is_tnew:
                pair_slice = get_slice(range(nrank_shm), job_list=self.close_pairs_global)[irank_shm]
                if pair_slice is not None:
                    for ipair in pair_slice:
                        read_idx0, read_idx1 = mp2.tmat_offsets_node[ipair]
                        save_idx0, save_idx1 = close_tmat_offsets[ipair]
                        #tmat_node = mp2.close_tmat if self.is_close(ipair) else mp2.remote_tmat
                        #tinit_node[save_idx0:save_idx1] = tmat_node[read_idx0:read_idx1]
                        #tinit_node[save_idx0:save_idx1] = mp2.tmat_node[read_idx0:read_idx1]
                        tinit_node[save_idx0:save_idx1] = mp2.close_tmat_node[read_idx0:read_idx1]
                comm_shm.Barrier()

            if len(pairlist) > 0:
                if mp2.use_gpu:
                    ene_new = res_kernel_gpu(tinit_node, close_tmat_offsets)
                else:
                    ene_new = 0.0
                    if run_remote:
                        for ipair in pairlist:
                            i = ipair // mp2.no
                            j = ipair % mp2.no
                            rmat_ij = self.get_rmat(ipair)
                            eff_del = 1 / (mp2.eo[i] + mp2.eo[j] - self.emui[i].reshape(-1,1) - self.emui[j].ravel())
                            effective_R = eff_del * multi_dot([self.xii[i].T, rmat_ij, self.xii[j]])
                            delta = multi_dot([self.xii[i], effective_R, self.xii[j].T])
                            self.T_new[ipair] += delta

                            ene_ij = 4 * np.dot(self.K_matrix[ipair].ravel(), self.T_new[ipair].ravel())
                            ene_new += ene_ij
                            self.pair_ene[ipair] = ene_ij

                    else:
                        '''if not self.use_dynt:
                            R_matrix = [None] * mp2.no**2
                            for ipair in pairlist:
                                R_matrix[ipair] = self.get_rmat(ipair)'''
                        for ipair in pairlist:
                            '''if self.use_dynt: #Dynamical amplitudes update
                                rmat_ij = self.get_rmat(ipair)
                            else:
                                rmat_ij = R_matrix[ipair]'''
                            rmat_ij = self.get_rmat(ipair)

                            effective_R = self.emu_ij[ipair] * multi_dot([self.X_matrix[ipair].T, rmat_ij, self.X_matrix[ipair]])
                            delta = multi_dot([self.X_matrix[ipair], effective_R, self.X_matrix[ipair].T])
                            self.T_new[ipair] += delta

                            T_bar_ij = 2 * self.T_new[ipair] - self.T_new[ipair].T
                            ene_ij = np.dot(self.K_matrix[ipair].ravel(), T_bar_ij.ravel())
                            if (ipair // mp2.no) != (ipair % mp2.no):
                                ene_ij *= 2
                            ene_new += ene_ij
                            self.pair_ene[ipair] = ene_ij
                '''if not run_remote and \
                   self.tmat_offsets_off_node is not None:
                    
                    for tidx0, tidx1 in self.tmat_offsets_off_node:
                        mp2.tmat_node[tidx0:tidx1] = 0.0'''
            else:
                ene_new = 0.0

            ene_node[irank_shm] = ene_new

            comm.Barrier()
            if nnode > 1:
                Acc_and_get_GA(ene_node)
                #Acc_and_get_GA(mp2.tmat_node)
                if not run_remote or mp2.cal_grad:
                    tmat_node = mp2.remote_tmat_node if run_remote else mp2.close_tmat_node
                    tnode_offsets = self.remote_tnode_offsets if run_remote else self.close_tnode_offsets
                    Get_from_other_nodes_GA(tmat_node, tnode_offsets)
                comm.Barrier()
            ene_new = np.sum(ene_node)

            var = abs(ene_old - ene_new)
            ene_old = ene_new
            
            if (len(pairlist) > 0):
                self.log.info("    Iter. %d: energy %.10f, by increment %.2E"%(ite+1, ene_new, var))

            #converged or not
            if (var < self.ene_tol):
                break
        
        free_win(win_ene_node)
        if not tinit_is_tnew:
            free_win(win_tinit)
        
        return ene_new

    def kernel(self):
        self.map_pairs_to_ranks()

        if (not self.mp2.use_gpu) or (
            self.mp2.int_storage == 3):
            self.load_mat()

        if self.exist_close:
            ene_close = self.run_iteration()
        else:
            ene_close = 0.0

        if self.exist_remote:
            ene_remote = self.run_iteration(run_remote=True)
        else:
            ene_remote = 0.0
        
        if self.check_ene:
            comm.Barrier()
            if irank == 0:
                #locfock = np.abs(self.mp2.loc_fock.ravel())
                locfock = np.abs(np.dot(self.mp2.o.T, self.mp2.o)).ravel()
                #print(locfock.shape)
                pairs = self.mp2.pairlist
                sort_ids = np.argsort(-locfock[pairs])
                msg = ""
                for ipair in pairs[sort_ids]:
                    #if abs(locfock[ipair]) < 1e-5:
                    #print(ipair, self.mp2.s_ratio[ipair], locfock[ipair], self.pair_ene[ipair])
                    msg += "%.5e %.5e %.5e\n"%(self.mp2.s_ratio[ipair], locfock[ipair], self.pair_ene[ipair])
                with open("pair_ene.log", "w") as f:
                    f.write(msg)
            comm.Barrier()
            free_win(self.win_pair_ene)
        return ene_close + ene_remote

class OSVLMP2():
    def __init__ (self, RHF, my_para):
        self.__dict__.update(my_para.__dict__)
        self.chkfile_ialp = my_para.chkfile_ialp_mp2
        self.chkfile_fitratio = self.chkfile_fitratio_mp2
        self.RHF = RHF
        self.t_hf = RHF.t_hf
        self.mol = RHF.mol
        self.mo_energy = RHF.mo_energy
        self.mo_occ = RHF.mo_occ
        self.mo_coeff = RHF.mo_coeff
        self.int_storage  = RHF.int_storage 
        self.shell_tol = RHF.shell_tol
        self.with_solvent = self.RHF.with_solvent
        
        #self.solvent = my_para.solvent
        self.atom_list = range(self.mol.natm)
        self.stdout = sys.stdout
        self.naoaux = self.naux_mp2# = my_para.naux_mp2
        self.shm_ranklist = range(len(self.rank_list))
        self.nao = self.mol.nao_nr()
        self.nao_pair = self.nao * (self.nao+1) // 2
        
        if not self.cal_grad:
            self.cposv_tol = self.osv_tol
        self.o = self.mo_coeff[:, self.mo_occ>0]
        self.v = self.mo_coeff[:, self.mo_occ==0]
        self.ev = self.mo_energy[self.mo_occ==0]
        self.nv = self.v.shape[1]
        self.nao = self.v.shape[0]
        self.no = self.nao - self.nv
        if self.use_frozen == True:
            self.use_sl = True
        if self.use_sl == True:
          #self.nocc_core = get_ncore(self.mol)
          #self.nocc_core = get_ncore(self)
            self.nocc_core = get_ncore(self.mol)
        else:
            self.nocc_core = None  
        if self.osv_tol == 0:
            self.use_cposv = False
        self.lg_dr = False
        if self.cal_grad:
            self.ene_tol = 1e-8
        else:
            self.ene_tol = 1e-6
        self.clus_type = 0

        if self.int_storage == 0:
            self.ijp_shape = True
        else:
            self.ijp_shape = False

        if (self.mol.pbc) or (not self.use_gpu) or (self.int_storage in {0, 3}):
            self.unsort_ialp_ao = True
        else:
            self.unsort_ialp_ao = False
            
    def localization(self, log):
        t1=get_current_time()
        log.info('\n--------------------------------Localization---------------------------------')

        #Initialise occupied mo list
        self.mo_list = np.arange(self.no)
        if self.use_frozen == True:
            log.info("Frozen core:ON")
            self.mo_list = self.mo_list[self.nocc_core:]
        else:
            log.info("Frozen core:OFF")
        self.nocc = len(self.mo_list)
        #pop_method = 'mul_melow'
        #pop_method = 'low_melow'
        #pop_method = 'mulliken'
        #pop_method = 'lowdin'
        
        self.win_o, self.o = get_shared((self.nao, self.no))
        self.win_eo, self.eo = get_shared(self.no)
        self.win_loc, self.loc_fock = get_shared((self.no, self.no))
        self.win_uo, self.uo = get_shared((self.no, self.no), set_zeros=True)
        
        if self.chkfile_loc is not None:
            if irank_shm == 0:
                log.info(f"Read loc matrices from check file:{self.chkfile_loc}")
                with h5py.File(self.chkfile_loc, 'r') as f:
                    f['o'].read_direct(self.o)
                    f['eo'].read_direct(self.eo)
                    f['loc_fock'].read_direct(self.loc_fock)
                    f['uo'].read_direct(self.uo)
        else:
            if self.use_frozen:
                use_sl = True
                frozen = False
            else:
                use_sl = self.use_sl
                frozen = self.use_frozen
            
            if self.local_type == 0:
                if irank_shm == 0:
                    #Canonical orbital
                    self.uo[:] = np.diag(np.ones(self.no))
                    self.o[:] = self.mo_coeff[:, self.mo_occ>0]
                    self.loc_fock[:] = np.diag(self.mo_energy[:self.no])
                    self.eo[:] = self.mo_energy[:self.no]
            else:
                occ_coeff = self.mo_coeff[:, self.mo_occ>0]
                
                uo = localization(self.mol, occ_coeff, local_type=self.local_type, 
                                    pop_method=self.pop_method, cal_grad=self.cal_grad, use_sl=use_sl, 
                                    frozen=frozen, loc_fit=self.loc_fit, verbose=self.verbose, log=log, 
                                    use_gpu=self.use_gpu, loc_tol=self.loc_tol, iop=1)
                if irank_shm == 0:
                    self.uo[:] = uo
                    np.dot(self.mo_coeff[:, :self.no], self.uo, out=self.o)
                    self.loc_fock[:] = multi_dot([self.uo.T, np.diag(self.mo_energy[:self.no]), self.uo])
                    self.eo[:] = np.diag(self.loc_fock)
                if irank == 0:
                    with h5py.File('loc_var.chk', 'w') as f:
                        f.create_dataset("uo", data=self.uo)
                        f.create_dataset("o", data=self.o)
                        f.create_dataset("loc_fock", data=self.loc_fock)
                        f.create_dataset("eo", data=self.eo)
                    if self.chkfile_save is not None:
                        if self.local_type == 1:
                            dir_loc = "%s/pm"%self.chkfile_save
                        elif self.local_type == 2:
                            dir_loc = "%s/boys"%self.chkfile_save
                        os.makedirs(dir_loc, exist_ok=True)
                        shutil.copy('loc_var.chk', dir_loc)
        comm_shm.Barrier()

        print_time(['localization', get_elapsed_time(t1)], log)
        self.t_loc = get_elapsed_time(t1)

        if self.use_gpu:
            free_cupy_mem()

    def int_trans(self, log):
        log.info("\n----------------------MP2 MO integral transformation------------------------")
        # Initialize density fitting class

        
        self.aux_atm_offset = self.with_df.auxmol.offset_nr_by_atom()
        self.ao_atm_offset = self.mol.offset_nr_by_atom()

        if self.use_gpu:
            self.memory_pool = cupy.get_default_memory_pool()
            self.stream_gpu = cupy.cuda.Stream()

            if not self.mol.pbc:
                self.intopt = VHFOpt(self.mol, self.with_df.auxmol, 'int2e')
                
                #self.intopt.build(1e-9, diag_block_with_triu=True, aosym=False)
                self.intopt.memory_pool = self.memory_pool
            
        elif not self.mol.pbc:
            if (self.cal_grad) and (self.RHF.shell_slice is None):
                self.RHF.shell_slice = int_prescreen.shell_prescreen(self.mol, self.RHF.with_df.auxmol, log, 
                                    shell_slice=self.RHF.shell_slice, shell_tol=self.shell_tol, qc_method='RHF')
                
            self.shell_slice = int_prescreen.shell_prescreen(self.mol, self.with_df.auxmol, log, shell_slice=None, 
                                                        shell_tol=self.shell_tol, qc_method='MP2')
        
        #Construct ri-mp2 3c2e integrals
        if (self.int_storage == 2) or (not self.cal_grad):
            self.t_feri_mp2 = create_timer()
        else:
            t0 = get_current_time()
            if self.mol.pbc:
                if self.int_storage == 0:
                    self.feri_node = self.RHF.feri_node
                elif self.int_storage == 1:
                    self.file_feri = self.RHF.file_feri
                self.ijp_shape = self.RHF.ijp_shape
            else:
                get_df_int3c2e(self, self.with_df, 'mp2', ijp_shape=self.ijp_shape, log=log)
                print_time(['mp2 3c2e integrals', get_elapsed_time(t0)], log)
            self.t_feri_mp2 = get_elapsed_time(t0)
        

        t1=get_current_time() 
        log.info('\nBegin calculation of (ial|P)...')

        if self.RHF.shared_int:
            if self.int_storage in {0, 3}:
                self.win_ialp = self.RHF.win_ialp
                self.ialp_node = self.RHF.ialp_node
            elif self.int_storage in {1, 2}:
                self.file_ialp = self.RHF.file_ialp
            
            loc_trans = True
        else:
            loc_trans = False

        get_df_ialp(self, self.with_df, 'mp2', log, ijp_shape=self.ijp_shape, loc_trans=loc_trans, 
                    unsort_ao=self.unsort_ialp_ao)
        if irank == 0:
            print_mem('ialp generation', self.pid_list, log)
        print_time(['ialp generation', get_elapsed_time(t1)], log)
        self.t_feri_mp2 += get_elapsed_time(t1)
    
        
        
        if self.use_gpu:
            free_cupy_mem()

    def OSV_generation(self, log):
        log.info("\n------------------------------OSV-based quantities-------------------------------")
        svd_method = "randomized SVD" if self.svd_method else "exact SVD"
        log.info("Begin OSV generation (%s)..."%svd_method)
        t1=get_current_time()
        generate_osv(self, log)
        print_time(['OSV generation', get_elapsed_time(t1)], log)

        if irank == 0:
            print_mem('OSV generation', self.pid_list, log)
        
        log.info("    " + "-"*40)
        msg_list = [["Full virtual orbitals", self.nv],
                    ['Average CP-OSVs (%.1E)'%self.cposv_tol, int(np.mean(self.nosv_cp[self.mo_list]))],
                    ['Average OSVs (%.1E)'%self.osv_tol, int(np.mean(self.nosv[self.mo_list]))]]
        print_align(msg_list, align='lr', indent=4, log=log)
        

        if self.loc_fit:
            ave_nfit = np.sum(self.nfit)/len(self.mo_list)
            #ave_nbfit = np.sum(self.nbfit)/len(self.mo_list)
            #log.info('\nAverage local fitting basis for MP2 (full %d):'%self.naoaux)
            msg_list = [["Full fitting", self.naoaux],
                        ['Average fitting (%.1E):'%self.fit_tol, int(ave_nfit)]]
                        #['Average block fitting (%.1E):'%self.fit_tol, int(ave_nbfit)]]
            log.info("    " + "-"*40)
            print_align(msg_list, align='lr', indent=4, log=log)
        log.info("    " + "-"*40)
        #log.info("%s"%self.nosv)
        #log.info("%s"%self.nfit)

        if irank == 0:
            msg_nosv = "".join([f"{i} {nosv}\n" for i, nosv in enumerate(self.nosv)])
            msg_nfit = "".join([f"{i} {nfit}\n" for i, nfit in enumerate(self.nfit)])

            with open(f"nosv_{self.molecule}_{self.basis}_%.1E.log"%self.osv_tol, 'w') as f:
                f.write(msg_nosv)
        
            with open(f"nfit_{self.molecule}_{self.basis}_%.1E.log"%self.fit_tol, 'w') as f:
                f.write(msg_nfit)

        if (sum(self.nosv)/self.nocc) == self.nv:
            self.use_cposv = False
        
        if self.use_gpu:
            free_cupy_mem()
        

    def get_sf(self, log):
        t1=get_current_time()
        log.info("\nBegin computing S and F matrices...")
        get_sf_GA(self)
        print_time(['S/F generation', get_elapsed_time(t1)], log)
        if irank == 0:
            print_mem('S/F generation', self.pid_list, log)
        
        if self.use_gpu:
            free_cupy_mem()
        
        if not self.cal_grad:
            free_win(self.win_qmat_node)
            self.qmat_node = None

    def classify_pairs(self, log):
        log.info("\nBegin pair classification...")

        self.fij_tol = 0 if self.local_type == 0 else 1e-6

        check_discarded = self.s_ratio[self.pairlist] < self.disc_tol
        self.pairlist_dicarded = self.pairlist[check_discarded]
        self.pairlist = self.pairlist[np.invert(check_discarded)]
        check_remote = self.s_ratio[self.pairlist] < self.remo_tol
        #check_remote[np.abs(self.loc_fock.ravel()[self.pairlist]) < self.fij_tol] = True
        check_close = np.invert(check_remote)

        pairs_ji = (self.pairlist%self.no) * self.no + (self.pairlist//self.no)

        self.is_close = np.zeros(self.no**2, dtype=bool)
        self.is_remote = np.zeros(self.no**2, dtype=bool)

        self.is_close[self.pairlist] = self.is_close[pairs_ji] = check_close
        self.is_remote[self.pairlist] = self.is_remote[pairs_ji] = check_remote

        self.pairlist_remote = self.pairlist[check_remote]
        self.pairlist_close = self.pairlist[check_close]

        orbs_full = np.arange(self.no)
        check_mo = np.zeros(self.no, dtype=bool)
        check_mo[self.pairlist_remote//self.no] = True
        check_mo[self.pairlist_remote%self.no] = True
        self.mo_remote = orbs_full[check_mo]
        check_mo = np.zeros(self.no, dtype=bool)
        check_mo[self.pairlist_close//self.no] = True
        check_mo[self.pairlist_close%self.no] = True
        self.mo_close = orbs_full[check_mo]

        pairs_full = np.arange(self.no**2)
        self.is_discarded = self.s_ratio < self.disc_tol
        is_kept = np.invert(self.is_discarded)
        self.pairlist_full = pairs_full[is_kept]
        #self.is_close = self.s_ratio >= self.remo_tol
        self.pairlist_close_full = pairs_full[self.is_close]
        #self.is_remote = np.copy(is_kept)
        #self.is_remote[self.pairlist_close_full] = False
        self.pairlist_remote_full = pairs_full[self.is_remote]

        self.jlist_close_full = {}#[None] * self.no
        self.jlist_close = {}
        self.jlist_remote = {}
        for i in self.mo_list:
            pairs_i = i * self.no + self.mo_list
            self.jlist_close_full[i] = self.mo_list[self.is_close[pairs_i]]

            js_tri = self.mo_list[self.mo_list >= i]
            pairs_tri = i * self.no + js_tri

            self.jlist_close[i] = js_tri[self.is_close[pairs_tri]]
            self.jlist_remote[i] = js_tri[self.is_remote[pairs_tri]]
        
        '''if self.lg_dr:
            check_offidag = (self.pairlist_close // self.no) != (self.pairlist_close % self.no)
            self.pairlist_offdiag = self.pairlist_close[check_offidag]
        else:
            self.pairlist_full = self.pairlist_close_full
            check_offidag = (self.pairlist // self.no) != (self.pairlist % self.no)
            self.pairlist_offdiag = self.pairlist[check_offidag]'''
        check_offidag = (self.pairlist_close // self.no) != (self.pairlist_close % self.no)
        self.pairlist_offdiag_close = self.pairlist_close[check_offidag]

        if len(self.mo_remote) > 0:
            check_offidag = (self.pairlist_remote // self.no) != (self.pairlist_remote % self.no)
            self.pairlist_offdiag_remote = self.pairlist_remote[check_offidag]

        self.refer_pairlist = self.pairlist

        #Get pair fitting domains
        if self.loc_fit:
            pair_slice_all = get_slice(job_list=self.pairlist, rank_list=range(nrank))
            pair_slice = pair_slice_all[irank]

            self.win_nfit_pair, self.nfit_pair = get_shared(self.no**2, dtype=np.int32, set_zeros=True)
            #self.fit_pair = [None] * self.no**2
            if self.cal_grad:
                self.win_nbfit_pair, self.nbfit_pair = get_shared(self.no**2, dtype=np.int32, set_zeros=True)
                #self.bfit_pair = [None] * self.no**2

            if pair_slice is not None:
                #fit_full = np.arange(self.naoaux, dtype=np.int32)
                for ipair in pair_slice:
                    i = ipair // self.no
                    j = ipair % self.no
                    pair_ji = j * self.no + i
                    #pidx = self.pair_indices[ipair]
                    '''check_fit = np.zeros(self.naoaux, dtype=bool)
                    if self.cal_grad:
                        check_bfit = np.zeros(self.naoaux, dtype=bool)
                    for k in [i, j]:
                        check_fit[self.fit_list[k]] = True
                        if self.cal_grad:
                            for p0, p1 in self.bfit_seg[k]:
                                check_bfit[p0:p1] = True'''
                    union_fit = np.union1d(self.fit_list[i], self.fit_list[j])
                    #self.fit_pair[ipair] = fit_full[check_fit]
                    self.nfit_pair[pair_ji] = self.nfit_pair[ipair] = len(union_fit)

                    if self.cal_grad:
                        #self.bfit_pair[ipair] = fit_full[check_bfit]                    
                        self.nbfit_pair[pair_ji] = self.nbfit_pair[ipair] = len(self.bfit_pair[ipair])

            comm.Barrier()
            if nnode > 1:
                Acc_and_get_GA(self.nfit_pair)
                if self.cal_grad:
                    Acc_and_get_GA(self.nbfit_pair)
                comm.Barrier()

            '''nfit_total = np.sum(self.nfit_pair[self.pairlist])
            self.win_fit_node, self.fit_node = get_shared(nfit_total, dtype=np.int32)#, set_zeros=True)
            fit_offsets = [None for i in range(self.no**2)]
            fit_idx0 = 0

            if self.cal_grad:
                nbift_total = np.sum(self.nbfit_pair[self.pairlist])
                self.win_bfit_node, self.bfit_node = get_shared(nbift_total, dtype=np.int32)#, set_zeros=True)           
                bfit_offsets = [None for i in range(self.no**2)]
                bfit_idx0 = 0
            for rank_i, pairs_i in enumerate(pair_slice_all):

                for ipair in pairs_i:
                    pair_ji = (ipair%self.no) * self.no + (ipair//self.no)

                    fit_idx1 = fit_idx0 + self.nfit_pair[ipair]

                    if self.cal_grad:
                        bfit_idx1 = bfit_idx0 + self.nbfit_pair[ipair]
                    if rank_i == irank:
                        self.fit_node[fit_idx0:fit_idx1] = self.fit_pair[ipair]
                        if self.cal_grad:
                            self.bfit_node[bfit_idx0:bfit_idx1] = self.bfit_pair[ipair]

                    self.fit_pair[pair_ji] = self.fit_pair[ipair] = self.fit_node[fit_idx0:fit_idx1]
                    fit_offsets[ipair] = [fit_idx0, fit_idx1]
                    fit_idx0 = fit_idx1

                    if self.cal_grad:
                        self.bfit_pair[pair_ji] = self.bfit_pair[ipair] = self.bfit_node[bfit_idx0:bfit_idx1]
                        bfit_offsets[ipair] = [bfit_idx0, bfit_idx1]
                        bfit_idx0 = bfit_idx1

            comm.Barrier()
            if nnode > 1:
                fit_node_offsets = get_node_offsets(pair_slice_all, fit_offsets)
                Get_from_other_nodes_GA(self.fit_node, fit_node_offsets)
                if self.cal_grad:
                    bfit_node_offsets = get_node_offsets(pair_slice_all, bfit_offsets)
                    Get_from_other_nodes_GA(self.bfit_node, bfit_node_offsets)
                
                comm.Barrier()'''
        

        if irank == 0:
            close_is, close_js = np.divmod(self.pairlist_close, self.no)
            close_nfits = self.nfit_pair[self.pairlist_close]
            remote_is, remote_js = np.divmod(self.pairlist_remote, self.no)
            remote_nfits = self.nfit_pair[self.pairlist_remote]

            msg_nfit_cpair = "".join([f"{i} {j} {nfit}\n" for i, j, nfit in 
                                      zip(close_is, close_js, close_nfits)])
            with open(f"nfit_close_pair_{self.molecule}_{self.basis}_%.1E.log"%self.remo_tol, 'w') as f:
                f.write(msg_nfit_cpair)
            

            msg_list = [['Pair screening threshold', '%.1E'%self.remo_tol],
                        ['Pair discarding threshold', '%.1E'%self.disc_tol],
                        ['Number of close pairs', len(self.pairlist_close)],
                        ['Number of remote pairs', len(self.pairlist_remote)],
                        ['Number of discared pairs', len(self.pairlist_dicarded)],
                        ['Average close pair fitting', int(np.mean(close_nfits))]]
            
            if len(remote_nfits) > 0:
                msg_nfit_rpair = "".join([f"{i} {j} {nfit}\n" for i, j, nfit in 
                                        zip(remote_is, remote_js, remote_nfits)])
                with open(f"nfit_remote_pair_{self.molecule}_{self.basis}_%.1E.log"%self.disc_tol, 'w') as f:
                    f.write(msg_nfit_rpair)
                msg_list.append(['Average remote pair fitting', int(np.mean(remote_nfits))])

            print_align(msg_list, align='lr', indent=4, log=log)

    def adjust_OSV(self, log):       
        msg = "\nThe number of OSVs will be fixed as %d\n"%self.nosv_ml
        msg += "    Adjusting the size of qmat and SF matrices"
        log.info(msg)
        t0=get_current_time()
        update_qmat_ml(self)
        update_sf_ml(self)
        print_time(['adjusting osv size', get_elapsed_time(t0)], log)

    def get_kmat(self, log):
        log.info("\nBegin K matrix computations...")
        get_kmatrix_GA(self)

        if self.use_gpu:
            free_cupy_mem()

    
    def get_precond(self, log):
        #Preconditioning
        log.info("\nBegin preconditioning...")

        if self.use_gpu:
            self.t_cal = create_timer()
            self.t_data = create_timer()

            self.recorder_cal = []
            self.recorder_data = []

        t0=get_current_time()
        get_precon_close(self)
        if len(self.mo_remote) > 0:
            get_precon_remote(self)
        
        if self.use_gpu:
            batch_accumulate_time(self.t_cal, self.recorder_cal)
            batch_accumulate_time(self.t_data, self.recorder_data)

            times_gpu = [['calculation', self.t_cal], ['GPU-CPU data', self.t_data]]
            times_gpu = get_max_rank_time_list(times_gpu)
            print_time(times_gpu, log)
            free_cupy_mem()
        print_time(['preconditioning', get_elapsed_time(t0)], log)

        if irank == 0:
            print_mem('preconditioning', self.pid_list, log)

    def get_mp2_ene(self, log):

        if self.method == 3:
           self.method == 4 

        if self.method == 1:
            method = 'MBE(3)-OSV-MP2 with global correction'
        elif self.method == 2:
            method = 'g-MBE(3)-OSV-MP2 without global correction'
        elif self.method == 3:
            method = 'OSV-MP2 with 2-body amplitudes'
        elif self.method == 4:
            method = 'OSV-MP2'
        log.info("\nBegin residual iterations...\n    Method: %s"%method)

        if self.use_gpu:
            self.t_cal = create_timer()
            self.t_data = create_timer()

            self.recorder_cal = []
            self.recorder_data = []

        if self.method != 4:
            t0=get_current_time()
            select_clusters(self, log)
            print_time(['cluster selection', get_elapsed_time(t0)], log)

        #self.win_tmat_node, self.tmat_node = get_shared(self.kmat_node.size, set_zeros=True)
        self.win_close_tmat_node, self.close_tmat_node = get_shared(self.close_kmat_node.size, 
                                                                    set_zeros=True)
        if len(self.pairlist_remote) > 0 and self.cal_grad:
            self.win_remote_tmat_node, self.remote_tmat_node = get_shared(self.remote_kmat_node.size)
        self.tmat_offsets_node = self.kmat_offsets_node
        self.tmat_dim = self.kmat_dim

        

        t0=get_current_time()
        if self.method == 1: #c-mbe-osv-mp2
            residualMBE(self, log)
            mp2_res = ResidualIterations(self, max_cycle=1)
            self.ene_mp2 = mp2_res.kernel()
        elif self.method == 2: #mbe-osv-mp2
            self.ene_mp2 = residualMBE(self, log)
            if irank==0: print(self.ene_mp2)
            mp2_res = ResidualIterations(self, self.pairlist_remote, max_cycle=1)
            self.ene_mp2 += mp2_res.kernel()
        elif self.method == 3: #osv-mp2
            residualMBE(self, log)
            mp2_res = ResidualIterations(self)
            self.ene_mp2 = mp2_res.kernel()
        elif self.method == 4: #original osv-mp2
            mp2_res = ResidualIterations(self)
            self.ene_mp2 = mp2_res.kernel()

        if self.use_gpu:
            batch_accumulate_time(self.t_cal, self.recorder_cal)
            batch_accumulate_time(self.t_data, self.recorder_data)
            
            times_gpu = [['calculation', self.t_cal], ['GPU-CPU data', self.t_data]]
            times_gpu = get_max_rank_time_list(times_gpu)
            print_time(times_gpu, log)
            free_cupy_mem()
        print_time(['residual iterations', get_elapsed_time(t0)], log)
        print_mem('residual iterations', self.pid_list, log)

    def kernel(self):
        log = lib.logger.Logger(self.stdout, self.verbose)
        #log.info('\n--------------------------------MP2 energy---------------------------------')
        self.t_feri_mp2 = create_timer()
        t_intopt = np.zeros_like(self.t_feri_mp2)
        if not self.mol.pbc:
            self.with_df = DF(self.mol)
            self.with_df.auxbasis = self.auxbasis_mp2
            self.with_df.auxmol = self.auxmol_mp2

        if self.ml_test and (not self.ml_mp2int):
            self.__dict__.update(self.RHF.__dict__)
            self.t_loc = create_timer()
        else:
            # Perform localization
            
            self.localization(log)

            if self.get_chk:
                sys.exit()

            if self.fully_direct:
                t0 = get_current_time()
                self.intopt = Int3c2eOpt(self.mol, self.with_df.auxmol).build(aux_group_size=512, cutoff=1e-10)
                t_intopt = get_elapsed_time(t0)
                self.t_feri_mp2 += t_intopt
                print_time(["creating int generator", t_intopt], log)
            else:
                # Get MP2 AO integrals
                self.int_trans(log)

        t0 = get_current_time()

        #initialize pairlists
        jlist, ilist = [i.ravel() for i in np.meshgrid(self.mo_list, self.mo_list)]
        sel_indices = ilist <= jlist
        self.pairlist = ilist[sel_indices] * self.no + jlist[sel_indices]
        '''loc_fock = np.abs(self.loc_fock.ravel())
        self.pairlist = self.pairlist[loc_fock[self.pairlist]>1e-5]'''

        #OSV generation
        self.OSV_generation(log)

        #Calculate S and F matrix
        self.get_sf(log)

        # Conduct pair classification        
        self.classify_pairs(log)

        # Adjust osvs for ml features
        if (self.ml_test) and (not self.ml_mp2int) and (self.nosv_ml is not None):
            self.adjust_OSV(log)

        # Compute K matrix
        self.get_kmat(log)

        #self.use_gpu = False

        # Perform preconditioning
        if (not self.ml_test) or (self.ml_mp2int):
            self.get_precond(log)

        #self.use_gpu = True

        self.t_osv_gen = get_elapsed_time(t0)

        if self.fully_direct:
            self.t_osv_gen -= (self.t_feri_mp2 - t_intopt)

        t0 = get_current_time()
        # Solve the residual equation
        self.get_mp2_ene(log)
        self.t_res = get_elapsed_time(t0)

        return self.ene_mp2
