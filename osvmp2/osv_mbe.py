import os
import numpy as np
from pyscf import lib
from osvmp2.__config__ import ngpu
from osvmp2.mpi_addons import *
from osvmp2.osvutil import *
import mpi4py
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
    from osvmp2.gpu.mp2_ene_cuda import mbe_residual_iter_cuda

    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)

    THREADS_PER_AXIS = 16

def select_clusters(self, log):

    log.info("\n    OSV many body expansion")
    log.info('    ---------------------------------------------')

    # Select 1b clusters
    self.oneb_clusters = np.copy(self.mo_list).reshape(-1, 1)
    self.oneb_pairs = (self.mo_list * self.no + self.mo_list).reshape(-1, 1)

    # Select close 2b clusters
    n2b_close = len(self.pairlist_offdiag_close)
    self.twobc_clusters = np.empty((n2b_close, 2), dtype=np.int64)
    self.twobc_pairs = np.empty((n2b_close, 3), dtype=np.int64)
    ilist_close = self.pairlist_offdiag_close // self.no
    jlist_close = self.pairlist_offdiag_close  % self.no
    self.twobc_clusters[:, 0] = ilist_close 
    self.twobc_clusters[:, 1] = jlist_close 
    self.twobc_pairs[:, 0] = ilist_close * self.no + ilist_close
    self.twobc_pairs[:, 1] = ilist_close * self.no + jlist_close
    self.twobc_pairs[:, 2] = jlist_close * self.no + jlist_close

    if len(self.mo_remote) > 0:
        n2b_remote = len(self.pairlist_offdiag_remote)
        self.twobr_clusters = np.empty((n2b_remote, 2), dtype=np.int64)
        self.twobr_pairs = np.empty((n2b_remote, 3), dtype=np.int64)
        ilist_remote = self.pairlist_offdiag_remote // self.no
        jlist_remote = self.pairlist_offdiag_remote  % self.no
        self.twobr_clusters[:, 0] = ilist_remote 
        self.twobr_clusters[:, 1] = jlist_remote 
        self.twobr_pairs[:, 0] = ilist_remote * self.no + ilist_remote
        self.twobr_pairs[:, 1] = ilist_remote * self.no + jlist_remote
        self.twobr_pairs[:, 2] = jlist_remote * self.no + jlist_remote
    else:
        self.twobr_clusters = []

    # Select 3b clusters
    nocc_core = self.no - len(self.mo_list)
    if self.method in [1, 2]:
        self.threeb_clusters = []
        for ipair in self.pairlist_close:
            i = ipair // self.no
            j = ipair % self.no
            if i == j:
                continue
            max_orb = max(i, j)
            if max_orb == self.no - 1:
                continue
            kidx0 = max_orb - nocc_core + 1
            for k in self.mo_list[kidx0:]:
                is_sel = True
                for l in [i, j]:
                    lk = l * self.no + k
                    #if self.is_remote[lk] or self.is_discarded[lk]:
                    if not self.is_close[lk]:
                        is_sel = False
                        break
                if is_sel:
                    ik = i * self.no + k
                    jk = j * self.no + k
                    sijk = np.sort(self.s_ratio[[ipair, ik, jk]])
                    if (np.mean(sijk) > self.threeb_tol) or (sijk[1] > self.threeb_tol):
                        self.threeb_clusters.append([i, j, k])

        self.threeb_clusters = np.asarray(self.threeb_clusters)
        ilist_threb = self.threeb_clusters[:, 0]
        jlist_threb = self.threeb_clusters[:, 1]
        klist_threb = self.threeb_clusters[:, 2]
        self.threeb_pairs = np.empty((len(self.threeb_clusters), 6), dtype=int)
        list_col = [ilist_threb, jlist_threb, klist_threb]
        col_idx = 0
        for idx, list0 in enumerate(list_col):
            for list1 in list_col[idx:]:
                self.threeb_pairs[:, col_idx] = list0 * self.no + list1
                col_idx += 1
    else:
        self.threeb_clusters = []
        self.threeb_pairs = []

    # Get counts of clusters for MBE
    self.oneb_counts = np.zeros(self.no**2, dtype=int)
    self.twob_counts_diag = np.zeros(self.no**2, dtype=int)
    self.twob_counts_offdiag = np.zeros(self.no**2, dtype=int)

    self.oneb_counts[self.mo_list * self.no + self.mo_list] = 1

    # 2b: 1b dT_ii(ij) =  T_ii(ij) - T_ii(i)
    twob_mo, twob_counts_mo = np.unique(self.twobc_clusters.ravel(), return_counts=True)
    np.subtract.at(self.oneb_counts, twob_mo*self.no+twob_mo, twob_counts_mo)

    self.twob_counts_diag[self.pairlist_offdiag_close] = 1
    self.twob_counts_offdiag[self.pairlist_offdiag_close] = 1
    if len(self.mo_remote) > 0:
        self.twob_counts_offdiag[self.pairlist_offdiag_remote] = 1

    if len(self.threeb_clusters) > 0:
        # 3b: 1b dT_ii(ijk) =  T_ii(ijk) - T_ii(ij) - T_ii(ik) + T_ii(i)
        threeb_mo, threeb_counts_mo = np.unique(self.threeb_clusters.ravel(), return_counts=True)
        np.add.at(self.oneb_counts, threeb_mo*self.no+threeb_mo, threeb_counts_mo)
        
        threeb_poffdiag, theeb_counts_poffdiag = np.unique(self.threeb_pairs[:, [1, 2, 4]].ravel(), return_counts=True)
        np.subtract.at(self.twob_counts_diag, threeb_poffdiag, theeb_counts_poffdiag)

        # 3b: 2b dT_ij(ijk) =  T_ij(ijk) - T_ij(ij)
        np.subtract.at(self.twob_counts_offdiag, threeb_poffdiag, theeb_counts_poffdiag)


    msg_list = [['Number of 1-Body clusters', len(self.oneb_clusters)]]
    print_align(msg_list, align='lr', indent=4, log=log)
    log.info('    ---------------------------------------------')
    l2bc = len(self.twobc_clusters)
    l2br = len(self.twobr_clusters)
    l2b = l2bc + l2br
    msg_list = [['2B selection threshold', '%.1e'%self.remo_tol],
                ['Number of 2-Body clusters', l2b], 
                ['Number of close 2-Body clusters', l2bc], 
                ['Number of remote 2-Body clusters', l2br]]
    print_align(msg_list, align='lr', indent=4, log=log)
    if (self.threeb_tol != 1) and (len(self.threeb_clusters) > 0):
        log.info('    -----------------------------------------------')
        msg_list = [['3B selection threshold', '%.1e'%self.threeb_tol],
                    ['Number of 3-Body clusters', len(self.threeb_clusters)]]
        print_align(msg_list, align='lr', indent=4, log=log)
    log.info('    -----------------------------------------------')





#MP2 iterations    
class MBEResidualIterations():
    def __init__(self, mp2, cluster, pairs=None, ene_tol=1e-8, use_tinit=False, 
                 use_dynt=True, k_tol=1e-5, max_cycle=None):
        self.mp2 = mp2
        self.cluster = cluster

        if pairs is None:
            self.pairs = np.array([i * mp2.no + j for i in cluster for j in cluster if i <= j])
        else:
            self.pairs = pairs

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

        self.pair_ene = [None] * mp2.no**2
        self.log = lib.logger.Logger(mp2.stdout, mp2.verbose)

    def load_mat(self):
        mp2 = self.mp2

        # Get OSV matrices from shared memory
        self.S_matrix = [None] * mp2.no**2
        self.F_matrix = [None] * mp2.no**2
        self.K_matrix = [None] * mp2.no**2
        self.X_matrix = [None] * mp2.no**2
        self.emu_ij = [None] * mp2.no**2
        self.T_matrix = [None] * mp2.no**2
        size_tmat = np.sum([np.prod(mp2.tmat_dim[ipair]) for ipair in self.pairs])
        tmat_buffer = np.zeros(size_tmat)

        tidx0 = 0
        for ipair in self.pairs:
            #self.K_matrix[ipair] = getMatFromNode(ipair, mp2.kmat_node, mp2.kmat_offsets_node, mp2.kmat_dim)
            self.K_matrix[ipair] = getMatFromNode(ipair, mp2.close_kmat_node, mp2.kmat_offsets_node, mp2.kmat_dim)
            self.X_matrix[ipair] = getMatFromNode(ipair, mp2.xmat_node, mp2.xmat_offsets_node, mp2.xmat_dim)
            self.emu_ij[ipair] = getMatFromNode(ipair, mp2.emuij_node, mp2.emuij_offsets_node, mp2.emuij_dim)
            self.S_matrix[ipair] = getMatFromNode(ipair, mp2.smat_node, mp2.sf_offsets_node, mp2.sf_dim)
            self.F_matrix[ipair] = getMatFromNode(ipair, mp2.fmat_node, mp2.sf_offsets_node, mp2.sf_dim)

            dim0, dim1 = mp2.tmat_dim[ipair]
            tidx1 = tidx0 + dim0 * dim1
            self.T_matrix[ipair] = tmat_buffer[tidx0:tidx1].reshape(dim0, dim1)

            if self.use_tinit:
                np.copyto(self.T_matrix[ipair], getMatFromNode(ipair, mp2.tmat_save, mp2.tmat_offsets_node, mp2.tmat_dim))

            tidx0 = tidx1

    def get_rmat(self, ipair):
        mp2 = self.mp2
        i = ipair // mp2.no
        j = ipair % mp2.no
        rmat = np.copy(self.K_matrix[ipair])

        F_ijij = generation_SuperMat([i, j, i, j], self.F_matrix, mp2.nosv, mp2.no)
        for k in self.cluster:
            if abs(mp2.loc_fock[k, j]) > self.k_tol:
                S_ikij = generation_SuperMat([i, k, i, j], self.S_matrix, mp2.nosv, mp2.no)

                B = -mp2.loc_fock[k, j] * S_ikij
                if (k==j):
                    B += F_ijij
                
                if i > k:
                    T_ik = flip_ij(k, i, self.T_matrix[k*mp2.no+i], mp2.nosv)
                else:
                    T_ik = self.T_matrix[i*mp2.no+k]

                rmat += multi_dot([S_ikij.T, T_ik, B])
            if abs(mp2.loc_fock[i, k]) > self.k_tol:
                S_ijkj = generation_SuperMat([i, j, k, j], self.S_matrix, mp2.nosv, mp2.no)
                C = -mp2.loc_fock[i, k] * S_ijkj
                if (i==k):
                    C += F_ijij
                if k > j:
                    T_kj = flip_ij(j, k, self.T_matrix[j*mp2.no+k], mp2.nosv)
                else:
                    T_kj = self.T_matrix[k*mp2.no+j]

                rmat += multi_dot([C, T_kj, S_ijkj.T])
        return rmat

    

    def kernel(self):
        self.load_mat()
        mp2 = self.mp2

        #diis is not needed for clusters
        ene_old = 0.0
        for ite in range(self.max_cycle):
            ene_new = 0.0
            
            #Dynamical amplitudes update
            if not self.use_dynt:
                R_matrix = [None] * mp2.no**2
                for ipair in self.pairs:
                    R_matrix[ipair] = self.get_rmat(ipair)

            for ipair in self.pairs:
                if self.use_dynt:
                    rmat_ij = self.get_rmat(ipair)
                else:
                    rmat_ij = R_matrix[ipair]

                effective_R = self.emu_ij[ipair] * multi_dot([self.X_matrix[ipair].T, rmat_ij, self.X_matrix[ipair]])
                delta = multi_dot([self.X_matrix[ipair], effective_R, self.X_matrix[ipair].T])
                self.T_matrix[ipair] += delta

                T_bar_ij = 2 * self.T_matrix[ipair] - self.T_matrix[ipair].T
                ene_ij = np.dot(self.K_matrix[ipair].ravel(), T_bar_ij.ravel())
                if (ipair // mp2.no) != (ipair % mp2.no):
                    ene_ij *= 2
                ene_new += ene_ij
                self.pair_ene[ipair] = ene_ij

            var = abs(ene_old - ene_new)
            ene_old = ene_new
            
            #self.log.info("    Iter. %d: energy %.10f, by increment %.2E"%(ite+1, ene_new, var))

            #converged or not
            if (var < self.ene_tol):
                break

        return ene_new
    
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



def save_tmat_batch(self, pairs_batch, tmat_batch):

    offsets_node = np.asarray([self.tmat_offsets_node[ipair] for ipair in pairs_batch])
    offsets_node = merge_intervals(offsets_node)

    #self.win_tmat_node.Lock(0)
    self.win_close_tmat_node.Lock(0)
    local_tidx0 = 0
    for tidx0, tidx1 in offsets_node:
        local_tidx1 = local_tidx0 + (tidx1 - tidx0)
        #self.tmat_node[tidx0: tidx1] += tmat_batch[local_tidx0: local_tidx1]
        self.close_tmat_node[tidx0: tidx1] += tmat_batch[local_tidx0: local_tidx1]
        local_tidx0 = local_tidx1
    #self.win_tmat_node.Unlock(0)
    self.win_close_tmat_node.Unlock(0)

def clusters_to_pairs(clusters, nocc):
    nmo_cluster = clusters.shape[1]
    if nmo_cluster == 1:
        return clusters * (nocc + 1)
    else:
        rows, cols = np.triu_indices(nmo_cluster)
        return clusters[:, rows] * nocc + clusters[:, cols]

def batch_clusters(clusters, ncluster_batch):
    
    clusters = np.asarray(clusters)
    total_ncluster = len(clusters)
    nmo_cluster = clusters.shape[1]

    if nmo_cluster == 2:
        clusters = clusters[np.lexsort((clusters[:, 1], clusters[:, 0]))]
    elif nmo_cluster == 3:
        clusters = clusters[np.lexsort((clusters[:, 2], clusters[:, 1], clusters[:, 0]))]

    batches = []

    for idx0 in np.arange(total_ncluster, step=ncluster_batch):
        idx1 = min(idx0+ncluster_batch, total_ncluster)
        batches.append(clusters[idx0: idx1])
    
    return batches

def residual_cluster(self, clusters, ene_tol, cidx_slice=None, use_tinit=False):
    nmo = len(clusters[0])
    nclus = len(clusters)

    max_cycle = self.max_cycle
    if cidx_slice is None:
        nosvs_mos = [self.nosv[clusters[:, idx]] for idx in range(nmo)]
        weight_list = np.zeros(nclus, dtype=int)
        for iidx in range(nmo):
            for jidx in range(nmo)[iidx:]:
                weight_list += (nosvs_mos[iidx] + nosvs_mos[jidx])**3
        cidx_slice = get_slice(range(nrank), job_size=nclus, weight_list=weight_list, 
                               sort=True, in_rank_sort=True)[irank]

    if cidx_slice is not None:
        clusters_slice = clusters[cidx_slice]
        npair_cluster = (nmo * (nmo + 1) // 2)
        max_nosv = np.max(self.nosv)
        max_memory = get_mem_spare(self.mol, 0.8)
        total_ncluster = len(cidx_slice)
        max_tmat_size = npair_cluster * 4 * max_nosv**2
        pair_ene_rank = np.zeros(self.no**2)

        if self.use_gpu:
            max_ncluster_gpu = get_ncols_for_memory(0.6*self.gpu_memory, max_tmat_size, total_ncluster)
            max_memory -= max_ncluster_gpu * max_tmat_size * 8 * 1e-6
            mbe_res_kern_cuda = mbe_residual_iter_cuda(self, nmo, cidx_slice, pair_ene_rank, ene_tol)
        else:
            max_ncluster_gpu = 0
        
        max_ncluster = get_ncols_for_memory(max_memory, max_tmat_size, total_ncluster)
        max_ncluster = max(max_ncluster, max_ncluster_gpu)

        cluster_batches = batch_clusters(clusters_slice, max_ncluster)

        tmat_sizesum_batch = np.empty(self.no**2, dtype=int)
        
        for clusters_batch in cluster_batches:
            pairs_clus_batch = clusters_to_pairs(clusters_batch, self.no)

            if nmo > 1:
                uniq_pairs_batch = np.unique(pairs_clus_batch.ravel())            
                ilist_batch, jlist_batch = np.divmod(uniq_pairs_batch, self.no)
                tmat_size_batch = (self.nosv[ilist_batch] + self.nosv[jlist_batch])**2
                tmat_sizesum_batch[uniq_pairs_batch] = np.cumsum(tmat_size_batch)
                tmat_batch = np.zeros(tmat_sizesum_batch[uniq_pairs_batch[-1]])
                #tmat_batch = np.zeros(np.sum(tmat_size_batch))
            else:
                uniq_pairs_batch = pairs_clus_batch.ravel()
                tmat_batch = None
                tmat_sizesum_batch = None

            if self.use_gpu:
                mbe_res_kern_cuda(clusters_batch, pairs_clus_batch, tmat_batch, tmat_sizesum_batch)
            else:
                
                #for cidx in cidx_slice:
                for cluster, pairlist in zip(clusters_batch, pairs_clus_batch):

                    res = MBEResidualIterations(self, cluster, pairlist, ene_tol=ene_tol, 
                                            use_tinit=use_tinit, max_cycle=max_cycle)
                    res.kernel()

                    save_mat_mbe(self, nmo, pairlist, res.T_matrix, res.pair_ene, tmat_batch, 
                                pair_ene_rank, tmat_sizesum_batch)

            
            #self.tmat_check[uniq_pairs_batch] = True

            not_to_save = (nmo == 1) or (self.method == 3) or (self.method == 2 and nmo == 3 and not self.cal_grad)
            if not not_to_save:
                save_tmat_batch(self, uniq_pairs_batch, tmat_batch)
            
        Accumulate_GA_shm(self.win_pairene_node, self.pairene_node, pair_ene_rank)
    comm.Barrier()
    '''if nnode > 1:
        Acc_and_get_GA(self.tmat_save)
        Acc_and_get_GA(self.tmat_check)
        comm.Barrier()'''
    
    



def residualMBE(self, log=None):

    if log is None:
        log = lib.logger.Logger(self.stdout, self.verbose)
    
    if self.method in [1, 2]:
        etol_2b = etol_3b = self.ene_tol
    elif self.method == 3:
        etol_2b = etol_3b = 1e-4

    use_tsave = True #(self.method != 2 or self.cal_grad)

    # Compute mp2 residual equation
    self.win_pairene_node, self.pairene_node = get_shared(self.no**2, set_zeros=True)
    #self.win_tmat_check, self.tmat_check = get_shared(self.no**2, dtype=bool, set_zeros=True)
    if use_tsave:
        #self.win_tmat_save, self.tmat_save = get_shared(self.kmat_node.size, set_zeros=True)
        self.win_tmat_save, self.tmat_save = get_shared(self.close_kmat_node.size)
    else:
        #self.win_tmat_save, self.tmat_save = self.win_tmat_node, self.tmat_node
        self.win_tmat_save, self.tmat_save = self.win_close_tmat_node, self.close_tmat_node

    if use_tsave:
        close_k_costs = self.kmat_dim[self.pairlist_close][:, 0]**2
        close_pair_slices = get_slice(range(nrank), job_list=self.pairlist_close, weight_list=close_k_costs)
        close_pair_slice = np.asarray(close_pair_slices[irank])
        pair_indices = np.empty(self.no**2, dtype=np.int32)
        pair_indices[self.oneb_pairs.ravel()] = np.arange(len(self.oneb_pairs), dtype=np.int32)
        pair_indices[self.twobc_pairs[:, 1]] = np.arange(len(self.twobc_clusters), dtype=np.int32)

        mask_diag = close_pair_slice//self.no == close_pair_slice%self.no
        cidx_oneb_slice = pair_indices[close_pair_slice[mask_diag]]
        cidx_twob_slice = pair_indices[close_pair_slice[np.invert(mask_diag)]]
    else:
        cidx_oneb_slice = cidx_twob_slice = None

    # One-body clusters
    t0 = get_current_time()
    residual_cluster(self, self.oneb_clusters, 1e-1, cidx_slice=cidx_oneb_slice)
    print_time(['one-body energy', get_elapsed_time(t0)], log)

    # Close two-body clusters
    t0 = get_current_time()
    residual_cluster(self, self.twobc_clusters, etol_2b, cidx_slice=cidx_twob_slice)
    print_time(['close two-body energy', get_elapsed_time(t0)], log)

    if use_tsave:
        tsave_node_offsets = get_node_offsets(close_pair_slices, self.tmat_offsets_node)
        Get_from_other_nodes_GA(self.tmat_save, tsave_node_offsets)

    # Three-body clusters
    if len(self.threeb_clusters) > 0:
        t0 = get_current_time()
        residual_cluster(self, self.threeb_clusters, etol_3b, use_tinit=True)
        print_time(['three-body energy', get_elapsed_time(t0)], log)


    comm.Barrier()
    Acc_and_get_GA(self.pairene_node)
    #Acc_and_get_GA(self.tmat_node)
    Acc_and_get_GA(self.close_tmat_node)
    comm.Barrier()

    if self.method == 2:
        self.ene_mp2 = np.sum(self.pairene_node)
        comm.Barrier()
    else:
        self.ene_mp2 = 0.0

    #DO SOMETHING
    wins = [self.win_pairene_node]#, self.win_tmat_check]
    if use_tsave:
        wins.append(self.win_tmat_save)
        self.tmat_save = None
    for win in wins:
        free_win(win)
    self.pairene_node = None #self.tmat_check = None

    return self.ene_mp2
    