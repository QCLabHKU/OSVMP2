import os
import shutil
import ctypes
import numbers
import h5py
import scipy
#import scipy.linalg.interpolative as sli
import numpy as np
from osvmp2.loc.loc_addons import slice_fit, get_fit_domain, get_bfit_domain
from pyscf.gto.moleintor import make_loc
from osvmp2 import int_prescreen
from osvmp2.__config__ import ngpu
from osvmp2.mpi_addons import *
from osvmp2.osvutil import *
from osvmp2.lib import randSvd
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
    from osvmp2.lib import osvMp2Cuda

    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)

    THREADS_PER_AXIS = 16


#def get_GA_slice_imba(offsets, slice_i):
def read_file(f_name, obj_name, idx0=None, idx1=None, buffer=None):
    var = None
    read_sucess = False; count = 0
    while (read_sucess == False) and (count<10):
        try:
            with h5py.File(f_name, 'r') as f:
                if idx0 is None:
                    if buffer is None:
                        var = f[obj_name][:]
                    else:
                        f[obj_name].read_direct(buffer, np.s_[:])
                elif idx1 is None:
                    if buffer is None:
                        var = f[obj_name][idx0]
                    else:
                        f[obj_name].read_direct(buffer, np.s_[idx0])
                else:
                    if buffer is None:
                        var = f[obj_name][idx0:idx1]
                    else:
                        f[obj_name].read_direct(buffer, np.s_[idx0:idx1])
                read_sucess = True
        except IOError as e:
            einfo = e
            read_sucess = False
        count += 1
    if read_sucess == False:
        raise IOError(einfo)
    if buffer is None:
        return var
    
def read_GA(offsets, slice_i, buffer, win, dtype=np.float64, list_col=None, dim_list=None, sup_dim=1, buf_idx_start=0):
    if dtype == 'f8':
        size_unit = 8
    elif dtype == 'i':
        size_unit = 4
    sup_shape = sup_dim
    sup_dim = np.prod(sup_dim)
    slice_i = list(sorted(set(slice_i)))
    buf_idx0 = buf_idx_start
    slice_kept = []
    if type(list_col) != type(None):
        for i in slice_i:
            if type(list_col[i]) == type(None):
                break
            slice_kept.append(i)
            slice_i.remove(i)
        for i in slice_kept:
            if sup_dim == 1:
                buf_idx1 = buf_idx0 + np.prod(dim_list[i])
            else:
                buf_idx1 = buf_idx0 + 1
            buf_idx0 = buf_idx1
    if slice_i != []:
        
        rank_list, seg_list, idx_list = get_GA_slice(offsets, slice_i)
        for idx, rank_i in enumerate(rank_list):
            recv_idx0, recv_idx1 = idx_list[idx]
            buf_idx1 = buf_idx0 + (recv_idx1-recv_idx0)
            win.Lock(rank_i, lock_type=MPI.LOCK_SHARED)
            win.Get(buffer[buf_idx0:buf_idx1], target_rank=rank_i, target=[recv_idx0*sup_dim*size_unit, (recv_idx1-recv_idx0)*sup_dim, MPI.DOUBLE])
            win.Unlock(rank_i)
            buf_idx0 = buf_idx1
    if type(list_col) != type(None):
        buf_idx0 = buf_idx_start
        for i in slice_kept:
            if sup_dim == 1:
                buf_idx1 = buf_idx0 + np.prod(dim_list[i])
            else:
                buf_idx1 = buf_idx0 + 1
            buf_idx0 = buf_idx1
        for idx, i in enumerate(slice_i):
            if sup_dim == 1:
                buf_idx1 = buf_idx0 + np.prod(dim_list[i])
                list_col[i] = buffer[buf_idx0:buf_idx1].reshape(dim_list[i])
            else:
                buf_idx1 = buf_idx0 + 1
                list_col[i] = buffer[buf_idx0:buf_idx1].reshape(sup_shape)
            buf_idx0 = buf_idx1
        return buffer, list_col
    else:
        return buffer

def get_buff_size(dim_list, slice_i):
    dim_sum = 0
    for i in slice_i:
        dim_sum += np.prod(dim_list[i])
    return dim_sum

def get_GA_node(self, slice_list, win_ga, offsets_ga, dim_list, len_offsets, sup_dim=1):
    if sup_dim == 1:
        buff_size = get_buff_size(dim_list, slice_list)
        win_buff, buff = get_shared(buff_size, dtype=np.float64)
    else:
        dim0 = len(slice_list)
        if type(sup_dim) == int:
            win_buff, buff = get_shared((dim0, sup_dim), dtype=np.float64)
        else:
            dim1, dim2 = sup_dim
            win_buff, buff = get_shared((dim0, dim1, dim2), dtype=np.float64)

    offsets = [None]*len_offsets
    idx0 = 0
    for i in slice_list:
        idx1 = idx0 + np.prod(dim_list[i])
        offsets[i] = [idx0, idx1]
        idx0 = idx1
    
    slice_i = get_slice(rank_list=self.shm_ranklist, job_list=slice_list)[irank_shm]
    if slice_i is not None:
        buff = read_GA(offsets_ga, slice_i, buff, win_ga, dtype=np.float64, dim_list=dim_list, sup_dim=sup_dim, buf_idx_start=offsets[slice_i[0]][0])
    comm_shm.Barrier()
    return offsets, win_buff, buff

def read_GA_node(slice_i, offsets_node, buf_node, dim_list, tmp_list=None, buff=None):
    if tmp_list is None:
        i = slice_i[0]
        recv_idx0, recv_idx1 = offsets_node[i]
        if buff is not None:
            buff[:] = buf_node[recv_idx0:recv_idx1].reshape(dim_list[i])
            return buff
        else:
            return buf_node[recv_idx0:recv_idx1].reshape(dim_list[i])
    else:
        tmp_list = [None]*len(tmp_list)
        buf_idx0 = 0
        for i in slice_i:
            buf_idx1 = buf_idx0 + np.prod(dim_list[i])
            recv_idx0, recv_idx1 = offsets_node[i]
            if buff is None:
                tmp_list[i] = buf_node[recv_idx0:recv_idx1].reshape(dim_list[i])
            else:
                buff[buf_idx0:buf_idx1] = buf_node[recv_idx0:recv_idx1]
                tmp_list[i] = buff[buf_idx0:buf_idx1].reshape(dim_list[i])
            buf_idx0 = buf_idx1
        return tmp_list



'''def get_pairslice(self):
    pair_slice_remote = get_slice(job_list=self.pairlist_remote, rank_list=range(nrank))
    pair_slice_close = get_slice(job_list=self.pairlist_close, rank_list=range(nrank))
    pair_slice = []
    for rank_i in range(nrank):
        if (pair_slice_remote[rank_i] == None) and (pair_slice_close[rank_i] == None):
            pair_slice.append(None)
        else:
            slice_i = []
            if pair_slice_remote[rank_i] is not None:
                slice_i.extend(pair_slice_remote[rank_i])
            if pair_slice_close[rank_i] is not None:
                slice_i.extend(pair_slice_close[rank_i])
            pair_slice.append(sorted(slice_i))
    return pair_slice'''

'''def get_nfit(auxmol, pairlist, atom_close, naux_close):
    from loc_addons import joint_fit_domains_by_atom
    nocc = len(atom_close)
    win_nfit, nfit = get_shared(nocc**2, dtype=np.int64)
    pair_slice = get_slice(job_list=pairlist, rank_list=range(nrank))[irank]
    if pair_slice is not None:
        for ipair in pair_slice:
            i = ipair//nocc
            j = ipair%nocc
            if i == j:
                nfit[ipair] = naux_close[i]
            else:
                nfit[ipair] = joint_fit_domains_by_atom(auxmol, [i, j], atom_close, joint_type='union')[1]
    comm.Barrier()
    Acc_and_get_GA(nfit)
    return win_nfit, nfit'''

'''def get_nfit(auxmol, pairlist, fit_list, nfit_close):
    from osvmp2.loc.loc_addons import joint_fit_domains_by_aux
    nocc = len(fit_list)
    win_nfit, nfit = get_shared(nocc**2, dtype=np.int64, set_zeros=True)
    pair_slice = get_slice(job_list=pairlist, rank_list=range(nrank))[irank]
    if pair_slice is not None:
        for ipair in pair_slice:
            i = ipair//nocc
            j = ipair%nocc
            if i == j:
                nfit[ipair] = nfit_close[i]
            else:
                nfit[ipair] = joint_fit_domains_by_aux(auxmol, [i, j], fit_list, joint_type='union')[1]
    comm.Barrier()
    Acc_and_get_GA(nfit)
    return win_nfit, nfit  '''      

def sort_pairlist(self, pairlist, is_full=False, print_info=False):
    is_remote = self.is_remote[pairlist[0]]
    nosv_list = []
    for ipair in pairlist:
        i = ipair // self.no
        j = ipair % self.no
        if is_full:
            nosv_list.append(self.nosv[j])
        else:
            #nosv_list.append(self.nosv[i]*self.nosv[j])
            nosv_list.append([self.nosv[i], self.nosv[j]])

    if self.loc_fit:
        #win_nfit, nfit = get_nfit(self.with_df.auxmol, pairlist, self.fit_list, self.nfit)

        pidx_list = [self.pair_indices[ipair] for ipair in pairlist]
        nfit = self.nfit_pair[pidx_list]
        log = lib.logger.Logger(self.stdout, self.verbose)
        if print_info:
            if is_remote:
                log.info('    Average scr pair nfit: %d'%(np.sum(nfit)/len(pairlist)))
            else:
                log.info('    Average rem pair nfit: %d'%(np.sum(nfit)/len(pairlist)))
    else:
        nfit = [self.naoaux]*len(pairlist)

    cost_list = []
    for pidx, ipair in enumerate(pairlist):
        nosv_i, nosv_j = nosv_list[pidx]
        nfit_ij = nfit[pidx]
        if is_remote:
            cost_list.append(nosv_i*nfit_ij*nosv_j)
        else:
            nosv_ij = nosv_i + nosv_j
            #cost_list.append(nosv_ij*nfit_ij*(self.nao+nosv_ij)+nfit_ij**3+2*nfit_ij**2)
            cost_virco = self.nao*self.nv*nosv_ij
            cost_trans = nosv_ij*self.nao*nfit_ij
            if (ipair//self.no) != (ipair%self.no):
                cost_trans *= 2
            cost_trans += nosv_ij*nfit_ij*nosv_ij
            cost_list.append(cost_virco+cost_trans)
    if self.loc_fit:
        comm_shm.Barrier()
        #free_win(win_nfit); nfit=None
    return [ipair for nosv, ipair in sorted(zip(cost_list, pairlist), reverse=True)]

def get_si(self, cluslist, on_node=False):
    '''rank_list = []
    for rank_i, node_i in itertools.product(range(nrank//self.nnode), range(self.nnode)):
        rank_list.append(self.rank_slice[node_i][rank_i])'''
    if on_node:
        no_rank = nrank_shm
        rank_list = range(no_rank)
    else:
        no_rank = nrank
        if nnode > 1:
            rank_list = np.asarray(self.rank_slice).T.ravel()
        else:
            rank_list = range(no_rank)

    job_slice = [None]*no_rank
    '''for idx, clus_i in enumerate(cluslist):
        rank_idx = rank_list[idx%nrank]
        if job_slice[rank_idx] == None:
            job_slice[rank_idx] = [clus_i]
        else:
            job_slice[rank_idx].append(clus_i)'''
    rank_idx = 0
    for clus_i in cluslist:
        rank_i = rank_list[rank_idx]
        if job_slice[rank_i] == None:
            job_slice[rank_i] = [clus_i]
        else:
            job_slice[rank_i].append(clus_i)
        rank_idx += 1
        if rank_idx == no_rank:
            rank_idx = 0
    return job_slice

def merge_pairslice(pair_slice_remote, pair_slice_close, sort=True):
    pair_slice = []
    for rank_i in range(nrank):
        if (pair_slice_remote[rank_i] == None) and (pair_slice_close[rank_i] == None):
            pair_slice.append(None)
        else:
            slice_i = []
            if pair_slice_remote[rank_i] is not None:
                slice_i.extend(pair_slice_remote[rank_i])
            if pair_slice_close[rank_i] is not None:
                slice_i.extend(pair_slice_close[rank_i])
            if sort:
                slice_i.sort()
            pair_slice.append(sorted(slice_i))
    return pair_slice

def get_pairslice(self, is_remote=True, is_full=False, pairlist=None, even_adjust=False, log=None):
    if even_adjust:
        if pairlist is None:
            if is_full:
                pairlist_full = sort_pairlist(self, self.pairlist_full)
                pair_slice = get_si(self, pairlist_full)
            else:
                pairlist_close = sort_pairlist(self, self.pairlist_close, print_info=True)
                if (is_remote) and (self.pairlist_remote != []):
                    pairlist_remote = sort_pairlist(self, self.pairlist_remote, print_info=True)
                    pair_slice = get_si(self, pairlist_close + pairlist_remote)
                else:
                    pair_slice = get_si(self, pairlist_close)
        else:
            pairlist = sort_pairlist(self, pairlist, print_info=True)
            pair_slice = get_si(self, pairlist)
    else:
        if pairlist is None:
            if is_full:
                pair_slice = get_slice(rank_list=range(nrank), job_list=self.refer_pairlist_full)
            else:
                pair_slice = get_slice(rank_list=range(nrank), job_list=self.refer_pairlist_close)
                if is_remote:
                    pair_slice_remote = get_slice(rank_list=range(nrank), job_list=self.refer_pairlist_remote)
                    pair_slice = merge_pairslice(pair_slice_remote, pair_slice)
        else:
            pair_slice = get_slice(rank_list=range(nrank), job_list=pairlist)
    return pair_slice


def get_read_batch(self, pairs_left, nocc_read, use_sratio=True):
    def get_pairsel(orb_select, is_select, check_valid):
        orb_select.sort()
        pairs_select = []
        orb_now = []
        for iidx, i in enumerate(orb_select):
            #for j in orb_select[iidx:]:
            for j in orb_select:
                ipair = i*self.no+j
                if (check_valid[ipair]) and (is_select[ipair] == False):
                    pairs_select.append(ipair)
                    is_select[ipair] = True
                    orb_now.extend([i, j])
        orbs_to_read = list(set(orb_now))
        return pairs_select, orbs_to_read, is_select
    def get_pairs_domain(nocc_read, mo_list, orb_distrib, is_select, check_valid):
        nocc_read = min(nocc_read, len(mo_list))
        orb_select_check = np.zeros(self.no, dtype=bool)
        pairs_collect = []
        orbs_collect = []
        orb_sel = []
        nread_collect = []
        while True:
            norb_check = []
            for i in mo_list:
                if orb_select_check[i] == False:
                    orb_distrib[i] = [j for j in orb_distrib[i] if orb_select_check[j] == False]
                    if len(orb_distrib[i]) > 0:
                        norb_check.append([len(orb_distrib[i]), i])
                else:
                    orb_distrib[i] = []
            if len(norb_check) == 0:
                if len(orb_sel) > 0:
                    pairs_sel, orb_sel, is_select = get_pairsel(orb_sel, is_select, check_valid)
                    pairs_collect.extend(pairs_sel)
                    nread_collect.append(len(orb_sel))
                    orbs_collect.extend(orb_sel)
                break
            norb_check.sort(reverse=True)
            norb_left = nocc_read - len(orb_sel)
            i_max = norb_check[0][1]
            orb_go = orb_distrib[i_max][:norb_left]
            orb_select_check[orb_go] = True
            orb_sel.extend(orb_go)
            if len(orb_sel) == nocc_read:
                pairs_sel, orb_sel, is_select = get_pairsel(orb_sel, is_select, check_valid)
                pairs_collect.extend(pairs_sel)
                nread_collect.append(len(orb_sel))
                orbs_collect.extend(orb_sel)
                orb_sel = []
                break
        pairs_collect.sort()
        orbs_collect = list(set(orbs_collect))
        orbs_collect.sort()
        return pairs_collect, orbs_collect, is_select
    
    def _get_read_batch(pairs_left):
        def orbs_from_pairs(pairlist):
            orbs_list = [[ipair//self.no, ipair%self.no] for ipair in pairlist]
            orbs_list = np.asarray(orbs_list).ravel()
            return list(set(orbs_list))
        log = lib.logger.Logger(self.stdout, self.verbose)
        is_select = [False] * self.no**2
        npairs_total = len(pairs_left)
        check_valid = np.zeros(self.no**2, dtype=bool)
        check_valid[pairs_left] = True
        orbs_read_last = np.zeros(self.no, dtype=bool)
        pairs_batch = []
        orbs_batch = []
        nread_list = []
        total_read = 0
        step = 0
        while len(pairs_left) > 0:
            mo_list = orbs_from_pairs(pairs_left)
            orb_distrib = [None] * self.no
            for i in mo_list:
                orb_distrib[i] = []
            for ipair in pairs_left:
                i = ipair // self.no
                j = ipair % self.no
                orb_distrib[i].append(j)
                if i != j:
                    orb_distrib[j].append(i)
            
            for i in mo_list:
                orb_distrib[i].append(i)
                orb_distrib[i] = set(orb_distrib[i])
                if use_sratio:
                    sr_check = [[self.s_ratio[i*self.no+j], j] for j in orb_distrib[i]]
                    sr_check.sort(reverse=True)
                    orb_distrib[i] = [j for s, j in sr_check]

            step += 1
            pairs_sel, orbs_to_read, is_select = get_pairs_domain(nocc_read, mo_list, orb_distrib, is_select, check_valid)
            pairs_batch.append(pairs_sel)
            orbs_batch.append(orbs_to_read)

            orbs_read_now = np.zeros(self.no, dtype=bool)
            orbs_read_now[orbs_to_read] = True
            orbs_exist = [i for i in orbs_to_read if orbs_read_last[i]]
            orbs_read_last = orbs_read_now
            nread = len(orbs_to_read)-len(orbs_exist)
            nread_list.append(nread)
            total_read += nread
            pairs_left = list(set(pairs_left) - set(pairs_sel))
            #log.info("norbs_read: %d, npairs_sel: %d, npairs_left: %d"%(nread, len(pairs_sel), len(pairs_left)))

        log.info("    Total paris: %d; Max nocc/batch: %d; Total reads: %d"%(npairs_total, max(nread_list), total_read))
        return pairs_batch, orbs_batch
    return _get_read_batch(pairs_left)

def get_pairslice_node(self, pair_slice, nocc_read):
    pairs_rank = [None] * nrank

    for node_i, rank_list in enumerate(self.rank_slice):
        pairs_node = []
        pair_slice_node = [pair_slice[rank_i] for rank_i in rank_list]
        for pairlist in pair_slice_node:
            if pairlist is not None:
                pairs_node.extend(pairlist)
        #pairs_batch, orbs_batch = get_read_batch(pairs_node)
        pairs_batch, orbs_batch = get_read_batch(self, pairs_node, nocc_read)
        
        pair_slice_batch = []
        
        for plist in pairs_batch:
            plist = sort_pairlist(self, plist)
            islice = get_si(self, plist, on_node=True)
            for rank_i, plist_i in enumerate(islice):
                if pairs_rank[rank_i] is None:
                    pairs_rank[rank_i] = plist_i
                elif plist_i is not None:
                    pairs_rank[rank_i] = pairs_rank[rank_i] + plist_i
            pair_slice_batch.append(islice)
        if irank//nrank_shm == node_i:
            pbatch, obatch = pair_slice_batch, orbs_batch
    for plist in pairs_rank:
        if plist is not None:
            plist.sort()
    return pbatch, obatch, pairs_rank


def get_sf_cp_GA(self):
    #pair_slice = get_slice(job_list=self.pairlist_full, rank_list=range(nrank))
    if self.int_storage == 2:
        self.dir_sf_cp = 'sf_cp_tmp'
        if irank == 0:
            os.makedirs(self.dir_sf_cp)
        comm.Barrier()
    pair_slice = get_pairslice(self, is_remote=False, is_full=True)
    self.sf_dim_cp = [None]*self.no**2
    self.sf_cp_offsets = [None]*self.no**2
    for rank_i, pair_i in enumerate(pair_slice):
        if pair_i is not None:
            sf_idx0 = 0
            for ipair in pair_i:
                i = ipair // self.no
                j = ipair % self.no
                self.sf_dim_cp[ipair] = (self.nosv_cp[i], self.nosv[j])
                sf_idx1 = sf_idx0 + self.nosv_cp[i]*self.nosv[j]
                self.sf_cp_offsets[ipair] = [rank_i, [sf_idx0, sf_idx1]]
                sf_idx0 = sf_idx1
    pair_slice = pair_slice[irank]
    if pair_slice is not None:
        dim_qmat = []
        dim_qcp = []
        dim_sf_cp = []
        for ipair in pair_slice:
            i = ipair // self.no
            j = ipair % self.no
            dim_qmat.append(self.nv*self.nosv[j])
            dim_qcp.append(self.nv*self.nosv_cp[i])
            dim_sf_cp.append(self.nosv_cp[i]*self.nosv[j])
            
        #buf_qmat = np.empty(max(dim_qmat), dtype=np.float64)
        buf_qcp = np.empty(max(dim_qcp), dtype=np.float64)
        #qmat_tmp = [None]*self.no
        if self.int_storage == 2:
            file_sf_cp = h5py.File("%s/sf_cp_%d.tmp"%(self.dir_sf_cp, irank), 'w')
            scp_save = file_sf_cp.create_dataset('scp', (sum(dim_sf_cp), ), dtype=np.float64)
            fcp_save = file_sf_cp.create_dataset('fcp', (sum(dim_sf_cp), ), dtype=np.float64)
            
        else:
            self.scp_ga = np.empty(sum(dim_sf_cp), dtype=np.float64)
            self.fcp_ga = np.copy(self.scp_ga)
        i_pre, j_pre = 0, 0
        for idx, ipair in enumerate(pair_slice):
            i = ipair // self.no
            j = ipair % self.no
            #buf_qmat, qmat_tmp = read_GA(self.qmat_offsets, [j], buf_qmat, self.win_qmat, dtype=np.float64, list_col=qmat_tmp, dim_list=self.qmat_dim)
            #qmat_tmp = buf_qmat[:self.nv*self.nosv[j]]
            #qmat_tmp = read_GA(self.qmat_offsets, [j], qmat_tmp, self.win_qmat, dtype=np.float64).reshape(self.nv, self.nosv[j])
            qidx0, qidx1 = self.qmat_offsets_node[j]
            qmat_j = self.qmat_node[qidx0:qidx1].reshape(self.qmat_dim[j])
            if (idx == 0) or (i != i_pre):
                #if self.int_storage == 2:
                #read_file('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'qcp', buffer=qcp_tmp)
                qcp_tmp = buf_qcp[:self.nv*self.nosv_cp[i]].reshape(self.nv, self.nosv_cp[i])
                read_file('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'qcp', buffer=qcp_tmp)
                '''else:
                    qcp_tmp = buf_qcp[:self.nv*self.nosv_cp[i]]
                    qcp_tmp = read_GA(self.qcp_offsets, [i], qcp_tmp, self.win_qcp, dtype=np.float64).reshape(self.nv, self.nosv_cp[i])'''
            sf_idx0, sf_idx1 = self.sf_cp_offsets[ipair][1]
            if self.int_storage == 2:
                scp_save[sf_idx0:sf_idx1] = np.dot(qcp_tmp.T, qmat_j).ravel()
                fcp_save[sf_idx0:sf_idx1] = np.dot(np.multiply(qcp_tmp.T, self.ev), qmat_j).ravel()
                #fcp_save[sf_idx0:sf_idx1] = multi_dot([qcp_tmp.T, self.ev_di, qmat_tmp]).ravel()
            else:
                self.scp_ga[sf_idx0:sf_idx1] = np.dot(qcp_tmp.T, qmat_j).ravel()
                #self.fcp_ga[sf_idx0:sf_idx1] = multi_dot([qcp_tmp.T, self.ev_di, qmat_tmp]).ravel()
                self.fcp_ga[sf_idx0:sf_idx1] = np.dot(np.multiply(qcp_tmp.T, self.ev), qmat_j).ravel()
            #qmat_tmp[j] = None
            i_pre, j_pre = i, j
        
    else:
        self.scp_ga = None
        self.fcp_ga = None
    if self.int_storage == 2:
        if pair_slice is not None:
            file_sf_cp.close()
        
    else:
        self.win_scp = create_win(self.scp_ga, comm=comm)
        self.win_fcp = create_win(self.fcp_ga, comm=comm)
        self.win_scp.Fence()
        self.win_fcp.Fence()   



def get_ijp_GA(self, mtype="mp2"):
    mo_slice = get_slice(range(nrank), job_list=self.mo_list)
    self.jlist = [None]*self.no
    for ipair in self.pairlist:
        i = ipair//self.no
        j = ipair%self.no
        if self.jlist[i] is None:
            self.jlist[i] = [j]
        else:
            self.jlist[i].append(j)
    self.ijp_offsets = [None]*self.no**2
    len_ijp_core = 0
    for rank_i, mo_i in enumerate(mo_slice):
        if mo_i is not None:
            ijp_idx = 0
            for i in mo_i:
                for j in self.jlist[i]:
                    self.ijp_offsets[i*self.no+j] = [rank_i, [ijp_idx, ijp_idx+1]]
                    ijp_idx += 1
                    if rank_i == irank:
                        len_ijp_core += 1
    
    if self.int_storage == 2:
        self.dir_ijp = 'ijp_%s_tmp'%mtype
        if irank == 0:
            os.makedirs(self.dir_ijp)
        comm.Barrier()
    mo_slice = mo_slice[irank]
    if mo_slice is not None:
        if self.int_storage == 2:
            file_ijp = h5py.File("%s/%d.tmp"%(self.dir_ijp, irank), "w")
            file_ijp.create_dataset('ijp', shape=(len_ijp_core, self.naoaux), dtype=np.float64)
            buf_ialp = np.empty((self.nao, self.naoaux))
        else:
            self.ijp_ga = np.empty((len_ijp_core, self.naoaux), dtype=np.float64)
        ijp_idx0 = 0
        for idx_i, i in enumerate(mo_slice):
            ijp_idx1 = ijp_idx0 + len(self.jlist[i])
            if self.int_storage == 2:
                file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, i)
                read_file(file_ialp, 'ialp', buffer=buf_ialp)
                ijp_i = np.dot(self.o[:, self.jlist[i]].T, buf_ialp)
                file_ijp['ijp'].write_direct(ijp_i, dest_sel=np.s_[ijp_idx0:ijp_idx1])
            else:
                ialp_tmp = self.ialp_mo[idx_i]
                self.ijp_ga[ijp_idx0:ijp_idx1] = np.dot(self.o[:, self.jlist[i]].T, ialp_tmp)
            ijp_idx0 = ijp_idx1
    else:
        self.ijp_ga = None
    comm.Barrier()
    
    if self.int_storage != 2:
        self.win_ijp = create_win(self.ijp_ga, comm=comm)
        self.win_ijp.Fence()
    




def get_gamma_GA(self):
    #if self.int_storage == 2:
    def update_cpn(idx, i, N):
        for m in range(self.nosv_cp[i]):
            for n in range(self.nosv[i]):
                delta = self.s_ga[idx][n] - self.s_ga[idx][m]
                #if (abs(delta) <= 1e-6):
                #if m == n:
                #if m <= n:
                #if (abs(delta) <= 1e-8):
                if m == n:
                    N[m, n] = np.float64(0)
                else:
                    if (abs(N[m, n]/delta) > 1):
                        N[m, n] = np.float64(0)
                    else:
                        N[m, n] = N[m, n]/delta
        '''scp_i = self.s_ga[idx, :self.nosv_cp[i]]
        s_i = self.s_ga[idx, :self.nosv[i]]
        delta = lib.direct_sum('a-b->ab', scp_i, s_i).ravel()
        delta *= -1
        idx_large = abs(delta) > 1e-6
        delta[idx_large] = 1/delta[idx_large]
        delta[np.invert(idx_large)] = 0
        N *= delta.reshape(N.shape)'''
        return N
    #Yi -> Yli, Yla, gamma_omug, PQ
    def preread(self):
        mo_slice = get_slice(range(nrank), job_list=self.mo_list)
        node_now = irank//self.nrank_shm
        rank_list = [node_now*comm_shm.size+irank_shm for irank_shm in range(comm_shm.size)]
        mo_slice_node = []
        for rank_i in rank_list:
            if mo_slice[rank_i] is not None:
                mo_slice_node.extend(mo_slice[rank_i])
        #j_list = []
        j_remote_node = []
        pairs_node = []
        for i in mo_slice_node:
            if i is not None:
                for j in self.mo_list:
                    if i < j:
                        ipair = i*self.no+j
                    else:
                        ipair = j*self.no+i
                    if self.is_discarded[ipair] == False:
                        pairs_node.append(ipair)
                        if self.is_remote[ipair]:
                            j_remote_node.append(j)
        self.win_imup_node, self.win_tmat_node = None, None

        j_remote_node = sorted(list(set(j_remote_node)))
        pairs_node = sorted(list(set(pairs_node)))
        if len(j_remote_node) > 0:
            if self.int_storage == 2:
                self.imup_offsets_node = {}
                size_imup = 0
                imup_idx0 = 0
                for j in j_remote_node:
                    isize = np.prod(self.imup_dim[j])
                    size_imup += isize
                    imup_idx1 = imup_idx0 + isize
                    self.imup_offsets_node[j] = [imup_idx0, imup_idx1]
                    imup_idx0 = imup_idx1
                self.win_imup_node, self.imup_node = get_shared(size_imup)
                jr_slice = get_slice(range(nrank_shm), job_list=j_remote_node)[irank_shm]
                if jr_slice is not None:
                    for j in jr_slice:
                        imup_idx0, imup_idx1 =  self.imup_offsets_node[j]
                        buff_imup = self.imup_node[imup_idx0:imup_idx1].reshape(self.imup_dim[j])
                        read_file(f"{self.dir_imup}/imup_{j}.tmp", "imup", buffer=buff_imup)
            else:
                self.imup_offsets_node, self.win_imup_node, self.imup_node = get_GA_node(self, j_remote_node, self.win_imup, self.imup_offsets, self.imup_dim, self.no)
        self.tmat_offsets_node, self.win_tmat_node, self.tmat_node = get_GA_node(self, pairs_node, self.win_tmat, self.tmat_offsets, self.tmat_dim, self.no**2)
        comm.Barrier()
        
    def clear_preread(self):
        for win_i in [self.win_imup_node, self.win_tmat_node]:
            if win_i is not None:
                free_win(win_i)
        self.imup_node, self.tmat_node = None, None
    
    ## NEW ##
    def transform_yi(self):
        #from osvmp2.loc.loc_addons import slice_fit, joint_fit_domains_by_atom, joint_fit_domains_by_aux
        #self.loc_fit = False
        if self.int_storage == 2:
            self.dir_yi = 'yi_tmp'
            self.dir_cal = '%s/cal'%self.dir_yi
            if irank == 0:
                for dir_i in [self.dir_yi, self.dir_cal]:#, self.dir_pq]:
                    os.makedirs(dir_i)
            comm.Barrier()
        preread(self)
        mo_slice = get_slice(range(nrank), job_list=self.mo_list)
        mo_offsets = [None]*(self.no+1)
        for rank_i, mo_i in enumerate(mo_slice):
            if mo_i is not None:
                for idx, num in enumerate(mo_i):
                    mo_offsets[num] = [rank_i, [idx, idx+1]]
        mo_slice = mo_slice[irank]
        win_lqa, Lqa_node = get_shared((self.nao, self.nao), dtype=np.float64, set_zeros=True)
        win_pq, pq_node = get_shared((naoaux, naoaux), dtype=np.float64, set_zeros=True)
        win_low, low_node = get_shared((naoaux, naoaux))
        if irank_shm == 0:
            if os.path.isfile('j2c_mp2.tmp'):
                read_file('j2c_mp2.tmp', 'low', buffer=low_node)
            else:
                j2c = self.auxmol.intor('int2c2e', hermi=1)
                low_node[:] = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
        comm_shm.Barrier()

        if mo_slice is not None:
            if irank_shm == 0:
                lqa_tmp = Lqa_node
                buf_pq = pq_node
            else:
                lqa_tmp = np.zeros((self.nao, self.nao))
                buf_pq = np.zeros((naoaux, naoaux))
            mo0, mo1 = mo_slice[0], mo_slice[-1]+1
            nocc_rank = len(mo_slice)
            #Set up buffers
            dim_qmat = []
            dim_tbar = []
            dim_imup = []
            if self.loc_fit:
                fit_ij = [None]*self.no**2
                nfit_ij = [None]*self.no**2
                dim_ialp_lfit = []
                dim_imup_lfit = []
                dim_pjij_lfit = []
                bfit_ij = [None]*self.no**2
                nbfit_ij = [None]*self.no**2
                dim_ialp_lbfit = []
                dim_imup_lbfit = []
                dim_pjij_lbfit = []

                for i in mo_slice:
                    for j in self.mo_list:
                        if i < j:
                            ipair = i*self.no+j
                        else:
                            ipair = j*self.no+i
                        if (self.is_discarded[ipair]):continue
                        if self.loc_fit:
                            pidx = self.pair_indices[ipair]
                            fit_ij[ipair] = self.fit_pair[pidx]
                            bfit_ij[ipair] = self.bfit_pair[pidx] 
                            nfit_ij[ipair] = self.nfit_pair[pidx]
                            nbfit_ij[ipair] = self.nbfit_pair[pidx]
                        if (self.is_remote[ipair]):
                            dim_imup.append(np.prod(self.imup_dim[j]))
                            '''if self.loc_fit:
                                dim_imup_lfit.append((self.nosv[i]+self.nosv[j])*nfit_ij[ipair])'''
                        elif self.loc_fit:
                            dim_ialp_lfit.append(2*self.nao*nfit_ij[ipair])
                            dim_pjij_lfit.append((self.nosv[i]+self.nosv[j])*nfit_ij[ipair])
                            dim_ialp_lbfit.append(2*self.nao*nbfit_ij[ipair])
                            dim_pjij_lbfit.append((self.nosv[i]+self.nosv[j])*nbfit_ij[ipair])
            if self.int_storage != 2:
                self.yi_mo = np.zeros((nocc_rank, self.nao, naoaux))
            if irank_shm == 0:
                DMP2 = self.DMP2
            else:
                DMP2 = self.dmp2_save
            for idx_i, i in enumerate(mo_slice):
                t0 = get_current_time()
                #Read ialp_i and Q'_i
                if self.int_storage == 2:
                    yi = np.zeros((self.nao, self.naoaux))
                else:
                    yi = self.yi_mo[idx_i]
                #Read qao_i
                qidx0, qidx1 = self.qao_offsets_node[i]
                qao_i = self.qao_node[qidx0:qidx1].reshape(self.qao_dim[i])

                #Initialse ialmunu and pjij
                j_list = []
                j_remote = []
                for j in self.mo_list:
                    if i < j:
                        ipair = i*self.no+j
                    else:
                        ipair = j*self.no+i
                    if self.is_discarded[ipair] == False:
                        j_list.append(j)
                        if self.is_remote[ipair]:
                            j_remote.append(j)
                exist_remote = False
                if len(j_remote) != 0:
                    exist_remote = True

                if len(j_remote) != 0:
                    pjij = np.zeros((naoaux, self.nosv[i]))
                if (self.use_cposv):
                    ialmunu = np.zeros((self.nao, self.nosv[i]))
                    #Read N_i
                    n_idx0, n_idx1 = self.cpn_offsets[i][1]
                    N_i = self.cpn_ga[n_idx0:n_idx1].reshape(self.cpn_dim[i])
                    accumulate_time(self.t_read, t0)
                
                tj = get_current_time()
                for j in j_list:#self.mo_list:
                    if i < j:
                        ipair = i*self.no+j
                    else:
                        ipair = j*self.no+i
                    #if self.is_discarded[ipair]:continue
                    
                    t0 = get_current_time()
                    #Read T_bar
                    tmat_tmp = read_GA_node([ipair], self.tmat_offsets_node, self.tmat_node, self.tmat_dim)

                    accumulate_time(self.t_read, t0)
                    if self.is_remote[ipair]:
                        t1 = get_current_time()
                        t0 = get_current_time()
                        #Read J_j Q_j
                        imup_tmp = read_GA_node([j], self.imup_offsets_node, self.imup_node, self.imup_dim)
                        accumulate_time(self.t_read, t0)

                        t0 = get_current_time()
                        tbar_tmp = 2*4*tmat_tmp
                        if i < j:
                            pjij += np.dot(imup_tmp, tbar_tmp.T)
                        else:
                            pjij += np.dot(imup_tmp, tbar_tmp)
                        accumulate_time(self.t_dk, t0)
                        accumulate_time(self.t_remote, t1)
                    else:
                        t1 = get_current_time()
                        t0 = get_current_time()
                        if i == j:
                            #qmat_j = qmat_i
                            qao_j = qao_i
                            jmunup = imunup = read_file(f"{self.dir_imunup}/imunup_{ipair}.tmp", 'imunup')
                            jialmunu = ijalmunu = read_file(f"{self.dir_ijalmunu}/ijalmunu_{ipair}.tmp", 'ijalmunu')
                            for imat in [imunup, ijalmunu]:
                                imat *= 4
                        else:
                            #qmat_j = read_GA_node([j], self.qmat_offsets_node, self.qmat_node, self.qmat_dim)
                            qidx0, qidx1 = self.qao_offsets_node[j]
                            qao_j = self.qao_node[qidx0:qidx1].reshape(self.qao_dim[j])
                            if i < j:
                                imunup = read_file(f"{self.dir_imunup}/imunup_{ipair}.tmp", 'imunup')
                                ijalmunu = ijalmunu = read_file(f"{self.dir_ijalmunu}/ijalmunu_{ipair}.tmp", 'ijalmunu')
                                jmunup = read_file(f"{self.dir_jmunup}/jmunup_{ipair}.tmp", 'jmunup')
                                jialmunu = read_file(f"{self.dir_jialmunu}/jialmunu_{ipair}.tmp", 'jialmunu')
                            else:
                                jmunup = read_file(f"{self.dir_imunup}/imunup_{ipair}.tmp", 'imunup')
                                jialmunu = ijalmunu = read_file(f"{self.dir_ijalmunu}/ijalmunu_{ipair}.tmp", 'ijalmunu')
                                imunup = read_file(f"{self.dir_jmunup}/jmunup_{ipair}.tmp", 'jmunup')
                                ijalmunu = read_file(f"{self.dir_jialmunu}/jialmunu_{ipair}.tmp", 'jialmunu')
                            for imat in [imunup, jmunup, ijalmunu, jialmunu]:
                                imat *= 4
                        accumulate_time(self.t_read, t0)

                        t0 = get_current_time()
                        #Compute first term of Y_i
                        if i < j:
                            #coeff = np.dot(self.v, np.concatenate((qmat_i, qmat_j), axis=1))
                            coeff = np.concatenate((qao_i, qao_j), axis=1)
                        else:
                            coeff = np.concatenate((qao_j, qao_i), axis=1)
                            #coeff = np.dot(self.v, np.concatenate((qmat_j, qmat_i), axis=1))
                        coeff *= 4
                        #Get J_j^dag(Q_i Q_j)
                        tbar_tmp = 2*tmat_tmp - tmat_tmp.T
                        if i < j:
                            pj_ij = np.dot(jmunup.T, tbar_tmp.T)
                        else:
                            pj_ij = np.dot(jmunup.T, tbar_tmp)
                                
                        
                        #Compute first term of Y_i
                        yi += np.dot(coeff/4, pj_ij.T) 
                        accumulate_time(self.t_yi, t0)

                        if (self.use_cposv):
                            #Compute derivative K
                            if i < j:
                                ialmunu += np.dot(ijalmunu, (tbar_tmp.T)[:, :self.nosv[i]])
                                ialmunu += np.dot(jialmunu, tbar_tmp[:, :self.nosv[i]])
                            else:
                                if i == j:
                                    ialmunu += 2*np.dot(ijalmunu, tbar_tmp[:, self.nosv[j]:])
                                else:
                                    ialmunu += np.dot(ijalmunu, tbar_tmp[:, self.nosv[j]:])
                                    ialmunu += np.dot(jialmunu, (tbar_tmp.T)[:, self.nosv[j]:])
                            accumulate_time(self.t_dk, t0)
                        coeff = None
                        imunup = jmunup = ijalmunu = jialmunu = None
                print_mem(f'({i}) j loop', self.pid_list, log)
                accumulate_time(self.t_jloop, tj)
                #######################################################################################

                if exist_remote:
                    t0 = get_current_time()
                    #yi += multi_dot([self.v, qmat_i, pjij.T])
                    yi += multi_dot([qao_i, pjij.T])
                    accumulate_time(self.t_yi, t0)

                if self.int_storage == 2:
                    ialp_i = read_file(f"{self.dir_ialp}/ialp_{i}.tmp", 'ialp')
                else:
                    ialp_i = self.ialp_mo[idx_i]
                if (self.use_cposv):
                    #if self.int_storage == 2:
                    t0 = get_current_time()
                    #Read qcp
                    file_qcp = f"{self.dir_qcp}/qcp_{i}.tmp"
                    qcp_i = read_file(file_qcp, 'qcp')
                    #Read qmat
                    file_qmat = f"{self.dir_qmat}/qmat_{i}.tmp"
                    qmat_i = read_file(file_qmat, 'qmat')
                    accumulate_time(self.t_read, t0)

                    t0 = get_current_time()
                    N_i += multi_dot([qcp_i.T, self.v.T, ialmunu])
                    accumulate_time(self.t_close, t1)
                    if exist_remote:
                        N_i += multi_dot([qcp_i.T, self.v.T, ialp_i, pjij])
                    N_i = update_cpn(idx_i, i, N_i)
                    eiab = lib.numpy_helper.direct_sum('a+b->ab', self.ev, self.ev)-2*self.eo[i]
                    omega = multi_dot([qcp_i, N_i, qmat_i.T])/eiab
                    accumulate_time(self.t_dk, t0)
                    accumulate_time(self.t_iloop, t0)

                    qmat_i = qcp_i = None

                    t0 = get_current_time()
                    xi = multi_dot([self.v, omega, self.v.T])
                    xi += xi.T
                    yi += np.dot(xi, ialp_i)
                    accumulate_time(self.t_yi, t0)
                    accumulate_time(self.t_iloop, t0)
                    
                    #omega += omega.T
                    #Read T_ii
                    t0 = get_current_time()
                    file_tii = f"{self.dir_tii}/ti_{i}.tmp"
                    tii = read_file(file_tii, 'tii')
                    accumulate_time(self.t_read, t0)

                    t0 = get_current_time()
                    omega *= tii
                    '''omega *= 2
                    DMP2[i, i] += np.sum(omega)
                    DMP2[self.no:, self.no:] -= omega'''
                    
                    DMP2[i, i] += 2*np.sum(omega)
                    DMP2[self.no:, self.no:] -= np.diag(np.sum(omega, axis=0) + 
                                                        np.sum(omega, axis=1))
                    accumulate_time(self.t_yi, t0)
                    accumulate_time(self.t_iloop, t0)

                    xi = eiab = omega = tii = None
                else:
                    file_tii = file_qcp = None

                t0 = get_current_time()
                #compute y_al be
                lqa_tmp += np.dot(ialp_i, yi.T)

                #compute PQ
                buf_pq += np.dot(ialp_i.T, yi)
                ialp_i = None

                scipy.linalg.solve_triangular(low_node.T, yi.T, lower=False, overwrite_b=True, 
                                              check_finite=False)
                print_mem(f'({i}) pq lqa', self.pid_list, log)
                accumulate_time(self.t_yi, t0)
                accumulate_time(self.t_iloop, t0)
                if self.int_storage == 2:
                    if file_tii is not None and self.chkfile_tii is None:
                        os.remove(file_tii)
                    if file_qcp is not None and self.chkfile_qcp is None:
                        os.remove(file_qcp)
                    t1 = get_current_time()
                    '''with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, irank), 'r+') as file_yi:
                        #file_yi['yi'][idx_i] = yi.T
                        file_yi['yi'].write_direct(yi, dest_sel=np.s_[idx_i])'''
                    with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, i), 'w') as file_yi:
                        file_yi.create_dataset("yi", data=yi)
                    accumulate_time(self.t_write, t1)
                    yi = yi.reshape(self.nao, naoaux)
            if irank_shm != 0:
                Accumulate_GA_shm(win_lqa, Lqa_node, lqa_tmp)
                Accumulate_GA_shm(win_pq, pq_node, buf_pq)
                lqa_tmp, buf_pq= None, None
        else:
            buf_pq = None
        #win_pq.Fence()
        if irank_shm != 0:
            Accumulate_GA_shm(self.win_dmp2, self.DMP2, self.dmp2_save)
        comm.Barrier()
        clear_preread(self)
        Acc_and_get_GA(var=self.DMP2)
        Acc_and_get_GA(Lqa_node)
        Accumulate_GA(var=pq_node)

        if irank_shm == 0:
            self.Yla = multi_dot([self.mo_coeff.T, Lqa_node, self.RHF.get_ovlp(), self.v])
        else:
            self.Yla = None
        comm.Barrier()
        free_win(win_lqa); Lqa_node=None; lqa_tmp=None

        t0 = get_current_time()
        if irank == 0:
            if self.int_storage == 2 and self.chkfile_ialp_mp2 is None:
                shutil.rmtree(self.dir_ialp)
            buf_pq *= 0.5
            scipy.linalg.solve_triangular(low_node.T, buf_pq.T, lower=False, overwrite_b=True, 
                                          check_finite=False)
            buf_pq = scipy.linalg.solve_triangular(low_node.T, buf_pq, lower=False,
                                                   check_finite=False).T
            buf_pq += buf_pq.T
        comm.Barrier()
        
        def save_pq(buf_pq):
            with h5py.File('pq.tmp', 'w') as file_pq:
                file_pq.create_dataset('pq', shape=(naoaux, naoaux), dtype=np.float64)
                file_pq['pq'].write_direct(buf_pq)
        if self.shared_disk:
            if irank == 0:
                save_pq(buf_pq)
        else:
            bcast_GA(buf_pq)
            if irank_shm == 0:
                save_pq(buf_pq)
        comm.Barrier()
        free_win(win_pq)
        free_win(win_low)
        print_time(['collecting PQ', get_elapsed_time(t0)], log)



    def get_pq_response(self):#, win_pq, pq_node):
        win_grad, grad_node = get_shared(len(self.atom_list)*3, dtype=np.float64, set_zeros=True)
        atom_slice = get_slice(range(nrank), job_list=self.atom_list)[irank]
        if atom_slice is not None:
            if irank_shm == 0:
                grad = grad_node
            else:
                grad = np.zeros(len(self.atom_list)*3)
            offset_atom = self.with_df.auxmol.aoslice_by_atom()
            atm0, atm1 = atom_slice[0], atom_slice[-1]
            AUX0, AUX1 = offset_atom[atm0][-2], offset_atom[atm1][-1]
            naux_list = []
            for atm_i in atom_slice:
                aux0, aux1 = offset_atom[atm_i][2:]
                naux_list.append(aux1-aux0)
            buf_PQq = np.empty(3*max(naux_list)*self.naoaux)
            buf_PQ = np.empty(((AUX1-AUX0), self.naoaux))
            with h5py.File('pq.tmp', 'r') as file_pq:
                file_pq['pq'].read_direct(buf_PQ, source_sel=np.s_[AUX0:AUX1])
            aux_idx0 = 0
            for atm_i in atom_slice:
                s0, s1, aux0, aux1 = offset_atom[atm_i]
                naux_seg = aux1 - aux0
                aux_idx1 = aux_idx0 + naux_seg
                s_slice = [s0, s1, 0, self.with_df.auxmol.nbas]
                PQq = aux_e2(auxmol=self.with_df.auxmol, intor='int2c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, hermi=0, out=buf_PQq)
                pq_tmp = buf_PQ[aux_idx0:aux_idx1]
                rank_master = (irank//comm_shm.size)*comm_shm.size
                idx0, idx1 = atm_i*3, (atm_i+1)*3
                grad[idx0:idx1] += np.dot(PQq.reshape(3,-1), pq_tmp.ravel())
                aux_idx0 = aux_idx1
            if irank_shm != 0:
                Accumulate_GA_shm(win_grad, grad_node, grad)
        comm.Barrier()
        if self.nnode > 1:
            Acc_and_get_GA(var=grad_node)
        comm_shm.Barrier()
        grad = np.copy(grad_node)
        comm_shm.Barrier()
        free_win(win_grad)
        return grad
    def collect_yi(self):
        def get_yial(self, buf_recv, mo_offsets):
            nao = self.nao
            nocc = self.no
            comm.Barrier()            
            win_yi = create_win(self.yi_mo, comm=comm)
            win_yi.Fence()
            occ_idx = [None]*(self.no+1)
            for idx, i in enumerate(list(self.mo_list)+[self.mo_list[-1]+1]):
                occ_idx[i] = idx
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                for rank_i, mo_i in mo_offsets:
                    mo0, mo1 = mo_i
                    nocc_seg = mo1 - mo0
                    idx0, idx1 = occ_idx[mo0], occ_idx[mo1]
                    dim_sup = nocc_seg*self.naoaux
                    target=[al0*dim_sup*8, nao_rank*dim_sup, MPI.DOUBLE]
                    Get_GA(win_yi, buf_recv[:nao_rank*dim_sup], target_rank=rank_i, target=target)
                    self.yi_ao[idx0:idx1] = buf_recv[:nao_rank*dim_sup].reshape(nao_rank, nocc_seg, self.naoaux).transpose(1,0,2)
            comm.Barrier()
            fence_and_free(win_yi)
        #Collect Y_i(al)
        if self.int_storage != 2:
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            nocc_list = []
            mo_offsets = []
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i is not None:
                    mo0, mo1 = mo_i[0], mo_i[-1]+1
                    nocc_list.append(len(mo_i))
                    mo_offsets.append([rank_i, [mo0, mo1]])
            idx_break = irank%len(mo_offsets)
            mo_offsets = mo_offsets[idx_break:] + mo_offsets[:idx_break]
            mo_slice = mo_slice[irank]
            if mo_slice is None:
                self.yi_mo = None
            else:
                self.yi_mo = contigous_trans(self.yi_mo.reshape(-1, self.nao, naoaux), order=(1, 0, 2))
            ao_slice = int_prescreen.get_slice_rank(self.shell_slice, aslice=True)[0][irank]
        
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                nocc_val = len(self.mo_list)
                ncore = self.no - nocc_val
                buf_recv = np.empty(nao_rank*max(nocc_list)*self.naoaux, dtype=np.float64)
                self.yi_ao = np.zeros((nocc_val, nao_rank, self.naoaux))
            else:
                buf_recv = None
                self.yi_ao = None
            get_yial(self, buf_recv, mo_offsets)
            self.yi_mo = None
            

    #Kernel
    mol = self.mol
    auxmol = self.with_df.auxmol
    ao_loc = make_loc(self.mol._bas, 'sph')
    naoaux = self.naoaux
    log = lib.logger.Logger(self.stdout, self.verbose)
    tt = get_current_time()
    t0 = get_current_time()
    self.t_jloop = create_timer()
    self.t_iloop = create_timer()
    self.t_dk = create_timer()
    self.t_yi = create_timer()
    self.t_read = create_timer()
    self.t_write = create_timer()
    self.t_slice = create_timer()
    self.t_remote = create_timer()
    self.t_close = create_timer()
    #self.opt = 1

    t0 = get_current_time()
    transform_yi(self)
    print_time(['Yi and dK', get_elapsed_time(t0)], log)

    t0 = get_current_time()
    grad = get_pq_response(self)
    print_time(['response of PQ', get_elapsed_time(t0)], log)

    t0 = get_current_time()
    collect_yi(self)
    print_time(['collection of Yi(P)', get_elapsed_time(t0)], log)


    time_list = [['j loop', self.t_jloop], ['i loop', self.t_iloop], 
                 ['dK', self.t_dk], ['Yi', self.t_yi], 
                 ['reading', self.t_read], ['writing', self.t_write]]
    if self.loc_fit:
        time_list += [['ldf slicing', self.t_slice]]
    time_list += [['Yi and dK', get_elapsed_time(tt)]]

    time_list = get_max_rank_time_list(time_list)

    if irank == 0:
        print_time(time_list, log)
        print_mem('Yi and dK', self.pid_list, log)
    comm.Barrier()
    
    if (self.int_storage == 2) and (irank == 0):
        if self.chkfile_qcp is None:
            shutil.rmtree(self.dir_qcp)
        if self.chkfile_tii is None:
            shutil.rmtree(self.dir_tii)
    return grad
    #sys.exit()

def mp2_dferi_GA(self):
    def get_seg(aux_loc, shell_seg, aux_seg):
        if len(aux_seg) < nrank:
            shell_slice = get_slice(rank_list=range(nrank), job_list=shell_seg)
            for idx, s_i in enumerate(shell_slice):
                if s_i is not None:
                    shell_slice[idx] = sorted(list(set(reduce(lambda x, y :x+y, s_i))))
        else:
            shell_slice = OptPartition(nrank, shell_seg, aux_seg)[0]
            if len(shell_slice) < nrank:
                shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
                for rank_i, s_i in enumerate(shell_slice):
                    if s_i is not None:
                        shell_slice[rank_i] = s_i[0]

        shell_slice = shell_slice[irank]
        if shell_slice is not None:
            s0, s1 = shell_slice[0], shell_slice[-1]
            aux0, aux1 = aux_loc[s0], aux_loc[s1]
            aux_slice = aux0, aux1
            aux_idx = [None]*(self.naoaux+1)
            for idx, num in enumerate(range(aux0, aux1+1)):
                aux_idx[num] = idx
        else:
            aux_slice, shell_slice, aux_idx = None, None, None
        return aux_slice, shell_slice, aux_idx
    def df_grad(index, buff_feri, yi_tmp, buff_yal, yli_tmp):
        def get_aoseg_by_atom(ao0, ao1, aoatoms):
            aolist = list(range(ao0, ao1))
            atmlist = aoatoms[ao0:ao1].tolist()
            aolist.append(ao1)
            atmlist.append(-2)
            idx_list = []
            atm_pre = -1
            for idx, atm in enumerate(atmlist):
                if atm != atm_pre:
                    idx_list.append(idx)
                atm_pre = atm
            aoslice = []
            for idx0, idx1 in zip(idx_list[:-1], idx_list[1:]):
                aoslice.append([atmlist[idx0], [aolist[idx0], aolist[idx1]]])
            return aoslice
        t0 = get_current_time()
        a0, a1, b0, b1 = index
        al0, al1, be0, be1 = [ao_loc[si] for si in index]
        nao0 = al1 - al0
        nao1 = be1 - be0
        grad = np.zeros(len(self.atom_list)*3)
        s_slice = (a0, a1, b0, b1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
        t0 = get_current_time()
        feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buff_feri)
        alpbe = feri_tmp.transpose(0, 2, 1).reshape(nao0*self.naoaux, -1)
        accumulate_time(self.t_eri, t0)

        t0 = get_current_time()
        yli_tmp[be0:be1, ncore:] += np.dot(yi_tmp, alpbe).T
        accumulate_time(self.t_yli, t0)

        for sidx, (sa0, sa1, sb0, sb1) in enumerate([[a0, a1, b0, b1], [b0, b1, a0, a1]]):
            s_slice = (sa0, sa1, sb0, sb1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
            t0 = get_current_time()
            feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
            accumulate_time(self.t_eri, t0)

            t0 = get_current_time()
            idx_list = [None]*(self.nao+1)
            if sidx == 0:
                for idx, aoi in enumerate(range(al0, al1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(al0, al1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,cbd->a', data_tmp, yal_tmp[:,idx0:idx1])
            else:
                for idx, aoi in enumerate(range(be0, be1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(be0, be1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[idx0:idx1])
            accumulate_time(self.t_gra, t0)

        s_slice = (b0, b1, a0, a1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
        t0 = get_current_time()
        feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip2_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
        accumulate_time(self.t_eri, t0)

        t0 = get_current_time()
        for atm, idx in get_aoseg_by_atom(0, self.naoaux, aux_atoms):
            p0, p1 = idx
            data_tmp = feri_tmp[:, :, :, p0:p1]
            grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[:,:,p0:p1])
        accumulate_time(self.t_gra, t0)
        return grad
    

    def collect_grad(gradient):
        if irank_shm == 0:
            win_col = create_win(gradient, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        if irank_shm == 0 and irank != 0:
            win_col.Lock(0)
            win_col.Accumulate(gradient, target_rank=0, op=MPI.SUM)
            win_col.Unlock(0)
        win_col.Fence()
        free_win(win_col)
        return gradient

    log = lib.logger.Logger(self.stdout, self.verbose)
    log.info("\nBegin MP2 derivative feri gradient...")
    tt = get_current_time()

    t1 = get_current_time()
    if self.with_df.auxmol is None:
        self.with_df.auxmol = addons.make_auxmol(self.mol, self.with_df.auxbasis)
    t0 = get_current_time()
    ao_atoms = np.zeros(self.nao, dtype=np.int64)
    for atm_i, si in enumerate(self.mol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        ao_atoms[ao0:ao1] = atm_i
    aux_atoms = np.zeros(self.naoaux, dtype=np.int64)
    for atm_i, si in enumerate(self.with_df.auxmol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        aux_atoms[ao0:ao1] = atm_i

    t0 = get_current_time()
    #Get integral slices and auxilary slices
    naoaux = self.naoaux
    ao_loc = make_loc(self.mol._bas, 'sph')
    auxmol = self.with_df.auxmol
    aux_loc = make_loc(auxmol._bas, 'sph')
    shell_seg = []
    naux_seg = []
    idx0 = idx1 = 0
    while idx1 < (len(aux_loc)-1):
        idx1 = idx0 + 1
        shell_seg.append([idx0, idx1])
        naux_seg.append(aux_loc[idx1]-aux_loc[idx0])
        idx0 = idx1
    aux_slice, shell_slice, aux_idx = get_seg(aux_loc, shell_seg, naux_seg)

    win_yli, yli_node = get_shared((self.nao, self.no), dtype=np.float64, set_zeros=True)
    win_grad, grad_node = get_shared(self.mol.natm*3, set_zeros=True)
    #Calculate gradient
    '''self.t_eri_gen = create_timer()
    self.t_eri_cal = create_timer()'''
    self.t_eri = create_timer()
    self.t_yli = create_timer()
    self.t_trans = create_timer()
    self.t_gra = create_timer()
    self.t_read = create_timer()
    if self.shell_slice is None:
        self.shell_slice = int_prescreen.shell_prescreen(self.mol, self.with_df.auxmol, log, self.shell_slice, 
                                                   self.shell_tot, meth_type='MP2')
    self.yal_type = 0
    ao_slice, shell_slice_rank = int_prescreen.get_slice_rank(self.shell_slice, aslice=True)
    max_memory = get_mem_spare(self.mol, 0.9)
    if shell_slice_rank is not None:
        if irank_shm == 0:
            grad_tmp = grad_node
        else:
            grad_tmp = np.zeros(len(self.atom_list)*3)
        atm_seg = np.arange(len(self.atom_list)*3).reshape(-1, 3)
        nocc_val = len(self.mo_list)
        ncore = self.no - nocc_val
        size_yi, size_feri, shell_slice_rank = int_prescreen.mem_control(self.mol, nocc_val, self.naoaux, shell_slice_rank, 
                                                           "derivative_feri", max_memory)
        loop_list = slice2seg(self.mol, shell_slice_rank)

        if irank_shm ==0:
            yli_tmp = yli_node
        else:
            yli_tmp = np.zeros((self.nao, self.no))
        
        '''buff_feri = np.empty((3*max(naop_list)*self.naoaux))
        buff_yal = np.empty(max(naop_list)*self.naoaux)'''
        buff_int = np.empty(size_feri)
        buff_feri = buff_int[:(size_feri*3//4)]
        buff_yal = buff_int[(size_feri*3//4):]
        if self.int_storage == 2:
            #buff_yi = np.empty((nocc_val*max(nal_list)*self.naoaux), dtype=np.float64)
            buff_yi = np.empty(size_yi, dtype=np.float64)
        else:
            buff_yi = None
        #for int_i in int_slice:
        occ_idx = [None]*(self.no+1)
        for idx, i in enumerate(list(self.mo_list) + [self.mo_list[-1]+1]):
            occ_idx[i] = idx
        ao_idx = [None]*(self.nao+1)
        al0, al1 = ao_slice[irank]
        for idx, al in enumerate(range(al0, al1+1)):
            ao_idx[al] = idx
        mo_coeff = self.o[:, ncore:]
        for a0, a1, be_list in loop_list:
            #Get y_ialp
            al0, al1 = ao_loc[a0], ao_loc[a1]
            nao0 = al1 - al0
            if self.int_storage == 2:
                t0 = get_current_time()
                yi_tmp = buff_yi[:nocc_val*nao0*self.naoaux].reshape(nocc_val, nao0, self.naoaux)
                '''mo_slice = get_slice(range(nrank), job_list=self.mo_list)
                mo_offsets = []
                for rank_i, mo_i in enumerate(mo_slice):
                    if mo_i is not None:
                        mo0, mo1 = mo_i[0], mo_i[-1]+1
                        mo_offsets.append([rank_i, [mo0, mo1]])
                idx_break = irank%len(mo_offsets)
                mo_offsets = mo_offsets[idx_break:] + mo_offsets[:idx_break]

                for rank_i, mo_i in mo_offsets:
                    mo0, mo1 = mo_i
                    idx0, idx1 = occ_idx[mo0], occ_idx[mo1]
                    with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, rank_i), 'r') as f:
                        #f['yi'].read_direct(yi_tmp, source_sel=np.s_[:, aux0:aux1], dest_sel=np.s_[mo0:mo1])
                        f['yi'].read_direct(yi_tmp[idx0:idx1], source_sel=np.s_[:, al0:al1])'''
                for iidx, i in enumerate(self.mo_list):
                    with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, i), 'r') as f:
                        #f['yi'].read_direct(yi_tmp, source_sel=np.s_[:, aux0:aux1], dest_sel=np.s_[mo0:mo1])
                        f['yi'].read_direct(yi_tmp[iidx], source_sel=np.s_[al0:al1])
                accumulate_time(self.t_read, t0)
                yi_tmp = yi_tmp.reshape(nocc_val, -1)
            else:
                idx0, idx1 = ao_idx[al0], ao_idx[al1]
                yi_tmp = self.yi_ao[:,idx0:idx1].reshape(nocc_val, -1)
            for (b0, b1) in be_list:
                be0, be1 = ao_loc[b0], ao_loc[b1]
                nao1 = be1 - be0
                t0 = get_current_time()
                yal_tmp = buff_yal[:nao1*nao0*self.naoaux].reshape(nao1, -1)
                yal_tmp = np.dot(mo_coeff[be0:be1], yi_tmp, out=yal_tmp).reshape(nao1, nao0, self.naoaux)
                accumulate_time(self.t_trans, t0)
                int_i = (a0, a1, b0, b1)
                grad_tmp += df_grad(int_i, buff_feri, yi_tmp, yal_tmp, yli_tmp)
        if self.int_storage != 2:
            self.yi_ao = None
        if irank_shm != 0:
            Accumulate_GA_shm(win_grad, grad_node, grad_tmp)
            #if self.int_storage == 2:
            Accumulate_GA_shm(win_yli, yli_node, yli_tmp)
    buff_int = None
    buff_feri = None
    buff_yal = None
    buff_yi = None
    comm.Barrier()
    
    #if self.int_storage == 2:
    Acc_and_get_GA(yli_node)
    if irank_shm == 0:
        self.Yli = np.dot(self.mo_coeff.T, yli_node)
    comm_shm.Barrier(); free_win(win_yli)
    if self.int_storage == 2 and irank == 0:
        shutil.rmtree(self.dir_yi)
    Gamma_omuG = None
    #Collect contribution of gradient from different nodes
    t_syn = get_current_time()
    if self.nnode > 1:
        Acc_and_get_GA(grad_node)
    comm_shm.Barrier()
    grad = np.copy(grad_node)
    comm_shm.Barrier()
    free_win(win_grad)
    t_syn = get_elapsed_time(t_syn)

    time_list = [['yli', self.t_yli], ['feri', self.t_eri], ['back trans', self.t_trans], ['grad', self.t_gra]]
    if self.int_storage == 2:
        time_list.append(['reading', self.t_read])
    time_list = get_max_rank_time_list(time_list)
    if irank == 0:
        print_time(time_list, log)
        print_time(['MP2 derivarive feri', get_elapsed_time(tt)], log)
        print_mem('MP2 derivarive feri', self.pid_list, log)
    return grad

def dfhf_response_ga(self, dm1, dm2, feri):#ialp, feri):
    def get_gamma_omug(dm1, dm2):
        naoaux= self.naux_hf
        win_a, A = get_shared(naoaux, dtype=np.float64, set_zeros=True)
        win_b, B = get_shared(naoaux, dtype=np.float64, set_zeros=True)
        if self.int_storage == 2:
            auxmol = self.with_df.auxmol
            naoaux = self.naux_hf
            ao_loc = make_loc(self.mol._bas, 'sph')
            if self.shell_slice is None:
                self.shell_slice = int_prescreen.shell_prescreen(self.mol, auxmol, log, shell_slice=self.shell_slice, 
                                                               shell_tot=self.shell_tot, meth_type='RHF')
            shellslice_rank = int_prescreen.get_slice_rank(self.shell_slice)
            max_memory = get_mem_spare(mol, 0.9)
            if shellslice_rank is not None:
                if irank_shm != 0:
                    A_tmp = np.zeros(naoaux)
                    B_tmp = np.zeros(naoaux)
                else:
                    A_tmp = A
                    B_tmp = B
                '''naop_list = []
                for si in shellslice_rank:
                    a0, a1, b0, b1 = si
                    naop_list.append((ao_loc[a1]-ao_loc[a0])*(ao_loc[b1]-ao_loc[b0]))
                buf_feri = np.empty(max(naop_list)*naoaux)'''
                size_feri, shellslice_rank = int_prescreen.mem_control(self.mol, self.no, naoaux, 
                                                         shellslice_rank, 0.9, max_memory)
                buf_feri = np.empty(size_feri)
                for idx, si in enumerate(shellslice_rank):
                    a0, a1, b0, b1 = si
                    al0, al1 = ao_loc[a0], ao_loc[a1]
                    be0, be1 = ao_loc[b0], ao_loc[b1]
                    nao0, nao1 = ao_loc[a1]-ao_loc[a0], ao_loc[b1]-ao_loc[b0]
                    s_slice = (a0, a1, b0, b1, self.mol.nbas, self.mol.nbas+auxmol.nbas)
                    #s_slice = (b0, b1, a0, a1, self.mol.nbas, self.mol.nbas+auxmol.nbas)
                    feri_tmp = aux_e2(self.mol, auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buf_feri)#.transpose(1,0,2)
                    dm1_tmp = dm1[al0:al1, be0:be1].ravel()
                    dm2_tmp = dm2[al0:al1, be0:be1].ravel()
                    A_tmp += np.dot(dm1_tmp, feri_tmp.reshape(-1, naoaux))
                    B_tmp += np.dot(dm2_tmp, feri_tmp.reshape(-1, naoaux))
                if irank_shm != 0:
                    Accumulate_GA_shm(win_a, A, A_tmp)
                    Accumulate_GA_shm(win_b, B, B_tmp)
            comm_shm.Barrier()
        else:
            #aux_slice = get_slice(range(nrank), job_size=self.naoaux)[irank]
            aux_slice = get_auxshell_slice(self.with_df.auxmol)[0][irank]
            if aux_slice is not None:
                feri_buffer_unpack = np.empty((self.nao, self.nao))
                if (self.int_storage == 1):
                    with h5py.File(self.feri_aux, 'r') as feri_aux:
                        for idx, num in enumerate(aux_slice):
                            lib.numpy_helper.unpack_tril(np.asarray(feri_aux['j3c'][idx]), out=feri_buffer_unpack)
                            A[num] = np.dot(feri_buffer_unpack.ravel(), dm1.ravel())
                            B[num] = np.dot(feri_buffer_unpack.ravel(), dm2.ravel())
                else:
                    for idx, num in enumerate(aux_slice):
                        lib.numpy_helper.unpack_tril(self.feri_aux[idx], out=feri_buffer_unpack)
                        A[num] = np.dot(feri_buffer_unpack.ravel(), dm1.ravel())
                        B[num] = np.dot(feri_buffer_unpack.ravel(), dm2.ravel())
                feri_buffer_unpack = None
                self.feri_aux = None
        comm.Barrier()
        
        Acc_and_get_GA(A)
        Acc_and_get_GA(B)
        if irank_shm == 0:
            if self.int_storage == 2:
                with h5py.File('j2c_hf.tmp', 'r') as f:
                    j2c = np.asarray(f['j2c'])
                    scipy.linalg.solve(j2c, A, overwrite_b=True)
                    scipy.linalg.solve(j2c, B, overwrite_b=True)
            else:
                with h5py.File('j2c_hf.tmp', 'r') as f:
                    low = np.asarray(f['low'])
                    scipy.linalg.solve_triangular(low.T, A, lower=False, overwrite_b=True, 
                                                  check_finite=False)
                    scipy.linalg.solve_triangular(low.T, B, lower=False, overwrite_b=True, 
                                                  check_finite=False)
        
        return win_a, A, win_b, B

    def get_seg(aux_loc, shell_seg, aux_seg):
        if len(aux_seg) < nrank:
            shell_slice = get_slice(rank_list=range(nrank), job_list=shell_seg)
            for idx, s_i in enumerate(shell_slice):
                if s_i is not None:
                    shell_slice[idx] = sorted(list(set(reduce(lambda x, y :x+y, s_i))))
        else:
            shell_slice = OptPartition(nrank, shell_seg, aux_seg)[0]
            if len(shell_slice) < nrank:
                shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
                for rank_i, s_i in enumerate(shell_slice):
                    if s_i is not None:
                        shell_slice[rank_i] = s_i[0]
        shell_slice = shell_slice[irank]
        if shell_slice is not None:
            s0, s1 = shell_slice[0], shell_slice[-1]
            aux0, aux1 = aux_loc[s0], aux_loc[s1]
            aux_slice = aux0, aux1
            aux_idx = [None]*(self.naoaux+1)
            for idx, num in enumerate(range(aux0, aux1+1)):
                aux_idx[num] = idx
        else:
            aux_slice, shell_slice, aux_idx = None, None, None
        return aux_slice, shell_slice, aux_idx

    def get_aoseg_by_atom(ao0, ao1, aoatoms):
        aolist = list(range(ao0, ao1))
        atmlist = aoatoms[ao0:ao1].tolist()
        aolist.append(ao1)
        atmlist.append(-2)
        idx_list = []
        atm_pre = -1
        for idx, atm in enumerate(atmlist):
            if atm != atm_pre:
                idx_list.append(idx)
            atm_pre = atm
        aoslice = []
        for idx0, idx1 in zip(idx_list[:-1], idx_list[1:]):
            aoslice.append([atmlist[idx0], [aolist[idx0], aolist[idx1]]])
        return aoslice

    def df_grad(index, buff_feri, yal_tmp):
        a0, a1, b0, b1 = index
        al0, al1, be0, be1 = [ao_loc[s] for s in index]
        nao0 = al1 - al0
        nao1 = be1 - be0
        grad = np.zeros(len(self.atom_list)*3)
        
        for sidx, (sa0, sa1, sb0, sb1) in enumerate([[a0, a1, b0, b1], [b0, b1, a0, a1]]):
            s_slice = (sa0, sa1, sb0, sb1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
            t0 = get_current_time()
            feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
            accumulate_time(self.t_eri, t0)

            t0 = get_current_time()
            idx_list = [None]*(self.nao+1)
            if sidx == 0:
                for idx, aoi in enumerate(range(al0, al1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(al0, al1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,cbd->a', data_tmp, yal_tmp[:,idx0:idx1])
            else:
                for idx, aoi in enumerate(range(be0, be1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(be0, be1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[idx0:idx1])
            accumulate_time(self.t_gra, t0)

        s_slice = (b0, b1, a0, a1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
        t0 = get_current_time()
        feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip2_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
        accumulate_time(self.t_eri, t0)

        t0 = get_current_time()
        for atm, idx in get_aoseg_by_atom(0, self.naoaux, aux_atoms):
            p0, p1 = idx
            data_tmp = feri_tmp[:, :, :, p0:p1]
            grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[:,:,p0:p1])
        accumulate_time(self.t_gra, t0)
        return grad

    def collect_grad(gradient):
        if irank_shm == 0:
            win_col = create_win(gradient, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        if irank_shm == 0 and irank != 0:
            win_col.Lock(0)
            win_col.Accumulate(gradient, target_rank=0, op=MPI.SUM)
            win_col.Unlock(0)
        win_col.Fence()
        free_win(win_col)
        return gradient

    tt = get_current_time()
    t1 = get_current_time()
    self.naoaux = self.naux_hf
    self.atom_list = range(self.mol.natm)
    if self.with_df.auxmol is None:
        self.with_df.auxmol = addons.make_auxmol(self.mol, self.with_df.auxbasis)
    log = lib.logger.Logger(self.stdout, self.verbose)
    nocc = self.no
    nao = self.nao
    naoaux = self.naux_hf
    ao_loc = make_loc(self.mol._bas, 'sph')
    mol = self.mol
    auxmol = self.with_df.auxmol
    aux_loc = make_loc(auxmol._bas, 'sph')
    ao_atoms = np.zeros(self.nao, dtype=np.int64)
    for atm_i, si in enumerate(self.mol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        ao_atoms[ao0:ao1] = atm_i
    aux_atoms = np.zeros(self.naoaux, dtype=np.int64)
    for atm_i, si in enumerate(self.with_df.auxmol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        aux_atoms[ao0:ao1] = atm_i
    atm_seg = np.arange(len(self.atom_list)*3).reshape(-1, 3)
    #if self.int_storage == 2:
    def transform_yi(self, A, B):
        mo_slice = get_slice(range(nrank), job_size=self.no)
        mo_offsets = [None]*self.no
        for rank_i, mo_i in enumerate(mo_slice):
            if mo_i is not None:
                for idx, i in enumerate(mo_i):
                    mo_offsets[i] = [rank_i, [idx, idx+1]]
        if self.int_storage == 2:
            self.dir_yi = "yi_hf_tmp"
            self.dir_cal = '%s/cal'%self.dir_yi
            if irank == 0:
                for dir_i in [self.dir_yi, self.dir_cal]:
                    os.makedirs(dir_i)
            comm.Barrier()
        mo_slice = mo_slice[irank]
        win_pq, pq_node = get_shared((naoaux, naoaux), set_zeros=True)
        win_low, low_node = get_shared((naoaux, naoaux))
        if irank_shm == 0:
            #read_file('j2c_hf.tmp', 'low_inv', buffer=low_node)
            read_file('j2c_hf.tmp', 'low', buffer=low_node)
        comm_shm.Barrier()
        if mo_slice is not None:
            if irank_shm == 0:
                buf_pq = pq_node
            else:
                buf_pq = np.zeros((naoaux, naoaux))
            nocc_rank = len(mo_slice)
            #occ_coeff = self.mo_coeff[:,:self.no]
            occ_coeff = self.o
            if self.int_storage == 2:
                file_yi = h5py.File('%s/yi_%d.tmp'%(self.dir_yi, irank), 'w')
                #file_yi.create_dataset('yi', shape=(nocc_rank, naoaux, nao), dtype=np.float64)
                file_yi.create_dataset('yi', shape=(nocc_rank, nao, naoaux), dtype=np.float64)
                yi = np.empty((self.nao, naoaux))
                ialp_i = np.empty((self.nao, naoaux))
            else:
                self.yi_mo = np.empty((nocc_rank, self.nao, naoaux))
                
            t_feri = create_timer()
            for idx_i, i in enumerate(mo_slice):
                #Read ialp_i
                if self.int_storage == 2:
                    file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, i)
                    read_file(file_ialp, 'ialp', buffer=ialp_i)
                    #os.remove(file_ialp)
                else:
                    ialp_i = self.ialp_mo[idx_i]
                    yi = self.yi_mo[idx_i]
                #Compute yi
                np.dot(dm2, ialp_i, out=yi)
                
                #compute PQ
                buf_pq -= np.dot(ialp_i.T, yi)
                scipy.linalg.solve_triangular(low_node.T, yi.T, lower=False, overwrite_b=True, 
                                              check_finite=False)
                #Save yi
                if self.int_storage == 2:
                    file_yi['yi'].write_direct(yi, dest_sel=np.s_[idx_i])
            if irank_shm != 0:
                Accumulate_GA_shm(win_pq, pq_node, buf_pq)
                buf_pq = None
        else:
            buf_pq = None
        comm.Barrier()
        Accumulate_GA(var=pq_node)
        if irank == 0:
            scipy.linalg.solve_triangular(low_node.T, buf_pq.T, lower=False, overwrite_b=True, 
                                          check_finite=False)
            buf_pq = scipy.linalg.solve_triangular(low_node.T, buf_pq, lower=False,
                                                   check_finite=False).T
            buf_pq += np.dot(A.reshape(-1, 1), B.reshape(1, -1))#np.einsum('i, j->ij', A, B)
            buf_pq += buf_pq.T
            with h5py.File('pq.tmp', 'w') as file_pq:
                file_pq.create_dataset('pq', shape=(naoaux, naoaux), dtype=np.float64)
                file_pq['pq'].write_direct(buf_pq)
        comm.Barrier()
        free_win(win_pq)
        free_win(win_low)


    def get_pq_response(self):#, pq_node):
        win_grad, grad_node = get_shared(len(self.atom_list)*3, dtype=np.float64, set_zeros=True)
        atom_slice = get_slice(range(nrank), job_list=self.atom_list)[irank]
        if atom_slice is not None:
            if irank_shm == 0:
                grad = grad_node
            else:
                grad = np.zeros(len(self.atom_list)*3)
            offset_atom = self.with_df.auxmol.aoslice_by_atom()
            atm0, atm1 = atom_slice[0], atom_slice[-1]
            AUX0, AUX1 = offset_atom[atm0][-2], offset_atom[atm1][-1]
            naux_list = []
            for atm_i in atom_slice:
                aux0, aux1 = offset_atom[atm_i][2:]
                naux_list.append(aux1-aux0)
            buf_PQq = np.empty(3*max(naux_list)*naoaux)
            buf_PQ = np.empty(((AUX1-AUX0), naoaux))
            with h5py.File('pq.tmp', 'r') as file_pq:
                file_pq['pq'].read_direct(buf_PQ, source_sel=np.s_[AUX0:AUX1])
            aux_idx0 = 0
            for atm_i in atom_slice:
                s0, s1, aux0, aux1 = offset_atom[atm_i]
                naux_seg = aux1 - aux0
                aux_idx1 = aux_idx0 + naux_seg
                s_slice = [s0, s1, 0, self.with_df.auxmol.nbas]
                PQq = aux_e2(auxmol=self.with_df.auxmol, intor='int2c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, hermi=0, out=buf_PQq)
                pq_tmp = buf_PQ[aux_idx0:aux_idx1]
                idx0, idx1 = atm_i*3, (atm_i+1)*3
                grad[idx0:idx1] += np.dot(PQq.reshape(3,-1), pq_tmp.ravel())
                aux_idx0 = aux_idx1
            if (irank_shm != 0):
                Accumulate_GA_shm(win_grad, grad_node, grad)
        comm.Barrier()
        return win_grad, grad_node
    #Collect Yi
    def collect_yi(self):
        def get_yial(self, buf_recv, mo_offsets):
            nao = self.nao
            nocc = self.no
            comm.Barrier()            
            win_yi = create_win(self.yi_mo, comm=comm)
            win_yi.Fence()
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                for rank_i, mo_i in mo_offsets:
                    mo0, mo1 = mo_i
                    nocc_seg = mo1 - mo0
                    dim_sup = nocc_seg*self.naoaux
                    target=[al0*dim_sup*8, nao_rank*dim_sup, MPI.DOUBLE]
                    Get_GA(win_yi, buf_recv[:nao_rank*dim_sup], target_rank=rank_i, target=target)
                    self.yi_ao[mo0:mo1] = buf_recv[:nao_rank*dim_sup].reshape(nao_rank, nocc_seg, self.naoaux).transpose(1,0,2)
            comm.Barrier()
            fence_and_free(win_yi)
        #Collect Y_i(al)
        if self.int_storage != 2:
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            nocc_list = []
            mo_offsets = []
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i is not None:
                    mo0, mo1 = mo_i[0], mo_i[-1]+1
                    nocc_list.append(len(mo_i))
                    mo_offsets.append([rank_i, [mo0, mo1]])
            idx_break = irank%len(mo_offsets)
            mo_offsets = mo_offsets[idx_break:] + mo_offsets[:idx_break]
            mo_slice = mo_slice[irank]
            if mo_slice is None:
                self.yi_mo = None
            else:
                self.yi_mo = contigous_trans(self.yi_mo.reshape(-1, self.nao, naoaux), order=(1, 0, 2))
            ao_slice = int_prescreen.get_slice_rank(self.shell_slice, aslice=True)[0][irank]
        
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                buf_recv = np.empty(nao_rank*max(nocc_list)*self.naoaux, dtype=np.float64)
                self.yi_ao = np.zeros((self.no, nao_rank, self.naoaux))
            else:
                buf_recv = None
                self.yi_ao = None
            get_yial(self, buf_recv, mo_offsets)
            self.yi_mo = None

    t0 = get_current_time()
    win_a, A, win_b, B = get_gamma_omug(dm1, dm2)
    print_time(['fitting term A and B', get_elapsed_time(t0)], log)

    #win_pq, pq_node = transform_yi(self, A, B)
    t0 = get_current_time()
    transform_yi(self, A, B)
    collect_yi(self)
    print_time(['Yi', get_elapsed_time(t0)], log)

    t0 = get_current_time()
    win_grad, grad_node = get_pq_response(self)#, pq_node)
    print_time(['gradient of (P|Q)', get_elapsed_time(t0)], log)

    
    shell_seg = []
    naux_seg = []
    for s0 in range(auxmol.nbas):
        s1 = s0 + 1
        shell_seg.append([s0, s1])
        naux_seg.append(aux_loc[s1]-aux_loc[s0])
    #aux_slice, shell_slice, aux_idx = get_seg(aux_loc, shell_seg, naux_seg)
    aux_slice, aux_offsets, shell_slice = get_auxshell_slice(auxmol)
    aux_slice = aux_slice[irank]
    if aux_slice is not None:
        aux_idx = [None]*(naoaux+1)
        for idx, p in enumerate(aux_slice + [aux_slice[-1]+1]):
            aux_idx[p] = idx
    shell_slice = shell_slice[irank]
    t_omug = get_elapsed_time(t1)
    ao_slice, shell_slice_rank = int_prescreen.get_slice_rank(self.shell_slice, aslice=True)
    max_memory = get_mem_spare(mol, 0.9)
    if shell_slice_rank is not None:
        size_yi, size_feri, shell_slice_rank = int_prescreen.mem_control(self.mol, self.no, self.naoaux, 
                                                           shell_slice_rank, "derivative_feri", max_memory)
        loop_list = slice2seg(self.mol, shell_slice_rank)
        buff_int = np.empty(size_feri)
        buff_feri = buff_int[:(size_feri*3//4)]
        buff_yal = buff_int[(size_feri*3//4):]
        if self.int_storage == 2:
            #buff_yi = np.empty((self.no*max(nal_list)*self.naoaux), dtype=np.float64)
            buff_yi = np.empty(size_yi, dtype=np.float64)
        else:
            buff_yi = None
        ao_idx = [None]*(self.nao+1)
        al0, al1 = ao_slice[irank]
        for idx, al in enumerate(range(al0, al1+1)):
            ao_idx[al] = idx
        #Calculate gradient
        atm_seg = np.arange(len(self.atom_list)*3).reshape(-1, 3)
        grad_tmp = np.zeros(len(self.atom_list)*3)
        mo_coeff = -2*self.o

        self.t_eri = create_timer()
        self.t_trans = create_timer()
        self.t_gra = create_timer()
        self.t_read = create_timer()

        
        for a0, a1, be_list in loop_list:
            #Get y_ialp
            al0, al1 = ao_loc[a0], ao_loc[a1]
            nao0 = al1 - al0
            if self.int_storage == 2:
                t0 = get_current_time()
                yi_tmp = buff_yi[:self.no*nao0*self.naoaux].reshape(self.no, nao0, self.naoaux)
                mo_slice = get_slice(range(nrank), job_list=range(self.no))
                mo_offsets = []
                for rank_i, mo_i in enumerate(mo_slice):
                    if mo_i is not None:
                        mo0, mo1 = mo_i[0], mo_i[-1]+1
                        mo_offsets.append([rank_i, [mo0, mo1]])
                idx_break = irank%len(mo_offsets)
                mo_offsets = mo_offsets[idx_break:] + mo_offsets[:idx_break]

                for rank_i, mo_i in mo_offsets:
                    mo0, mo1 = mo_i
                    with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, rank_i), 'r') as f:
                        f['yi'].read_direct(yi_tmp[mo0:mo1], source_sel=np.s_[:, al0:al1])
                accumulate_time(self.t_read, t0)
                yi_tmp = yi_tmp.reshape(self.no, -1)
            else:
                idx0, idx1 = ao_idx[al0], ao_idx[al1]
                yi_tmp = self.yi_ao[:,idx0:idx1].reshape(self.no, -1)
            for (b0, b1) in be_list:
                be0, be1 = ao_loc[b0], ao_loc[b1]
                nao1 = be1 - be0
                t0 = get_current_time()
                yal_tmp = buff_yal[:nao1*nao0*self.naoaux].reshape(nao1, -1)
                yal_tmp = np.dot(mo_coeff[be0:be1], yi_tmp, out=yal_tmp).reshape(-1, self.naoaux)
                yal_tmp += np.dot(dm2[be0:be1, al0:al1].reshape(-1,1), A.reshape(1,-1))
                yal_tmp += np.dot(dm1[be0:be1, al0:al1].reshape(-1,1), B.reshape(1,-1))
                yal_tmp = yal_tmp.reshape(nao1, nao0, self.naoaux)
                accumulate_time(self.t_trans, t0)
                int_i = (a0, a1, b0, b1)
                grad_tmp += df_grad(int_i, buff_feri, yal_tmp)
        if self.int_storage != 2:
            self.yi_ao = None
        
        Accumulate_GA_shm(win_grad, grad_node, grad_tmp)
    buff_int = None
    buff_feri = None
    buff_yal = None
    buff_yi = None
    comm.Barrier()
    
    for win in [win_a, win_b]:
        free_win(win)
    A, B, Gamma_omuG = None, None, None
    
    #Collect contribution of gradient from different nodes
    t_syn = get_current_time()
    if self.nnode > 1:
        #gradient = collect_grad(gradient)
        Acc_and_get_GA(var=grad_node)
    comm_shm.Barrier()
    grad = np.copy(grad_node)
    comm_shm.Barrier()
    free_win(win_grad)
    t_syn = get_elapsed_time(t_syn)

    time_list = [['feri', self.t_eri], ['back trans', self.t_trans], ['grad', self.t_gra]]
    if self.int_storage == 2:
        time_list.append(['reading', self.t_read])
    time_list = get_max_rank_time_list(time_list)

    if irank == 0:
        print_time(time_list, log)
        #print_time(['RHF gradient', get_elapsed_time(tt)], log)
        if self.int_storage == 2:
            shutil.rmtree(self.dir_yi)
            if self.chkfile_ialp_hf is None:
                shutil.rmtree(self.dir_ialp)
    return grad