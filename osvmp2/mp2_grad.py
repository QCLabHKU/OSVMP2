import os
import time
import shutil
import h5py
import numpy as np
from pyscf import lib, ao2mo, scf, gto, lib, df, grad
from pyscf.lib import logger
from pyscf.df.incore import *
from osvmp2.mm.solvation import get_grad_sol
from osvmp2 import hf_grad, osv_grad, int_prescreen
from osvmp2.osvutil import *
from osvmp2.ga_addons import *
from osvmp2.loc.loc_addons import *
from osvmp2.loc.CPL_meta import solve_CPL
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nnode = nrank/comm_shm.size
ztype = 1
def get_T(self, T, i, j):
    ij = i*self.no+j
    ji = j*self.no+i
    if (self.is_remote[i*self.no+j]):
        if i > j:
            T_ji = T[ji]
            T_ij = T_ji.T
        else:
            T_ij = T[ij]
    else:
        if i > j:
            T_ji = T[ji][:self.nosv[j], self.nosv[j]:]
            T_ij = T_ji.T
        else:
            T_ij = T[ij][:self.nosv[i], self.nosv[i]:]
    return T_ij

def get_DM_ij(self, ipair, T_matrix, T_bar, DMP2):
    def check_redundant(self, i, j, k):
        def get_pair(i, j):
            if i < j:
                return i*self.no+j
            else:
                return j*self.no+i
        if (i == k) or (j == k):
            return False
        else:
            is_red = False
            pairs = [get_pair(i, k), get_pair(j, k)]
            for ipair in pairs:
                if self.is_remote[ipair] or self.is_discarded[ipair]:
                    is_red = True
                    break
            return is_red
    i = ipair//self.no
    j = ipair % self.no
    D_ij = 0
    mo_list = self.mo_list
    if self.use_mbe:
        #if self.mbe_mode == 0 or self.mbe_mode == 2:
        if self.clus_type == 20:
            if i == j:
                mo_list = [k for k in self.mo_list if k != i]
        elif self.clus_type == 3:
            mo_list = [k for k in self.mo_list if (k != i) and (k != j)]
    for k in mo_list:
        #print(self.mo_list, i, j, k)
        if check_redundant(self, i, j, k):continue
        if (self.is_remote[ipair]):
            A = multi_dot([get_T(self, T_matrix, k, i), self.S_matrix[i*self.no+j]])
            D_ij += ddot(A.ravel(), get_T(self, T_bar, k, j).ravel())
            B = multi_dot([get_T(self, T_matrix, k, j), self.S_matrix[j*self.no+i]])
            D_ij += ddot(B.ravel(), get_T(self, T_bar, k, i).ravel())
        else:
            if k > i:
                ik = i*self.no+k
                T_matrix_ki = flip_ij(i, k, T_matrix[ik], self.nosv)
                T_bar_ki = flip_ij(i, k, T_bar[ik], self.nosv)
            else:
                ki = k*self.no+i 
                T_matrix_ki = T_matrix[ki]
                T_bar_ki = T_bar[ki]
            if k > j:
                jk = j*self.no+k
                T_matrix_kj = flip_ij(j, k, T_matrix[jk], self.nosv)
                T_bar_kj = flip_ij(j, k, T_bar[jk], self.nosv)
            else:
                kj = k*self.no+j
                T_matrix_kj = T_matrix[kj]
                T_bar_kj = T_bar[kj]
            S_kikj = generation_SuperMat([k, i, k, j], self.S_matrix, self.nosv, self.no)
            S_kjki = S_kikj.T
            if i == j:
                #D_ij += 2*np.trace(multi_dot([S_kikj, T_matrix_kj, S_kjki, T_bar_ki.T]))
                D_ij += 2*np.dot(multi_dot([S_kikj, T_matrix_kj, S_kjki]).ravel(), T_bar_ki.ravel())
            else:
                #D_ij += np.trace(multi_dot([S_kjki, T_matrix_ki, S_kikj, T_bar_kj.T]))
                #D_ij += np.trace(multi_dot([S_kikj, T_matrix_kj, S_kjki, T_bar_ki.T]))
                D_ij += np.dot(multi_dot([S_kikj, T_matrix_kj, S_kjki]).ravel(), T_bar_ki.ravel())
                D_ij += np.dot(multi_dot([S_kjki, T_matrix_ki, S_kikj]).ravel(), T_bar_kj.ravel())
    if (i == j):
        DMP2[i, j] -= D_ij
    else:
        DMP2[i, j] -= D_ij
        DMP2[j, i] -= D_ij
    return DMP2

def get_DM_ab(self, ipair, T_matrix, T_bar, DMP2):
    i = ipair//self.no
    j = ipair % self.no
    if self.is_remote[ipair]:
        result1 = ddot(T_matrix[ipair], T_bar[ipair].T)
        result2 = ddot(T_bar[ipair].T, T_matrix[ipair])
        if (i != j):
            result1 = 2*result1
            result2 = 2*result2
        DMP2[self.no:, self.no:] += multi_dot([self.Q_matrix[i], result1, self.Q_matrix[i].T]) + \
                                    multi_dot([self.Q_matrix[j], result2, self.Q_matrix[j].T])
    else:
        S_ijij = generation_SuperMat([i, j, i, j], self.S_matrix, self.nosv, self.no)
        result = multi_dot([T_matrix[ipair], S_ijij, T_bar[ipair].T])
        result += multi_dot([T_bar[ipair].T, S_ijij, T_matrix[ipair]])
        if (i != j):
            result = 2*result
        try:
            coeff = np.concatenate((self.Q_matrix[i], self.Q_matrix[j]), axis=1)
        except ValueError:
            print(self.clus_type, self.Q_matrix[i], self.Q_matrix[j])
        DMP2[self.no:, self.no:] += multi_dot([coeff, result, coeff.T])
    return DMP2

#MO-basis
def MO_basis_A(self, T_matrix):
    t_g_mo = get_current_time()
    T_bar = [None]*self.no**2
    for ipair in self.pairlist:
        if (self.is_remote[ipair]):
            T_bar[ipair] = 2 * T_matrix[ipair]
        else:
            T_bar[ipair] = 2 * T_matrix[ipair] - T_matrix[ipair].T
    log = lib.logger.Logger(self.stdout, self.verbose)
    tt = get_current_time()
    log.info("\n----------------------Begin the gradient calculation in MO basis--------------------------")
    if len(self.pairlist_remote) > 0:
        self.lg_correct = True
    else:
        self.lg_correct = False
    #Compute osv_grad residue
    if (self.use_osv_grad):
        if self.use_mbe == False:
            def get_cp(ipair):
                def cal_cp(i, j, ipair):
                    S_matrix_cp[ipair] = ddot(self.Q_matrix_cp[i].T, self.Q_matrix[j])
                    #F_matrix_cp[ipair] = multi_dot([self.Q_matrix_cp[i].T, self.ev_di, self.Q_matrix[j]])
                    F_matrix_cp[ipair] = ddot(np.multiply(self.Q_matrix_cp[i].T, self.ev), self.Q_matrix[j])
                i = ipair // self.no
                j = ipair % self.no
                pair_ji = j*self.no+i
                cal_cp(i, j, ipair)
                if i != j:
                    cal_cp(j, i, pair_ji)
            def get_tco(num):
                self.Q_ao_cp[num] = ddot(self.v, self.Q_matrix_cp[num])
            self.Q_ao_cp = [None]*self.no
            S_matrix_cp = [None]*self.no**2
            F_matrix_cp = [None]*self.no**2
            #self.mo_list = range(self.no)
            for ipair in self.pairlist:#self.pairlist_close:
                get_cp(ipair)
            for num in range(self.no):
                get_tco(num)
        else:
            S_matrix_cp = self.S_matrix_cp
            F_matrix_cp = self.F_matrix_cp
            
        t0 = get_current_time()
        if self.lg_dr:
            self.lg_correct = True
        else:
            self.lg_correct = False
        N = osv_grad.derivative_R(self, T_matrix, T_bar, S_matrix_cp, F_matrix_cp)
        self.lg_correct = True
        print_time(['derivative residue', get_elapsed_time(t0)], log)
    else:
        N = None
    t0 = get_current_time()
    #Calculate DM_ij, DM_ab 
    
    def DM_to_AO(self, DMP2, T_bar):
        def get_dm(ipair, DMP2):
            t0 = get_current_time()
            DMP2 = get_DM_ij(self, ipair, T_matrix, T_bar, DMP2)
            t_dm_ij = get_elapsed_time(t0)

            t0 = get_current_time()
            DMP2 = get_DM_ab(self, ipair, T_matrix, T_bar, DMP2)
            t_dm_ab = get_elapsed_time(t0)
            if (self.is_remote[ipair]):
                self.t_ij_remote += t_dm_ij
                self.t_ab_remote += t_dm_ab
            else:
                self.t_ij_close += t_dm_ij
                self.t_ab_close += t_dm_ab
            return DMP2
        
        self.t_ij_close, self.t_ij_remote = create_timer(), create_timer()
        self.t_ab_close, self.t_ab_remote = create_timer(), create_timer()

        if self.use_mbe:
            if self.clus_type == 1:
                for ipair in self.pairlist:
                    DMP2 = get_dm(ipair, DMP2)
            elif self.clus_type == 21:
                DMP2 = get_dm(self.pairlist[1], DMP2)
            elif self.clus_type == 20:
                DMP2 = get_DM_ab(self, self.pairlist[1], T_matrix, T_bar, DMP2)
                for ipair in self.pairlist:
                    DMP2 = get_DM_ij(self, ipair, T_matrix, T_bar, DMP2)
            elif self.clus_type == 3:
                for ipair in self.pairlist:
                    if ipair//self.no != ipair%self.no:
                        DMP2 = get_DM_ij(self, ipair, T_matrix, T_bar, DMP2)
        else:
            if (self.lg_correct):
                for ipair in self.pairlist:
                    DMP2 = get_dm(ipair, DMP2)
            else:
                for ipair in self.pairlist_close:
                    DMP2 = get_dm(ipair, DMP2)


        time_list = [['DM_ij for close pairs', self.t_ij_close], ['DM_ij for remote pairs', self.t_ij_remote], 
                     ['DM_ab for close pairs', self.t_ab_close], ['DM_ab for remote pairs', self.t_ab_remote]]
        time_list = get_max_rank_time_list(time_list)
        print_time(time_list, log)
        return DMP2
    DMP2 = np.zeros((self.nao, self.nao))
    DMP2 = DM_to_AO(self, DMP2, T_bar)
    print_time(['Density matrix calculation done', get_elapsed_time(t0)], log)
    print_time(['OSV gradient in MO-basis finished', get_elapsed_time(tt)], log)
    self.t_g_mo = get_elapsed_time(t_g_mo)
    #return Gamma_ialp, DMP2
    #return Gamma_ialp, N, DMP2
    return T_bar, N, DMP2

def MO_basis_B(self, T_bar=None, N=None, Gamma_ialp=None):
    Gamma_ialp, N = get_gamma_GA(self, T_bar, N, Gamma_ialp)
    return Gamma_ialp, N

#From here, AO BASIS!!!!!!!!
def AO_basis(self):
    def get_grad_idx():
        lst = []
        idex = []
        for i in self.atom_list:
            lst.append(self.mol.offset_nr_by_atom()[i])
            idex.append(3*i)
            idex.append(3*i+1)
            idex.append(3*i+2)
        return lst, idex    
    def get_cpl_coef():
        theta = ddot(self.loc_fock, self.DMP2[:self.no, :self.no])-ddot(
                self.DMP2[:self.no, :self.no], self.loc_fock)+ddot(self.uo.T, self.Yli[:self.no])
        if (self.nocc_core is None):
            list = range(self.no)
        else:
            list = range(self.nocc_core, self.no)
            list2 = range(self.nocc_core)
            aloc1, S_loc1 = solve_CPL(self, theta, list=list2)
        aloc, S_loc = solve_CPL(self, theta, list=list)
        return aloc, S_loc
    def update_DMP2():
        if irank_shm == 0:
            if (self.uo is not None):#and (self.loc_fit == False):
                self.Yli = ddot(self.Yli, self.uo.T)
                self.DMP2[:self.no, :self.no] = multi_dot([self.uo, self.DMP2[:self.no, :self.no], self.uo.T])
            dm_hf = scf.hf.make_rdm1(self.mo_coeff, self.mo_occ)
            self.dm_mo = np.copy(self.DMP2)
        win_dm_ur, dm_unrelaxed = get_shared((self.nao, self.nao))
        if irank_shm == 0:
            dm_unrelaxed[:] = dm_hf
            dm_unrelaxed += multi_dot([self.mo_coeff, self.DMP2, self.mo_coeff.T])
        comm_shm.Barrier()
        if self.pop_uremp2:
            analysis(self.mol, dm_unrelaxed, 'unrelaxed MP2', charge_method=self.charge_method, save_data='unrelaxed_MP2', log=log)
        free_win(win_dm_ur)
        if irank_shm == 0:
            if (self.use_cpl):
                if (self.local_type == 0):
                    theta = ddot(self.loc_fock, self.DMP2[:self.no, :self.no])-ddot(
                        self.DMP2[:self.no, :self.no], self.loc_fock)+self.Yli[:self.no]
                    for i in range(0, self.no):
                        for j in range(0, i):
                            if (i == j):
                                continue
                            else:
                                if (abs(self.eo[j]-self.eo[i]) < 1.0e-7):
                                    continue
                                else:
                                    self.DMP2[i, j] += theta[i, j]/(self.eo[j]-self.eo[i])
                                    self.DMP2[j, i] += theta[j, i]/(self.eo[i]-self.eo[j])

                elif (self.local_type == 1):
                    theta = ddot(np.diag(self.eo), self.DMP2[:self.no, :self.no])-ddot(
                        self.DMP2[:self.no, :self.no], np.diag(self.eo))+self.Yli[:self.no]
                    if (self.nocc_core is not None):
                        for i in range(0, self.nocc_core):
                            for j in range(self.nocc_core, self.no):
                                if (i == j):
                                    continue
                                else:
                                    if (abs(self.eo[j]-self.eo[i]) < 1.0e-7):
                                        continue
                                    else:
                                        self.DMP2[i, j] += theta[i, j]/(self.eo[j]-self.eo[i])
                                        self.DMP2[j, i] += theta[j, i]/(self.eo[i]-self.eo[j])
            if ztype == 1:
                if (self.use_osv_grad is not False):# and 1==0:
                    D_ab = self.DMP2[self.no:, self.no:]
                    f_ab = lib.direct_sum('a-b->ab', self.ev, self.ev)
                    theta = f_ab*D_ab + self.Yla[self.no:]
                    y_ab = self.Yla[self.no:]
                    for i in range(0, self.nv):
                        for j in range(0, i):
                            if (i == j):
                                continue
                            else:
                                if (abs(self.ev[j]-self.ev[i]) < 1.0e-7):
                                    continue
                                else:
                                    a = self.no+i
                                    b = self.no+j
                                    self.DMP2[a, b] += theta[i, j] / (self.ev[j]-self.ev[i])
                                    self.DMP2[b, a] += theta[j, i] / (self.ev[i]-self.ev[j])
                    '''theta = ddot(np.diag(self.ev), self.DMP2[self.no:, self.no:])-ddot(
                    self.DMP2[self.no:, self.no:], np.diag(self.ev))+self.Yla[self.no:]
                    for i in range(0, self.nv):
                        for j in range(0, i):
                            if (i == j):
                                continue
                            else:
                                if (abs(self.ev[j]-self.ev[i]) < 1e-7):
                                    continue
                                else:
                                    self.DMP2[self.no+i, self.no+j] += theta[i, j] / \
                                        (self.ev[j]-self.ev[i])
                                    self.DMP2[self.no+j, self.no+i] += theta[j, i] / \
                                        (self.ev[i]-self.ev[j])'''
        
    #To solve Z vector equation to get z_vec
    def Z_vector_get_veff(dm=None, z_bej=None):
        #dm = multi_dot([c_gaj, c_beb, z_bj]) + multi_dot([c_bej, c_gab, z_bj])
        def cal_jk(num, ialp_i, eri_tmp, jmat, kmat):
            t0 = get_current_time()
            jmat += (ialp_i.T*np.dot(eri_tmp.ravel(), dm.ravel())).T
            accumulate_time(self.t_j, t0)
            t0 = get_current_time()
            kmat += multi_dot([ialp_i.T, dm, eri_tmp]).T
            accumulate_time(self.t_k, t0)
            return jmat, kmat
        t_k = get_current_time()
        win_jmat, j_mat = get_shared((self.nao, self.no), dtype='f8', set_zeros=True)
        win_kmat, k_mat = get_shared((self.nao, self.no), dtype='f8', set_zeros=True)
        split_z = False
        if dm is None:
            split_z = True
        if self.direct_int:
            mol = self.mol
            auxmol = self.RHF.with_df.auxmol
            nao = self.nao
            naoaux = self.naux_hf
            nocc = self.no
            ao_loc = make_loc(self.mol._bas, 'sph')

            def get_uip():
                mo_slice = get_slice(job_size=self.no, rank_list=range(nrank))
                mo_address = []
                for rank_i, mo_i in enumerate(mo_slice):
                    if mo_i is not None:
                        mo0, mo1 = mo_i[0], mo_i[-1]+1
                        mo_address.append([rank_i, [mo0, mo1]])
                win_low, low_node = get_shared((naoaux, naoaux))
                if irank_shm == 0:
                    '''read_file('j2c_hf.tmp', 'low_inv', buffer=low_node)
                    low_node = contigous_trans(low_node)'''
                    read_file('j2c_hf.tmp', 'low', buffer=low_node)
                comm_shm.Barrier()
                self.dir_iup = 'iup_tmp'
                file_list = os.listdir(os.getcwd())
                if self.dir_iup not in file_list:
                    if irank == 0:
                        make_dir(self.dir_iup)
                comm.Barrier()
                mo_slice = mo_slice[irank]
                if mo_slice is not None:
                    mo0, mo1 = mo_slice[0], mo_slice[-1]+1
                    nocc_rank = len(mo_slice)
                    mo_idx = [None]*(self.no+1)
                    for idx, i in enumerate(list(mo_slice)+[mo1]):
                        mo_idx[i] = idx
                else:
                    nocc_rank = None
                max_mo = get_buff_len(self.mol, size_sub=self.nao*naoaux, ratio=0.6, max_len=nocc_rank)
                if mo_slice is not None:
                    seg_idx = np.append(np.arange(mo0, mo1, step=max_mo), mo1)
                    mo_seg = [[mo0, mo1] for mo0, mo1 in zip(seg_idx[:-1], seg_idx[1:])]
                    buf_iup = np.zeros((max_mo, nao, naoaux))
                    buf_ialp = np.empty((nao, naoaux))
                    
                    if 'iup_%d.tmp'%mo0 not in file_list:
                        '''file_iup = h5py.File('%s/iup_%d.tmp'%(self.dir_iup, irank), 'w')
                        file_iup.create_dataset('iup', shape=(nocc_rank, nao, naoaux), dtype='f8')'''
                        with h5py.File('%s/iup_%d.tmp'%(self.dir_iup, irank), 'w') as file_iup:
                            file_iup.create_dataset('iup', shape=(nocc_rank, nao, naoaux), dtype='f8')
                    '''else:
                        file_iup = h5py.File('%s/iup_%d.tmp'%(self.dir_iup, irank), 'r+')'''
                    
                    for mo_i in mo_seg:
                        mo0, mo1 = mo_i
                        for oidx, i in enumerate(range(mo0, mo1)):
                            t1 = get_current_time()
                            read_file("%s/ialp_%d.tmp"%(self.RHF.dir_ialp, i), 'ialp', buffer=buf_ialp)
                            accumulate_time(self.t_read, t1)

                            t1 = get_current_time()
                            ddot(dm, buf_ialp, out=buf_iup[oidx])
                            accumulate_time(self.t_k, t1)
                        t1 = get_current_time()
                        oidx0, oidx1 = mo_idx[mo0], mo_idx[mo1]
                        nocc_seg = mo1 - mo0
                        iup_tmp = buf_iup[:nocc_seg].reshape(-1, naoaux)
                        #ddot(iup_tmp, low_node, out=iup_tmp)
                        scipy.linalg.solve_triangular(low_node.T, iup_tmp.T, lower=False, overwrite_b=True, 
                                                    check_finite=False)
                        iup_tmp = iup_tmp.reshape(nocc_seg, nao, naoaux)
                        accumulate_time(self.t_k, t1)

                        t1 = get_current_time()
                        with h5py.File('%s/iup_%d.tmp'%(self.dir_iup, irank), 'r+') as file_iup:
                            file_iup['iup'].write_direct(iup_tmp, dest_sel=np.s_[oidx0:oidx1])
                        accumulate_time(self.t_write, t1)
                    #file_iup.close()
                comm.Barrier()
                free_win(win_low); low_node=None
                
            def get_k_gammaq(k_mat=None):
                tt = get_current_time()
                ao_slice, shell_slice_rank = int_prescreen.get_slice_rank(self.RHF.shell_slice, aslice=True)
                if split_z:
                    self.address_uip = []
                    for rank_i, slice_i in enumerate(ao_slice):
                        if slice_i is not None:
                            #ao0, ao1 = slice_i
                            self.address_uip.append([rank_i, slice_i])
                    self.dir_ialpz = 'ialp_z_tmp'
                    if irank == 0:
                        make_dir(self.dir_ialpz)
                    comm.Barrier()
                else:
                    #shell_slice_rank = int_prescreen.get_slice_rank(self.RHF.shell_slice, aslice=True)
                    win_gammaq, gammaq_node = get_shared(naoaux, set_zeros=True)
                mo_slice = get_slice(job_size=self.no, rank_list=range(nrank))
                mo_address = []
                for rank_i, mo_i in enumerate(mo_slice):
                    if mo_i is not None:
                        mo0, mo1 = mo_i[0], mo_i[-1]+1
                        mo_address.append([rank_i, [mo0, mo1]])
                idx_break = irank%len(mo_address)
                mo_address = mo_address[idx_break:] + mo_address[:idx_break]
                if shell_slice_rank is not None:
                    if split_z == False:
                        if irank_shm == 0:
                            gammaq = gammaq_node
                            k_tmp = k_mat
                        else:
                            gammaq = np.zeros(naoaux)
                            k_tmp = np.zeros((self.nao, self.no), dtype='f8')
                max_memory = get_mem_spare(mol, 0.9)
                if shell_slice_rank is not None:
                    if split_z:
                        size_ialp, size_feri, shell_slice_rank = int_prescreen.mem_control(mol, nocc, naoaux, shell_slice_rank, 
                                                                             "half_trans", max_memory, nbfit=self.RHF.nbfit_z)
                    else:
                        size_ialp, size_feri, shell_slice_rank = int_prescreen.mem_control(mol, nocc, naoaux, shell_slice_rank, 
                                                                             "half_trans", max_memory)
                    buf_ialp = np.empty(size_ialp)
                    buf_feri = np.empty(size_feri)
                    prod_no_naux = nocc*naoaux
                    if split_z:
                        if self.loc_fit:
                            prod_no_naux = sum([nfit for nfit in self.RHF.nbfit_z if nfit is not None])
                            fit_idx = [[None]*(naoaux+1) for i in self.mo_list]
                            for i, fit_i in enumerate(self.RHF.bfit_seg_z):
                                pidx = 0
                                for p0, p1 in fit_i:
                                    for p in range(p0, p1+1):
                                        fit_idx[i][p] = pidx
                                        pidx += 1
                                    pidx -= 1
                            atm_offset = auxmol.offset_nr_by_atom()
                        t1 = get_current_time()
                        #f_ialpz = h5py.File('%s/ialp_%d.tmp'%(self.dir_ialpz, irank), 'w')
                        nao_rank = ao_slice[irank][1] - ao_slice[irank][0]
                        with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialpz, irank), 'w') as f_ialpz:
                            if self.loc_fit:
                                for i in self.mo_list:
                                    naux_i = self.RHF.nbfit_z[i]
                                    f_ialpz.create_dataset(str(i), shape=(nao_rank, naux_i), dtype='f8')
                            else:
                                f_ialpz.create_dataset('ialp', shape=(nao_rank, nocc, naoaux), dtype='f8')
                        accumulate_time(self.t_write, t1)
                    
                    al0, al1 = ao_slice[irank]
                    idx_ao_rank = [None]*(nao+1)
                    for idx, aoi in enumerate(range(al0, al1+1)):
                        idx_ao_rank[aoi] = idx
                    SHELL_SEG = slice2seg(mol, shell_slice_rank, max_nao=buf_ialp.size//prod_no_naux)
                    for seg_i in SHELL_SEG:
                        A0, A1 = seg_i[0][0], seg_i[-1][1]
                        AL0, AL1 = ao_loc[A0], ao_loc[A1]
                        nao_seg = AL1 - AL0
                        if split_z:
                            if self.loc_fit:
                                iup_tmp = [None]*self.no
                                buf_idx0 = 0
                                for i, naux_i in enumerate(self.RHF.nbfit_z):
                                    buf_idx1 = buf_idx0 + naux_i*nao_seg
                                    iup_tmp[i] = buf_ialp[buf_idx0:buf_idx1].reshape(nao_seg, naux_i)
                                    iup_tmp[i][:] = 0
                                    buf_idx0 = buf_idx1
                            else:
                                iup_tmp = buf_ialp[:nao_seg*naoaux*nocc].reshape(nao_seg, nocc, naoaux)
                                iup_tmp[:] = 0
                        else:
                            #iup_tmp = buf_ialp[:nao_seg*naoaux*nocc].reshape(nocc, nao_seg, naoaux)
                            iup_tmp = buf_ialp[:nao_seg*naoaux*nocc].reshape(nao_seg, naoaux, nocc)
                            t1 = get_current_time()
                            for rank_i, mo_i in mo_address:
                                mo0, mo1 = mo_i[0], mo_i[-1]
                                with h5py.File('%s/iup_%d.tmp'%(self.dir_iup, rank_i), 'r') as file_iup:
                                    #file_iup['iup'].read_direct(iup_tmp[mo0:mo1], source_sel=np.s_[:, AO0:AO1])
                                    iup_tmp[:, :, mo0:mo1] = file_iup['iup'][:, AL0:AL1].transpose(1,2,0)
                            accumulate_time(self.t_read, t1)
                        buf_idx0 = 0
                        for a0, a1, b_list in seg_i:
                            al0, al1 = ao_loc[a0], ao_loc[a1]
                            nao0 = al1 - al0
                            buf_idx1 = buf_idx0 + nao0
                            for b0, b1 in b_list:
                                be0, be1 = ao_loc[b0], ao_loc[b1]
                                nao1 = be1 - be0
                                s_slice = (a0, a1, b0, b1, mol.nbas, mol.nbas+auxmol.nbas)
                                t1 = get_current_time()
                                feri_tmp = aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buf_feri)
                                accumulate_time(self.t_feri, t1)
                                if split_z:
                                    #For K matrix
                                    if self.loc_fit:
                                        #TODO
                                        '''t1 = get_current_time()
                                        feri_tmp = feri_tmp.transpose(2,0,1)
                                        for seg_atm, seg_mo, seg_act in self.RHF.cal_seg_z:
                                            atm0, atm1 = seg_atm[0], seg_atm[-1]
                                            p0, p1 = atm_offset[atm0][-2], atm_offset[atm1][-1]
                                            idx_seg = {} #[None]*(naoaux+1)
                                            for idx_p, p in enumerate(range(p0, p1+1)):
                                                idx_seg[p] = idx_p
                                            pui_tmp = ddot(feri_tmp[p0:p1].reshape(-1, nao1), z_bej[be0:be1, seg_mo]).reshape((p1-p0), nao0, -1)
                                            for idx_i, i in enumerate(seg_mo):
                                                atm_seg_i = list2seg(seg_act[i])
                                                for atm0, atm1 in atm_seg_i:
                                                    q0, q1 = atm_offset[atm0][-2], atm_offset[atm1-1][-1]
                                                    cal_idx0, cal_idx1 = idx_seg[q0], idx_seg[q1]
                                                    save_idx0, save_idx1 = fit_idx[i][q0], fit_idx[i][q1]
                                                    iup_tmp[i][buf_idx0:buf_idx1, save_idx0:save_idx1] += pui_tmp[cal_idx0:cal_idx1, :, idx_i].T'''
                                        accumulate_time(self.t_k, t1)
                                                
                                    else:
                                        t1 = get_current_time()
                                        iup_tmp[buf_idx0:buf_idx1] += ddot(z_bej[be0:be1].T, feri_tmp.transpose(1,0,2).reshape(nao1, -1)).reshape(nocc, nao0, naoaux).transpose(1,0,2)
                                        accumulate_time(self.t_k, t1)
                                else:
                                    #For K matrix
                                    t1 = get_current_time()
                                    feri_tmp = feri_tmp.transpose(1,0,2)
                                    k_tmp[be0:be1] += ddot(feri_tmp.reshape(nao1, -1), iup_tmp[buf_idx0:buf_idx1].reshape(-1, nocc))
                                    accumulate_time(self.t_k, t1)

                                    #For J matrix
                                    t1 = get_current_time()
                                    gammaq += ddot(dm[be0:be1, al0:al1].ravel(), feri_tmp.reshape(-1, naoaux))
                                    accumulate_time(self.t_j, t1)
                            buf_idx0 = buf_idx1
                            if split_z:
                                t1 = get_current_time()
                                idx0, idx1 = idx_ao_rank[AL0], idx_ao_rank[AL1]
                                with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialpz, irank), 'r+') as f_ialpz:
                                    if self.loc_fit:
                                        for i in self.mo_list:
                                            f_ialpz[str(i)].write_direct(iup_tmp[i], dest_sel=np.s_[idx0:idx1])
                                    else:
                                        f_ialpz['ialp'].write_direct(iup_tmp, dest_sel=np.s_[idx0:idx1])
                                accumulate_time(self.t_write, t1)
                    if split_z == False:
                        if irank_shm != 0:
                            Accumulate_GA_shm(win_gammaq, gammaq_node, gammaq)
                    
                else:
                    gammaq = None
                    k_tmp = None
                accumulate_time(self.t_kga, tt)
                comm.Barrier()
                if split_z==False:
                    win_gammaq_col = get_win_col(gammaq_node)
                    Accumulate_GA(win_gammaq_col, gammaq_node)
                    win_gammaq_col.Fence()
                    if irank == 0:
                        with h5py.File('j2c_hf.tmp', 'r') as f:
                            '''low_inv = np.asarray(f['low_inv'])
                            ddot(gammaq_node, low_inv, out=gammaq_node)'''
                            low = np.asarray(f['low'])
                            scipy.linalg.solve_triangular(low, gammaq_node, lower=True, overwrite_b=True, 
                                                        check_finite=False)

                    comm.Barrier()
                    if irank_shm == 0 and irank != 0:
                        Get_GA(win_gammaq_col, gammaq_node)
                    fence_and_free(win_gammaq_col)
                    return win_gammaq, gammaq_node, k_tmp
            def get_ijpz_gammaq():
                self.dir_ijpz = 'ijp_z_tmp'
                if irank == 0:
                    make_dir(self.dir_ijpz)
                comm.Barrier()
                mo_slice = get_slice(range(nrank), job_list=self.mo_list)[irank]
                naoaux = self.naux_hf
                win_gammaq, gammaq_node = get_shared(naoaux, set_zeros=True)
                if mo_slice is not None:
                    if irank_shm == 0:
                        gammaq = gammaq_node
                    else:
                        gammaq = np.zeros(naoaux)
                    mo0, mo1 = mo_slice[0], mo_slice[-1]+1
                    nocc_rank = len(mo_slice)
                    mo_idx = [None]*(self.no+1)
                    for idx, i in enumerate(list(mo_slice)+[mo1]):
                        mo_idx[i] = idx
                    size_sub = self.no*naoaux
                else:
                    size_sub = None
                    nocc_rank = None
                max_mo = get_buff_len(self.mol, size_sub=size_sub, ratio=0.6, max_len=nocc_rank)
                if mo_slice is not None:
                    seg_idx = np.append(np.arange(mo0, mo1, step=max_mo), mo1)
                    mo_seg = [[mo0, mo1] for mo0, mo1 in zip(seg_idx[:-1], seg_idx[1:])]
                    buf_ialp = np.empty((self.nao, naoaux))
                    buf_ijp = np.empty(max_mo*self.no*naoaux)
                    with h5py.File('%s/ijp_%d.tmp'%(self.dir_ijpz, irank), 'w') as f_ijp:
                        f_ijp.create_dataset('ijp', shape=(nocc_rank, self.no, self.naux_hf), dtype='f8')

                    oidx0 = 0
                    for mo_i in mo_seg:
                        mo0, mo1 = mo_i
                        nocc_seg = mo1 - mo0
                        #for i in mo_slice:
                        ijp_tmp = buf_ijp[:nocc_seg*self.no*naoaux].reshape(nocc_seg, self.no, naoaux)
                        for idx, i in enumerate(range(mo0, mo1)):
                            t1 = get_current_time()
                            file_ialp = '%s/ialp_%d.tmp'%(self.RHF.dir_ialp, i)
                            read_file(file_ialp, 'ialp', buffer=buf_ialp)
                            accumulate_time(self.t_read, t1)

                            t1 = get_current_time()
                            ddot(z_bej.T, buf_ialp, out=ijp_tmp[idx])
                            accumulate_time(self.t_k, t1)

                            #For j
                            t1 = get_current_time()
                            gammaq += ddot(z_bej[:, i], buf_ialp)
                            accumulate_time(self.t_j, t1)

                        t1 = get_current_time()
                        oidx1 = oidx0 + nocc_seg
                        with h5py.File('%s/ijp_%d.tmp'%(self.dir_ijpz, irank), 'r+') as f_ijp:
                            f_ijp['ijp'].write_direct(ijp_tmp, dest_sel=np.s_[oidx0:oidx1])
                        accumulate_time(self.t_write, t1)
                        oidx0 = oidx1
                    t_p2 = get_elapsed_time(t0)
                    if irank_shm != 0:
                        Accumulate_GA_shm(win_gammaq, gammaq_node, gammaq)
                comm.Barrier()
                Acc_and_get_GA(gammaq_node)
                return win_gammaq, gammaq_node
                
            def get_jk(j_mat, k_mat, gamma_node):
                def step2(j_mat, k_mat, gamma_node):
                    t1 = get_current_time()
                    buf_ialp = np.empty((nao, naoaux))
                    mo_slice = get_slice(job_size=self.no, rank_list=range(nrank))[irank]
                    #mo_slice = mo_slice[irank]
                    if mo_slice is not None:
                        mo0, mo1 = mo_slice[0], mo_slice[-1]+1
                        nocc_rank = len(mo_slice)
                        mo_idx = [None]*(self.no+1)
                        for idx, i in enumerate(list(mo_slice)+[mo1]):
                            mo_idx[i] = idx
                    else:
                        nocc_rank = None
                    if self.loc_fit == False:
                        max_mo = get_buff_len(self.mol, size_sub=self.nao*naoaux, 
                                               ratio=0.6, max_len=nocc_rank)
                    max_mem = get_mem_spare(self.mol, ratio=0.5)*1e6
                    if mo_slice is not None:
                        if self.loc_fit:
                            #naux_fit = [self.RHF.nbfit_z[i] for i in mo_slice]
                            #naux = [self.RHF.nbfit_z[i] for i in mo_slice]
                            max_mo = 0
                            mem0 = 0
                            mem1 = 0
                            for i in mo_slice:
                                mem_next0 = mem0+(self.nao*self.RHF.nbfit_z[i])*8
                                mem_next1 = mem1+(self.nao*self.RHF.nfit[i])*8
                                if max(mem_next0, mem_next1) > max_mem:break
                                mem0 = mem_next0
                                mem1 = mem_next1
                                max_mo += 1
                            if max_mo == 0:
                                max_mo = 1
                            seg_idx = np.append(np.arange(mo0, mo1, step=max_mo), mo1)
                            mo_seg = [[mo0, mo1] for mo0, mo1 in zip(seg_idx[:-1], seg_idx[1:])]
                            buf_ialp = np.empty(max(mem0, mem1)//8)
                            buf_ijp = np.empty(self.no*buf_ialp.size//self.nao)
                            ialp_read = np.empty((self.nao, naoaux))
                            ijp_read = np.empty((self.no, naoaux))
                        else:
                            seg_idx = np.append(np.arange(mo0, mo1, step=max_mo), mo1)
                            mo_seg = [[mo0, mo1] for mo0, mo1 in zip(seg_idx[:-1], seg_idx[1:])]
                            buf_ialp = np.empty(max_mo*self.nao*naoaux)
                            #buf_ialp = np.empty((nao*naoaux))
                            buf_ijp = np.empty(max_mo*self.no*naoaux)
                            ialp_read = np.empty((nao, naoaux))
                    
                        if irank_shm == 0:
                            j_tmp = j_mat
                            k_tmp = k_mat
                        else:
                            j_tmp = np.zeros((self.nao, self.no), dtype='f8')
                            k_tmp = np.zeros((self.nao, self.no), dtype='f8')
                            
                        for (mo0, mo1) in mo_seg:
                            nocc_seg = mo1 - mo0
                            #Compute K1
                            #read ijp
                            if self.loc_fit:
                                for idx, i in enumerate(range(mo0, mo1)):
                                    t1 = get_current_time()
                                    naux_i = self.RHF.nbfit_z[i]
                                    #ialp_ldf = buf_ialp[:naux_i*self.nao].reshape(naux_i, self.nao)
                                    ialp_ldf = buf_ialp[:naux_i*self.nao].reshape(self.nao, naux_i)
                                    ijp_ldf = buf_ijp[:naux_i*self.no].reshape(self.no, naux_i)
                                    
                                    #read ialp_z
                                    for (rank_i, idx_list) in self.address_uip:
                                        ao0, ao1 = idx_list
                                        with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialpz, rank_i), 'r') as f:
                                            f[str(i)].read_direct(ialp_ldf[ao0:ao1])
                                            #fidx0, fidx1 = f['address'][i]
                                            #f['ialp'].read_direct(ialp_ldf[ao0:ao1], source_sel=np.s_[:, fidx0:fidx1])                                                
                                    #read ijp
                                    with h5py.File('%s/ijp_%d.tmp'%(self.RHF.dir_ijp, i), 'r') as f:
                                            f['ijp'].read_direct(ijp_ldf)
                                    accumulate_time(self.t_read, t1)

                                    t1 = get_current_time()
                                    #k_tmp += ddot(ialp_ldf.T, ijp_ldf.T)
                                    k_tmp += ddot(ialp_ldf, ijp_ldf.T)
                                    accumulate_time(self.t_k, t1)
                                
                                        
                            else:
                                #read ialp_z
                                t1 = get_current_time()
                                ialp_tmp = buf_ialp[:nocc_seg*nao*naoaux].reshape(nao, nocc_seg, naoaux)
                                for (rank_i, idx_list) in self.address_uip:
                                    ao0, ao1 = idx_list
                                    with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialpz, rank_i), 'r') as f:
                                        f['ialp'].read_direct(ialp_tmp[ao0:ao1], source_sel=np.s_[:, mo0:mo1])
                                #read ijp
                                ijp_tmp = buf_ijp[:nocc_seg*self.no*naoaux].reshape(nocc_seg, self.no, naoaux)
                                for idx, i in enumerate(range(mo0, mo1)):
                                    with h5py.File('%s/ijp_%d.tmp'%(self.RHF.dir_ijp, i), 'r') as f:
                                        f['ijp'].read_direct(ijp_tmp[idx])
                                accumulate_time(self.t_read, t1)

                                t1 = get_current_time()
                                k_tmp += ddot(ialp_tmp.reshape(nao, -1), ijp_tmp.transpose(0,2,1).reshape(-1, self.no))
                                accumulate_time(self.t_k, t1)

                            #Compute K2
                            t1 = get_current_time()
                            rank_list, seg_list, idx_list = get_GA_slice(self.RHF.ialp_mo_address, range(self.no))
                            if self.loc_fit:
                                for idx, i in enumerate(range(mo0, mo1)):
                                    '''naux_i = self.RHF.nbfit_z[i]
                                    ialp_ldf = np.empty((self.nao, naux_i))
                                    ialp_ldf = slice_fit(ialp_ldf, ialp_tmp[:, idx], self.RHF.fit_seg[i], axis=1)
                                    ijp_ldf = np.empty((naux_i, self.no))
                                    ijp_ldf = slice_fit(ijp_ldf, ijp_tmp[idx], self.RHF.fit_seg[i], axis=0)
                                    k_tmp += ddot(ialp_ldf, ijp_ldf)'''
                                    t1 = get_current_time()
                                    naux_i = self.RHF.nfit[i]
                                    with h5py.File('%s/ialp_%d.tmp'%(self.RHF.dir_ialp, i), 'r') as f:
                                        f['ialp'].read_direct(ialp_read)
                                    accumulate_time(self.t_read, t1)

                                    #For j:
                                    t1 = get_current_time()
                                    j_tmp[:, i] += ddot(ialp_read, gamma_node)
                                    accumulate_time(self.t_j, t1)

                                    #For k:
                                    t1 = get_current_time()
                                    for idx, rank_i in enumerate(rank_list):
                                        i0, i1 = seg_list[idx]
                                        f_idx0, f_idx1 = idx_list[idx]
                                        with h5py.File('%s/ijp_%d.tmp'%(self.dir_ijpz, rank_i), 'r') as f:
                                            f['ijp'].read_direct(ijp_read[i0:i1].reshape(i1-i0, 1, -1), source_sel=np.s_[:, i:i+1])
                                    accumulate_time(self.t_read, t1)

                                    t1 = get_current_time()
                                    ijp_ldf = buf_ijp[:naux_i*self.no].reshape(self.no, naux_i)
                                    ijp_ldf = slice_fit(ijp_ldf, ijp_read, self.RHF.fit_seg[i], axis=1)
                                    ialp_ldf = buf_ialp[:naux_i*self.nao].reshape(self.nao, naux_i)
                                    ialp_ldf = slice_fit(ialp_ldf, ialp_read, self.RHF.fit_seg[i], axis=1)
                                    accumulate_time(self.t_slice, t1)

                                    t1 = get_current_time()
                                    k_tmp += ddot(ialp_ldf, ijp_ldf.T)
                                    accumulate_time(self.t_k, t1)
                            else:
                                #read ialp
                                t1 = get_current_time()
                                ialp_tmp = buf_ialp[:nocc_seg*nao*naoaux].reshape(nao, nocc_seg, naoaux)
                                for idx, i in enumerate(range(mo0, mo1)):
                                    with h5py.File('%s/ialp_%d.tmp'%(self.RHF.dir_ialp, i), 'r') as f:
                                        #f['ialp'].read_direct(ialp_tmp, dest_sel=np.s_[:, idx])
                                        f['ialp'].read_direct(ialp_read)
                                        #For j:
                                        j_tmp[:, i] += ddot(ialp_read, gamma_node)
                                        ialp_tmp[:, idx] = ialp_read
                                #read ijp_z
                                ijp_tmp = buf_ijp[:nocc_seg*self.no*naoaux].reshape(nocc_seg, naoaux, self.no)
                                for idx, rank_i in enumerate(rank_list):
                                    i0, i1 = seg_list[idx]
                                    f_idx0, f_idx1 = idx_list[idx]
                                    with h5py.File('%s/ijp_%d.tmp'%(self.dir_ijpz, rank_i), 'r') as f:
                                        ijp_tmp[:, :, i0:i1] = f['ijp'][:, mo0:mo1].transpose(1,2,0)
                                accumulate_time(self.t_read, t1)

                                t1 = get_current_time()
                                k_tmp += ddot(ialp_tmp.reshape(nao, -1), ijp_tmp.reshape(-1, self.no))
                                accumulate_time(self.t_k, t1)
                    else:
                        j_tmp = None
                        k_tmp = None
                    return j_tmp, k_tmp
                j_tmp, k_tmp = step2(j_mat, k_mat, gamma_node)
                return j_tmp, k_tmp

            #Step 2 for J matrix
            def get_j(j_mat, gammaq_node=None):
                t1 = get_current_time()
                '''if split_z:
                    win_gammaq, gammaq_node = get_shared(naoaux)'''
                buf_ialp = np.empty((nao, naoaux))
                mo_slice = get_slice(job_size=self.no, rank_list=range(nrank))[irank]
                '''if split_z:
                    if mo_slice is not None:
                        if irank_shm == 0:
                            gammaq = gammaq_node
                        else:
                            gammaq = np.zeros(naoaux)
                        for i in mo_slice:
                            read_file("%s/ialp_%d.tmp"%(self.RHF.dir_ialp, i), 'ialp', buffer=buf_ialp)
                            gammaq += ddot(z_bej[:, i], buf_ialp)
                        if irank_shm != 0:
                            Accumulate_GA_shm(win_gammaq, gammaq_node, gammaq)
                    comm.Barrier()
                    Acc_and_get_GA(gammaq_node)
                '''
                if mo_slice is not None:
                    if irank_shm == 0:
                        j_tmp = j_mat
                    else:
                        j_tmp = np.zeros((self.nao, self.no), dtype='f8')
                    for i in mo_slice:
                        read_file("%s/ialp_%d.tmp"%(self.RHF.dir_ialp, i), 'ialp', buffer=buf_ialp)
                        j_tmp[:, i] = ddot(buf_ialp, gammaq_node)
                else:
                    j_tmp = None
                accumulate_time(self.t_j, t1)
                comm.Barrier()
                if split_z:
                    free_win(win_gammaq)
                return j_tmp

            if split_z:
                t1 = get_current_time()
                win_gammaq, gammaq_node = get_ijpz_gammaq()
                accumulate_time(self.t_ijpz, t1)
                #if self.loc_fit==False:
                #t1 = get_current_time()
                get_k_gammaq()
                #self.t_kga += get_current_time() - t1

                t1 = get_current_time()
                j_tmp, k_tmp = get_jk(j_mat, k_mat, gammaq_node)
                accumulate_time(self.t_jk, t1)
                #j_tmp = get_j(j_mat)
            else:
                get_uip()
                win_gammaq, gammaq_node, k_tmp = get_k_gammaq(k_mat)
                j_tmp = get_j(j_mat, gammaq_node)
            free_win(win_gammaq)

        else:
            #aux_slice = get_slice(rank_list=range(nrank), job_size=self.naoaux)[irank]
            auxmol = self.RHF.with_df.auxmol
            aux_slice = get_auxshell_slice(auxmol)[0][irank]
            if aux_slice != None:
                if irank_shm != 0:
                    j_tmp = np.zeros((self.nao, self.no), dtype='f8')
                    k_tmp = np.zeros((self.nao, self.no), dtype='f8')
                feri_buffer_unpack = np.empty((self.nao, self.nao))
                if (self.use_ga):
                    if (self.outcore):
                        with h5py.File(self.RHF.feri_aux, 'r') as feri_aux:
                            for idx, num in enumerate(aux_slice):
                                lib.numpy_helper.unpack_tril(np.asarray(feri_aux['j3c'][idx]), out=feri_buffer_unpack)
                                ialp_tmp = self.RHF.ialp_aux[idx]
                                if irank_shm == 0:
                                    j_mat, k_mat = cal_jk(num, ialp_tmp, feri_buffer_unpack, j_mat, k_mat)
                                else:
                                    j_tmp, k_tmp = cal_jk(num, ialp_tmp, feri_buffer_unpack, j_tmp, k_tmp)
                    else:
                        for idx, num in enumerate(aux_slice):
                            lib.numpy_helper.unpack_tril(self.RHF.feri_aux[idx], out=feri_buffer_unpack)
                            ialp_tmp = self.RHF.ialp_aux[idx]
                            if irank_shm == 0:
                                j_mat, k_mat = cal_jk(num, ialp_tmp, feri_buffer_unpack, j_mat, k_mat)
                            else:
                                j_tmp, k_tmp = cal_jk(num, ialp_tmp, feri_buffer_unpack, j_tmp, k_tmp)
                else:
                    if (self.outcore):
                        #file_list, aux_seg = get_ialp_seg(self, aux_slice, 'aux_hf')
                        ialp_tmp = np.empty((self.no, self.nao), dtype='f8')
                        feri_tmp = np.empty((self.nao_pair), dtype='f8')
                        #for f_idx, seg_i in enumerate(aux_seg):
                        for idx, num in enumerate(aux_slice):
                            read_sucess = False
                            while (read_sucess==False):
                                try:
                                    #with h5py.File(file_list[f_idx], 'r') as ialp:
                                    with h5py.File('%s/ialp_aux_%d.tmp'%(self.RHF.dir_ialp_aux, irank), 'r') as ialp:
                                        #for num in seg_i:
                                            feri.read_direct(feri_tmp, source_sel=np.s_[idx])
                                                #feri_tmp[:] = feri[idx]
                                            lib.numpy_helper.unpack_tril(feri_tmp, out=feri_buffer_unpack)
                                            ialp_tmp[:] = ialp[str(num)]
                                            if irank_shm == 0:
                                                j_mat, k_mat = cal_jk(num, ialp_tmp, feri_buffer_unpack, j_mat, k_mat)
                                            else:
                                                j_tmp, k_tmp = cal_jk(num, ialp_tmp, feri_buffer_unpack, j_tmp, k_tmp)
                                    read_sucess = True
                                except IOError:
                                    log.info("rank %d, Warning:%s cannot be opened"%(irank, file_list[f_idx]))
                                    read_sucess = False
                    else:
                        for num in aux_slice:
                            lib.numpy_helper.unpack_tril(feri[num], out=feri_buffer_unpack)
                            #ialp_tmp = self.ialp[num]
                            ialp_tmp = self.RHF.ialp[num]
                            if irank_shm == 0:
                                j_mat, k_mat = cal_jk(num, ialp_tmp.T, feri_buffer_unpack, j_mat, k_mat)
                            else:
                                j_tmp, k_tmp = cal_jk(num, ialp_tmp.T, feri_buffer_unpack, j_tmp, k_tmp)
                feri_buffer_unpack, ialp_tmp = None, None
            else:
                j_tmp = None
                k_tmp = None

        if irank_shm != 0:
            if j_tmp is not None:
                Accumulate_GA_shm(win_jmat, j_mat, j_tmp)
            if k_tmp is not None:
                Accumulate_GA_shm(win_kmat, k_mat, k_tmp)
            j_tmp, k_tmp = None, None    
        t_syn = get_current_time()
        comm_shm.Barrier()
        if irank_shm == 0:
            if split_z:
                veff = 4*j_mat-k_mat
            else:
                veff = 2*j_mat-k_mat
        else:
            veff = None
        comm.Barrier()
        if self.nnode > 1:
            Acc_and_get_GA(veff)
        if self.loc_fit and irank_shm == 0:
            ddot(veff, self.uo.T, out=veff)
        accumulate_time(self.t_jk_syn, t_syn)
        return veff

    def Z_vector(feri):
        log = lib.logger.Logger(self.stdout, self.verbose)
        '''if irank == 0:
            print_test(self.Yla, 'Yla')
            print_test(self.Yli, 'Yli')
            print_test(self.DMP2, 'DMP2')'''
        self.kry_type = 1
        def krylov(aop, b, x0=None, tol=1e-6, max_cycle=30, x1=None, dot=ddot,
            lindep=1e-15, use_diis=False, hermi=False, verbose=5):
            #Krylov subspace method to solve  (1+a) x = b
            if self.kry_type == 0:
                ax = [aop(b)]
                if irank_shm == 0:
                    xs = [b]
                    innerprod = [dot(xs[0].conj(), xs[0])]
                    h = np.empty((max_cycle, max_cycle), dtype=ax[0].dtype)
            else:
                x1 = x1.reshape(1, -1)
                if irank_shm == 0:
                    b = b.reshape(1, -1)
                    x, rmat = lib.linalg_helper._qr(b, dot, lindep)
                    x1[:] = x
                    for i in range(len(x1)):
                        x1[i] *= rmat[i,i]
                    innerprod = [dot(xi.conj(), xi).real for xi in x1]
                    max_innerprod = max(innerprod)
                    if max_innerprod < lindep or max_innerprod < tol**2:
                        raise ValueError("Linear dependency lower than tolerance")
                    xs = []
                    ax = []
            
            nroots, ndim = 1, x1.size
            if (use_diis) and (irank_shm == 0):
                adiis = lib.diis.DIIS()

            comm_shm.Barrier()
            win_conv, conv = get_shared(1, dtype='i', set_zeros=True)
            max_cycle = min(max_cycle, ndim)
            for cycle in range(max_cycle):
                if self.kry_type == 0:
                    if irank_shm == 0:
                        x1[:] = ax[-1]
                        # Schmidt orthogonalization
                        for i in range(cycle+1):
                            s12 = h[i, cycle] = dot(xs[i].conj(), ax[-1])        # (*)
                            x1[:] -= (s12/innerprod[i]) * xs[i]
                        h[cycle, cycle] += innerprod[cycle]                      # (*)
                        innerprod.append(dot(x1.conj(), x1).real)
                        if irank==0:log.info('    krylov cycle %d  r = %g', cycle, np.sqrt(innerprod[-1]))
                        if innerprod[-1] < lindep or innerprod[-1] < tol**2:
                            conv[0] = 1
                        xs.append(x1.copy())
                    comm_shm.Barrier()
                    if conv:
                        break
                    ax1 = aop(x1)
                    if irank_shm == 0:
                        if (use_diis):
                            ax1 = adiis.update(ax1, (ax1-x1))
                        ax.append(ax1)
                else:
                    axt = aop(x1)
                    if irank_shm == 0:
                        if use_diis:
                            axt = adiis.update(axt, (axt-x1))
                        axt = axt.reshape(1,ndim)
                        
                        xs.extend(x1.copy())
                        ax.extend(axt)
                        x1[:] = axt
                        for i in range(len(xs)):
                            xsi = numpy.asarray(xs[i])
                            for j, axj in enumerate(axt):
                                x1[j] -= xsi * (dot(xsi.conj(), axj) / innerprod[i])                            
                        axt = None
                        max_innerprod = 0
                        idx = []
                        for i, xi in enumerate(x1):
                            innerprod1 = dot(xi.conj(), xi).real
                            max_innerprod = max(max_innerprod, innerprod1)
                            if innerprod1 > lindep and innerprod1 > tol**2:
                                idx.append(i)
                                innerprod.append(innerprod1)
                        if irank == 0:
                            log.debug('    krylov cycle %d  r = %g', cycle, max_innerprod**.5)
                        if max_innerprod < lindep or max_innerprod < tol**2:
                            conv[0] = 1
                    comm_shm.Barrier()
                    if conv:
                        break
            if self.kry_type == 0:
                if irank_shm == 0:
                    #x = np.empty(self.nv*self.no, dtype='f8')
                    if irank==0:log.debug('final cycle = %d', cycle)
                    nd = cycle + 1
                    for i in range(nd):
                        for j in range(i):
                            h[i, j] = dot(xs[i].conj(), ax[j])
                    g = np.zeros(nd, dtype=b.dtype)
                    g[0] = innerprod[0]
                    c = np.linalg.solve(h[:nd, :nd], g)
                    x1[:] = xs[0] * c[0]
                    for i in range(1, len(c)):
                        x1 += c[i] * xs[i]
                comm_shm.Barrier()
            elif irank_shm == 0:
                nd = cycle + 1
                h = numpy.empty((nd,nd), dtype=x1.dtype)

                for i in range(nd):
                    xi = numpy.asarray(xs[i])
                    if hermi:
                        for j in range(i+1):
                            h[i,j] = dot(xi.conj(), ax[j])
                            h[j,i] = h[i,j].conj()
                    else:
                        for j in range(nd):
                            h[i,j] = dot(xi.conj(), ax[j])
                    xi = None

                # Add the contribution of I in (1+a)
                for i in range(nd):
                    h[i,i] += innerprod[i]

                g = numpy.zeros((nd,nroots), dtype=x1.dtype)
                # Restore the first nroots vectors, which are array b or b-(1+a)x0
                for i in range(min(nd, nroots)):
                    xsi = numpy.asarray(xs[i])
                    for j in range(nroots):
                        g[i,j] = dot(xsi.conj(), b[j])

                c = numpy.linalg.solve(h, g)
                x1[:] = lib.linalg_helper._gen_x0(c, xs)
                if b.ndim == 1:
                    x1 = x1[0]
            comm_shm.Barrier()
            if use_diis:
                ax = aop(x1)
                if irank_shm == 0:
                    x1.ravel()[:] = b.ravel() - ax
            return x1
        
        def Z_vector_get_B():#            
            win_dmb, dmb = get_shared((self.nao, self.nao), dtype='f8')
            if irank_shm == 0:
                dmb[:] = multi_dot([self.mo_coeff, self.DMP2, self.mo_coeff.T])
                dmb += dmb.T
                
            comm_shm.Barrier()
            veff = Z_vector_get_veff(dmb)
            
            win_b0, B0 = get_shared((self.nv, self.no), dtype='f8', set_zeros=True)
            if irank_shm == 0:
                ddot(-self.v.T, veff, out=B0)
                B0 += self.Yla[:self.no].T
                B0 -= self.Yli[self.no:]
                if (self.use_cpl is not False and self.local_type is 1):
                    if (self.nocc_core is None or self.nocc_core == 0):
                        B0 -= ddot(aloc[self.no:], self.uo.T)
                    else:
                        B0 -= ddot(aloc[self.no:], self.uo[:, self.nocc_core:].T)
                        B0 -= ddot(aloc1[self.no:], self.uo[:, :self.nocc_core].T)
                B0 *= e_ai_inv
            comm_shm.Barrier()
            free_win(win_dmb)
            return win_b0, B0
        #t0 = get_current_time()
        
        self.step = 1
        def AZ(z_vec):
            if self.direct_int:
            #if self.direct_int and 1==0:
                win_z_bej, z_bej = get_shared((self.nao, self.no), dtype='f8')
                if irank_shm == 0:
                    if self.loc_fit:
                        z_bej[:] = multi_dot([self.v, z_vec.reshape(self.nv, self.no), self.uo])
                    else:
                        ddot(self.v, z_vec.reshape(self.nv, self.no), out=z_bej)

                comm_shm.Barrier()
                veff = Z_vector_get_veff(z_bej=z_bej)
                free_win(win_z_bej)
            else:
                win_dm1, dm1 = get_shared((self.nao, self.nao), dtype='f8')
                if irank_shm == 0:
                    #dm1[:] = multi_dot([self.v, z_vec.reshape(self.nv, self.no), self.o.T])
                    moco_can = self.mo_coeff[:, :self.no]
                    dm1[:] = multi_dot([self.v, z_vec.reshape(self.nv, self.no), moco_can.T])
                    dm1 += dm1.T
                comm_shm.Barrier()
                veff = Z_vector_get_veff(dm1)
                free_win(win_dm1)
            if irank_shm == 0:
                result = (ddot(self.v.T, veff)*e_ai_inv).ravel()
                #result = (2*multi_dot([self.v.T, veff, self.o])*e_ai_inv).ravel()
            else:
                result = None
            comm_shm.Barrier()
            return result
        self.t_feri = create_timer()
        self.t_j = create_timer()
        self.t_k = create_timer()
        self.t_read = create_timer()
        self.t_write = create_timer()
        self.t_dm_send = create_timer()
        self.t_jk_syn = create_timer()
        self.t_ijpz = create_timer()
        self.t_kga = create_timer()
        self.t_jk = create_timer()
        self.t_slice = create_timer()

        e_ai_inv = 1.0 / lib.direct_sum('a-i->ai', self.ev, self.eo)
        win_b, B = Z_vector_get_B()
        win_z, z_vec = get_shared(self.nv*self.no, dtype='f8')
        z_con = krylov(AZ, B.ravel(), tol=1.0e-5, max_cycle=20, x1=z_vec, verbose=5)
        if irank_shm == 0:
            z_vec[:] = z_con
            z_vec = z_vec.reshape(self.nv, self.no)
        time_list = []
        if self.direct_int:
            time_list += [['ijpz', self.t_ijpz], ['kga', self.t_kga], ['JK', self.t_jk],
                          ['reading', self.t_read], ['writing', self.t_write]]
            if self.loc_fit:
                time_list.append(['slicing', self.t_slice])
        time_list += [['feri', self.t_feri], ['J matrix', self.t_j], ['K matrix', self.t_k]]
        time_list = get_max_rank_time_list(time_list)
        print_time(time_list, log)
        comm_shm.Barrier()
        #free_win(win_b); B = None
        az1 = None
        if irank == 0:
            #print_time(['Z vector', get_elapsed_time(t0)], log)
            if self.direct_int:
                for dir_i in [self.RHF.dir_ijp, self.dir_ijpz, self.dir_ialpz]:
                    shutil.rmtree(dir_i)
        #sys.exit()
        return win_z, z_vec, win_b, B
    def make_rdm1e(mo_energy, mo_coeff, mo_occ):
        '''Energy weighted density matrix'''
        mo0 = mo_coeff[:, mo_occ > 0]
        mo0e = mo0 * (mo_energy[mo_occ > 0] * mo_occ[mo_occ > 0])
        return np.dot(mo0e, mo0.T.conj())
    def get_w(z_vec, B=None):
        win_pi, Pi = get_shared((self.nao, self.nao), dtype='f8')
        win_w, W = get_shared((self.nao, self.nao), dtype='f8')
        #To be continued
        if ztype == 0:
            if irank_shm == 0:
                Pi[:] = self.DMP2
                Pi[self.no:, self.no:] *= 2
                Pi[:self.no, self.no:] -= z_vec.T
                Pi[:] = multi_dot([self.mo_coeff, Pi, self.mo_coeff.T])

                Eab = np.zeros((self.nao, self.nao), dtype='f8')
                #E^ij_ab
                Eab[:self.no, :self.no] = lib.direct_sum(
                    'p+q->pq', self.eo, self.eo)
                #E^ab_ab
                Eab[self.no:, self.no:] = lib.direct_sum(
                    'p+q->pq', self.ev, self.ev)
                #W[:] = self.DMP2 * Eab
                W[:self.no, self.no:] = z_vec.T
                W += self.DMP2 * Eab + 0.5*np.trace(B)

            comm_shm.Barrier()
            if irank_shm == 0:
                W[:] = 0.5*multi_dot([self.mo_coeff, W, self.mo_coeff.T])
                W += make_rdm1e(self.mo_energy, self.mo_coeff, self.mo_occ)
                W += W.T
                Pi[:] = (Pi+Pi.T)/2

        else:
            #if irank == 0:
            if irank_shm == 0:
                Pi[:] = self.DMP2
                Pi[self.no:, :self.no] += z_vec
                Pi[:] = multi_dot([self.mo_coeff, Pi, self.mo_coeff.T])
                Eab = np.zeros((self.nao, self.nao), dtype='f8')
                #E^ij_ab
                Eab[:self.no, :self.no] = lib.direct_sum(
                    'p+q->pq', self.eo, self.eo)
                #E^ab_ab
                Eab[self.no:, self.no:] = lib.direct_sum(
                    'p+q->pq', self.ev, self.ev)
                W[:] = self.DMP2 * Eab
                W[self.no:, :self.no] += z_vec*self.eo*2
                W[:self.no, :self.no] += self.Yli[:self.no]
                W[self.no:, self.no:] += self.Yla[self.no:]
                W[:self.no, self.no:] += 2*self.Yla[:self.no]

            comm_shm.Barrier()
            #if ztype == 1:
            #if self.direct_int == False:
            veff = Z_vector_get_veff(Pi+Pi.T)
            mo_co = self.mo_coeff[:, :self.no]
            if irank_shm == 0:
                w_tmp = ddot(mo_co.T, veff)
            else:
                w_tmp = None
            comm_shm.Barrier()
            if irank == 0 and self.direct_int:
                shutil.rmtree('iup_tmp')
            if irank_shm == 0:
                W[:self.no, :self.no] += w_tmp
            #if irank == 0:
            if irank_shm == 0:
                W[:] = 0.5*multi_dot([self.mo_coeff, W, self.mo_coeff.T])
                if (self.use_cpl is not False and self.local_type == 1):
                    if (self.nocc_core is None or self.nocc_core == 0):
                        W += 0.5*multi_dot([mo_co, aloc[:self.no], mo_co.T])
                    else:
                        W += 0.5 * multi_dot(
                                [mo_co[:, self.nocc_core:], aloc[self.nocc_core:self.no], mo_co[:, self.nocc_core:].T])
                        W += 0.5 * multi_dot([mo_co[:, :self.nocc_core], 
                                        aloc1[:self.nocc_core], mo_co[:, :self.nocc_core].T])
                    W -= S_loc
                    if (self.nocc_core is not None and self.nocc_core > 0):
                        W -= S_loc1

                #W += make_rdm1e(self.mo_energy, self.mo_coeff, self.mo_occ)
                W += W.T
                Pi[:] = (Pi+Pi.T)/2
        return win_pi, Pi, win_w, W

    
    #Kernel
    t_g_ao = get_current_time()
    log = lib.logger.Logger(self.stdout, self.verbose)
    tt = get_current_time()
    lst, idex = get_grad_idx()
    self.eo = self.mo_energy[self.mo_occ>0]
    if self.loc_fit == False:
        self.o = self.mo_coeff[:, self.mo_occ>0]

    #Update MP2 density matrix
    t0 = get_current_time()
    #if irank == 0:
    if irank_shm == 0:
        if (self.use_cpl and self.local_type == 1):
            aloc, S_loc = get_cpl_coef()
    update_DMP2()
    gradient = np.zeros(self.mol.natm*3)

    #Generate RHF ialp
    self.naoaux = self.naux_hf
    self.RHF.nao = self.nao
    self.RHF.naux_hf = self.naux_hf
    self.RHF.naux_mp2 = self.naux_mp2
    self.RHF.frozen = self.use_frozen
    self.RHF.nocc_core = self.nocc_core
    self.RHF.o = self.o
    self.RHF.naoaux = self.naux_hf
    self.RHF.no = self.no
    self.mo_list = list(range(self.no))
    self.RHF.mo_list = self.mo_list
    self.RHF.grad_cal = self.grad_cal
    self.RHF.loc_fit = self.loc_fit
    self.RHF.fit_tol = self.fit_tol
    self.RHF.bfit_tol = self.bfit_tol
    if (self.wrap_test) and (self.direct_int == False):
        from osvmp2.OSVL import parallel_eri
        parallel_eri(self.RHF, self.RHF.with_df, 'hf', log)
    if (self.use_ga):
        feri, eri = None, None
    else:
        if (self.outcore):
            read_sucess = False
            while (read_sucess==False):
                try:
                    #eri = h5py.File(self.RHF.with_df._cderi, 'r')
                    eri = h5py.File('%s/feri_tmp_%d.tmp'%(self.RHF.dir_feri, irank), 'r')
                    read_sucess = True
                except IOError:
                    log.info("rank %d, Warning:%s cannot be opened"%(irank, self.RHF.with_df._cderi))
                    read_sucess = False
            feri = eri['j3c']
        else:
            feri = self.RHF.with_df._cderi

    #RHF (ial|P)
    from osvmp2.OSVL import get_ialp
    t1 = get_current_time()
    if irank == 0:log.info('\nRHF ialp calculation starts...')
    get_ialp(self.RHF, self.RHF.with_df, 'hf', log)
    print_mem('RHF ialp', self.pid_list, log)
    print_time(['ialp generation', get_elapsed_time(t1)], log)
    if self.loc_fit:
        ave_nfit = np.sum(self.RHF.nfit)/self.no
        ave_nbfit = np.sum(self.RHF.nbfit)/self.no
        ave_nbfit_z = np.sum(self.RHF.nbfit_z)/self.no
        log.info('\nAverage local fitting basis for RHF (full %d):'%self.naoaux)
        msg_list = [['Fitting (%.1E):'%self.fit_tol, int(ave_nfit)],
                    ['Block fitting (%.1E):'%self.fit_tol, int(ave_nbfit)],
                    ['Block fitting for Z-vector (%.1E):'%self.bfit_tol, int(ave_nbfit_z)]]
        print_align(msg_list, align='lr', indent=4, log=log)
    self.t_grad_hf = get_elapsed_time(t1)

    if irank == 0:log.info('\nBegin solving Z vector equation...')
    t1 = get_current_time()
    win_z, z_vec, win_b, B = Z_vector(feri)
    if irank == 0:
        self.dm_mo[self.no:, :self.no] += z_vec
        self.dm_mo = multi_dot([self.mo_coeff, self.dm_mo, self.mo_coeff.T])
    t_z = get_elapsed_time(t1)

    t1 = get_current_time()
    win_pi, Pi, win_w, W = get_w(z_vec, B)
    t_w = get_elapsed_time(t1)

    for win in [win_z, win_b]:
        free_win(win)
    self.t_zvec = t_z + t_w
    time_list = [["CPHF solution", t_z], ["term Pi and W", t_w], ["Z vector", self.t_zvec]]
    time_list = get_max_rank_time_list(time_list)
    print_time(time_list, log)
    z_vec = None; B = None
    print_mem('Z vector', self.pid_list, log)
    

    tt = get_current_time()
    # Compute RHF density matrix and energy-weighted density matrix.
    win_dm0, dm0 = get_shared((self.nao, self.nao), dtype='f8')
    win_dme0, dme0 = get_shared((self.nao, self.nao), dtype='f8')
    if irank_shm == 0:
        dm0[:] = self.RHF.make_rdm1(self.mo_coeff, self.mo_occ)
        dme0[:] = make_rdm1e(self.mo_energy, self.mo_coeff, self.mo_occ)
    comm_shm.Barrier()
    
    
    #print(self.solvent)
    if self.solvent is not None:
        t0 = get_current_time()
        log.info('\nBegin the computation of solvation gradient...')
        g_sol = get_grad_sol(self.with_solvent, (Pi+dm0), log)
        print_time(["solvation gradient", get_elapsed_time(t0)], log)
        gradient += g_sol.ravel()
    win_drelaxed, dm_relaxed = get_shared((self.nao, self.nao), dtype='f8')
    if irank_shm == 0:
        dm_relaxed[:] = Pi
        dm_relaxed += dm0
    comm_shm.Barrier()
    if self.pop_remp2:
        analysis(self.mol, dm_relaxed, 'relaxed MP2', charge_method=self.charge_method_mp2, 
                save_data='relaxed_MP2', log=log)#, gen_mep=True)
    free_win(win_drelaxed)
    t0 = get_current_time()
    log.info('\nBegin the computation of RHF gradient...')
    gradient += dfhf_response_ga(self.RHF, dm0, Pi+0.5*dm0, feri)#ialp, feri)
    #gradient += dfhf_response_ga(self, dm0, 0.5*dm0, feri)
    comm_shm.Barrier()
    if (self.outcore==False) and (self.use_ga==False):
        free_win(self.RHF.win_ialp)
        self.RHF.ialp = None
    elif (self.outcore):
        if eri is not None:
            eri.close()
        feri = None
    #Gradient of core hamiltonian
    gradient += hf_grad.grad_hcore_nuc(self.RHF, dm0, dme0, Pi, W, log=log)
    for win in [win_pi, win_w, win_dm0]:
        free_win(win)
    self.t_grad_hf += get_elapsed_time(tt)
    return gradient

