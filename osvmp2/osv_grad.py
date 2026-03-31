import os
import time
import numpy as np
from osvmp2.osvutil import *


def generation_BigMat(self, ijkl, matrix, dim1, dim2, ndim):
    i, j, k, l  = ijkl
    SuperMat  = np.empty((dim1[i]+dim1[j], dim2[k]+dim2[l]))
    SuperMat[:dim1[i], :dim2[k]]  = matrix[i*ndim+k]
    SuperMat[dim1[i]:, :dim2[k]]  = matrix[j*ndim+k]
    SuperMat[:dim1[i], dim2[k]:]  = matrix[i*ndim+l]
    SuperMat[dim1[i]:, dim2[k]:]  = matrix[j*ndim+l]
    return SuperMat

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

def derivative_R(self, T_matrix, T_bar, S_matrix_cp, F_matrix_cp):
    def check_redundant(self, i, j, k):
        def get_pair(i, j):
            if i < j:
                return i*self.no+j
            else:
                return j*self.no+i
        result = False
        if (i != k) and (j != k):
            pairs = [get_pair(i, k), get_pair(j, k)]
            for ipair in pairs:
                if self.is_remote[ipair] or self.is_discarded[ipair]:
                    result = True
                    break
        return result
    def residue_eval(ipair, N):
        def get_dm(tmat, xmat, tbar):
            return multi_dot([tmat, xmat, tbar.T]) + multi_dot([tmat.T, xmat, tbar])
        def get_dm_remote(tmat, tbar, xmat_11=None, xmat_22=None):
            if xmat_11 is None:
                return ddot(tmat, tbar.T), ddot(tmat.T, tbar)
            else:
                return multi_dot([tmat, xmat_22, tbar.T]), multi_dot([tmat.T, xmat_11, tbar])
        i = ipair//self.no
        j = ipair % self.no

        if (self.is_remote[ipair]):
            '''R_top = np.zeros((self.nosv_cp[i], self.nosv[j]))
            R_bot = np.zeros((self.nosv_cp[j], self.nosv[i]))'''
            #todo
            Df_ii, Df_jj = get_dm_remote(T_matrix[ipair], T_bar[ipair], 
                                      self.F_matrix[i*self.no+i], self.F_matrix[j*self.no+j])
            Ds_ii_1, Ds_jj_1 = get_dm_remote(T_matrix[ipair], T_bar[ipair])
            Ds_ii_2, Ds_ij = get_dm_remote(get_T(self, T_matrix, i, i), T_bar[ipair], 
                                      self.S_matrix[i*self.no+i], self.S_matrix[i*self.no+j])
            Ds_ji, Ds_jj_2 = get_dm_remote(get_T(self, T_matrix, j, j), T_bar[ipair], 
                                      self.S_matrix[j*self.no+i], self.S_matrix[j*self.no+j])
            f_iijj = self.loc_fock[i,i] + self.loc_fock[j,j]
            R_ii = ddot(S_matrix_cp[i*self.no+i], Df_ii - f_iijj*Ds_ii_1)
            R_ii += ddot(F_matrix_cp[i*self.no+i], Ds_ii_1)
            R_ii -= self.loc_fock[i,j]*(ddot(S_matrix_cp[i*self.no+i], Ds_ii_2) +
                                        ddot(S_matrix_cp[i*self.no+j], Ds_ji))
            R_jj = ddot(S_matrix_cp[j*self.no+j], Df_jj - f_iijj*Ds_jj_1)
            R_jj += ddot(F_matrix_cp[j*self.no+j], Ds_jj_1)
            R_jj -= self.loc_fock[i,j]*(ddot(S_matrix_cp[j*self.no+i], Ds_ij) +
                                        ddot(S_matrix_cp[j*self.no+j], Ds_jj_2))
        else:
            skip_iijj = False

            if self.clus_type == 20:
                if i == j:
                    skip_iijj = True
            elif self.clus_type == 3:
                skip_iijj = True
            if skip_iijj:
                R_ijij = np.zeros((self.nosv_cp[i]+self.nosv_cp[j], self.nosv[i]+self.nosv[j]))
            else:
                F_ijij = generation_SuperMat([i, j, i, j], self.F_matrix, self.nosv, self.no)
                S_ijij = generation_SuperMat([i, j, i, j], self.S_matrix, self.nosv, self.no)
                Df_ijij = get_dm(T_matrix[ipair], F_ijij, T_bar[ipair])
                Ds_ijij = get_dm(T_matrix[ipair], S_ijij, T_bar[ipair])
                F_ijij, S_ijij = None, None
                F_cp_ijij = generation_BigMat(self, [i, j, i, j], F_matrix_cp, self.nosv_cp, self.nosv, self.no)
                S_cp_ijij = generation_BigMat(self, [i, j, i, j], S_matrix_cp, self.nosv_cp, self.nosv, self.no)
                f_iijj = self.loc_fock[i,i] + self.loc_fock[j,j]
                R_ijij = ddot(S_cp_ijij, Df_ijij-f_iijj*Ds_ijij) + ddot(F_cp_ijij, Ds_ijij)
                F_cp_ijij, S_cp_ijij = None, None
        
            mo_list = self.mo_list

            if self.clus_type == 20:
                if i == j:
                    mo_list = [k for k in self.mo_list if k != i]
            elif self.clus_type == 3:
                mo_list = []
                for k in self.mo_list:
                    if (k != i) and (k != j):
                        mo_list.append(k)
            for k in mo_list:
                if check_redundant(self, i, j, k): continue
                if k != j:
                    if i > k:
                        T_ik = flip_ij(k, i, T_matrix[k*self.no+i], self.nosv)
                    else:
                        T_ik = T_matrix[i*self.no+k]
                    S_cp_ijik = generation_BigMat(self, [i, j, i, k], S_matrix_cp, self.nosv_cp, self.nosv, self.no)
                    S_ikij = generation_SuperMat([i, k, i, j], self.S_matrix, self.nosv, self.no)
                    Ds_ikij = get_dm(T_ik, S_ikij, T_bar[ipair])
                    R_ijij -= self.loc_fock[k,j] * ddot(S_cp_ijik, Ds_ikij)
                    S_cp_ijik, S_ikij, Ds_ijij = None, None, None
                if k != i:
                    if k > j:
                        T_kj = flip_ij(j, k, T_matrix[j*self.no+k], self.nosv)
                    else:
                        T_kj = T_matrix[k*self.no+j]
                    S_cp_ijkj = generation_BigMat(self, [i, j, k, j], S_matrix_cp, self.nosv_cp, self.nosv, self.no)
                    S_kjij = generation_SuperMat([k, j, i, j], self.S_matrix, self.nosv, self.no)
                    Ds_kjij = get_dm(T_kj, S_kjij, T_bar[ipair])
                    R_ijij -= self.loc_fock[i,k] * ddot(S_cp_ijkj, Ds_kjij)
                    S_cp_ijkj, S_kjij, Ds_kjij = None, None, None
            
        if (self.is_remote[ipair]):
            N_i = 2*R_ii
            N_j = 2*R_jj
        else:
            R_ijij *= 2
            N_i = R_ijij[:self.nosv_cp[i], :self.nosv[i]]
            N_j = R_ijij[self.nosv_cp[i]:, self.nosv[i]:]
            if (i != j):
                N_i *= 2
                N_j *= 2
        N[i] += N_i
        N[j] += N_j
        return N
    
    if self.lg_correct:
        pairlist = self.pairlist
    else:
        pairlist = self.pairlist_close

    if (self.clus_type == 3) or (self.clus_type == 21):
        pairlist_ij = []
        for ipair in pairlist:
            if ipair//self.no != ipair%self.no:
                pairlist_ij.append(ipair)
        pairlist = pairlist_ij

    if len(pairlist) == 0:
        N = None
    else:
        N = [None]*self.no
        for i in self.mo_list:
            N[i] = np.zeros((self.nosv_cp[i], self.nosv[i]))
        for ipair in pairlist:
            N = residue_eval(ipair, N)
    return N 

def update_cpn(self, N):
    for i in self.mo_list:
        for m in range(self.nosv_cp[i]):
            for n in range(self.nosv[i]):
                delta = self.s[i][n] - self.s[i][m]
                if (abs(delta) <= 1.0e-6):
                    N[i][m, n] = np.float64(0)
                else:
                    N[i][m, n] = N[i][m, n]/delta
    return N
