import os
import time
import numpy as np
from pyscf import lib, lib
from pyscf.lib.numpy_helper import ddot
from osvmp2.osvutil import *
import scipy
from pyscf.lo import orth, nao
def flip_ij(index,no):
    i = index// no
    j = index % no
    return j*no+i
def pack_mat(A,list):
    lst =[]
    for i in list:
         for j in list:
            if (i<j):   
              lst.append(A[i,j]-A[j,i])
    
    return lst

def pack_mat_upper(A,dim):
    list =[]
    for i in range(0,dim):
         for j in range(0,i):

              list.append(A[i,j])
    return list

def unpack_mat(list,dim,pair,lst):
    A = np.zeros((len(lst),len(lst)))
    for ij in pair:
         ele = list[pair.index(ij)]
         i = ij // dim
         j = ij %  dim
         i = lst.index(i)
         j = lst.index(j)
         A[i,j]= ele
         A[j,i]= -ele
    return A       
def unpack_mat_upper(list,dim,pair):
    A = np.zeros((dim,dim))
    for ij in pair:
         ele = list[pair.index(ij)]
         i = ij //dim
         j = ij % dim
         A[i,j]= ele
    return A
def generate_upper(A,no):
    B= np.zeros(A.shape) 
    for i in range(0,no):
           for j in range(0,no):
               if (i>j):
                   B[i,j]= A[i,j]
    return B 


def genAtomPartition(mol):
   part = {}
   for iatom in range(mol.natm):
      part[iatom]=[]
   ncgto = 0
   for binfo in mol._bas:
      atom_id = binfo[0]
      lang = binfo[1]
      ncntr = binfo[3]
      nbas = ncntr*(2*lang+1)
      part[atom_id]+=range(ncgto,ncgto+nbas)
      ncgto += nbas
   partition = []
   for iatom in range(mol.natm):
      partition.append(part[iatom])
   return partition

def solve_CPL(self,theta,list):
   
   mol = self.mol
   log = lib.logger.Logger(self.RHF.stdout, self.RHF.verbose)
   t0=(logger.process_clock(), logger.perf_counter())

   s = Ovlp = self.RHF.get_ovlp()
   
   pair = [] 
   for i in list:
       for j in list:
           if (i<j):
               pair.append(i*self.no+j)
   
   partition = genAtomPartition(self.mol)
   natm = len(partition)
   coeff = np.concatenate((self.o,self.v), axis=1)
   Qapi= np.zeros((natm,self.dim,self.no))
   c = restore_ao_character(mol, 'ANO')

   pre_orth_ao = project_to_atomic_orbitals(mol, "ANO")

   weight = numpy.ones(pre_orth_ao.shape[0])
   core_lst, val_lst, rydbg_lst = nao._core_val_ryd_list(mol)
   nbf = mol.nao_nr()
   pre_nao = pre_orth_ao.astype(s.dtype)
   cnao = numpy.empty((nbf,nbf), dtype=s.dtype)
   if core_lst:
        c = pre_nao[:,core_lst].copy()
        s1 = reduce(lib.dot, (c.conj().T, s, c))
        cnao[:,core_lst] = c1 = lib.dot(c, lowdin(s1))
        c = pre_nao[:,val_lst].copy()
        c -= reduce(lib.dot, (c1, c1.conj().T, s, c))
        
        hat_c_core= bar_c_core = pre_nao[:,core_lst].copy()
        s_core = s1.copy()
        x_core, X_core = scipy.linalg.eigh(s_core)
        s_core_lowdin = lowdin(s_core)
        c_core = c1.copy()
      
        bar_c_val = pre_nao[:,val_lst].copy()
        hat_c_val = c.copy()
        D_core = np.einsum('i,j->ij',x_core,x_core**0.5)
        D_core += D_core.T

   else:
        c = pre_nao[:,val_lst]

        bar_c_val = hat_c_val = pre_nao[:,val_lst].copy()
        

   if val_lst:
        s1 = reduce(lib.dot, (c.conj().T, s, c))
        wt = weight[val_lst]
        cnao[:,val_lst] = lib.dot(c, lowdin(s1))
        
        s_val = s1.copy()
        x_val, X_val = scipy.linalg.eigh(s_val)
        c_val = cnao[:,val_lst].copy()
        s_val_lowdin = lowdin(s_val)
        D_val = np.einsum('i,j->ij',x_val,x_val**0.5)
        D_val += D_val.T

   if rydbg_lst:
        cvlst = core_lst + val_lst
        c1 = cnao[:,cvlst].copy()
        c = pre_nao[:,rydbg_lst].copy()
        c -= reduce(lib.dot, (c1, c1.conj().T, s, c))
        s1 = reduce(lib.dot, (c.conj().T, s, c))
        cnao[:,rydbg_lst] = lib.dot(c, lowdin(s1))
        
        bar_c_vir = pre_nao[:,rydbg_lst].copy()
        hat_c_vir = c.copy()
        c_vir = cnao[:,rydbg_lst]
        s_vir = s1.copy()
        x_vir, X_vir = scipy.linalg.eigh(s_vir)
        s_vir_lowdin = lowdin(s_vir)
        D_vir = np.einsum('i,j->ij',x_vir,x_vir**0.5)
        D_vir += D_vir.T
   
   c_orth = cnao

   csc = reduce(lib.dot, (coeff.T, s, c_orth))
   csc_list =[]
   for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            Qapi[i] = numpy.dot(csc[:,p0:p1], csc[:self.no,p0:p1].conj().T)
            csc_list.append(c_orth[:,p0:p1])
   plen = len(pair)
   #print s_core_lowdin-s_core_lowdin.T
   #print s_vir_lowdin-s_vir_lowdin.T
   #print s_val_lowdin-s_val_lowdin.T
   #for ij in pair:
   #      i = ij // self.no
   #      j = ij %  self.no
   #      print ((Qapi[:,i,i]-Qapi[:,j,j])*Qapi[:,i,j]).sum()
   B_ijkl = np.zeros((plen,plen))
   for ij in pair:
      for kl in pair:
           i1 = pair.index(ij)
           i2 = pair.index(kl)
           i= ij//self.no
           j= ij% self.no
           k= kl//self.no
           l= kl %self.no
           if (i==k):

               B_ijkl[i1,i2]+=2*np.dot(Qapi[:,i,j],Qapi[:,k,l])
               B_ijkl[i1,i2] += 0.5*(Qapi[:,j,l]*(Qapi[:,i,i]+Qapi[:,k,k]-Qapi[:,j,j]-Qapi[:,l,l])).sum()
           if (i==l):
               B_ijkl[i1,i2]-=2*np.dot(Qapi[:,i,j],Qapi[:,l,k])
               B_ijkl[i1,i2] -= 0.5*(Qapi[:,j,k]*(Qapi[:,i,i]+Qapi[:,l,l]-Qapi[:,j,j]-Qapi[:,k,k])).sum()
           if (j==k):
               B_ijkl[i1,i2]-=2*np.dot(Qapi[:,j,i],Qapi[:,k,l])
               B_ijkl[i1,i2] -= 0.5*(Qapi[:,i,l]*(Qapi[:,j,j]+Qapi[:,k,k]-Qapi[:,i,i]-Qapi[:,l,l])).sum()
           if (j==l):
               B_ijkl[i1,i2]+=2*np.dot(Qapi[:,j,i],Qapi[:,l,k])
               B_ijkl[i1,i2] += 0.5*(Qapi[:,i,k]*(Qapi[:,j,j]+Qapi[:,l,l]-Qapi[:,i,i]-Qapi[:,k,k])).sum()
   theta_cpl = np.asarray(pack_mat(theta,list))
   
   zloc = scipy.linalg.solve(B_ijkl,-theta_cpl)
   for i in range(zloc.shape[0]):
         if (abs(zloc[i])>100):
                  zloc[i]=0.0
          
   zloc = unpack_mat(zloc,self.no,pair,list)
   b = np.zeros((self.mol.natm,len(list),len(list)))
   for i in range(0,self.mol.natm):
           b[i,:,:]+= 2*np.diag(np.einsum('ij,ij->i',zloc,Qapi[i,list][:,list]))
           S = np.diagonal(Qapi[i,list][:,list])
           Skl = lib.numpy_helper.direct_sum('k,l->kl',-S,S)
           b[i,:,:]+= Skl*zloc
   azloc = np.einsum('api,aij->apj',Qapi[:,:,list],b)
   azloc = np.einsum('api->pi',azloc)
   s = Ovlp = self.RHF.get_ovlp()

   tmp_A = ddot(self.o[:,list].T,s)
   A = pmmul([tmp_A]*len(csc_list),csc_list)
   A  = pmmul(b,A)
   
   B = pmmul([tmp_A]*len(csc_list),A,trans_a=1)
   B = np.concatenate(B,axis =1)
   I = np.zeros((self.dim,self.dim))
   if (rydbg_lst):
             E_vir = -ddot(hat_c_vir.T,B[:,rydbg_lst])
             E_vir = reduce(ddot,(X_vir.T,E_vir,X_vir))
             E_vir = E_vir/D_vir
             E_vir = reduce(ddot,(X_vir,E_vir,X_vir.T))
             F_vir = -reduce(ddot,(s,hat_c_vir,E_vir+E_vir.T,bar_c_vir.T))- reduce(ddot,(B[:,rydbg_lst],s_vir_lowdin,bar_c_vir.T))
             if (core_lst):
                    H_vir = reduce(ddot,(F_vir,s,c_core))+ reduce(ddot,(s,F_vir.T,c_core))
                    I += reduce(ddot,(c_core,c_core.T,F_vir))
             if (val_lst):
                    G_vir = reduce(ddot,(F_vir,s,c_val))+ reduce(ddot,(s,F_vir.T,c_val))
                    I += reduce(ddot,(c_val,c_val.T,F_vir))
             I += reduce(ddot,(hat_c_vir,E_vir,hat_c_vir.T))

   if (val_lst):
      G_val = B[:,val_lst]

      if (rydbg_lst):
             G_val += G_vir
      E_val = -ddot(hat_c_val.T,G_val)
      E_val = reduce(ddot,(X_val.T,E_val,X_val))
      E_val = E_val/D_val
      E_val = reduce(ddot,(X_val,E_val,X_val.T))
      I += reduce(ddot,(hat_c_val,E_val,hat_c_val.T))
      F_val = -reduce(ddot,(s,hat_c_val,E_val+E_val.T,bar_c_val.T))-reduce(ddot,(G_val,s_val_lowdin,bar_c_val.T))
      if (core_lst):
                    H_val = reduce(ddot,(F_val,s,c_core))+ reduce(ddot,(s,F_val.T,c_core))
                    I += reduce(ddot,(c_core,c_core.T,F_val))

   if (core_lst):
        H_core = B[:,core_lst]
        if (rydbg_lst):
                 H_core += H_vir
        if (val_lst):
                 H_core += H_val

        E_core = -ddot(hat_c_core.T,H_core)
        E_core = reduce(ddot,(X_core.T,E_core,X_core))
        E_core = E_core/D_core
        E_core = reduce(ddot,(X_core,E_core,X_core.T))
        I   += reduce(ddot,(bar_c_core,E_core,bar_c_core.T))
   
   S_loc = np.concatenate(A,axis =1)
   S_loc = np.dot(S_loc,c_orth.T)
   S_loc = np.dot(self.o[:,list],S_loc)
   S_loc += I.copy()
   #S_loc  = I.copy()
   return azloc,S_loc
