import time
import numpy as np
from pyscf import lib,ao2mo,scf,gto,lib,df
from pyscf.lib import logger
from pyscf.scf import _vhf,cphf
from pyscf.lib.numpy_helper import transpose,ddot
from pyscf.scf import _vhf
import OSVL
from pyscf import grad
from OSVL import generation_SuperMat
from  osvutil import *
import os
from df_addons import *
from addons import *
from CPOSV  import cposv
import scipy
global B_ijkl
def flip_ij(index,no):
    i = index// no
    j = index % no
    return j*no+i
def pack_mat(A,dim):
    list =[]
    for i in range(0,dim):
         for j in range(0,i): 
              
              list.append(A[i,j]-A[j,i])
    return list
def pack_mat_upper(A,dim):
    list =[]
    for i in range(0,dim):
         for j in range(0,i):

              list.append(A[i,j])
    return list

def unpack_mat(list,dim,pair):
    A = np.zeros((dim,dim))
    for ij in pair:
         ele = list[pair.index(ij)]
         i = ij // dim
         j = ij %  dim
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

def solve_CPL(self,theta):
   log = lib.logger.Logger(self.RHF.stdout, self.RHF.verbose)
   t0=get_current_time()

   Ovlp = self.RHF.get_ovlp()

   pair = [] 
   for i in range(self.no):
       for j in range(0,i):
               pair.append(i*self.no+j)
   
   partition = genAtomPartition(self.mol)
   natm = len(partition)
   coeff = np.concatenate((self.o,self.v), axis=1)
   Sapi= np.zeros((natm,self.dim,self.no))
   Ovlp = ddot(Ovlp,coeff)
   
   Lqi   = []
   Lqq   = []
   Ovlp_buff  =[]
   Ovlpo_buff =[]
   for index in partition:
       Lqi.append(self.o[index])
       Lqq.append(coeff[index])
       Ovlp_buff.append(Ovlp[index])
       Ovlpo_buff.append(Ovlp[index,:self.no])

   Sapi += np.asarray(pmmul(Lqi,Ovlp_buff,trans_a=1)).transpose(0,2,1)
   Sapi += np.asarray(pmmul(Lqq,Ovlpo_buff,trans_a=1))
   Sapi = 0.5*Sapi
   Ovlp_buff = None
   Ovlpo_buff = None
   Lqq = None 
   plen = len(pair)
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

               B_ijkl[i1,i2]+=2*np.dot(Sapi[:,i,j],Sapi[:,k,l])
               B_ijkl[i1,i2] += 0.5*(Sapi[:,j,l]*(Sapi[:,i,i]+Sapi[:,k,k]-Sapi[:,j,j]-Sapi[:,l,l])).sum()
           if (i==l):
               B_ijkl[i1,i2]-=2*np.dot(Sapi[:,i,j],Sapi[:,l,k])
               B_ijkl[i1,i2] -= 0.5*(Sapi[:,j,k]*(Sapi[:,i,i]+Sapi[:,l,l]-Sapi[:,i,i]-Sapi[:,k,k])).sum()
           if (j==k):
               B_ijkl[i1,i2]-=2*np.dot(Sapi[:,j,i],Sapi[:,k,l])
               B_ijkl[i1,i2] -= 0.5*(Sapi[:,i,l]*(Sapi[:,j,j]+Sapi[:,k,k]-Sapi[:,i,i]-Sapi[:,l,l])).sum()
           if (j==l):
               B_ijkl[i1,i2]+=2*np.dot(Sapi[:,j,i],Sapi[:,l,k])
               B_ijkl[i1,i2] += 0.5*(Sapi[:,i,k]*(Sapi[:,j,j]+Sapi[:,l,l]-Sapi[:,i,i]-Sapi[:,k,k])).sum()
           
   theta_cpl = np.asarray(pack_mat(theta,self.no))
   zloc = scipy.linalg.solve(B_ijkl,-theta_cpl)

   zloc = unpack_mat(zloc,self.no,pair)
   B = np.zeros((self.mol.natm,self.no,self.no))
   for i in range(0,self.mol.natm):
           B[i,:,:]+= 2*np.diag(np.einsum('ij,ij->i',zloc,Sapi[i,:self.no]))
           S = np.diagonal(Sapi[i,:self.no])
           Skl = lib.numpy_helper.direct_sum('k,l->kl',-S,S)
           B[i,:,:]+= Skl*zloc
   azloc = np.einsum('api,aij->apj',Sapi,B)
   azloc = np.einsum('api->pi',azloc)
   S_loc = np.concatenate(pmmul(Lqi,B),axis=0)
   S_loc = ddot(S_loc,self.o.T)
   return azloc,S_loc
