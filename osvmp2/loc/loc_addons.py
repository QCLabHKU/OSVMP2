import time
from functools import reduce
import numpy as np
from scipy.optimize import leastsq, least_squares
import types
import h5py
import copy

import os
import pyscf
from pyscf import lib, lo, df, gto
from pyscf.lib import logger
from pyscf.lo import orth
from pyscf.soscf import ciah
from pyscf.gto.moleintor import make_loc
from pyscf.tools.cubegen import Cube
from pyscf.dft import numint

from mpi4py import MPI
#from osvmp2.loc import orth, ciah
from osvmp2.loc.hirshfeld import hirshfeld_chg
from osvmp2.loc import loc_jacobi
from osvmp2.__config__ import inputs
from osvmp2.osvutil import *
from osvmp2.ga_addons import *
#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()    # Size of communicator
irank = comm.Get_rank()    # Ranks in communicator
inode = MPI.Get_processor_name()     # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//nrank_shm

EA2DEBYE = 4.80320
ANG2BOHR = 1.8897259886

def get_ncore(mol):
    ncore = 0
    for atm in range(mol.natm):
        if mol.atom_charge(atm) < 5:
            ncore += 0
        elif 5 <= mol.atom_charge(atm) <= 12:
            ncore += 1
        elif 13 <= mol.atom_charge(atm) <= 30:
            ncore += 5
        elif 31 <= mol.atom_charge(atm) <= 38:
            ncore += 9
        elif 39 <= mol.atom_charge(atm) <= 48:
            ncore += 14
        elif 49 <= mol.atom_charge(atm) <= 56:
            ncore += 18
        else:
            raise ValueError('No core orbital options for atom %s'%mol.atom[atm][0])
    return ncore

def atomic_pops(self, mol, mo_coeff, method='meta_lowdin'):
    s = self.s
    nmo = mo_coeff.shape[1]
    proj = np.empty((mol.natm,nmo,nmo))
    #t0 = get_current_time()
    if method.lower() == 'mulliken':
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            csc = multi_dot([mo_coeff[p0:p1].conj().T, s[p0:p1], mo_coeff])
            proj[i] = (csc + csc.conj().T) * .5

    elif method.lower() in ('lowdin', 'meta_lowdin'):
        if method.lower() == 'lowdin':
            orthao = orth.orth_ao(mol, method, None, s=s)
        else:
            c = lo.orth.restore_ao_character(mol, 'ANO')
            orthao = orth.orth_ao(mol, method, c, s=s)
        
        csc = multi_dot([mo_coeff.conj().T, s, orthao])
        for atm, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            proj[atm] = np.dot(csc[:,p0:p1], csc[:,p0:p1].conj().T)
        
    else:
        raise KeyError('method = %s' % method)
    #self.t_chg += get_current_time() - t0
    #print_time(['POPS!!!', get_current_time() - t0])
    self.csc = csc
    #self.pop = proj
    return proj


def kernel(self, mo_coeff=None, pop_method=None, callback=None, verbose=3):
    def atomic_init_guess(mol, mo_coeff, s, c):
        import scipy.linalg.interpolative as sli
        mo = multi_dot([c.conj().T, s, mo_coeff])
        idx = np.argsort(np.einsum('pi,pi->p', mo.conj(), mo))
        nmo = mo.shape[1]
        idx = sorted(idx[-nmo:])

        # Rotate mo_coeff, make it as close as possible to AOs
        u, w, vh = np.linalg.svd(mo[idx])
        return lib.dot(u, vh).conj().T
    def get_ovlp_int():
        if getattr(self.mol, 'pbc_intor', None):  # whether mol object is a cell
            s = self.mol.pbc_intor('int1e_ovlp_sph', hermi=1)
        else:
            s = self.mol.intor_symmetric('int1e_ovlp')

        s12 = orth.lowdin(s)
        #if irank == 0:
        with h5py.File('int1e_ovlp_%d.tmp'%irank, 'w') as f:
            f.create_dataset('s', shape=s.shape, dtype='f8')
            f.create_dataset('s12', shape=s12.shape, dtype='f8')
            f['s'][:] = s
            f['s12'].write_direct(s12)
        return s, s12
        
    from pyscf.tools import mo_mapping
    #setattr(self, 't_chg', create_timer())
    #self.t_chg = create_timer()
    if pop_method == 'mul_melow':
        dual_loc = True
        self.pop_method = 'mulliken'
    elif pop_method == 'low_melow':
        dual_loc = True
        self.pop_method = 'lowdin'
    else:
        dual_loc = False
        if pop_method != None:
            self.pop_method = pop_method

    if mo_coeff is not None:
        self.mo_coeff = np.asarray(mo_coeff, order='C')
    nao, nocc = self.mo_coeff.shape
    if nocc <= 1:
        return np.ones((nocc, nocc))

    if self.verbose >= logger.WARN:
        self.check_sanity()
    #self.dump_flags()

    cput0 = get_current_time()
    log = logger.new_logger(self, verbose=verbose)

    if self.conv_tol_grad is None:
        conv_tol_grad = np.sqrt(self.conv_tol*.1)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)
    else:
        conv_tol_grad = self.conv_tol_grad

    #Compute overlap matrix S and S^-1/2
    self.s, s12 = get_ovlp_int()

    if mo_coeff is None:
        if getattr(self, 'mol', None) and self.mol.natm == 0:
            # For customized Hamiltonian
            uo = self.get_init_guess('random')
        else:
            uo = atomic_init_guess(self.mol, self.mo_coeff, self.s, s12)
    else:
        uo = self.get_init_guess(None)

    #cput01 = print_time(['initial guess of uo', get_elapsed_time(cput0)], log)

    rotaiter = ciah.rotate_orb_cc(self, uo, conv_tol_grad, verbose=verbose)
    u, g_orb, stat = next(rotaiter)
    #log = logger.new_logger(self, verbose=5)
    cput1 = print_time(['initializing CIAH', get_elapsed_time(cput0)], log)

    tot_kf = stat.tot_kf
    tot_hop = stat.tot_hop
    conv = False
    e_last = 0
    for imacro in range(self.max_cycle):
        norm_gorb = np.linalg.norm(g_orb)
        uo = lib.dot(uo, u)
        e = self.cost_function(uo)
        e_last, de = e, e-e_last

        log.info('macro= %d  f(x)= %.14g  delta_f= %g  |g|= %g  %d KF %d Hx',
                 imacro+1, e, de, norm_gorb, stat.tot_kf+1, stat.tot_hop)
        #cput1 = print_time(['cycle= %d'%(imacro+1), get_elapsed_time(cput1)], log)

        if (norm_gorb < conv_tol_grad and abs(de) < self.conv_tol):
            meth_now = self.pop_method.lower()
            if dual_loc and (meth_now == 'mulliken' or meth_now == 'lowdin'):
                self.pop_method = 'meta_lowdin'
            else:
                conv = True
        if callable(callback):
            callback(locals())
        if conv:
            break

        u, g_orb, stat = rotaiter.send(uo)
        tot_kf += stat.tot_kf
        tot_hop += stat.tot_hop

    rotaiter.close()
    log.info('macro X = %d  f(x)= %.14g  |g|= %g  %d intor %d KF %d Hx',
             imacro+1, e, norm_gorb,
             (imacro+1)*2, tot_kf+imacro+1, tot_hop)
# Sort the localized orbitals, to make each localized orbitals as close as
# possible to the corresponding input orbitals
    if irank == 0:
        nocc = self.mol.nelectron//2
        nmo = self.mo_coeff.shape[1]
        ncore = get_ncore(self.mol)
        #pcharge = np.einsum('xii->xi', self.pop)**2
        '''pcharge = self.pop
        with h5py.File('pcharge.tmp', 'r+') as file_pc:
            if nmo == (nocc-ncore) or (nmo > nocc/2 and nmo < nocc):
                file_pc['charge'][:, ncore:] = pcharge
            else:
                file_pc['charge'][:, :nmo] = pcharge'''
        pcharge = np.zeros((nmo, self.mol.nbas))
        ao_loc = make_loc(self.mol._bas, 'sph')
        for s in range(self.mol.nbas):
            p0, p1 = ao_loc[s], ao_loc[s+1]
            charge_i = np.dot(self.csc[:,p0:p1], self.csc[:,p0:p1].conj().T)
            pcharge[:, s] = np.diag(charge_i)
        '''with h5py.File('pcharge.tmp', 'r+') as file_pc:
            if nmo == (nocc-ncore) or (nmo > nocc/2 and nmo < nocc):
                file_pc['charge'][ncore:] = pcharge
            else:
                file_pc['charge'][:nmo] = pcharge'''
    sorted_idx = mo_mapping.mo_1to1map(uo)
    '''self.mo_coeff = lib.dot(self.mo_coeff, uo[:,sorted_idx])
    return self.mo_coeff'''
    #print_time(['charge', self.t_chg], log)
    
    return uo[:,sorted_idx]

def pivoted_cholesky_decompose(A, tol=1e-6, low=True, max_no=None):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    if max_no is None:
        max_no = n 

    L = np.zeros((n, n), dtype=float)

    d = np.diag(A).copy()
    error = np.sum(np.abs(d))  # L1 norm
    pi = np.arange(n)
    m = 0
    
    while error > tol and m < max_no:
        # Fix 1: Add m to get absolute index
        i = m + np.argmax(d[pi[m:n]])

        # Swap pi[m] and pi[i]
        old_pi_m = pi[m].copy()
        pi[m] = pi[i]
        pi[i] = old_pi_m
        #pi[m], pi[i] = pi[i], pi[m]

        L[m, pi[m]] = np.sqrt(d[pi[m]])

        for i in range(m+1, n):
            # Fix 2: Use :m instead of :m-1 (m here denotes a 0-based index, not a number)
            sum_term = np.dot(L[:m, pi[m]], L[:m, pi[i]])
            L[m, pi[i]] = (A[pi[m], pi[i]] - sum_term) / L[m, pi[m]]
            d[pi[i]] -= L[m, pi[i]]**2
        
        # Fix 3: Correct error calculation
        error = np.sum(d[pi[m+1:n]])
        m += 1


    return L[:m].T if low else L[:m]

def localization(mol, mo_coeff, local_type=1, pop_method='low_melow', smat=None, 
                 cal_grad=True, use_sl=False, frozen=False, loc_fit=False, verbose=3,
                 log=None, use_cpp=True, use_gpu=False, loc_tol=1e-6, iop=1):
    def loc_mo(mo_coeff, smat=None):
        
        if local_type == 1:
            '''loc = lo.PM(mol, mo_coeff)
            loc.atomic_pops = types.MethodType(atomic_pops, loc)
            loc.kernel = types.MethodType(kernel, loc)
            uo = loc.kernel(pop_method=pop_method, verbose=verbose)'''
            
            uo = loc_jacobi.loc(mol, mo_coeff, iop=iop, smat=smat, verbose=verbose, 
                                use_cpp=use_cpp, use_gpu=use_gpu, tol=loc_tol) 
        elif local_type == 2:
            uc = lo.Boys(mol, mo_coeff).kernel(verbose=verbose)
            uo = multi_dot([mo_coeff.T, smat, uc])
        return uo
    def get_uo(smat):
        nocc = mol.nelectron//2
        ncore = get_ncore(mol)
        if local_type == 3:
            if irank_shm == 0:
                if (use_sl and ncore != 0) or (frozen):
                    uo = np.diag(np.ones(nocc))

                    mo_coeff_core = mo_coeff[:, :ncore]
                    dm_core = np.dot(mo_coeff_core, mo_coeff_core.T)
                    uc_core = pivoted_cholesky_decompose(dm_core, tol=1e-6, low=True, max_no=nocc)
                    uo[:ncore, :ncore] = multi_dot([mo_coeff_core.T, smat, uc_core])

                    mo_coeff_noncore = mo_coeff[:, ncore:]
                    dm_noncore = np.dot(mo_coeff_noncore, mo_coeff_noncore.T)
                    uc_noncore = pivoted_cholesky_decompose(dm_noncore, tol=1e-6, low=True, max_no=nocc)
                    uo[ncore:, ncore:] = multi_dot([mo_coeff_noncore.T, smat, uc_noncore])
                    
                else:
                    dm = np.dot(mo_coeff, mo_coeff.T)
                    uc = pivoted_cholesky_decompose(dm, tol=1e-6, low=True, max_no=nocc)
                    uo = multi_dot([mo_coeff.T, smat, uc])
                    #if irank == 0: print(np.linalg.norm(dm - uc @ uc.T))
                
            else:
                uo = None
        else:
            
            if (use_sl and ncore != 0) or (frozen):
                
                uo_val = loc_mo(mo_coeff[:, ncore:], smat)

                if irank_shm == 0:
                    uo = np.diag(np.ones(nocc))
                    uo[ncore:, ncore:] = uo_val
                else:
                    uo = None

                if frozen or ncore == 1:
                    if irank_shm == 0:
                        uo[:ncore, :ncore] = np.eye(ncore)
                else:
                    uo_core = loc_mo(mo_coeff[:, :ncore], smat)
                    if irank_shm == 0:
                        uo[:ncore, :ncore] = uo_core
            else:
                uo = loc_mo(mo_coeff, smat)
        return uo       
    if irank == 0:
        if verbose > 4:
            verbose = 4
        else:
            verbose = verbose
    else:
        verbose = 0
    if cal_grad and loc_fit and frozen:
        use_sl = True
        frozen = False
    ncore = get_ncore(mol)
    #Localized orbitals(1: Pipek-Mezey, 2: Foster-Boys)
    if use_sl or frozen:
        log.info("Split localization: ON (%d core orbitals)"%ncore)
    if smat is None:
        smat = get_ovlp(mol)
    if local_type == 1:
        if pop_method == 'mul_melow':
            pop_msg = "Mulliken-Meta_Lowdin"
        elif pop_method == "low_melow":
            pop_msg = "Lowdin-Meta_Lowdin"
        else:
            pop_msg = pop_method.title()
        log.info('Localization method: Pipek-Mezey with %s'%pop_msg)
    elif local_type == 2:
        log.info('Localization method: Foster-Boys')
    uo = get_uo(smat)
    return uo

def LMO_domains(mol, mo_coeff, frozen=False, tol_lmo=1e-6):
    nocc = mo_coeff.shape[1]
    mo_list = range(nocc)
    if frozen:
        ncore = get_ncore(mol)
        mo_list = mo_list[ncore:]
    lmo_close = [None]*nocc
    nlmo_close = [None]*nocc
    atm_offset = mol.offset_nr_by_atom()
    for i in mo_list:
        lmo_close_i = []
        nlmo_i = 0
        for atm, (s0, s1, al0, al1) in enumerate(atm_offset):
            if np.sum(mo_coeff[al0:al1, i]**2) > tol_lmo:
                nlmo_i += al1 - al0
                if lmo_close_i == []:
                    lmo_close_i.append([al0, al1])
                else:
                    if lmo_close_i[-1][-1] == al0:
                        lmo_close_i[-1][-1] = al1
                    else:
                        lmo_close_i.append([al0, al1])
        lmo_close[i] = lmo_close_i
        nlmo_close[i] = nlmo_i
    return lmo_close, nlmo_close

#def moco_fit(mol, mo_coeff, lmo_close, lmo_remote, nlmo_close):
def moco_fit(mol, mo_coeff, lmo_close, nlmo_close, frozen=False):
    def screenig(moco, lmo_remote):
        for i, lmo_slice in enumerate(lmo_remote):
            if lmo_slice is not None:
                for al0, al1 in lmo_slice:
                    moco[al0:al1, i] = 0.0
        return moco
    def residue(x):
        dif = 1 - np.dot(x, vi)
        '''if irank == 0:
            print(irank, i, dif)'''
        return dif
    nao, nocc = mo_coeff.shape
    #moco_fit = np.zeros_like(mo_coeff)
    win_moco_fit, moco_fit = get_shared((nao, nocc), set_zeros=True)
    win_S, S = get_shared((nao, nao))
    if irank_shm == 0:
        S[:] = mol.intor_symmetric('int1e_ovlp')
    comm_shm.Barrier()
    mo_list = range(nocc)
    if frozen:
        ncore = get_ncore(mol)
        mo_list = mo_list[ncore:]
    mo_slice = mo_slice = get_slice(job_list=mo_list, core_list=range(nrank))[irank]
    use_fit = True
    if mo_slice is not None:
        for i in mo_slice:
            if nlmo_close[i] is not None:
                phi_i = np.sum(mo_coeff[:, i])
                Sa = np.empty((nlmo_close[i], nlmo_close[i]))
                Sv = np.empty((nlmo_close[i], nao))
                moco_i = np.empty(nlmo_close[i])
                idx_al0 = 0
                for al0, al1 in lmo_close[i]:
                    idx_al1 = idx_al0 + (al1-al0)
                    #Sa[idx_al0:idx_al1, idx_al0:idx_al1] = S[al0:al1, al0:al1]
                    idx_be0 = 0
                    for be0, be1 in lmo_close[i]:
                        idx_be1 = idx_be0 + (be1-be0)
                        Sa[idx_al0:idx_al1, idx_be0:idx_be1] = S[al0:al1, be0:be1]
                        idx_be0 = idx_be1
                    Sv[idx_al0:idx_al1] = S[al0:al1]
                    moco_i[idx_al0:idx_al1] = mo_coeff[al0:al1, i]
                    idx_al0 = idx_al1
                if use_fit:
                    vi = np.dot(Sv, mo_coeff[:, i])
                    moco_x = least_squares(residue, moco_i).x
                else:
                    moco_x = moco_i
                idx_al0 = 0
                for al0, al1 in lmo_close[i]:
                    idx_al1 = idx_al0 + (al1-al0)
                    moco_fit[al0:al1, i] = moco_x[idx_al0:idx_al1]
                    idx_al0 = idx_al1
    comm.Barrier()
    Acc_and_get_GA(moco_fit)
    if irank_shm == 0:
        mo_coeff[:] = moco_fit
    comm_shm.Barrier()
    for win_i in [win_moco_fit, win_S]:
        free_win(win_i)
    return mo_coeff
    #return least_squares(residue, moco_init).x.reshape(nao, nocc)


def AO_domains(mol, lmo_close, shell_slice):
    #shell_slice = reduce(lambda x, y :x+y, shell_slice)
    nocc = len(lmo_close)
    ao_loc = make_loc(mol._bas, 'sph')
    ao_close = [None]*nocc
    for i, lmo_slice in enumerate(lmo_close):
        if lmo_slice is not None:
            ao_close[i] = []
            for BE0, BE1 in lmo_slice:
                for a0, a1, b0, b1 in shell_slice:
                    al0, al1, be0, be1 = [ao_loc[s0] for s0 in (a0, a1, b0, b1)]
                    if (BE0 <= be1) and (be0 <= BE1):
                        ao_close[i] += range(al0, al1)
            ao_close[i] = len(set(ao_close[i]))
    return ao_close

def atomic_dist(mol, unit='ang'):
    atm_dist = np.zeros((mol.natm, mol.natm))
    atom_list = range(mol.natm)
    for atm0 in atom_list:
        co0 = np.asarray(mol.atom[atm0][1])
        atm1_list = atom_list[atm0:]
        for atm1 in atm1_list:
            co1 = np.asarray(mol.atom[atm1][1])
            atm_dist[atm0, atm1] = np.linalg.norm(co1-co0)
            atm_dist[atm1, atm0] = atm_dist[atm0, atm1]
    if unit == 'bohr':
        atm_dist *= 1.8897161646320724
    return atm_dist

def FIT_domains(mol, auxmol, mo_coeff, pcharge, frozen=False, tot_prime=0.2, tot_nsup=2):
    tot_dist = 2*tot_nsup+1
    nocc = mo_coeff.shape[1]
    mo_list = range(nocc)
    if frozen:
        ncore = get_ncore(mol)
        mo_list = mo_list[ncore:]
    aux_loc = make_loc(auxmol._bas, 'sph')
    shell_list = range(mol.nbas)
    shell_close = [None]*nocc
    fit_close = [None]*nocc
    naux_close = [None]*nocc
    for i in mo_list:
        shell_fit = []
        shell_prime = []
        #Primary fitting domains
        for s in shell_list:
            if (pcharge[i, s] > tot_prime):
                
                shell_prime.append(s)
        shell_fit += shell_prime
        '''#Extension of fitting domains
        for s0 in shell_prime:
            atoms_sorted = [s for dist, s in sorted(zip(s_dist[s0], shell_list))]
            #Extend based on connectivity
            for s1 in atoms_sorted[:tot_nsup]:
                if atoms_sel[s1]==False:
                    shell_fit.append(s1)
                    atoms_sel[s1] = True
            #Extend based on distance
            for s1 in atoms_sorted[tot_nsup:]:
                if s_dist[s0, s1] > tot_dist: break
                if (atoms_sel[s1]==False):
                    shell_fit.append(s1)
                    atoms_sel[s1] = True'''
        shell_close[i] = sorted(shell_fit)
        shell_seg = list2seg(shell_close[i])
        fit_close[i] = []
        naux_close[i] = 0
        for s0, s1 in shell_seg:
            p0, p1 = aux_loc[s0], aux_loc[s1]
            fit_close[i].append([p0, p1])
            naux_close[i] += (p1-p0)
        #fit_close[i] = list2seg(fit_close[i])
    return shell_close, fit_close, naux_close

'''def FIT_domains(mol, auxmol, mo_coeff, pcharge, frozen=False, tot_prime=0.2, tot_nsup=2):
    tot_dist = 2*tot_nsup+1
    atm_dist = atomic_dist(mol, unit='bohr')
    nocc = mo_coeff.shape[1]
    mo_list = range(nocc)
    if frozen:
        ncore = get_ncore(mol)
        mo_list = mo_list[ncore:]
    atm_list = range(mol.natm)
    aux_atm_offset = auxmol.offset_nr_by_atom()
    atom_close = [None]*nocc
    fit_close = [None]*nocc
    naux_close = [None]*nocc
    for i in mo_list:
        atom_fit = []
        atom_prime = []
        atoms_sel = [False] * mol.natm
        #Primary fitting domains
        for atm0 in atm_list:
            if (pcharge[atm0, i] > tot_prime) and (atoms_sel[atm0]==False):
                atom_prime.append(atm0)
                atoms_sel[atm0] = True
        atom_fit += atom_prime
        #Extension of fitting domains
        for atm0 in atom_prime:
            atoms_sorted = [atm for dist, atm in sorted(zip(atm_dist[atm0], atm_list))]
            #Extend based on connectivity
            for atm1 in atoms_sorted[:tot_nsup]:
                if atoms_sel[atm1]==False:
                    atom_fit.append(atm1)
                    atoms_sel[atm1] = True
            #Extend based on distance
            for atm1 in atoms_sorted[tot_nsup:]:
                if atm_dist[atm0, atm1] > tot_dist: break
                if (atoms_sel[atm1]==False):
                    atom_fit.append(atm1)
                    atoms_sel[atm1] = True
        atom_close[i] = sorted(atom_fit)
        fit_close[i] = []
        naux_close[i] = 0
        for atm in atom_close[i]:
            s0, s1, p0, p1 = aux_atm_offset[atm]
            #fit_close[i] += range(p0, p1)
            if fit_close[i] == []:
                fit_close[i].append([p0, p1])
            else:
                if fit_close[i][-1][-1] == p0:
                    fit_close[i][-1][-1] = p1
                else:
                    fit_close[i].append([p0, p1])
            naux_close[i] += (p1-p0)
        #fit_close[i] = list2seg(fit_close[i])
    return atom_close, fit_close, naux_close'''

'''def joint_fit_domains(auxmol, mo_list, atom_close, joint_type='union'):
    #union or intersection fitting domains
    atom_joint = set(atom_close[mo_list[0]])
    for j in mo_list[1:]:
        if joint_type == 'union':
            atom_joint = atom_joint.union(atom_close[j])
        else:
            atom_joint = atom_joint.intersection(atom_close[j])
    atom_joint = list2seg(atom_joint)
    atom_offset = auxmol.offset_nr_by_atom()
    naux_joint = 0
    do_joint = []
    for atm0, atm1 in atom_joint:
        atm1 = atm1 - 1
        p0, p1 = atom_offset[atm0][-2], atom_offset[atm1][-1]
        do_joint.append([p0, p1])
        naux_joint += p1 - p0
    return do_joint, naux_joint'''


def list2seg(lst):
    if len(lst) == 0:
        return []
    arr = np.array(lst)
    # Find indices where the difference between consecutive elements > 1
    split_indices = np.where(np.diff(arr) != 1)[0] + 1
    # Split the array into consecutive segments
    segments = np.split(arr, split_indices)
    # Convert segments to [start, end] format
    return [[seg[0], seg[-1] + 1] for seg in segments]

'''def get_fit_domain(mol, auxmol, aux_ratio, fit_tol):
    nocc = mol.nelectron//2
    mo_list = range(nocc)
    fit_list = [[] for i in mo_list]
    fit_seg = [None]*nocc
    nfit = [0]*nocc
    for i in mo_list:
        for p, ratio_p in enumerate(aux_ratio[i]):
            if ratio_p > fit_tol:
                fit_list[i].append(p)
        fit_seg[i] = list2seg(fit_list[i])
        nfit[i] = len(fit_list[i])
    return fit_list, fit_seg, nfit'''

def get_fit_domain(aux_ratio, fit_tol):
    nocc, naux = aux_ratio.shape
    mo_list = range(nocc)
    fit_list = {}
    fit_seg = {}
    nfit = np.zeros(nocc, dtype=int)
    aux_list = np.arange(naux, dtype=np.int32)
    for i in mo_list:
        fit_list[i] = aux_list[aux_ratio[i] > fit_tol]
        nfit[i] = len(fit_list[i])
        fit_seg[i] = list2seg(fit_list[i])
        
    return fit_list, fit_seg, nfit

def get_bfit_domain(mol, auxmol, aux_ratio, bfit_tol):
    nocc = mol.nelectron//2
    natm = auxmol.natm
    mo_list = range(nocc)
    atom_close = [[] for i in mo_list]
    atom_mos = [[] for atm in range(natm)]
    aux_atm_offset = auxmol.offset_nr_by_atom()

    '''for i in mo_list:
        for atm, (_, _, p0, p1) in enumerate(aux_atm_offset):
            if np.amax(aux_ratio[i, p0:p1]) > bfit_tol:
                atom_close[i].append(atm)

    for i, atm_list in enumerate(atom_close):
        for atm in atm_list:
            atom_mos[atm].append(i)'''
    
    p0s = np.array([x[2] for x in aux_atm_offset])
    p1s = np.array([x[3] for x in aux_atm_offset])

    max_ratio = np.zeros((nocc, natm), dtype=np.float32)
    for atm, (p0, p1) in enumerate(zip(p0s, p1s)):
        max_ratio[:, atm] = np.max(aux_ratio[:, p0:p1], axis=1)

    # Boolean mask for atom inclusion
    mask_sel = max_ratio > bfit_tol

    # Precompute atom-orbital mappings 
    atom_close = [np.flatnonzero(row) for row in mask_sel]
    #atom_mos = [np.flatnonzero(mask_sel[:, atm]) for atm in range(natm)]

    bfit_seg = {}
    nbfit = np.zeros(nocc, dtype=int)
    for i in mo_list:
        iseg = []
        infit = 0
        for atm0, atm1 in list2seg(atom_close[i]):
            p0 = aux_atm_offset[atm0, 2]
            p1 = aux_atm_offset[atm1-1, 3]
            iseg.append([p0, p1])
            infit += (p1-p0)
        if len(iseg) > 0:
            bfit_seg[i] = iseg
            nbfit[i] = infit

    '''cal_seg = []
    seg_atm = []
    seg_mo = []
    seg_act = [[] for i in mo_list]
    for atm in range(natm):
        if (atm == 0) or (seg_mo == atom_mos[atm]):
            seg_atm.append(atm)
            if atm == 0:
                seg_mo = [i for i in atom_mos[atm]]
            for i in atom_mos[atm]:
                seg_act[i].append(atm)
        else:
            cal_seg.append([seg_atm, seg_mo, seg_act])
            seg_atm = [atm]
            seg_mo = [i for i in atom_mos[atm]]
            seg_act = [[] for i in mo_list]
            for i in atom_mos[atm]:
                seg_act[i].append(atm)
        if atm == (natm-1):
            cal_seg.append([seg_atm, seg_mo, seg_act])'''
    return atom_close, bfit_seg, nbfit#, cal_seg
        
def joint_fit_domains_by_atom(auxmol, mo_list, atom_close, joint_type='union'):
    #union or intersection fitting domains
    try:
        atom_joint = set(atom_close[mo_list[0]])
    except TypeError:
        print(atom_close[mo_list[0]])

    for j in mo_list[1:]:
        if joint_type == 'union':
            atom_joint = atom_joint.union(atom_close[j])
        else:
            atom_joint = atom_joint.intersection(atom_close[j])

    atom_joint = list2seg(atom_joint)
    aux_loc = make_loc(auxmol._bas, 'sph')
    atm_offset = auxmol.offset_nr_by_atom()
    naux_joint = 0
    do_joint_all = []
    do_joint = []
    for atm0, atm1 in atom_joint:
        p0, p1 = atm_offset[atm0][-2], atm_offset[atm1-1][-1]
        do_joint.append([p0, p1])
        do_joint_all = np.append(do_joint_all, np.arange([p0, p1]))
        naux_joint += p1 - p0
    return do_joint_all, do_joint, naux_joint

def joint_fit_domains_by_aux(auxmol, mo_list, fit_close, joint_type='union'):
    #union or intersection fitting domains
    try:
        fit_joint_all = set(fit_close[mo_list[0]])
    except TypeError:
        print(fit_close[mo_list[0]])

    for j in mo_list[1:]:
        if joint_type == 'union':
            fit_joint_all = fit_joint.union(fit_close[j])
        else:
            fit_joint_all = fit_joint.intersection(fit_close[j])
    fit_joint_all = list(fit_joint_all)
    naux_joint = len(fit_joint_all)
    fit_joint = list2seg(fit_joint_all)
    return fit_joint_all, fit_joint, naux_joint

    
def get_ao_domains(be0, be1, lmo_close, slice_i, i):
    lmo_slice = lmo_close[i]
    if (lmo_slice is None):
        return None
    elif (lmo_slice[0][0] >= be1) or (lmo_slice[-1][-1] <= be0):
        return None
    else:
        cal_slice = []
        for BE0, BE1 in lmo_slice:
            if BE0 >= be1: break
            if BE1 > be0:
                cal_slice.append([max(be0, BE0), min(be1, BE1)])
        #if irank == 0: print(be0, be1, lmo_slice, cal_slice)
        return cal_slice

def half_trans(mol, feri, mo_coeff, lmo_close, fit_close, slice_i, i, buf_feri=None, buf_moco=None, dot=np.dot, out=None):
    al0, al1, be0, be1 = slice_i
    be_idx = [None] * (be1+1)
    for idx, be in enumerate(range(be0, be1+1)):
        be_idx[be] = idx
    cal_slice = get_ao_domains(be0, be1, lmo_close, slice_i, i)
    if cal_slice is None or cal_slice == []:
        return None
    else:
        nao0 = al1 - al0
        nao1 = sum([(BE1-BE0) for BE0, BE1 in cal_slice])
        naux = sum([(p1-p0) for p0, p1 in fit_close[i]])
        
        if buf_feri == None:
            feri_tmp = np.empty((nao1, nao0, naux))
        else:
            feri_tmp = buf_feri[:nao0*nao1*naux].reshape(nao1, nao0, naux)
        if buf_moco == None:
            moco_tmp = np.empty(nao1)
        else:
            moco_tmp = buf_moco[:nao1]
        
        idx_BE0 = 0
        for BE0, BE1 in cal_slice:
            idx_BE1 = idx_BE0 + (BE1-BE0)
            moco_tmp[idx_BE0:idx_BE1] = mo_coeff[BE0:BE1, i]
            idx_be0, idx_be1 = be_idx[BE0], be_idx[BE1]
            idx_p0 = 0
            for p0, p1 in fit_close[i]:
                idx_p1 = idx_p0 + (p1-p0)
                feri_tmp[idx_BE0:idx_BE1, :, idx_p0:idx_p1] = feri[idx_be0:idx_be1, :, p0:p1]
                idx_p0 = idx_p1
            idx_BE0 = idx_BE1
        '''moco_tmp = mo_coeff[be0:be1, i]
        idx_p0 = 0
        for p0, p1 in fit_close[i]:
            idx_p1 = idx_p0 + (p1-p0)
            feri_tmp[:, :, idx_p0:idx_p1] = feri[:, :, p0:p1]
            idx_p0 = idx_p1'''
        try:
            if out is None:
                return dot(moco_tmp.T, feri_tmp.reshape(nao1, -1)).reshape(nao0, naux)
            else:
                return dot(moco_tmp.T, feri_tmp.reshape(nao1, -1), out=out).reshape(nao0, naux)
        except ValueError:
            print('DAMIT', cal_slice)

def slice_fit(var_slice, var_ori, fit_close=None, fit_flat=None, axis=0, reverse=False, accumulate=False):
    idx_p0 = 0
    for p0, p1 in fit_close:
        idx_p1 = idx_p0 + (p1-p0)
        if axis == 0:
            if reverse:
                if accumulate:
                    var_ori[p0:p1] += var_slice[idx_p0:idx_p1]
                else:
                    #var_ori[p0:p1] = var_slice[idx_p0:idx_p1]
                    np.copyto(var_ori[p0:p1], var_slice[idx_p0:idx_p1])
            else:
                if accumulate:
                    var_slice[idx_p0:idx_p1] += var_ori[p0:p1]
                else:
                    #var_slice[idx_p0:idx_p1] = var_ori[p0:p1]
                    np.copyto(var_slice[idx_p0:idx_p1], var_ori[p0:p1])

        elif axis == 1:
            if reverse:
                if accumulate:
                    var_ori[:, p0:p1] += var_slice[:, idx_p0:idx_p1]
                else:
                    #var_ori[:, p0:p1] = var_slice[:, idx_p0:idx_p1]
                    np.copyto(var_ori[:, p0:p1], var_slice[:, idx_p0:idx_p1])
            else:
                if accumulate:
                    var_slice[:, idx_p0:idx_p1] += var_ori[:, p0:p1]
                else:
                    #var_slice[:, idx_p0:idx_p1] = var_ori[:, p0:p1]
                    np.copyto(var_slice[:, idx_p0:idx_p1], var_ori[:, p0:p1])
        elif axis == 2:
            idx_q0 = 0
            for q0, q1 in fit_close:
                idx_q1 = idx_q0 + (q1-q0)
                if reverse:
                    if accumulate:
                        var_ori[p0:p1, q0:q1] += var_slice[idx_p0:idx_p1, idx_q0:idx_q1]
                    else:
                        #var_ori[p0:p1, q0:q1] = var_slice[idx_p0:idx_p1, idx_q0:idx_q1]
                        np.copyto(var_ori[p0:p1, q0:q1], var_slice[idx_p0:idx_p1, idx_q0:idx_q1])
                else:
                    if accumulate:
                        var_slice[idx_p0:idx_p1, idx_q0:idx_q1] += var_ori[p0:p1, q0:q1]
                    else:
                        #var_slice[idx_p0:idx_p1, idx_q0:idx_q1] = var_ori[p0:p1, q0:q1]
                        np.copyto(var_slice[idx_p0:idx_p1, idx_q0:idx_q1], var_ori[p0:p1, q0:q1])
                idx_q0 = idx_q1
        
        idx_p0 = idx_p1
    if reverse:
        return var_ori
    else:
        return var_slice


def get_ovlp(mol):
    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        return mol.pbc_intor('int1e_ovlp', hermi=1)
    else:
        return mol.intor_symmetric('int1e_ovlp')

def pop_analysis(mol, dm, charge_method="meta_lowdin", s=None, pre_orth_ao="ANO", log=None):
    r'''Mulliken population analysis
    .. math:: M_{ij} = D_{ij} S_{ji}
    Mulliken charges
    .. math:: \delta_i = \sum_j M_{ij}
    Inputs:
        dm: density matrix
        charge_method: meta_lowdin, lowdin, mulliken
    Returns:
        A list : pop, charges
        pop : nparray
                Mulliken population on each atomic orbitals
        charges : nparray
                Mulliken charges
    '''
    pop, chg = None, None
    charge_method = charge_method.lower()
    if charge_method in ["hirshfeld", "cm5"]:
        H = hirshfeld_chg.Hirshfeld(mol, dm)
        chg = H.charges()
    elif irank == 0:
        if s is None: s = get_ovlp(mol)
        if charge_method.lower() != "mulliken":
            pre_orth_ao = orth.pre_orth_ao(mol)
            C = orth.orth_ao(mol, method=charge_method, pre_orth_ao=pre_orth_ao, s=s)
            if "lowdin" in charge_method.lower():
                c_inv = np.dot(C.T, s)
                dm = multi_dot([c_inv, dm, c_inv.T.conj()])
            s = np.eye(mol.nao_nr())
        
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            pop = np.einsum('ij,ji->i', dm, s).real
        else: # ROHF
            pop = np.einsum('ij,ji->i', dm[0]+dm[1], s).real
        
        chg = np.zeros(mol.natm)
        for i, s in enumerate(mol.ao_labels(fmt=None)):
            chg[s[0]] += pop[i]
        chg = mol.atom_charges() - chg
    return pop, chg

def center_of_mass(mol):
    masses = mol.atom_mass_list(mol)
    return tuple(np.average(mol.atom_coords(), axis=0, weights=masses))

def dip_moment(mol, dm, unit='Debye', origin_mass=True, log=None):
    r''' Dipole moment calculation
    .. math::
        \mu_x = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|x|\mu) + \sum_A Q_A X_A\\
        \mu_y = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|y|\mu) + \sum_A Q_A Y_A\\
        \mu_z = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|z|\mu) + \sum_A Q_A Z_A
    where :math:`\mu_x, \mu_y, \mu_z` are the x, y and z components of dipole
    moment
    Args:
            mol: an instance of :class:`Mole`
            dm : a 2D ndarrays density matrices
    Return:
        A list: the dipole moment on x, y and z component
    '''

    #log = logger.new_logger(mol, verbose)

    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        # UHF denisty matrices
        dm = dm[0] + dm[1]

    if origin_mass:
        origin = center_of_mass(mol)
    else:
        origin = (0,0,0)
    
    with mol.with_common_orig(origin):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', ao_dip, dm).real

    charges = mol.atom_charges()
    coords  = mol.atom_coords() - np.asarray(origin)
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    mol_dip = nucl_dip - el_dip
    log.info("    Dipole moment: ")
    msg_list = [['', 'X', 'Y', 'Z'],
                    ["Electronic"]+['%8.5f'%i for i in el_dip],
                    ["Nuclear"]+['%8.5f'%i for i in nucl_dip],
                    ["Total (a.u.)"] + ['%8.5f'%i for i in mol_dip],
                    ["Total (Debye)"] + ['%8.5f'%i for i in mol_dip * nist.AU2DEBYE]] 
    print_align(msg_list, align='lrrr', align_1='lccc', indent=4, log=log)
    
    return mol_dip*nist.AU2DEBYE

def create_mep(mol, outfile, dm, nx=40, ny=40, nz=40, resolution=None,
        margin=5, log=None, atom_sel=None):
    
    """Calculates the molecular electrostatic potential (MEP) and write out in
    cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    t1 = get_current_time()
    #atom_sel = range(45)
    if atom_sel is not None:
        mol_sel = gto.Mole()
        mol_sel.atom = [mol._atom[ia] for ia in atom_sel]
        mol_sel.unit = 'Bohr'
        mol_sel.basis = mol.basis
        mol_sel.build()
        cc = Cube(mol_sel, nx, ny, nz, resolution, margin)
    else:
        cc = Cube(mol, nx, ny, nz, resolution, margin)
    print_time(["get cc", get_elapsed_time(t1)], log=log)

    t1 = get_current_time()
    coords = cc.get_coords()

    # Nuclear potential at given points
    npoint = coords.shape[0]
    job_slice = get_slice(job_size=mol.natm, rank_list=range(nrank))[irank]
    win_vnuc, vnuc_node = get_shared(npoint, dtype='f8', set_zeros=True)
    Vnuc = np.zeros(npoint)
    if job_slice is not None:
    #for i in range(mol.natm):
        for i in job_slice:
            r = mol.atom_coord(i)
            Z = mol.atom_charge(i)
            rp = r - coords
            Vnuc += Z / np.einsum('xi,xi->x', rp, rp)**.5
        Accumulate_GA_shm(win_vnuc, vnuc_node, Vnuc)
    
    comm_shm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(vnuc_node)
        comm.Barrier()

    # Potential of electron density
    Vele = np.empty_like(Vnuc)
    win_vele, vele_node = get_shared(npoint, dtype='f8', set_zeros=True)
    nao = dm.shape[0]
    job_slice = get_slice(job_size=npoint, rank_list=range(nrank))[irank]
    if job_slice is not None:
        max_len = len(job_slice)
    else:
        max_len = None
    blksize = get_buff_len(mol, size_sub=nao**2, ratio=0.8, max_len=max_len, min_len=1)

    if job_slice is not None:
        #print(job_slice[0], job_slice[-1]+1)
        for p0, p1 in lib.prange(job_slice[0], job_slice[-1]+1, blksize):
            fakemol = gto.fakemol_for_charges(coords[p0:p1])
            ints = df.incore.aux_e2(mol, fakemol)
            #Vele[p0:p1] = np.einsum('ijp,ij->p', ints, dm)
            vele_node[p0:p1] = np.dot(dm.ravel(), ints.reshape(-1, (p1-p0)))
    comm_shm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(vele_node)
        comm.Barrier()
    if irank == 0:
        MEP = vnuc_node - vele_node     # MEP at each point
        #print(MEP[:10])
        MEP = MEP.reshape(cc.nx,cc.ny,cc.nz)
        print_time(["get mep", get_elapsed_time(t1)], log=log)

        t1 = get_current_time()
        # Write the potential
        cc.write(MEP, outfile, 'Molecular electrostatic potential in real space')
        print_time(["write mep", get_elapsed_time(t1)], log=log)
    else:
        MEP = None
    comm.Barrier()
    for win in [win_vnuc, win_vele]:
        free_win(win)
    return MEP

def create_density(mol, outfile, dm, nx=40, ny=40, nz=40, resolution=None,
            margin=5, atom_sel=None):
    """Calculates electron density and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    from pyscf.pbc.gto import Cell

    if atom_sel is not None:
        mol_sel = gto.Mole()
        mol_sel.atom = [mol._atom[ia] for ia in atom_sel]
        mol_sel.unit = 'Bohr'
        mol_sel.basis = mol.basis
        mol_sel.build()
        cc = Cube(mol_sel, nx, ny, nz, resolution, margin)
    else:
        cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    nao = dm.shape[0]
    job_slice = get_slice(job_size=ngrids, rank_list=range(nrank))[irank]
    if job_slice is not None:
        max_len = len(job_slice)
    else:
        max_len = None
    blksize = get_buff_len(mol, size_sub=nao, ratio=0.8, max_len=max_len, min_len=1)
    #blksize = min(8000, ngrids)
    #rho = np.empty(ngrids)
    win_rho, rho_node = get_shared(ngrids, dtype='f8', set_zeros=True)
    if job_slice is not None:
        for ip0, ip1 in lib.prange(job_slice[0], job_slice[-1]+1, blksize):
            ao = mol.eval_gto(GTOval, coords[ip0:ip1])
            rho_node[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    comm_shm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(rho_node)
        comm.Barrier()
    if irank == 0:
        rho = rho_node.reshape(cc.nx,cc.ny,cc.nz)
        # Write out density to the .cube file
        cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    else:
        rho = None
    free_win(win_rho)
    return rho

def get_dip_charge(coords, charges, origin=(0,0,0), log=None):
    '''if unit == "ea":
        coef_unit = 1
    elif unit == "debye":
        coef_unit = EA2DEBYE
    elif unit == "au":
        coef_unit = ANG2BOHR'''

    coords = np.asarray(coords) #angstrom
    charges = np.asarray(charges)
    origin = np.asarray(origin)

    dip_total = np.dot(charges, coords-origin)

    log.info("    Dipole moment from charges: ")
    msg_list = [['', 'X', 'Y', 'Z'],
                    ["Total (a.u.)"] + ['%8.5f'%i for i in dip_total * ANG2BOHR],
                    ["Total (Debye)"] + ['%8.5f'%i for i in dip_total * EA2DEBYE]] 
    print_align(msg_list, align='lrrr', align_1='lccc', indent=4, log=log)

    return dip_total * EA2DEBYE

def analysis(mol, dm, meth='RHF', charge_method="meta_lowdin", save_data=None, 
             print_pop=False, pre_orth_ao="ano", charge_test=False, log=None,
             gen_mep=False, gen_den=False):
    def data2file(fname, data):
        try:
            with open(fname, 'a') as f:
                f.write(data)
        except IOError:
            with open(fname, 'w') as f:
                f.write(data)
    def save_chg(charge_method, orthao, chg):
        cmethod = charge_method.lower()
        if orthao is not None:
            cmethod += "_%s"%orthao.lower()
        #fname = 'charge_%s_%s_%s.dat'%(cmethod, mol.name, save_data)
        fname = 'charge_%s_%s.dat'%(cmethod, save_data)
        chg_dat = '#Charges for %s with %s\n'%(mol.name, cmethod)
        for ia in range(mol.natm):
            chg_dat += '%s     %15.10f\n'%(mol.atom_symbol(ia), chg[ia])
        data2file(fname, chg_dat)
    def save_dip(fname, mol_dip):
        x, y, z = mol_dip
        dip_dat = '%10.6f     %10.6f     %10.6f\n'%(x, y, z)
        data2file(fname, dip_dat)
    
    t1 = get_current_time()
    #charge_test = bool(int(os.environ.get("charge_test", 0)))
    charge_test = inputs["charge_test"]
    #pre_orth_ao can be ANO/MINAO
    if irank == 0:
        s = get_ovlp(mol)
    else:
        s = None
    #charge_test = True
    if charge_test:
        charge_dic = {}
        for ichg in ["mulliken", "lowdin", "meta_lowdin", "cm5"]:
            if "lowdin" in ichg:
                charge_dic[ichg] = {}
                #for orthao in ["ano"]:#, "minao"]:
                orthao = "ano"
                pop, chg = pop_analysis(mol, dm, charge_method=ichg, s=s, pre_orth_ao=orthao)
                if irank == 0:
                    charge_dic[ichg][orthao] = {"pop": pop, "chg": chg}
                    save_chg(ichg, orthao, chg)
            else:
                pop, chg = pop_analysis(mol, dm, charge_method=ichg, pre_orth_ao=None, s=s)
                if irank == 0:
                    if ichg == "cm5":
                        for idx_chg, chg_name in enumerate(["hirshfeld", "cm5"]):
                            charge_dic[chg_name] = {"pop": pop, "chg": chg[idx_chg]}
                            save_chg(chg_name, None, chg[idx_chg])
                    else:
                        charge_dic[ichg] = {"pop": pop, "chg": chg}
                        save_chg(ichg, None, chg)
        if irank == 0:
            if (pre_orth_ao is not None) and ("lowdin" in charge_method):
                pop = charge_dic[charge_method][pre_orth_ao]["pop"]
                chg = charge_dic[charge_method][pre_orth_ao]["chg"]
            else:
                pop = charge_dic[charge_method]["pop"]
                if charge_method in ["hirshfeld", "cm5"]:
                    chg = (charge_dic["hirshfeld"]["chg"], charge_dic["cm5"]["chg"])
                else:
                    chg = charge_dic[charge_method]["chg"]
            
    else:
        pop, chg = pop_analysis(mol, dm, charge_method, s=s, pre_orth_ao=pre_orth_ao)

    
    if irank == 0:
        log.info("\nProperties of %s"%meth)
        log.info('    '+'-'*30)
        
        chg_method = charge_method.replace("_", " ").title().replace(" ", "-")
        if charge_method == "cm5":
            chg_method = "CM5"
        if ("lowdin" in charge_method) and (pre_orth_ao is not None):
            chg_method = "%s (%s)"%(chg_method, pre_orth_ao)
        if (print_pop) and (pop is not None):
            log.info('    %s populations:'%chg_method)
            msg_list = []
            for i, si in enumerate(mol.ao_labels()):
                si_split = si.split()
                idx, ia = si_split[:2]
                imo = ""
                for ssidx, ss in enumerate(si_split[2:]):
                    if ss == si_split[2:][-1]:
                        imo += ss
                    else:
                        imo += "%s "%ss
                msg_list.append(["%s   "%idx, ia, imo, "%.5f"%pop[i]])
                #log.info('pop of  %s %10.5f', s, pop[i])
            print_align(msg_list, align='rllr', indent=4, log=log)
            log.info('    '+'-'*30)
        if charge_method in ["hirshfeld", "cm5"]:
            chg_ha, chg_cm5 = chg
            msg_list = [["", "", "Hirshfeld", "  CM5"]]
            #for ia in range(mol.natm):
            for ia, (cha, ccm5) in enumerate(zip(chg_ha, chg_cm5)):
                symb = mol.atom_symbol(ia)
                msg_list.append(['%d'%ia, '%s'%symb, '%.5f'%cha, '%.5f'%ccm5])
            print_align(msg_list, align='rrrr', align_1="rrrc", indent=4, log=log)
            chg_test = chg_cm5
        else:
            log.info('    %s atomic charges:'%chg_method)
            msg_list = []
            for ia in range(mol.natm):
                symb = mol.atom_symbol(ia)
                msg_list.append(['%d'%ia, '%s'%symb, '%.5f'%chg[ia]])
            print_align(msg_list, align='rrr', indent=4, log=log)
            chg_test = chg
        
        log.info('    '+'-'*49)
        coords = np.asarray([ico for iasym, ico in mol._atom]) / ANG2BOHR
        dip_chg = get_dip_charge(coords, chg_test, log=log)


        log.info('    '+'-'*49)
        mol_dip = dip_moment(mol, dm, unit='Debye', log=log)
        log.info('    '+'-'*49)
        
        if (save_data is not None):# and hasattr(mol, 'md_step'):
            if charge_test == False:
                if charge_method in ["hirshfeld", "cm5"]:
                    for idx_chg, chg_name in enumerate(["hirshfeld", "cm5"]):
                        save_chg(chg_name, None, chg[idx_chg])
                else:
                    save_chg(charge_method, pre_orth_ao, chg)
            #save_dip('dip_mom_%s_%s.dat'%(mol.name, save_data), mol_dip)
            save_dip(f'dip_mom_{charge_method}_{save_data}.dat', dip_chg)
            save_dip('dip_mom_%s.dat'%(save_data), mol_dip)
    if gen_mep:
        create_mep(mol, 'mep_%s.cube'%(save_data), dm, log=log)
    if gen_den:
        create_density(mol, 'den_%s.cube'%(save_data), dm)#, log=log)
    print_time(["pop analysis", get_elapsed_time(t1)], log=log)