import numpy as np
import time
from osvmp2.ga_addons import *
from osvmp2.__config__ import inputs, ngpu
if inputs["qm_atoms"] is not None:
    from osvmp2.mm import qmmm
from osvmp2.osvutil import *

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    mo0 = mo_coeff[:, mo_occ > 0]
    mo0e = mo0 * (mo_energy[mo_occ > 0] * mo_occ[mo_occ > 0])
    return np.dot(mo0e, mo0.T.conj())

def grad_nuc(mol, atmlst=None):
    gs = np.zeros((mol.natm, 3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.sqrt(np.dot(r1-r2, r1-r2))
                gs[j] -= q1 * q2 * (r2-r1) / r**3
    if atmlst is not None:
        gs = gs[atmlst]
    return gs

def grad_hcore_nuc(self, dm0, dme0, Pi=None, W=None, log=None):
    "Get the gradient of core Hamiltonian and nuclei"
    def get_hcore(mol):
        nao = mol.nao_nr()
        win_hcore, hcore = get_shared((3, nao, nao))
        atom_slice = get_slice(range(nrank), job_size=mol.natm)[irank]
        if atom_slice is not None:
            shell_by_atm = mol.offset_nr_by_atom()
            for ia in atom_slice:
                s0, s1, al0, al1 = shell_by_atm[ia]
                s_slice = [s0, s1, 0, mol.nbas]
                hcore[:, al0:al1] = self.mol.intor('int1e_ipkin', comp=3, shls_slice=s_slice)
                if mol._pseudo:
                    NotImplementedError('Nuclear gradients for GTH PP')
                else:
                    hcore[:, al0:al1] += mol.intor('int1e_ipnuc', comp=3, shls_slice=s_slice)
                if mol.has_ecp():
                    hcore[:, al0:al1] += mol.intor('ECPscalar_ipnuc', comp=3, shls_slice=s_slice)
        comm.Barrier()
        return win_hcore, hcore
    def hcore_deriv(mol, atm_id, hcore):
        with_ecp = mol.has_ecp()
        s0, s1, ao0, ao1 = self.mol.offset_nr_by_atom()[atm_i]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = self.mol.intor('int1e_iprinv', comp=3, out=buf_eri1) # <\nabla|1/r|>
            vrinv *= -mol.atom_charge(atm_id)
            if with_ecp and atm_id in with_ecp.ecp_atoms:
                vrinv += mol.intor('ECPscalar_iprinv', comp=3)
        vrinv[:, ao0:ao1] -= hcore[:, ao0:ao1]#get_hcore(mol, s0, s1)
        vrinv += vrinv.transpose(0, 2, 1)
        return vrinv
    t1 = get_current_time()
    win_dm_tmp, dm_tmp = get_shared((self.nao, self.nao))
    win_dme_tmp, dme_tmp = get_shared((self.nao, self.nao))
    if irank_shm == 0:
        if Pi is None:
            dm_tmp[:] = dm0
        else:
            dm_tmp[:] = Pi+dm0
        if W is None:
            dme_tmp[:] = dme0 * 2
        else:
            dme_tmp[:] = W + dme0 * 2
    comm_shm.Barrier()
    win_hcore, hcore = get_hcore(self.mol)
    if self.mol_mm is not None:
        t0 = get_current_time()
        h_qmmm, g_mm = qmmm.grad_nuc_ele_qmmm(self, dm_tmp)
        if h_qmmm is not None:
            Accumulate_GA_shm(win_hcore, hcore, h_qmmm)
        h_qmmm = None
        print_time(["QMMM grad", get_elapsed_time(t0)], log)
    comm.Barrier()
    if self.nnode > 1:
        Acc_and_get_GA(var=hcore)
    win_grad, grad_node = get_shared(len(self.atom_list)*3, set_zeros=True)
    comm_shm.Barrier()
    atom_slice = get_slice(range(nrank), job_list=self.atom_list)[irank]
    if atom_slice is not None:
        if irank_shm == 0:
            grad = grad_node
        else:
            grad = np.zeros(len(self.atom_list)*3)
        offset_atom = self.mol.offset_nr_by_atom()
        size_list = []
        for atm_i in atom_slice:
            ao0, ao1 = offset_atom[atm_i][2:]
            size_list.append((3*(ao1-ao0)*self.nao + 3*self.nao**2))
        buf_eri = np.empty(max(size_list))
        buf_eri1 = buf_eri[:3*self.nao**2]
        buf_eri2 = buf_eri[3*self.nao**2:]
        
        self.base = None
        #hcore_deriv = hcore_generator(self, self.mol)
        for atm_i in atom_slice:
            s0, s1, ao0, ao1 = self.mol.offset_nr_by_atom()[atm_i]
            idx0, idx1 = atm_i*3, (atm_i+1)*3
            s_slice = [s0, s1, 0, self.mol.nbas]
            S1 = self.mol.intor('cint1e_ipovlp_sph', comp=3, shls_slice=s_slice, out=buf_eri2).reshape(3, -1)
            #S1 *= -1
            grad[idx0:idx1] += ddot(S1, dme_tmp[ao0:ao1].ravel())
            h1ao = hcore_deriv(self.mol, atm_i, hcore)
            grad[idx0:idx1] += ddot(h1ao.reshape(3, -1), dm_tmp.ravel())
        if irank_shm != 0:
            Accumulate_GA_shm(win_grad, grad_node, grad)
        buf_eri, buf_eri1, buf_eri2 = None, None, None
        if irank == 0:
            Accumulate_GA_shm(win_grad, grad_node, grad_nuc(self.mol).ravel())
    comm.Barrier()
    if self.nnode > 1:
        Acc_and_get_GA(var=grad_node)
    comm_shm.Barrier()
    grad = np.copy(grad_node)
    comm_shm.Barrier()
    if self.mol_mm is not None:
        gnuc_qm, gnuc_mm = qmmm.grad_nuc_nuc_qmmm(self)
        grad += gnuc_qm.ravel()
        g_mm += gnuc_mm
        self.mol_mm.grad = g_mm
    for win_i in [win_dm_tmp, win_dme_tmp, win_hcore, win_grad]:
        free_win(win_i)
    print_time(["core hamiltonian grad", get_elapsed_time(t1)], log)
    return grad
    

def kernel(self):
    log = lib.logger.Logger(self.stdout, self.verbose)
    tt = t1 = get_current_time()
    self.no = self.mol.nelectron//2
    self.o = self.mo_coeff[:,:self.no]
    self.mo_list = range(self.no)
    self.atom_list = range(self.mol.natm)
    self.shell_slice = int_prescreen.shell_prescreen(self.mol, self.with_df.auxmol, log, 
                                shell_slice=self.shell_slice, shell_tol=self.shell_tol, meth_type='RHF')
    log.info('\nRHF ialp calculation starts...')
    get_ialp_GA(self, self.with_df, 'hf', log)
    print_time(['ialp generation', get_elapsed_time(t1)], log)
    
    win_dm0, dm0 = get_shared((self.nao, self.nao), dtype='f8')
    win_dme0, dme0 = get_shared((self.nao, self.nao), dtype='f8')
    if irank_shm == 0:
        dm0[:] = self.make_rdm1(self.mo_coeff, self.mo_occ)
        dme0[:] = make_rdm1e(self.mo_energy, self.mo_coeff, self.mo_occ)
    comm_shm.Barrier()

    log.info('\nRHF gradient calculation starts...')
    self.gradient = dfhf_response_ga(self, dm0, 0.5*dm0, None)
    self.gradient += grad_hcore_nuc(self, dm0, dme0, log=log)
    if self.solvent is not None:
        t0 = get_current_time()
        log.info('\nBegin the computation of solvation gradient...')
        g_sol = get_grad_sol(self.with_solvent, dm0, log)
        print_time(["solvation gradient", get_elapsed_time(t0)], log)
        self.gradient += g_sol.ravel()
    self.t_grad_hf = get_elapsed_time(tt)
    comm.Barrier()
    for win in [win_dm0, win_dme0]:
        win.Free()
    return self.gradient
