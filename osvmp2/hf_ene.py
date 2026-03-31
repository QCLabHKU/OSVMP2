##################################################################

# Import the packages needed
import os
import sys
import numpy
import types
import h5py
import shutil
import scipy
import numpy as np
from numpy.linalg import multi_dot
import pyscf
from pyscf import gto, lib
from pyscf.scf import atom_hf, diis
from pyscf.scf import addons as scf_addons
from pyscf.scf.hf import damping
from pyscf.df import DF
from pyscf.lib import logger
from pyscf.solvent import ddcosmo
#from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from osvmp2.__config__ import inputs, ngpu
if inputs["qm_atoms"] is not None:
    from osvmp2.mm import qmmm
#from int_prescreen import shell_prescreen
from osvmp2.mm.solvation import get_veff_sol
from osvmp2.loc import loc_addons
from osvmp2 import int_prescreen
from osvmp2.pbc.gamma_hf_ene import get_hcore as pbc_get_hcore
from osvmp2.pbc.gamma_hf_ene import gamma_get_ialp_bp, gamma_get_j_step2, correct_vk
from osvmp2.osvutil import *
from osvmp2.mpi_addons import *
from osvmp2.int_3c2e import get_df_int3c2e
from osvmp2.int_2c2e import get_j2c_low
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
#inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm= comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//comm_shm.size
inode = irank // nrank_shm

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)
if ngpu:
    import cupy
    import gpu4pyscf
    from gpu4pyscf.pbc.gto import int1e
    from gpu4pyscf.scf import diis
    from gpu4pyscf.scf.hf import damping, level_shift
    from gpu4pyscf.scf.hf import get_hcore as get_hcore_cuda
    from gpu4pyscf.scf.hf import init_guess_by_minao as init_guess_by_minao_cuda
    from osvmp2.gpu.int_3c2e_cuda import VHFOpt
    from osvmp2.gpu.int4c_jk_cuda import JKOpt, compute_jk
    from osvmp2.gpu.int4c_md_j_cuda import MdJOpt, compute_j
    from osvmp2.gpu.hf_ene_cuda import eigh_cuda, CDIIS, get_occ_cuda, get_jk_direct_cuda, get_jk_with_int_cuda
    from osvmp2.gpu.cuda_utils import avail_gpu_mem
    from osvmp2.pbc.gpu.gamma_int_3c2e_cuda import SRInt3c2eOpt

    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)
#OSV_grad = None


def init_guess_by_minao(mol):
    def minbeasis(symb, nelec_ecp):
        stdsymb = gto.mole._std_symbol(symb)
        basis_add = gto.basis.load('ano', stdsymb)
        occ = []
        basis_ano = []
# coreshl defines the core shells to be removed in the initial guess
        coreshl = gto.ecp.core_configuration(nelec_ecp)
        #coreshl = (0,0,0,0)  # it keeps all core electrons in the initial guess
        for l in range(4):
            ndocc, frac = atom_hf.frac_occ(stdsymb, l)
            if coreshl[l] > 0:
                occ.extend([0]*coreshl[l]*(2*l+1))
            if ndocc > coreshl[l]:
                occ.extend([2]*(ndocc-coreshl[l])*(2*l+1))
            if frac > 1e-15:
                occ.extend([frac]*(2*l+1))
                ndocc += 1
            if ndocc > 0:
                basis_ano.append([l] + [b[:ndocc+1] for b in basis_add[l][1:]])

        if nelec_ecp > 0:
            occ4ecp = []
            basis4ecp = []
            nelec_valence_left = gto.mole.charge(stdsymb) - nelec_ecp
            for l in range(4):
                if nelec_valence_left <= 0:
                    break
                ndocc, frac = atom_hf.frac_occ(stdsymb, l)
                assert(ndocc >= coreshl[l])

                n_valenc_shell = ndocc - coreshl[l]
                l_occ = [2] * (n_valenc_shell*(2*l+1))
                if frac > 1e-15:
                    l_occ.extend([frac] * (2*l+1))
                    n_valenc_shell += 1

                shell_found = 0
                for bas in mol._basis[symb]:
                    if shell_found >= n_valenc_shell:
                        break
                    if bas[0] == l:
                        off = n_valenc_shell - shell_found
                        # b[:off+1] because the first column of bas[1] is exp
                        basis4ecp.append([l] + [b[:off+1] for b in bas[1:]])
                        shell_found += len(bas[1]) - 1

                nelec_valence_left -= int(sum(l_occ[:shell_found*(2*l+1)]))
                occ4ecp.extend(l_occ)

            if nelec_valence_left > 0:
                logger.debug(mol, 'Characters of %d valence electrons are '
                             'not identified in the minao initial guess.\n'
                             'Electron density of valence ANO for %s will '
                             'be used.', nelec_valence_left, symb)
                return occ, basis_ano

# Compared to ANO valence basis, to check whether the ECP basis set has
# reasonable AO-character contraction.  The ANO valence AO should have
# significant overlap to ECP basis if the ECP basis has AO-character.
            atm1 = gto.Mole()
            atm2 = gto.Mole()
            atom = [[symb, (0.,0.,0.)]]
            atm1._atm, atm1._bas, atm1._env = atm1.make_env(atom, {symb:basis4ecp}, [])
            atm2._atm, atm2._bas, atm2._env = atm2.make_env(atom, {symb:basis_ano}, [])
            atm1._built = True
            atm2._built = True
            s12 = gto.intor_cross('int1e_s1e', atm1, atm2)[:,numpy.array(occ)>0]
            if abs(numpy.linalg.det(s12)) > .1:
                occ, basis_ano = occ4ecp, basis4ecp
            else:
                logger.debug(mol, 'Density of valence part of ANO basis '
                             'will be used as initial guess for %s', symb)
        return occ, basis_ano

    atmlst = set([mol.atom_symbol(ia) for ia in range(mol.natm)])

    nelec_ecp_dic = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in nelec_ecp_dic:
            nelec_ecp_dic[symb] = mol.atom_nelec_core(ia)

    basis = {}
    occdic = {}
    for symb in atmlst:
        if not gto.is_ghost_atom(symb):
            nelec_ecp = nelec_ecp_dic[symb]
            occ_add, basis_add = minbeasis(symb, nelec_ecp)
            occdic[symb] = occ_add
            basis[symb] = basis_add
    occ = []
    new_atom = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if not gto.is_ghost_atom(symb):
            occ.append(occdic[symb])
            new_atom.append(mol._atom[ia])
    occ = numpy.hstack(occ)

    pmol = gto.Mole()
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(new_atom, basis, [])
    pmol._built = True
    c = scf_addons.project_mo_nr2nr(pmol, numpy.eye(pmol.nao_nr()), mol)

    dm = numpy.dot(c*occ, c.conj().T)
# normalize eletron number
#    s = mol.intor_symmetric('int1e_s1e')
#    dm *= mol.nelectron / (dm*s).sum()
    return c, occ, dm

def make_rdm1(mo_coeff, mo_occ, out=None):
    occ_coeff = mo_coeff[:,mo_occ>0]
    #dm = numpy.dot(occ_coeff*mo_occ[mo_occ>0], occ_coeff.conj().T, out=out)
    dm = (occ_coeff*mo_occ[mo_occ>0]).dot(occ_coeff.conj().T, out=out)
    return dm


def get_jk_direct(self, dm, loc_df=False, log=None):
    
    def get_ialp_bp():
        nocc = self.occ_coeff.shape[1]
        mol = self.mol
        auxmol = self.with_df.auxmol
        nao = mol.nao_nr()
        naoaux = auxmol.nao_nr()
        ao_loc = mol.ao_loc

        ao_slice, shell_slice_rank = int_prescreen.get_slice_rank(self.shell_slice, aslice=True)

        nao = self.nao
        naoaux = self.naoaux
        ao_loc = self.mol.ao_loc
        
        win_bp, bp_node = get_shared(naoaux, set_zeros=True)

        if loc_df:
            if nocc > mol.nelectron//2:
                win_uo, uo = get_shared((nocc, nocc))
                if irank_shm == 0:
                    uo[:] = loc_addons.localization(mol, self.occ_coeff, verbose=0)
                comm_shm.Barrier()
                self.occ_coeff[:] = np.dot(self.occ_coeff, uo)
                comm_shm.Barrier()
                free_win(win_uo); uo = None
            else:
                if irank_shm == 0:
                    uo = loc_addons.localization(mol, self.occ_coeff, verbose=0)
                    np.dot(self.occ_coeff, uo, out=self.occ_coeff)
                comm.Barrier()
            win_pcharge, pcharge = get_shared((mol.natm, nocc))
            if irank_shm == 0:
                with h5py.File('pcharge.tmp', 'r') as file_pc:
                    file_pc['charge'].read_direct(pcharge)
            comm_shm.Barrier()
            self.lmo_close, self.lmo_remote, self.nlmo_close = LMO_domains(mol, self.occ_coeff, tol_lmo=1e-6)
            tot_prime = 0#1e-2
            self.fit_close, self.naux_close = FIT_domains(mol, auxmol, self.occ_coeff, pcharge, tot_prime=tot_prime, tot_nsup=4)
            self.occ_coeff[:] = moco_fit(mol, self.occ_coeff, self.lmo_close, self.lmo_remote, self.nlmo_close)
            comm_shm.Barrier()
            free_win(win_pcharge); pcharge=None

        else:
            if nocc != self.nocc_pre:
                file_ialp = h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm)
                nao_chunk = min(536870912 // naoaux, nao) # chunks must be smaller than 4GB
                '''ialp_data = file_ialp.create_dataset('ialp', (nocc, nao, naoaux), dtype='f8', 
                                                     chunks=(1, nao_chunk, naoaux))'''
                print(type(nocc), type(nao), type(naoaux))
                nocc = int(nocc); nao = int(nao); naoaux = int(naoaux)
                ialp_data = create_h5py_dataset(file_ialp, 'ialp', (nocc, nao, naoaux), 
                                                dtype=np.float64, chunks=(1, nao_chunk, naoaux))
            else:
                file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
                ialp_data = file_ialp['ialp']

        max_memory = get_mem_spare(mol, 0.8)
        if shell_slice_rank is not None:
            ao0, ao1 = ao_slice[irank][0], ao_slice[irank][1]
            nao_rank = ao1 - ao0

            bp = np.zeros(naoaux)
            
            size_ialp, size_feri, shell_slice_rank = int_prescreen.mem_control(mol, nocc, naoaux, shell_slice_rank, "half_trans", max_memory)
            buf_ialp = np.empty(size_ialp)
            buf_feri = np.empty(size_feri)
            c_buf_feri = buf_feri.ctypes.data_as(ctypes.c_void_p)
            s_slice = np.empty(6, dtype=np.int32)
            c_s_slice = s_slice.ctypes.data_as(ctypes.c_void_p)
            
            SHELL_SEG = slice2seg(mol, shell_slice_rank, max_nao=size_ialp//(nocc*naoaux))
            ao_idx0 = 0
            for seg_i in SHELL_SEG:
                A0, A1 = seg_i[0][0], seg_i[-1][1]
                AL0, AL1 = ao_loc[A0], ao_loc[A1]
                nao_seg = AL1 - AL0

                ialp_tmp = buf_ialp[:nocc*nao_seg*naoaux].reshape(nocc, nao_seg, naoaux)
                ialp_tmp[:] = 0
                buf_idx0 = 0

                for a0, a1, b_list in seg_i:
                    al0, al1 = ao_loc[a0], ao_loc[a1]
                    nao0 = al1 - al0
                    buf_idx1 = buf_idx0 + nao0
                    for b0, b1 in b_list:
                        be0, be1 = ao_loc[b0], ao_loc[b1]
                        nao1 = be1 - be0
                        #s_slice[:] = (a0, a1, b0, b1, mol.nbas, mol.nbas+auxmol.nbas)
                        s_slice[:] = (b0, b1, a0, a1, mol.nbas, mol.nbas+auxmol.nbas)
                        t1 = get_current_time()
                        #feri_go = aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buf_feri)
                        #feri_go = np.copy(feri_go)
                        aux_3c2e(auxmol, c_s_slice, c_buf_feri)
                        feri_size_seg = nao0 * nao1 * naoaux
                        #feri_tmp = buf_feri[:feri_size_seg].reshape(naoaux, nao0, nao1).transpose(2, 1, 0)
                        feri_tmp = buf_feri[:feri_size_seg].reshape(naoaux, -1)
                        accumulate_time(self.t_feri, t1)

                        #For J matrix
                        t1 = get_current_time()
                        #bp += np.dot(feri_tmp, dm[be0:be1, al0:al1].ravel())
                        accumulate_time(self.t_j, t1)

                        #For K matrix
                        t1 = get_current_time()
                        #feri_tmp = feri_tmp.T.reshape(nao1, -1)
                        #ialp_tmp[:, buf_idx0:buf_idx1] += np.dot(self.occ_coeff[be0:be1].T, feri_tmp).reshape(nocc, nao0, naoaux)
                        feri_tmp = feri_tmp.reshape(-1, nao1)
                        mocc_tmp = self.occ_coeff[be0:be1]
                        ialp_tmp[:, buf_idx0:buf_idx1] += np.dot(feri_tmp, mocc_tmp).reshape(naoaux, nao0, nocc).transpose(2, 1, 0)
                        if irank == 0:
                            print_mem('Half trans', self.pid_list, log)
                        accumulate_time(self.t_k, t1)
                    buf_idx0 = buf_idx1

                t1 = get_current_time()
                bp += np.einsum("ij,jip->p", self.occ_coeff[AL0:AL1], ialp_tmp)
                accumulate_time(self.t_j, t1)

                t1 = get_current_time()
                if loc_df:
                    pass
                else:
                    ialp_data.write_direct(ialp_tmp, dest_sel=np.s_[:, AL0:AL1])
                accumulate_time(self.t_write, t1)

            Accumulate_GA_shm(win_bp, bp_node, bp)

        file_ialp.close()
        comm_shm.Barrier()

        return win_bp, bp_node
    

    def fit_bp(bp_node, j2c_type="j2c"):
        if nnode > 1 and irank_shm == 0:
            win_bp_col = get_win_col(bp_node)
            Accumulate_GA(win_bp_col, bp_node)
            win_bp_col.Fence()
        
        if irank == 0:
            t1 = get_current_time()
            #j2c = read_file(self.file_j2c, j2c_type)#, buffer=low_node)#buf_pq)
            with h5py.File(self.file_j2c, 'r') as f:
                j2c = np.asarray(f[j2c_type])
            accumulate_time(self.t_read, t1)

            t1 = get_current_time()
            scipy.linalg.solve(j2c, bp_node, overwrite_b=True)    
            accumulate_time(self.t_j, t1)

            #print(get_current_time() - tt) 
            j2c = None
        comm.Barrier()
        
        #buf_pq = None
        if nnode > 1 and irank_shm == 0:
            if irank != 0:
                Get_GA(win_bp_col, bp_node)
            win_bp_col.Fence()
            free_win(win_bp_col)
        
        
        return bp_node

    #Step 2 for J matrix
    def get_j_step2(win_vj, vj_node, bp_node):
        
        shell_slice_rank = int_prescreen.get_slice_rank(self.shell_slice)
        max_memory = get_mem_spare(mol, 0.8)
        if shell_slice_rank is not None:
            max_memory -= nao * nao * 8 * 1e-6
            size_feri, shell_slice_rank = int_prescreen.mem_control(mol, nocc, naoaux, shell_slice_rank, 0.9, max_memory)
            buf_feri = np.empty(size_feri)
            c_buf_feri = buf_feri.ctypes.data_as(ctypes.c_void_p)
            s_slice = np.empty(6, dtype=np.int32)
            c_s_slice = s_slice.ctypes.data_as(ctypes.c_void_p)
            
            '''if irank_shm == 0:
                vj = vj_node
            else:'''
            vj = np.zeros((nao, nao))

            for idx, si in enumerate(shell_slice_rank):
                a0, a1, b0, b1 = si
                al0, al1 = ao_loc[a0], ao_loc[a1]
                be0, be1 = ao_loc[b0], ao_loc[b1]
                nao0 = al1 - al0
                nao1 = be1 - be0

                #s_slice = (a0, a1, b0, b1, mol.nbas, mol.nbas+auxmol.nbas)
                s_slice[:] = (b0, b1, a0, a1, mol.nbas, mol.nbas+auxmol.nbas)

                t1 = get_current_time()
                #feri_tmp = aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buf_feri)#.transpose(1,0,2)
                aux_3c2e(auxmol, c_s_slice, c_buf_feri)
                feri_size_seg = nao0 * nao1 * naoaux
                feri_tmp = buf_feri[:feri_size_seg]#.reshape(naoaux, -1)#.T #(naoaux, nao0, nao1)
                accumulate_time(self.t_feri, t1)
                
                t1 = get_current_time()
                #vj[al0:al1, be0:be1] += np.dot(feri_tmp.reshape(-1, naoaux), bp_node).reshape(nao0, nao1)
                vj[al0:al1, be0:be1] += np.dot(bp_node, feri_tmp.reshape(naoaux, -1)).reshape(nao0, nao1)
                accumulate_time(self.t_j, t1)
            buf_feri = None
            
            #if irank_shm != 0:
            Accumulate_GA_shm(win_vj, vj_node, vj)
            
        else:
            vj = None
        comm_shm.Barrier()
        return vj
    

    #Step 2 for K matrix
    def get_k_step2(win_vk, vk_node):
        mo_slice = get_slice(job_size=nocc, rank_list=range(nrank))[irank]
        win_low, low_node = get_shared((naoaux, naoaux))
        if irank_shm == 0:
            t1 = get_current_time()
            if loc_df:
                read_file(self.file_j2c, 'j2c', buffer=low_node)
            else:
                read_file(self.file_j2c, 'low', buffer=low_node)
            accumulate_time(self.t_read, t1)
        comm_shm.Barrier()
        if mo_slice is not None:
            '''if irank_shm == 0:
                vk = vk_node
            else:'''
            vk = np.zeros((nao, nao))
            if loc_df:
                naux_tot = sum([self.naux_close[i] for i in mo_slice])
                size_sub = nao*max(self.naux_close)
            else:
                size_sub = nao*naoaux
            nocc_rank = len(mo_slice)
        else:
            size_sub = None
            nocc_rank = None
        mem_avail = get_mem_spare(mol)
        max_memory = mem_avail #- (8*naoaux*naoaux*1e-6)
        if max_memory < 0:
            raise MemoryError("No sufficient memory")
        max_mo = get_buff_len(mol, size_sub=size_sub, ratio=0.4, max_len=nocc_rank, max_memory=max_memory)
        
        if self.shared_int:
            file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
        else:
            file_ialp = h5py.File(self.file_ialp, 'r')

        if mo_slice is not None:
            buf_ialp = np.empty(max_mo*size_sub)
            mo0, mo1 = mo_slice[0], mo_slice[-1]+1
            mo_idx = np.append(np.arange(mo0, mo1, step=max_mo), mo1)
            mo_seg = [[mo0, mo1] for mo0, mo1 in zip(mo_idx[:-1], mo_idx[1:])]
            
            for mo_i in mo_seg:
                mo0, mo1 = mo_i
                nocc_seg = mo1 - mo0
                if loc_df:
                    pass
                else:
                    ialp_tmp = buf_ialp[:nocc_seg*nao*naoaux].reshape(nocc_seg, nao, naoaux)
                
                if not self.debug:
                    t1 = get_current_time()
                    file_ialp["ialp"].read_direct(ialp_tmp, np.s_[mo0:mo1])
                    accumulate_time(self.t_read, t1)
                
                if loc_df:
                    pass
                else:

                    ialp_tmp = ialp_tmp.reshape(-1, naoaux)
                    t1 = get_current_time()
                    scipy.linalg.solve_triangular(low_node, ialp_tmp.T, lower=True, overwrite_b=True, check_finite=False)
                    ialp_tmp = ialp_tmp.reshape(-1, nao, naoaux)
                    for idx in range(nocc_seg):
                        ialp_i = ialp_tmp[idx]
                        vk += np.dot(ialp_i, ialp_i.T)
                    accumulate_time(self.t_k, t1)


                    if self.shared_int:
                        file_ialp["ialp"].write_direct(ialp_tmp, dest_sel=np.s_[mo0:mo1])

            Accumulate_GA_shm(win_vk, vk_node, vk)  

        else:
            vk = None   

        file_ialp.close()

        comm_shm.Barrier()
        free_win(win_low); low_node=None
        
        #return vk

    ###########################################################################
    if log is None:
        log = logger.Logger(sys.stdout, self.verbose)

    nao = self.mol.nao_nr()
    naoaux = self.with_df.auxmol.nao_nr()
    nocc = self.occ_coeff.shape[-1]
    ao_loc = make_loc(self.mol._bas, 'sph')
    mol = self.mol
    auxmol = self.with_df.auxmol

    win_vj, vj_node = get_shared((nao, nao), set_zeros=True)
    win_vk, vk_node = get_shared((nao, nao), set_zeros=True)

    comm.Barrier()

    t0 = get_current_time()
    if self.mol.pbc:
        win_bp, bp_node = gamma_get_ialp_bp(self, dm)
    else:
        win_bp, bp_node = get_ialp_bp()
    bp_node = fit_bp(bp_node)
    if irank == 0:
        print_mem('Half trans (%.2f secs)'%(get_elapsed_time(t0))[1], self.pid_list, log)
    
    t0 = get_current_time()
    if self.mol.pbc:
        gamma_get_j_step2(self, win_vj, vj_node, bp_node)
    else:
        get_j_step2(win_vj, vj_node, bp_node)
    free_win(win_bp)
         
    if irank == 0:
        print_mem('J matrix (%.2f secs)'%(get_elapsed_time(t0))[1], self.pid_list, log)

    t0 = get_current_time()
    
    get_k_step2(win_vk, vk_node)
    
    if nnode > 1:
        Acc_and_get_GA(vj_node)
        Acc_and_get_GA(vk_node)
        comm.Barrier()

    if irank_shm == 0:
        vj = np.copy(vj_node)
        vk = np.copy(vk_node)

    else:
        vj, vk = None, None
    #sys.exit()
    comm_shm.Barrier()
    for win in [win_vj, win_vk]:
        free_win(win)

    if irank == 0:
        print_mem('K matrix (%.2f secs)'%(get_elapsed_time(t0))[1], self.pid_list, log)

    self.nocc_pre = nocc

    return vj, vk


def get_jk_with_int(self, dm, loc_df=False, log=None):

    if log is None:
        log = logger.Logger(sys.stdout, self.verbose)
    nao = self.mol.nao_nr()
    naoaux = self.with_df.auxmol.nao_nr()
    nocc = self.occ_coeff.shape[-1]

    win_vj, vj_node = get_shared((nao, nao), set_zeros=True)
    win_vk, vk_node = get_shared((nao, nao), set_zeros=True)
    
    aux_slice = get_slice(range(nrank), job_size=naoaux)[irank]
    if self.int_storage == 0:
        feri_save = self.feri_node
        if self.shared_int:
            ialp_data = self.ialp_node
    elif self.int_storage == 1:
        file_feri = h5py.File(self.file_feri, 'r')
        feri_save = file_feri["feri"]

        if self.shared_int:
            file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
            ialp_data = file_ialp["ialp"]

    if aux_slice is not None:

        vj = numpy.zeros((nao, nao), dtype='f8')
        vk = numpy.zeros((nao, nao), dtype='f8')
        naux_slice = len(aux_slice)

        max_naux = get_ncols_for_memory(0.6*self.max_memory, nao*nao, naux_slice)

        for pidx0 in np.arange(naux_slice, step=max_naux):
            pidx1 = min(pidx0 + max_naux, naux_slice)
            p0 = aux_slice[pidx0]
            p1 = aux_slice[pidx1-1] + 1
            naux_seg = pidx1 - pidx0

            if self.ijp_shape:
                t1 = get_current_time()
                feri = feri_save[:, :, p0:p1] #(nao, nao, naux_seg)
                accumulate_time(self.t_read, t1)

                t1 = get_current_time()
                ialp = np.dot(self.occ_coeff.T, feri.reshape(nao, -1)).reshape(nocc, nao, naux_seg)
                alip = ialp.transpose(1, 0, 2).reshape(nao, -1)
                vk += np.dot(alip, alip.T)
                accumulate_time(self.t_k, t1)

                t1 = get_current_time()
                gammq = np.einsum('ij,jip->p', self.occ_coeff, ialp)  # Shape:(naux_seg,)
                vj += np.einsum('ijp,p->ij', feri, gammq)  # Shape:(nao, nao)
                accumulate_time(self.t_j, t1)

                if self.shared_int:
                    ialp_data[:, :, p0:p1] = ialp
            else:
                t1 = get_current_time()
                feri = feri_save[p0:p1] #(naux_seg, nao, nao)
                accumulate_time(self.t_read, t1)

                t1 = get_current_time()
                pali = np.dot(feri.reshape(-1, nao), self.occ_coeff).reshape(naux_seg, nao, nocc)
                accumulate_time(self.t_k, t1)

                t1 = get_current_time()
                gammq = np.dot(pali.reshape(naux_seg, -1), self.occ_coeff.ravel())  # Shape:(naux_seg,)
                vj += np.einsum('p,pij->ij', gammq, feri)  # Shape:(nao, nao)
                accumulate_time(self.t_j, t1)

                t1 = get_current_time()
                #vk += np.einsum("pji,pli->jl", pali, pali, optimize="optimal")
                pial = pali.transpose(0, 2, 1).reshape(-1, nao)
                vk += np.dot(pial.T, pial)
                accumulate_time(self.t_k, t1)

                if self.shared_int:
                    ialp_data[:, :, p0:p1] = pali.transpose(2, 1, 0)


        #if irank_shm != 0:
        Accumulate_GA_shm(win_vj, vj_node, vj)
        Accumulate_GA_shm(win_vk, vk_node, vk)

    if self.int_storage == 1:
        file_feri.close()
        if self.shared_int:
            file_ialp.close()

    comm.Barrier()

    if nnode > 1:
        Acc_and_get_GA(vj_node)
        Acc_and_get_GA(vk_node)
        '''if self.int_storage == 1:
            Acc_and_get_GA(self.ialp_node)'''
        comm.Barrier()

    if irank_shm == 0:
        '''vj = np.copy(vj_node)
        vk = np.copy(vk_node)'''
        np.copyto(vj, vj_node)
        np.copyto(vk, vk_node)
    else:
        vj, vk = None, None

    comm_shm.Barrier()
    for win in [win_vj, win_vk]:
        free_win(win)
    
    return vj, vk
#from gpu4pyscf.scf import j_engine
#from gpu4pyscf.scf.jk import _VHFOpt, get_j, get_jk
#from osvmp2.gpu.jk import _VHFOpt, get_jk
def get_hf_jk(self, dm, loc_df=False):

    if not self.use_df_hf:
        if irank_shm == 0 or self.use_gpu:
            dm_now = dm.copy()
            dm -= self.last_dm
            self.last_dm = dm_now
        if not self.use_gpu:
            comm_shm.Barrier()
    if self.int_storage == 2:
        log = logger.Logger(sys.stdout, self.verbose)
        if self.use_gpu:
            if self.use_df_hf:
                vj, vk = get_jk_direct_cuda(self, dm, loc_df)
            else:
                #_, vk = compute_jk(self.mol, dm, hermi=1, with_j=False)
                #vj = self.RHF.get_j(self.mol, dm, hermi=1)
                #vk = self.RHF.get_k(self.mol, dm, hermi=1)
                t0 = get_current_time()
                vj = compute_j(self.mol, dm, hermi=1, vhfopt=self.j_intopt)
                tj = get_elapsed_time(t0)
                self.t_j += tj
                print_time([f"{self.cycle} hf J", tj], log)

                t0 = get_current_time()
                vk = compute_jk(self.mol, dm, hermi=1, with_j=False, vhfopt=self.k_intopt)[1]
                tk = get_elapsed_time(t0)
                self.t_k += tk
                print_time([f"{self.cycle} hf k", tk], log)
        else:
            vj, vk = get_jk_direct(self, dm, loc_df)
    else:
        if self.use_gpu:
            vj, vk = get_jk_with_int_cuda(self, dm)
        else:
            vj, vk = get_jk_with_int(self, dm, loc_df)
    
    if irank_shm == 0 and self.mol.pbc:
        if self.mol.gamma_only:
            t0 = get_current_time()
            #_ewald_exxdiv_for_G0(self.mol, self.with_df.kpts[0], dm.reshape(-1, self.nao, self.nao),
            correct_vk(self.mol, self.with_df.kpts[0], dm.reshape(-1, self.nao, self.nao),
                       vk.reshape(-1, self.nao, self.nao), self.use_gpu)
            accumulate_time(self.t_k, t0)

    if not self.use_df_hf:
        if irank_shm == 0 or self.use_gpu:
            dm[:] = self.last_dm
        if irank_shm == 0:
            vj += self.last_vj
            vk += self.last_vk
            self.last_vj = vj
            self.last_vk = vk
    return vj, vk

def get_veff(self, mol=None, dm=None):
    if mol is None:mol = self.mol
    if dm is None:dm = make_rdm1()

    vj, vk = get_hf_jk(self, dm)

    if irank_shm == 0:
        vhf = vj - vk * .5
    else:
        vhf = None
    return vhf

def get_occ(self, mo_energy=None, mo_coeff=None):
    if mo_energy is None:mo_energy = self.mo_energy
    e_idx = numpy.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nmo = mo_energy.size
    mo_occ = numpy.zeros(nmo)
    nocc = self.mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    if self.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(self, '  mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)
    return mo_occ

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    '''e1 = numpy.einsum('ij,ji->', h1e, dm)
    e_coul = numpy.einsum('ij,ji->', vhf, dm) * .5'''
    e1 = float(h1e.ravel().dot(dm.ravel()))
    e_coul = float(vhf.ravel().dot(dm.ravel())) * .5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    return (e1+e_coul).real, e_coul


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    nuc = mf.energy_nuc()
    if mf.mol_mm is not None:
        nuc = qmmm.energy_nuc_qmmm(mf, nuc)
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    mf.scf_summary['nuc'] = nuc.real
    return e_tot


def access_chkfile(chkfile, mode, arrays, cycle=None):
    #The order of the buffer has to be:dm, mo_energy, mo_coeff, mo_occ, occ_coeff, e_tot
    key_list = ["dm", "mo_energy", "mo_coeff", "mo_occ", "occ_coeff", "e_tot"]
    array_dic = {}
    for idx, key_i in enumerate(key_list):
        array_dic[key_i] = arrays[idx]
    with h5py.File(chkfile, mode) as f:
        if mode == 'w':
            for idx, key_i in enumerate(key_list):
                f.create_dataset("scf/%s"%key_i, data=array_dic[key_i])
        else:
            keys_file = f["scf"].keys()
            if mode == 'r+':
                for idx, key_i in enumerate(key_list):
                    if key_i in keys_file:
                        f["scf/%s"%key_i].write_direct(array_dic[key_i])
                    else:
                        f.create_dataset("scf/%s"%key_i, data=array_dic[key_i])
            elif mode == 'r':
                nochk_list = []
                for idx, key_i in enumerate(key_list):
                    if array_dic[key_i] is None:
                        continue
                    if key_i in keys_file:
                        f["scf/%s"%key_i].read_direct(array_dic[key_i])
                    else:
                        #dm, occ_coeff
                        nochk_list.append(key_i)
                for key_i in nochk_list:
                    if key_i == "dm":
                        #array_dic[key_i][:] = make_rdm1(array_dic["mo_coeff"], array_dic["mo_occ"])
                        make_rdm1(array_dic["mo_coeff"], array_dic["mo_occ"], out=array_dic[key_i])
                    elif key_i == "occ_coeff": #for backward compatibility
                        #f["scf/mocc"].read_direct(array_dic[key_i])
                        mo_occ = array_dic["mo_occ"]
                        #print(array_dic[key_i].shape, array_dic["mo_coeff"][:, mo_occ>0]*(mo_occ[mo_occ>0]**0.5).shape)
                        array_dic[key_i][:] = array_dic["mo_coeff"][:, mo_occ>0]*(mo_occ[mo_occ>0]**0.5)

    if mode == 'r':
        return arrays
    '''elif self.chkfile_save is not None:
        shutil.copy(chkfile, "%s/hf_mat_cycle%d.chk"%(self.chkfile_save, cycle))
        if cycle > 0:
            os.remove("%s/hf_mat_cycle%d.chk"%(self.chkfile_save, cycle-1))'''

def get_gpu_intopt(self, log, use_df=True):
    self.memory_pool = cupy.get_default_memory_pool()
    if self.mol.pbc:
        omega = self.with_df.df_builder.omega
        self.intopt = SRInt3c2eOpt(self.mol, self.auxmol, -omega, fitting=True)
    else:
        if use_df:
            self.intopt = VHFOpt(self.mol, self.with_df.auxmol, 'int2e')
            self.intopt.memory_pool = self.memory_pool
            self.intopt.recorder_cart2sph = []
            self.intopt.recorder_ferikern = []
            self.intopt.recorder_prep = []
            self.intopt.recorder_feritot = []
        else:
            self.k_intopt = JKOpt(self.mol)
            self.j_intopt = MdJOpt(self.mol)
            self.k_intopt.build()
            self.j_intopt.build()
            

    

    #self.intopt.build(1e-9, diag_block_with_triu=True, aosym=False)
    
    self.stream_gpu = cupy.cuda.Stream()

    log.info("Set up GPU integral class")

def eig(h, s):
    '''Solver for generalized eigenvalue problem

    .. math:: HC = SCE
    '''
    e, c = scipy.linalg.eigh(h, s)
    idx = numpy.argmax(abs(c.real), axis=0)
    c[:,c[idx,numpy.arange(len(e))].real<0] *= -1
    return e, c

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    '''F = h^{core} + V^{HF}

    Special treatment (damping, DIIS, or level shift) will be applied to the
    Fock matrix if diis and cycle is specified (The two parameters are passed
    to get_fock function during the SCF iteration)

    Kwargs:
        h1e : 2D ndarray
            Core hamiltonian
        s1e : 2D ndarray
            Overlap matrix, for DIIS
        vhf : 2D ndarray
            HF potential matrix
        dm : 2D ndarray
            Density matrix, for DIIS
        cycle : int
            Then present SCF iteration step, for DIIS
        diis : an object of :attr:`SCF.DIIS` class
            DIIS object to hold intermediate Fock and error vectors
        diis_start_cycle : int
            The step to start DIIS.  Default is 0.
        level_shift_factor : float or int
            Level shift (in AU) for virtual space.  Default is 0.
    '''
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    f = h1e + vhf
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
        f = damping(f, fock_last, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        if mf.use_gpu and (not mf.use_df_hf):
            dm = cupy.asarray(dm)
        f = diis.update(s1e, dm, f, mf, h1e, vhf)#, f_prev=fock_last)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f

def kernel(self, conv_tol=1e-8, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    tt = cput0 = get_current_time()
    mol = self.mol
    self.nao = self.mol.nao_nr()
    self.vk_last = None
    self.delta_e = 1
    self.chk_hf = "hf_mat.chk"
    self.file_ialp = "ialp_hf.tmp"
    self.nocc_pre = 0
    self.t_j = create_timer()
    self.t_k = create_timer()
    self.t_feri = create_timer()
    self.t_write = create_timer()
    self.t_read = create_timer()
    self.t_gpu = create_timer()
    self.t_data = create_timer()
    self.cycle = -1

    log = logger.Logger(self.stdout, self.verbose)
    #self.sol_eps = 78.3553
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        log.info('Set gradient conv threshold to %g'%conv_tol_grad)

    if (not self.use_gpu) and (not self.mol.pbc):
        self.shell_slice = int_prescreen.shell_prescreen(mol, self.with_df.auxmol, log, 
                                                         shell_slice=self.shell_slice, 
                                                         shell_tol=self.shell_tol)

    t0 = get_current_time()
    

    self.debug = False

    if not self.debug:

        win_s1e, self.s1e = get_shared((self.nao, self.nao))
        if irank_shm == 0:
            if self.mol.pbc:
                s1e_gpu = int1e.int1e_s1e(mol, self.mol.kpts[0])
                cupy.asnumpy(s1e_gpu, out=self.s1e)
            else:
                self.s1e[:] = self.get_ovlp(mol)
        comm_shm.Barrier()
        print_time([['s1e', get_elapsed_time(t0)]], log=log)
        #self.use_gpu = self.with_df.use_gpu = self.with_df.df_builder.use_gpu = False
        if self.mol.pbc:
            h1e = pbc_get_hcore(self.with_df)
        else:
            if irank_shm == 0:
                if self.use_gpu:
                    h1e = get_hcore_cuda(mol)
                    if self.use_df_hf:
                        h1e = h1e.get()
                else:
                    h1e = self.get_hcore(mol)
            else:
                h1e = None
            if self.mol_mm is not None:
                h1e = qmmm.get_hcore_qmmm(self, h1e)

        if irank_shm == 0:
            if self.use_gpu and (not self.use_df_hf):
                h1e = cupy.asarray(h1e)
                s1e = cupy.asarray(self.s1e)
            else:
                s1e = self.s1e
    else:
        h1e = 0.0
        self.s1e = None
    print_time([['core Hamiltonian', get_elapsed_time(t0)]], log=log)

    if self.use_df_hf:
        if self.int_storage == 2:
            if not self.debug:
                t0 = get_current_time()
                self.file_j2c = "j2c_hf.tmp"
                get_j2c_low(self, file_name=self.file_j2c, save_file_only=True)
                print_time([['j2c and CD', get_elapsed_time(t0)]], log=log)
            
        else:
            get_df_int3c2e(self, self.with_df, 'hf', ijp_shape=self.ijp_shape, ovlp=self.s1e, log=log)


    use_pin = self.use_gpu 

    
    win_conv, conv_hf = get_shared(1, dtype=np.int64)
    win_nocc, nocc = get_shared(1, dtype=np.int64)
    win_dm, dm = get_shared((self.nao, self.nao), dtype='f8', use_pin=use_pin)
    self.win_ene_hf, self.mo_energy = get_shared(self.nao, dtype='f8', use_pin=use_pin)
    self.win_occ_hf, self.mo_occ = get_shared(self.nao, dtype='f8', use_pin=use_pin)
    self.win_coef_hf, self.mo_coeff = get_shared((self.nao, self.nao), dtype='f8', use_pin=use_pin)
    win_hfe, hfe = get_shared(1)

    if not self.use_df_hf:
        if self.use_gpu:
            self.last_dm = cupy.zeros((self.nao, self.nao))
            if irank_shm == 0:
                self.last_vj = cupy.zeros((self.nao, self.nao))
                self.last_vk = cupy.zeros((self.nao, self.nao))
        else:
            self.last_dm = np.zeros((self.nao, self.nao))
            if irank_shm == 0:
                self.last_vj = np.zeros((self.nao, self.nao))
                self.last_vk = np.zeros((self.nao, self.nao))

    if self.use_gpu and (not self.use_df_hf):
        dm_cal = cupy.empty((self.nao, self.nao))
    else:
        dm_cal = dm

    '''if "opt" in self.cal_mode:
        if os.path.isfile("hf_mat.chk"):
            self.chkfile_init = "hf_mat.chk"'''
    if irank_shm == 0:
        conv_hf[0] = 0

    if self.chkfile_init is None:
        t0 = get_current_time()
        if irank_shm == 0:
            if self.use_df_hf:
                occ_coeff, mo_occ, dm0 = init_guess_by_minao(mol)
                occ_coeff *= (mo_occ**0.5)
                nocc[0] = occ_coeff.shape[-1]
                dm[:] = dm0
            else:
                dm0_gpu = init_guess_by_minao_cuda(mol)
                cupy.asnumpy(dm0_gpu, out=dm)
            #self.mo_occ[:] = mo_occ
            
            mo_occ, dm0 = None, None
            print_time([['Initial RHF DM', get_elapsed_time(t0)]], log=log)
        
        if self.use_df_hf:
            comm_shm.Barrier()
            win_cvi, self.occ_coeff = get_shared((self.nao, nocc[0]), dtype='f8')
            if irank_shm == 0:
                self.occ_coeff[:] = occ_coeff
                occ_coeff = None
                
        comm_shm.Barrier()

        if self.use_gpu and (not self.use_df_hf):
            dm_cal.set(dm)
        vhf = self.get_veff(mol, dm_cal)
        if self.use_df_hf:
            free_win(win_cvi); self.occ_coeff = None
        win_cvi, self.occ_coeff = get_shared((self.nao, self.mol.nelectron//2), dtype='f8', use_pin=use_pin)
    else:
        win_cvi, self.occ_coeff = get_shared((self.nao, self.mol.nelectron//2), dtype='f8', use_pin=use_pin)
        if irank_shm == 0:
            access_chkfile(self.chkfile_init, 'r', [dm, self.mo_energy, self.mo_coeff, self.mo_occ, self.occ_coeff, hfe])
        comm_shm.Barrier()
        if self.use_gpu and (not self.use_df_hf):
            dm_cal.set(dm)
        
        vhf = self.get_veff(mol, dm_cal)

    if self.sol_eps is not None:
        if (self.with_solvent.frozen):
            if hasattr(self.with_solvent, "e"):
                e_sol, v_sol = self.with_solvent.e, self.with_solvent.v
            else:
                e_sol, v_sol = get_veff_sol(self.with_solvent, dm)
                self.with_solvent.e, self.with_solvent.v = e_sol, v_sol
        else:
            e_sol, v_sol = get_veff_sol(self.with_solvent, dm)
    
    if irank_shm == 0:
        self.e_tot = self.energy_tot(dm_cal, h1e, vhf)
        
        if self.sol_eps is not None:
            self.e_tot += e_sol
        hfe[0] = self.e_tot
    comm_shm.Barrier()
    self.e_tot = hfe[0]
    logger.info(self, 'init E= %.15g', self.e_tot)
    
    self.converged = False
    #mo_energy = mo_coeff = mo_occ = None
    if irank_shm == 0:
        #s1e = self.get_s1e(mol)
        '''cond = lib.cond(self.s1e)
        logger.debug(self, 'cond(S) = %s', cond)
        if numpy.max(cond)*1e-17 > conv_tol:
            logger.warn(self, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                        'SCF may be inaccurate and hard to converge.', numpy.max(cond))'''

        # Skip SCF iterations. Compute only the total energy of the initial density
        if self.max_cycle <= 0:
            if self.sol_eps is not None:
                fock = get_fock(self, h1e, s1e, vhf+v_sol, dm_cal)
            else:
                fock = get_fock(self, h1e, s1e, vhf, dm_cal)  # = h1e + vhf, no DIIS
            self.mo_energy[:], self.mo_coeff[:] = scipy.linalg.eigh(fock, self.s1e) #self.eig(fock, self.s1e)
            self.mo_occ[:] = self.get_occ(self.mo_energy, self.mo_coeff)
            self.occ_coeff[:] = self.mo_coeff[:, self.mo_occ>0]*(self.mo_occ[self.mo_occ>0]**0.5)
            return self.converged, self.e_tot, self.mo_energy, self.mo_coeff, self.mo_occ
        if isinstance(self.diis, lib.diis.DIIS):
            self_diis = self.diis
        elif self.diis:
            assert issubclass(self.DIIS, lib.diis.DIIS)
            if self.use_gpu and (not self.use_df_hf):
                self_diis = CDIIS(self, self.diis_file)
            else:
                self_diis = self.DIIS(self, self.diis_file)
            self_diis.space = self.diis_space
            self_diis.rollback = self.diis_space_rollback
        else:
            self_diis = None
    print_time(['initialize scf', get_elapsed_time(cput0)], log)

    create_ialp = False
    for cycle in range(self.max_cycle):
        '''if irank_shm == 0:
            dm_last = np.copy(dm)'''
        
        self.cycle += 1

        last_hf_e = self.e_tot
        
        if irank_shm == 0:
            if (self.sol_eps is not None) and (v_sol is not None):
                fock = get_fock(self, h1e, s1e, vhf+v_sol, dm_cal, cycle, self_diis)
            else:
                fock = get_fock(self, h1e, s1e, vhf, dm_cal, cycle, self_diis)

            #self.mo_energy[:], self.mo_coeff[:] = self.eig(fock, self.s1e)
            #self.mo_energy[:], self.mo_coeff[:] = scipy.linalg.eigh(fock, self.s1e) #eig(fock, self.s1e)
            if self.use_gpu:
                t0 = get_current_time()
                fock_gpu = cupy.asarray(fock)
                s1e_gpu = cupy.asarray(s1e)
                mo_ene_gpu, mo_coef_gpu = eigh_cuda(fock_gpu, s1e_gpu)
                
                mo_occ_gpu = get_occ_cuda(self, mo_ene_gpu, mo_coef_gpu)
                #dm_gpu = make_rdm1(mo_coef_gpu, mo_occ_gpu)
                if self.use_df_hf:
                    dm_gpu = make_rdm1(mo_coef_gpu, mo_occ_gpu)
                else:
                    make_rdm1(mo_coef_gpu, mo_occ_gpu, out=dm_cal)
                    dm_gpu = dm_cal
                occ_coef_gpu = mo_coef_gpu[:, mo_occ_gpu>0]*(mo_occ_gpu[mo_occ_gpu>0]**0.5)
                t0 = get_current_time()
                cupy.asnumpy(mo_ene_gpu, out=self.mo_energy)
                cupy.asnumpy(mo_coef_gpu, out=self.mo_coeff)
                cupy.asnumpy(mo_occ_gpu, out=self.mo_occ)
                cupy.asnumpy(dm_gpu, out=dm)
                cupy.asnumpy(occ_coef_gpu, out=self.occ_coeff)
                cupy.cuda.Stream.null.synchronize()
            else:
                self.mo_energy[:], self.mo_coeff[:] = scipy.linalg.eigh(fock, self.s1e) #eig(fock, self.s1e)
                self.mo_occ[:] = self.get_occ(self.mo_energy, self.mo_coeff)
                self.occ_coeff[:] = self.mo_coeff[:, self.mo_occ>0]*(self.mo_occ[self.mo_occ>0]**0.5)

                dm[:] = make_rdm1(self.mo_coeff, self.mo_occ)
            #dm[:] = lib.tag_array(dm, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)

        comm_shm.Barrier()
        if self.use_gpu and (not self.use_df_hf):
            dm_cal.set(dm)
        #print(f"{irank}", dm_cal.min(), dm_cal.max(), dm_cal.mean(), flush=True)
        #comm_shm.Barrier()
        
        vhf = self.get_veff(mol, dm_cal)

        t0 = get_current_time()
        if (self.sol_eps is not None) and (not self.with_solvent.frozen):
            e_sol, v_sol = get_veff_sol(self.with_solvent, dm)
        if irank_shm == 0:
            self.e_tot = self.energy_tot(dm_cal, h1e, vhf)
            if self.sol_eps is not None:
                self.e_tot += e_sol
            hfe[0] = self.e_tot
            if irank == 0:
                if cycle == 0:
                    access_chkfile(self.chk_hf, 'w', [dm, self.mo_energy, self.mo_coeff, self.mo_occ, self.occ_coeff, hfe], cycle)
                else:
                    access_chkfile(self.chk_hf, 'r+', [dm, self.mo_energy, self.mo_coeff, self.mo_occ, self.occ_coeff, hfe], cycle)
            
        comm_shm.Barrier()
        self.e_tot = hfe[0]
        self.delta_e = self.e_tot-last_hf_e

        if (abs(self.delta_e) < self.conv_tol * 100) and (
            self.method != 0) and (
            self.mol.pbc or 
            (self.ml_test and not self.ml_mp2int)):
            self.shared_int = True
        
        if (not create_ialp) and (self.shared_int):
            nocc = self.mol.nelectron//2

            if self.int_storage in {0, 3}:
                self.win_ialp, self.ialp_node = get_shared((nocc, self.nao, self.naoaux))
            elif self.int_storage == 1:
                nao_chunk = min(536870912 // self.naoaux, self.nao) # chunks must be smaller than 4GB
                with h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm) as file_ialp:
                    '''file_ialp.create_dataset('ialp', (nocc, self.nao, self.naoaux), dtype='f8', 
                                             chunks=(1, nao_chunk, self.naoaux))'''
                    create_h5py_dataset(file_ialp, "ialp", (nocc, self.nao, self.naoaux), 
                                        dtype=np.float64, chunks=(1, nao_chunk, self.naoaux))
            
            create_ialp = True

        if irank_shm == 0:
            #norm_gmo = np.linalg.norm(self.get_grad(self.mo_coeff, self.mo_occ, fock))
            #norm_gmo = norm_gmo / np.sqrt(norm_gmo.size)
            #norm_ddm = np.linalg.norm(dm-dm_last)
            #logger.info(self, 'cycle= %d E= %.12f, delta_E= %.2E, |g|=%.2E, |ddm|=%.2E', cycle+1, self.e_tot, self.delta_e, norm_gmo, norm_ddm)
            logger.info(self, 'cycle= %d E= %.12f, delta_E= %.2E', cycle+1, self.e_tot, self.delta_e)
            if (abs(self.e_tot-last_hf_e) < conv_tol):#and (norm_gmo < conv_tol_grad):
                self.converged = True
            if self.converged:
                conv_hf[0] = 1
        comm_shm.Barrier()
        self.converged = bool(conv_hf[0])
        if self.converged:
            break
        
    if self.converged:
        #self.dm = dm
        if (self.chkfile_save is not None) and (irank == 0):
            shutil.copy(self.chk_hf, self.chkfile_save)
            os.remove("%s/hf_mat_cycle%d.chk"%(self.chkfile_save, cycle))

        if self.shared_int:
            if nnode > 1: 
                if self.int_storage == 0:
                    Acc_and_get_GA(self.ialp_node)
                comm.Barrier()


    comm.Barrier()

    if use_pin and irank_shm == 0:
        for pin_array in [dm, self.mo_energy, self.mo_occ,
                          self.mo_coeff, self.occ_coeff]:
            unregister_pinned_memory(pin_array)

    for win in [win_hfe, win_dm, win_conv, win_nocc, win_s1e]:
        free_win(win)
    
    if not self.use_df_hf:
        self.last_dm = None
        if irank_shm == 0:
            self.last_vj = None
            self.last_vk = None
    self.s1e = None

    if (self.int_storage == 2) and (self.use_gpu) and (not self.use_df_hf):
        self.intopt = None
    if (irank == 0) and (self.use_df_hf) and (self.int_storage == 2) and (not self.shared_int):
        os.remove(self.file_ialp)
    
    self.t_hf = get_elapsed_time(tt)
    
    time_list = [['J matrix', self.t_j], ['K matrix', self.t_k]]
    if self.int_storage == 2:
        time_list.append(['feri', self.t_feri])
    time_list.append(["calculation", self.t_j+self.t_k+self.t_feri])
    if self.int_storage not in {0, 3}:
        time_list += ['reading', self.t_read], ['writing', self.t_write]
    if self.use_gpu:
        time_list.append(["GPU-CPU data", self.t_data])
    time_list.append(['RHF energy', self.t_hf])
    time_list = get_max_rank_time_list(time_list)
    if irank == 0:
        print_time(time_list, log)
        print_mem('RHF energy', self.pid_list, log)

        
def scf_ene(self, dm0=None, **kwargs):
    #cput0 = get_current_time()
    kernel(self, self.conv_tol, self.conv_tol_grad,
            dm0=dm0, callback=self.callback,
            conv_check=self.conv_check, **kwargs)

    #print_time(['SCF', get_elapsed_time(cput0)], log)
    self._finalize()
    return self.e_tot

def add_instance(hf, my_para):
    hf.__dict__.update(my_para.__dict__)
    if not hf.mol.pbc:
        hf.with_df = DF(hf.mol)
        hf.with_df.auxbasis = hf.auxbasis_hf
        hf.with_df.auxmol = hf.auxmol_hf
    hf.chkfile_ialp = hf.chkfile_ialp_hf
    hf.chkfile_fitratio = hf.chkfile_fitratio_hf
    hf.auxmol = hf.auxmol_hf
    hf.nao = hf.mol.nao_nr()
    hf.naoaux = hf.naux_hf
    hf.nocc_core = loc_addons.get_ncore(hf.mol)
    if irank == 0:
        hf.verbose = min(my_para.verbose, 4)
    hf.shm_ranklist = range(nrank//nnode)
    hf.conv_tol = 1e-8
    hf.max_cycle = 30
    funcType = types.MethodType
    hf.get_veff = funcType(get_veff, hf)
    hf.get_occ = funcType(get_occ, hf)
    hf.scf = funcType(scf_ene, hf)
    hf.energy_tot = funcType(energy_tot, hf)
    hf.shell_slice = None
    hf.ijp_shape = False
    hf.shared_int = False
    hf.debug = False
    return hf

def build_solvent(hf, log):
    if hf.sol_eps is not None:
        log.info('Set up solvation model with dielectric constant: %.2f'%(hf.sol_eps))
        t0 = get_current_time()
        if not hasattr(hf, 'with_solvent'):
            hf.with_solvent = ddcosmo.DDCOSMO(hf.mol)
            hf.with_solvent.eps = hf.sol_eps
            hf.with_solvent.frozen = False
            hf.with_solvent.build()
            hf.with_solvent.grid_int = None
            hf.with_solvent.int3c2e = None
            hf.with_solvent.int_storage = hf.int_storage
            hf.with_solvent.outcore = hf.outcore
        print_time([['solvation model setup', get_elapsed_time(t0)]], log=log)
    else:
        hf.with_solvent = None

def scf_parallel(hf, my_para):
    hf = add_instance(hf, my_para)
    log = logger.Logger(hf.stdout, hf.verbose)
    log.info('\n--------------------------------RHF energy---------------------------------')
    build_solvent(hf, log)
    if my_para.chkfile_hf is not None:
        t1 = get_current_time()
        log.info("Read HF matrices from check file: %s"%my_para.chkfile_hf)
        hf.win_ene_hf, hf.mo_energy = get_shared(hf.nao, dtype='f8')
        hf.win_occ_hf, hf.mo_occ = get_shared(hf.nao, dtype='f8')
        hf.win_coef_hf, hf.mo_coeff = get_shared((hf.nao, hf.nao), dtype='f8')
        hf.win_dm_hf, hf.dm = get_shared((hf.nao, hf.nao), dtype='f8')
        hf.win_moe_hf, e_tot = get_shared(1, dtype='f8')
        if irank_shm == 0:
            '''with h5py.File(my_para.chkfile_hf, 'r') as f:
                f["scf/mo_energy"].read_direct(hf.mo_energy)
                f["scf/mo_occ"].read_direct(hf.mo_occ)
                f["scf/mo_coeff"].read_direct(hf.mo_coeff)
                f["scf/dm"].read_direct(hf.dm)
                f["scf/e_tot"].read_direct(e_tot)'''

            arrays = [hf.dm, hf.mo_energy, hf.mo_coeff, hf.mo_occ, None, e_tot]
            access_chkfile(my_para.chkfile_hf, 'r', arrays)
        comm_shm.Barrier()
        print_time(["reading checkfile", get_elapsed_time(t1)], log)
        hfe = hf.e_tot = e_tot[0]
        if (not hf.int_storage == 2) and (hf.cal_grad):
            get_df_int3c2e(hf, hf.with_df, 'hf', ijp_shape=hf.ijp_shape, log=log)
        hf.t_hf = get_elapsed_time(t1)
        
    else:
        if hf.use_gpu:
            get_gpu_intopt(hf, log, use_df=hf.use_df_hf)
        
        hfe= hf.kernel()
        
    if hf.pop_hf:
        loc_addons.analysis(hf.mol, hf.dm, meth='RHF', charge_method=hf.charge_method, save_data='RHF', log=log)
    if (hf.ml_test) and (not hf.ml_mp2int):
        ml_rhf(hf, log)
    return hfe



