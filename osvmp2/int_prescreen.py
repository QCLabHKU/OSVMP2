import math
import numpy as np
import itertools
import ctypes
from pyscf import lib
from pyscf.gto.moleintor import make_loc
from pyscf.scf import _vhf
from osvmp2.osvutil import *
from osvmp2.mpi_addons import get_shared, get_slice, Acc_and_get_GA
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm

libcgto = lib.load_library('libcgto')
null = lib.c_null_ptr()

def combine_shells(shell_slice):
    shell_com = []
    b_list = []
    a0_pre = shell_slice[0][0]
    for idx, (a0, a1, b0, b1) in enumerate(shell_slice):
        if a0 != a0_pre:
            shell_com.append(b_list)
            b_list = [[a0, a1, b0, b1]]
        else:
            if b_list == []:
                b_list.append([a0, a1, b0, b1])
            else:
                if b0 == b_list[-1][-1]:
                    b_list[-1][-1] = b1
                else:
                    b_list.append([a0, a1, b0, b1])
        if idx == len(shell_slice)-1:
            shell_com.append(b_list)
        a0_pre = a0
    idx_list = []
    idx_i = []
    for idx, shell_i in enumerate(shell_com):
        b_list = []
        for a0, a1, b0, b1 in shell_i:
            b_list.extend([b0, b1])
        if idx == 0:
            idx_i.append(idx)
        else:
            if b_list == blist_pre:
                idx_i.append(idx)
            else:
                idx_list.append(idx_i)
                idx_i = [idx]
        if idx == len(shell_com)-1:
            idx_list.append(idx_i)
        blist_pre = b_list
    shell_slice = []
    for idx_i in idx_list:
        a0, a1 = shell_com[idx_i[0]][0][0], shell_com[idx_i[-1]][0][1]
        for ax0, ax1, b0, b1 in shell_com[idx_i[0]]:
            shell_slice.append([a0, a1, b0, b1])
    return shell_slice

def collect_slice(shell_seg, naop_shell, max_sum):
    shell_i = []
    len_si = 0
    shell_slice = []
    len_list = []
    for idx, len_i in enumerate(naop_shell):
        shell_i.append(shell_seg[idx])
        len_si += len_i
        idx_next = idx+1
        if idx_next > len(naop_shell)-1:
            idx_next = idx
        if (len_si + naop_shell[idx_next] > max_sum) or idx==(len(naop_shell)-1):
            shell_slice.append(shell_i)
            shell_i = []
            len_list.append(len_si)
            len_si = 0
    return shell_slice, len_list

def OPTPartition(n_slice, shell_seg, naop_shell, match_num=False):#match_num=True):
    #max_sum = get_max_sum(naop_shell, len(naop_shell), n_slice)
    max_sum = sum(naop_shell)//n_slice + 10
    shell_slice, len_list = collect_slice(shell_seg, naop_shell, max_sum)
    len_slice = len(len_list)
    shell_pre = shell_slice
    len_pre = len_list
    step = 0
    var_i = 10
    if (len_slice != n_slice):#and match_num:
        while len_slice != n_slice:
            shell_slice, len_list = collect_slice(shell_seg, naop_shell, max_sum)
            len_slice = len(len_list)
            if len_slice < n_slice:
                if (step > 20) and (len(len_pre)>n_slice):
                    break
                if step > 50:
                    break
                max_sum -= var_i
                shell_pre = shell_slice
                len_pre = len_list
            elif len_slice > n_slice:
                max_sum += var_i
                shell_pre = shell_slice
                len_pre = len_list
            step += 1
            var_i = var_i//2
            if var_i == 0:
                var_i = 1
    shell_slice_com = []
    for shell_i in shell_slice:
        shell_slice_com.append(combine_shells(shell_i))
    return shell_slice_com, len_list


def even_partition(a, m):
    """
    Partitions a list into m contiguous fragments with sums as close as possible.
    
    Args:
        a (list):List of numbers to partition.
        m (int):Number of fragments (1 <= m <= len(a)).
    
    Returns:
        list:List of m indices, each a sublist of a.
    
    Raises:
        ValueError:If m is less than 1 or greater than the list length.
    """
    # Convert input list to NumPy array for efficient operations
    a = np.array(a)
    n = len(a)
    
    # Validate input
    if m < 1 or m > n:
        raise ValueError("m must be between 1 and the length of the list")
    
    # Compute total sum and cumulative sum
    S = np.sum(a)
    csum = np.cumsum(a)  # csum[i] = sum(a[0:i+1])
    ideal_sum = S / m
    
    # Initialize split indices starting with 0
    split_indices = [0]
    
    # Find split points for fragments 1 to m-1
    for k in range(1, m):
        target = k * ideal_sum
        # Find index j where csum[j] is closest to target
        j = np.argmin(np.abs(csum - target))
        j = max(j, split_indices[-1] + 1)
        split_indices.append(j)
        #split_indices.append(j + 1)  # +1 because csum[j] includes a[j], and we split after a[j]
    
    # Add the end of the list as the final index
    split_indices.append(n)
    
    # Create fragments using the split indices
    fragments = [[split_indices[i], split_indices[i+1]] for i in range(m)]
    return fragments


def alloc_shell(shell_slice, naop_shell):
    shell_slice, len_list = OPTPartition(nrank, shell_slice, naop_shell)
    if len(len_list) < nrank:
        shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
        for idx, si in enumerate(shell_slice):
            if si is not None:
                shell_slice[idx] = si[0]
    return shell_slice




def collect_shell_pairs(shell_pairs, naop_shell_pairs, ao_loc):
    alist = shell_pairs[:, 0]
    shells, starts = np.unique(alist, return_index=True)
    nshell = len(shells)
    npairs = len(shell_pairs)
    shell_slice_uni = {}
    naop_list = np.empty(nshell, dtype=int)
    for a in shells:
        start = starts[a]
        end = npairs if (a == nshell-1) else starts[a + 1]

        shell_slice_uni[a] = shell_pairs[start:end]

        naop_list[a] = naop_shell_pairs[start:end].sum()
    if nshell < nrank:
        slice_offsets = [[a, a+1] for a in shells]
    else:
        slice_offsets = even_partition(naop_list, nrank)
    
    '''shell_slice = [None] * nrank
    for irank, (sidx0, sidx1) in enumerate(slice_offsets):
        shell_slice[irank] = {}
        for a in range(sidx0, sidx1):
            shell_slice[irank][a] = shell_slice_uni[a]'''
    ao_slice = [None] * nrank
    for rank_i, (sidx0, sidx1) in enumerate(slice_offsets):
        ao_slice[rank_i] = [ao_loc[sidx0], ao_loc[sidx1]]
    if irank < len(slice_offsets):
        sidx0, sidx1 = slice_offsets[irank]
        shell_list = np.arange(sidx0, sidx1)
        merged_b = {}
        for a in shell_list:
            slice_a = [np.copy(shell_slice_uni[a][0][2:])]
            for _, _, b0, b1 in shell_slice_uni[a][1:]:
                if (b0 - slice_a[-1][1]) < 5:
                    slice_a[-1][1] = b1
                else:
                    slice_a.append([b0, b1])
            merged_b[a] = np.asarray(slice_a)

        a0 = shell_list[0]
        merged_a = [[[a0, a0+1], merged_b[a0]]]

        for a in shell_list[1:]:
            if np.array_equal(merged_b[a], merged_a[-1][1]):
                merged_a[-1][0][1] = a + 1
            else:
                merged_a.append([[a, a+1], merged_b[a]])
        slice_rank = []
        for (a0, a1), bsegs in merged_a:
            for b0, b1 in bsegs:
                slice_rank.append([a0, a1, b0, b1])
        slice_rank = np.asarray(slice_rank)
        #print(slice_rank)
    else:
        slice_rank = None

    shell_slice = [ao_slice, slice_rank]

    return shell_slice


def merge_shell_pairs(shell_slice):
    pass


def get_gr_ref(mol, auxmol):
    opt = _vhf.VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond')
    #opt.direct_scf_tol = 1e-13
    # q_cond part 1: the regular int2e (ij|ij) for mol's basis
    opt.init_cvhf_direct(mol, 'int2e', 'CVHFsetnr_direct_scf')
    mol_q_cond = lib.frompointer(opt._this.contents.q_cond, mol.nbas**2)
    # Update q_cond to include the 2e-integrals (auxmol|auxmol)
    j2c = auxmol.intor('int2c2e', hermi=1)
    j2c_diag = np.sqrt(abs(j2c.diagonal()))
    aux_loc = auxmol.ao_loc
    aux_q_cond = [j2c_diag[i0:i1].max()
                    for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
    #if mo_coeff is not None:
    nshell = mol.nbas

    return (mol_q_cond*np.mean(aux_q_cond)).reshape(nshell, nshell)

def get_gr(mol, auxmol, mode=1):
    nshell = mol.nbas
    win_gr, gr = get_shared((nshell, nshell), set_zeros=True)
    
    if mode == 0:
        nshell_aux = auxmol.nbas
        win_qp, qp = get_shared(nshell_aux, set_zeros=True)

        if irank == 0:
            np.fill_diagonal(gr, 1)
        alist, blist = np.triu_indices(nshell, k=1)
        spidx_slice = get_slice(range(nrank), job_size=len(alist))[irank]
        
        t0 = get_current_time()
        if spidx_slice is not None:
            atm = np.asarray(mol._atm, dtype=np.int32, order='C')
            bas = np.asarray(mol._bas, dtype=np.int32, order='C')
            env = np.asarray(mol._env, dtype=np.double, order='C')
            shls = np.zeros(4, dtype=np.int32)

            ngto_all = (bas[:, 1] * 2 + 1) * bas[:, 3]

            ngto_as = ngto_all[alist]
            ngto_bs = ngto_all[blist]
            bra_eri_sizes = (ngto_as * ngto_bs)
            eri_sizes = bra_eri_sizes**2
            max_eri_size = np.max(eri_sizes)
            buf_eri = np.empty(max_eri_size, np.double, order='F')

            c_buf_eri = buf_eri.ctypes.data_as(ctypes.c_void_p)
            c_atm = atm.ctypes.data_as(ctypes.c_void_p)
            c_bas = bas.ctypes.data_as(ctypes.c_void_p)
            c_env = env.ctypes.data_as(ctypes.c_void_p)
            c_shls = shls.ctypes.data_as(ctypes.c_void_p)
            natm = ctypes.c_int(atm.shape[0])
            nbas = ctypes.c_int(bas.shape[0])
            fintor = getattr(libcgto, "int2e_sph")
            

            for pidx in spidx_slice:
                a = alist[pidx]
                b = blist[pidx]
                shls[:] = (a, b, a, b)
                
                # (al be | al be)
                #eri_shell_test = mol.intor_by_shell('int2e_sph', shls)
                fintor(c_buf_eri,
                    null, c_shls,
                    c_atm, natm,
                    c_bas, nbas,
                    c_env, null, null)
                
                bra_size = bra_eri_sizes[pidx]
                eri_shell = buf_eri[:eri_sizes[pidx]].reshape(bra_size, bra_size).diagonal()
                #np.absolute(eri_shell, out=eri_shell)
                max_eri = eri_shell.max()
                qab = math.sqrt(max_eri)
                gr[a, b] = qab
                gr[b, a] = qab
                
        print_time(['gr', get_elapsed_time(t0)])

        aux_shell_slice = get_slice(range(nrank), job_size=nshell_aux)[irank]
        if aux_shell_slice is not None:
            for a in aux_shell_slice:
                shls = (a, a)
                # (p|p)
                eri_shell = np.diag(auxmol.intor_by_shell('int2c2e', shls))
                #eri_shell = np.absolute(eri_shell)
                qp[a] = math.sqrt(eri_shell.max())

        comm.Barrier()
        #Acc_and_get_GA(gr)
        Acc_and_get_GA(qp)
        comm.Barrier()

        if shm_rank == 0:
            gr *= np.mean(qp)
        shm_comm.Barrier()
        free_win(win_qp)

    elif mode == 1:
        if shm_rank == 0:
            opt = _vhf.VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond')
            #opt.direct_scf_tol = 1e-13
            # q_cond part 1:the regular int2e (ij|ij) for mol's basis
            opt.init_cvhf_direct(mol, 'int2e', 'CVHFsetnr_direct_scf')
            gr[:] = lib.frompointer(opt._this.contents.q_cond, mol.nbas**2).reshape(nshell, nshell)
            # Update q_cond to include the 2e-integrals (auxmol|auxmol)
            j2c = auxmol.intor('int2c2e', hermi=1)
            j2c_diag = abs(j2c.diagonal())
            aux_loc = auxmol.ao_loc
            aux_q_cond = [np.sqrt(j2c_diag[i0:i1].max())
                            for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
            
            gr *= np.mean(aux_q_cond)
        shm_comm.Barrier()

    return win_gr, gr

def shell_prescreen(mol, auxmol, log, shell_slice=None, shell_tol=1e-10, 
                    qc_method='RHF', use_atom_pairs=False):
    nshell = mol.nbas
    nao = mol.nao_nr()
    naoaux = auxmol.nao_nr()
    ao_loc = make_loc(mol._bas, 'sph')
    if shell_slice is None:
        log.info('\nBegin Cauchy-Schwartz prescreening for %s...'%qc_method)

        t0 = get_current_time()
        win_gr, gr = get_gr(mol, auxmol)
        print_time(['Schwartz integral', get_elapsed_time(t0)], log=log)
        
        ao_atm_offset = mol.offset_nr_by_atom()
        natm = len(ao_atm_offset)
        
        #kept_atoms = np.full((mol.natm, mol.natm), False, dtype=bool)
        shell_list = np.arange(nshell)
        alist, blist = np.meshgrid(shell_list, shell_list, indexing='ij')
        alist = alist.reshape(-1, 1)
        blist = blist.reshape(-1, 1)
        shell_pairs = np.hstack([alist, alist+1, blist, blist+1])
        t0 = get_current_time()

        if use_atom_pairs:
            mask_kept = np.zeros((nshell, nshell), dtype=bool)
            for atm0, (a0, a1, _, _) in enumerate(ao_atm_offset):
                mask_kept[a0:a1, a0:a1] = True
                if atm0 == (natm - 1):
                    continue
                for b0, b1, _, _ in ao_atm_offset[atm0+1:]:
                    if (gr[a0:a1, b0:b1].max() > shell_tol):
                        mask_kept[a0:a1, b0:b1] = True
                        mask_kept[b0:b1, a0:a1] = True
        else:
            mask_kept = gr > shell_tol

        shell_pairs = shell_pairs[mask_kept.ravel()]
        nal = ao_loc[shell_pairs[:, 1]] - ao_loc[shell_pairs[:, 0]]
        nbe = ao_loc[shell_pairs[:, 3]] - ao_loc[shell_pairs[:, 2]]
        naop_shell_pairs = nal * nbe
        
        total_npairs = nao ** 2
        kept_npairs = np.sum(naop_shell_pairs)
        scr_npairs = total_npairs - kept_npairs
        scr_percent = 100 * scr_npairs / total_npairs

        shell_slice = collect_shell_pairs(shell_pairs, naop_shell_pairs, ao_loc)

        print_time(['shell pairs selection', get_elapsed_time(t0)], log=log)

        #####
        #shell_slice = alloc_shell(shell_pairs, naop_shell_pairs)
        
        shm_comm.Barrier()
        win_gr.Free()
        if irank == 0:
            msg = "    Threshold for AO pair screening:%.2E\n"%shell_tol
            msg += "    %d out of %d AO pairs are screened, sparsity:%.2f percent"%(scr_npairs, total_npairs, scr_percent)
            #log.info(msg)
            print(msg)
        
        #sys.exit()
    
    return shell_slice

def get_shell_segs(mol, slice_rank):
    shell_check = np.zeros(mol.nbas+1, dtype=bool)
    shell_seg = []
    for a0, a1, b0, b1 in slice_rank:
        if shell_check[a0]:
            shell_seg[-1][2].append([b0, b1])
        else:
            shell_seg.append([a0, a1, [[b0, b1]]])
            shell_check[a0] = True
    return shell_seg

def get_slice_rank(shell_slice, aslice=False):
    if aslice:
        return shell_slice
    else:
        return shell_slice[1]


def mem_control(mol, nocc, naoaux, slice_rank, instance, max_memory=None, nbfit=None):
    
    def mem_con_ashell(slice_rank, size_feri, max_nao=None, nao0_list=None):
        shell_seg = get_shell_segs(mol, slice_rank)
        shellslice_a = []

        for a0, a1, b_seg in shell_seg:
            #max_nao1 = max([(ao_loc[b1]-ao_loc[b0]) for b0, b1 in b_seg])
            b_seg = np.asarray(b_seg)
            max_nao1 = np.max(ao_loc[b_seg[:, 1]] - ao_loc[b_seg[:, 0]])
            al0, al1 = ao_loc[a0], ao_loc[a1]
            nao0 = al1 - al0

            if max_nao is None:
                max_nao0 = size_feri // (max_nao1*naoaux)
            else:
                max_nao0 = max_nao
            
            if (nao0 > max_nao0) and (a1 - a0 > 1):
                aslices = [[a0, a0+1]]
                for a in np.arange(a0+1, a1):
                    nao_slice = ao_loc[a+1] - ao_loc[aslices[-1][0]]
                    if nao_slice > max_nao0:
                        aslices.append([a, a+1])
                    else:
                        aslices[-1][1] = a+1
            else:
                aslices = [[a0, a1]]

            for sa0, sa1 in aslices:
                for b0, b1 in b_seg:
                    shellslice_a.append([sa0, sa1, b0, b1])
        
        return shellslice_a
    
    def mem_con_bshell(shellslice_a, size_feri):
        shellslice_rank = []
        for shell_i in shellslice_a:
            a0, a1, b0, b1 = shell_i
            al0, al1, be0, be1 = [ao_loc[s] for s in shell_i]
            nao0 = al1 - al0
            nao1 = be1 - be0
            if nao0*nao1*naoaux > size_feri:
                bi_0 = b0
                shell_temp = []
                for bi in range(b0, b1):
                    if nao0*(ao_loc[bi]-ao_loc[bi_0])*naoaux > size_feri:
                        shell_temp.append([a0, a1, bi_0, bi])
                        bi_0 = bi
                    if bi == b1-1:
                        shell_temp.append([a0, a1, bi_0, bi+1])
                shellslice_rank.extend(shell_temp)
            else:
                shellslice_rank.append([a0, a1, b0, b1])
        return shellslice_rank
    
    def get_size_feri(shellslice_rank):
        '''naop_list = []
        for shell_i in shellslice_rank:
            al0, al1, be0, be1 = [ao_loc[s] for s in shell_i]
            naop_list.append((al1-al0)*(be1-be0))'''
        shellslice_rank = np.asarray(shellslice_rank)
        nao0_list = ao_loc[shellslice_rank[:, 1]] - ao_loc[shellslice_rank[:, 0]]
        nao1_list = ao_loc[shellslice_rank[:, 3]] - ao_loc[shellslice_rank[:, 2]]
        return np.max(nao0_list * nao1_list) * naoaux
    
    def get_size_ialp(shellslice_rank, max_nao, prod_no_naux, nbfit=None):
        shell_check = [False]*mol.nbas
        nal_list = []
        for shell_i in shellslice_rank:
            if shell_check[shell_i[0]] == False:
                al0, al1, be0, be1 = [ao_loc[s] for s in shell_i]
                nal_list.append(al1-al0)
        if max_nao >= sum(nal_list):
            max_nao = sum(nal_list)
        elif max_nao <= max(nal_list):
            max_nao = max(nal_list)
        else:
            max_nal_pre = max(nal_list)
            for n in range(2, len(nal_list)):
                sum_list = []
                for idx0 in range(len(nal_list)-n+1):
                    sum_list.append(sum(nal_list[idx0:idx0+n]))
                max_nal = max(sum_list)
                if max_nao < max_nal:
                    max_nao = max_nal_pre
                    break
                max_nal_pre = max_nal

        size_ialp = max_nao*prod_no_naux
        if nbfit is None:
            size_ialp_f = max(nal_list)*prod_no_naux
        else:
            size_ialp_f = max(nal_list)*max([nfit_i for nfit_i in nbfit if nfit_i is not None])
        return size_ialp, size_ialp_f
    
    ao_loc = make_loc(mol._bas, 'sph')
    if max_memory is None:
        max_memory = 0.9*get_mem_spare(mol)

    if type(instance) == float:
        r_feri = instance
        size_feri = r_feri*max_memory*1e6/8
        shellslice_a = mem_con_ashell(slice_rank, size_feri=size_feri)
        shellslice_rank = mem_con_bshell(shellslice_a, size_feri)
        max_feri_size = get_size_feri(shellslice_rank)
        return max_feri_size, shellslice_rank
    
    elif instance == "feri":
        max_size_feri = max_memory * 1e6 / 8 
        sidx0 = slice_rank[0][0]
        sidx1 = slice_rank[-1][1]
        shells_rank = np.arange(sidx0, sidx1)
        nao_rank = ao_loc[sidx1] - ao_loc[sidx0]

        min_nao0 = np.max(ao_loc[shells_rank+1] - ao_loc[shells_rank])

        shellslice_rank = []
        size_feri_list = []

        shell_seg = get_shell_segs(mol, slice_rank)
        for a0, a1, b_seg in shell_seg:
            nao0 = ao_loc[a1] - ao_loc[a0]
            as_seg = np.arange(a0, a1)
            min_nao0_seg = np.max(ao_loc[as_seg+1] - ao_loc[as_seg])
            b_seg = np.asarray(b_seg)
            max_nao1_check = np.max(ao_loc[b_seg[:, 1]] - ao_loc[b_seg[:, 0]])
            max_nao0_check = max_size_feri / (naoaux * max_nao1_check)
            max_nao0_seg = min(max(max_nao0_check, min_nao0_seg), nao_rank)

            if max_nao0_seg >= nao0:
                a_segs = [[a0, a1]]
            else:
                a_segs = [[a0, a0+1]]
                for a in as_seg:
                    if ao_loc[a+1] - ao_loc[a_segs[-1][0]] > max_nao0_seg:
                        a_segs.append([a, a+1])
                    else:
                        a_segs[-1][1] = a+1

            for a0_now, a1_now in a_segs:
                nao0_now = ao_loc[a1_now] - ao_loc[a0_now]
                max_nao1_now = max_size_feri // (naoaux * nao0_now)
                for b0, b1 in b_seg:
                    nao1 = ao_loc[b1] - ao_loc[b0]
                    if max_nao1_now >= nao1:
                        shellslice_rank.append([a0_now, a1_now, b0, b1])
                        size_feri_list.append(nao0_now*nao1*naoaux)
                    else:
                        b_segs = [[b0, b0+1]]
                        for b in range(b0, b1):
                            if ao_loc[b+1] - ao_loc[b_segs[-1][0]] > max_nao1_now:
                                b_segs.append([b, b+1])
                            else:
                                b_segs[-1][1] = b+1
                        
                        for b0_now, b1_now in b_seg:
                            nao1_now = ao_loc[b1_now] - ao_loc[b0_now]
                            shellslice_rank.append([a0_now, a1_now, b0_now, b1_now])
                            size_feri_list.append(nao0_now*nao1_now*naoaux)


        shellslice_rank = np.asarray(shellslice_rank)
        size_feri = np.max(size_feri_list)
        return size_feri, shellslice_rank
    
    elif instance == "half_trans":
        #r_feri = 0.2
        r_ialp = 0.6
 
        max_memory_f8 = max_memory * 1e6 / 8 
        sidx0 = slice_rank[0][0]
        sidx1 = slice_rank[-1][1]
        shells_rank = np.arange(sidx0, sidx1)
        nao_rank = ao_loc[sidx1] - ao_loc[sidx0]
        min_nao0 = np.max(ao_loc[shells_rank+1] - ao_loc[shells_rank])
        
        nao_ialp = min(max(int(r_ialp * max_memory_f8  / (nocc * naoaux)), min_nao0), nao_rank)
        size_ialp = nocc * nao_ialp * naoaux

        max_size_feri = max_memory_f8 - size_ialp
        shellslice_rank = []
        size_feri_list = []

        shell_seg = get_shell_segs(mol, slice_rank)
        for a0, a1, b_seg in shell_seg:
            nao0 = ao_loc[a1] - ao_loc[a0]
            as_seg = np.arange(a0, a1)
            min_nao0_seg = np.max(ao_loc[as_seg+1] - ao_loc[as_seg])
            b_seg = np.asarray(b_seg)
            max_nao1_check = np.max(ao_loc[b_seg[:, 1]] - ao_loc[b_seg[:, 0]])
            max_nao0_check = max_size_feri / (naoaux * max_nao1_check)
            max_nao0_seg = min(max(max_nao0_check, min_nao0_seg), nao_ialp)
            if max_nao0_seg >= nao0:
                a_segs = [[a0, a1]]
            else:
                a_segs = [[a0, a0+1]]
                for a in as_seg:
                    if ao_loc[a+1] - ao_loc[a_segs[-1][0]] > max_nao0_seg:
                        a_segs.append([a, a+1])
                    else:
                        a_segs[-1][1] = a+1

            for a0_now, a1_now in a_segs:
                nao0_now = ao_loc[a1_now] - ao_loc[a0_now]
                max_nao1_now = max_size_feri // (naoaux * nao0_now)
                for b0, b1 in b_seg:
                    nao1 = ao_loc[b1] - ao_loc[b0]
                    if max_nao1_now >= nao1:
                        shellslice_rank.append([a0_now, a1_now, b0, b1])
                        size_feri_list.append(nao0_now*nao1*naoaux)
                    else:
                        b_segs = [[b0, b0+1]]
                        for b in range(b0, b1):
                            if ao_loc[b+1] - ao_loc[b_segs[-1][0]] > max_nao1_now:
                                b_segs.append([b, b+1])
                            else:
                                b_segs[-1][1] = b+1
                        
                        for b0_now, b1_now in b_seg:
                            nao1_now = ao_loc[b1_now] - ao_loc[b0_now]
                            shellslice_rank.append([a0_now, a1_now, b0_now, b1_now])
                            size_feri_list.append(nao0_now*nao1_now*naoaux)


        shellslice_rank = np.asarray(shellslice_rank)
        size_feri = np.max(size_feri_list)
        return size_ialp, size_feri, shellslice_rank
    elif instance == "derivative_feri":
        r_ialp = 0.3
        mem_cost_pre = 0
        for istep in range(int((1-r_ialp)//0.05) + 1):
            nao0_list = []
            naop_list = []
            nao_rank = 0
            al0_pre = -1
            for a0, a1, b0, b1 in slice_rank:
                al0, al1, be0, be1 = [ao_loc[s] for s in [a0, a1, b0, b1]]
                nao0 = al1 - al0
                nao1 = be1 - be0
                nao0_list.append(nao0)
                naop_list.append(nao0*nao1)
                if al0 != al0_pre:
                    nao_rank += nao0
                al0_pre= al0
            #size_feri = max(naop_list)*naoaux
            size_ialp = max(nao0_list)*nocc*naoaux
            if size_ialp*8*1e-6 < r_ialp*max_memory:
                max_nao = max(nao0_list)
            else:
                nal_list = []
                for a0, a1, b0, b1 in slice_rank:
                    for ai in range(a0, a1):
                        nal_list.append(ao_loc[ai+1] - ao_loc[ai])
                max_nao = int(max(r_ialp*max_memory//(nocc*naoaux*8*1e-6), max(nal_list)))
                size_ialp = max_nao*nocc*naoaux
            if max_nao < max(nao0_list):
                shellslice_a = mem_con_ashell(slice_rank, max_nao=max_nao, nao0_list=nao0_list)
            else:
                shellslice_a = slice_rank
            size_ialp = get_size_ialp(shellslice_a, max_nao, nocc*naoaux)[1]
            size_feri = (max_memory - (size_ialp*8*1e-6))//(4*8*1e-6)
            shellslice_rank = mem_con_bshell(shellslice_a, size_feri)
            size_feri = 4*get_size_feri(shellslice_rank)
            mem_cost = (size_ialp+size_feri)*8*1e-6
            #if irank == 0:print(r_ialp, "%.2f MB, %.2f MB"%((size_ialp+size_feri)*8*1e-6, max_memory))
            if (mem_cost > max_memory) or (r_ialp == 1) or (mem_cost == mem_cost_pre):
                break
            else:
                r_ialp += 0.05
                mem_cost_pre = mem_cost
        return size_ialp, size_feri, shellslice_rank
    else:
        raise IOError('No memory control scheme for process:"%s"'%instance)