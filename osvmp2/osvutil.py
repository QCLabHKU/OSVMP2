import time
import types
import psutil
import ctypes
import sys
import os
import warnings
import gc
import numpy as np
from numpy.linalg import svd, eigh, multi_dot
from pyscf import lib, gto
from pyscf.gto.moleintor import make_loc, make_cintopt
from osvmp2.__config__ import ngpu
from osvmp2.mpi_addons import get_shared, free_win, Acc_and_get_GA
ddot = np.dot
import h5py
import mpi4py
from mpi4py import MPI
#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()    # Size of communicator
irank = comm.Get_rank()    # Ranks in communicator
#inode = MPI.Get_processor_name()     # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//nrank_shm
inode = irank // nrank_shm

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)

if ngpu:
    #set up gpu
    import cupy
    import cupyx
    import cupyx.scipy.linalg
    
    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)

#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore")
if sys.version_info[0] >= 3:
    from functools import reduce

def set_omp(nthread):
    nthread = str(nthread)
    os.environ["OMP_NUM_THREADS"] = nthread
    os.environ["OPENBLAS_NUM_THREADS"] = nthread
    os.environ["MKL_NUM_THREADS"] = nthread
    os.environ["VECLIB_MAXIMUM_THREADS"] = nthread
    os.environ["NUMEXPR_NUM_THREADS"] = nthread

def print_align(msg_list, align='l', align_1=None, indent=0, log=None, printout=True, space=2):
    if align_1 is None:
        align_1 = align
    align_list = []
    for align_i in [align_1, align]:
        align_format = []
        for i in list(align_i):
            if i == 'l':
                align_format.append('<')
            elif i == 'c':
                align_format.append('^')
            elif i == 'r':
                align_format.append('>')
        align_list.append(align_format)
    len_col = []
    for col_i in zip(*msg_list):
        len_col.append(max([len(str(i)) for i in col_i])+space)
    msg = ''
    for idx, msg_i in enumerate(msg_list):
        if idx == 0:
            align_i = align_list[0]
        else:
            align_i = align_list[1]
        msg += ' ' * indent
        msg += ''.join([('{:%s%d} '%(ali, li)).format(mi) for ali, li, mi in zip(align_i, len_col, msg_i)])
        if idx != len(msg_list)-1:
            msg += '\n'
    if printout:
        if log is None:
            print(msg, flush=True)
        else:
            log.info(msg)
    return msg

def create_timer(use_gpu=ngpu):
    if use_gpu:
        return np.zeros(3, dtype=np.float32)
    else:
        return np.zeros(2, dtype=np.float32)

def get_current_time(use_gpu=ngpu):
    if sys.version_info < (3, 0):
        return time.clock(), time.time()
    elif use_gpu:
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()
        current_time = (time.process_time(), 
                        time.perf_counter(), 
                        start_event, end_event)
        start_event.record()
        return current_time
    else:
        return time.process_time(), time.perf_counter()

def get_elapsed_time(start_time):
    use_gpu = len(start_time) == 4
    end_time = get_current_time(use_gpu=False)
    elapsed_cpu_time = end_time[0] - start_time[0]
    elapsed_wall_time = end_time[1] - start_time[1]
    elapsed_time = [elapsed_cpu_time, elapsed_wall_time]
    if use_gpu:
        start_event, end_event = start_time[2:4]
        end_event.record()
        end_event.synchronize()
        elapsed_gpu_time = cupy.cuda.get_elapsed_time(start_event, end_event) / 1000
        elapsed_time.append(elapsed_gpu_time)
    
    return np.asarray(elapsed_time, dtype=np.float32)
    
def accumulate_time(timer, start_time):
    assert isinstance(timer, np.ndarray), f"Timer must be a NumPy array, got {type(timer)}"
    timer += get_elapsed_time(start_time)
    return timer

def record_elapsed_time(start_time):
    use_gpu = len(start_time) == 4
    end_time = get_current_time(use_gpu=False)
    elapsed_cpu_time = end_time[0] - start_time[0]
    elapsed_wall_time = end_time[1] - start_time[1]

    if use_gpu:
        start_event, end_event = start_time[2:4]
        end_event.record()
        return [elapsed_cpu_time, elapsed_wall_time, 
                start_event, end_event]
    else:
        return [elapsed_cpu_time, elapsed_wall_time]

def sum_elapsed_time(recorder):
    total_cpu_time = np.sum([t[0] for t in recorder])
    total_wall_time = np.sum([t[1] for t in recorder])
    if ngpu == 0:
        return np.asarray([total_cpu_time, total_wall_time])
    
    gpu_times = []
    for _, _, start_event, end_event in recorder:
        end_event.synchronize()
        elapsed_gpu_time = cupy.cuda.get_elapsed_time(start_event, end_event) / 1000
        gpu_times.append(elapsed_gpu_time)
    return np.asarray([total_cpu_time, total_wall_time, np.sum(gpu_times)])

def batch_accumulate_time(timer, recorder):
    assert isinstance(timer, np.ndarray), f"Timer must be a NumPy array, got {type(timer)}"
    timer += sum_elapsed_time(recorder)
    return timer

def get_max_rank_time_list(time_list):
    '''time_list = [[label, t], ...]'''
    tlist = get_max_rank_times([t for _, t in time_list])
    return [[l, t] for (l, _), t in zip(time_list, tlist)]

def get_max_rank_times(times_rank):
    times_rank = np.asarray(times_rank, dtype=np.float32)
    time_shape = times_rank.shape
    if len(time_shape) == 1:
        ntime = 1
        nttype = len(times_rank)
    else:
        ntime, nttype = time_shape
    win_times_ranks, times_ranks = get_shared((nrank, ntime, nttype), dtype=np.float32, set_zeros=True)
    times_ranks[irank] = times_rank
    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(times_ranks)
        comm.Barrier()
    max_times = np.max(times_ranks, axis=0).reshape(time_shape)
    comm.Barrier()
    win_times_ranks.Free()
    return max_times

    
def print_time(time_list, log=None, left_align=False):

    def get_columns(process, elapsed_time):
        #return ['CPU time for %s'%process, '%.2f sec,'%time_i[0], 'wall time', '%.2f sec'%time_i[1]]
        time_msg = [f"Time for {process}", " ", "CPU:", "%.2f s,"%elapsed_time[0], " ", "wall:", "%.2f s"%elapsed_time[1]]
        if use_gpu:
            time_msg[-1] += ","
            time_msg.extend([" ", "GPU:", "%.2f s"%elapsed_time[2]])
        return time_msg

    if (len(time_list)==2) and (type(time_list[0])==str):
        time_list = [time_list]

    use_gpu = (len(time_list[0][1]) == 3)

    table_data = []
    for process, elapsed_time in time_list:
        table_data.append(get_columns(process, elapsed_time))

    if left_align:
        indent = 0
    else:
        indent = 4
    
    align = "lrrrrrrrrrr" if use_gpu else "lrrrrrrr"
    print_align(table_data, align=align, indent=indent, log=log, space=0)

    return get_current_time()

def print_size(var, varname, log=None):
    msg = 'Size of %s is %.2f MB'%(varname, var.size*8*1e-6)
    if log is None:
        print(msg, flush=True)
    else:
        log.info(msg)

def mem_from_pid(pid):
    "Memory usgae with shared memory"
    mem_dic = psutil.Process(int(pid)).memory_info()
    return (mem_dic[0] - mem_dic[2] + mem_dic[2]/comm_shm.size)*1e-6

def print_test(a, varname=None, flush=True):
    a = a.ravel()
    msg = '%.8E  %.8E  %.8E'%(a.min(), a.max(), a.mean())
    if varname is not None:
        msg = '%s: '%varname + msg
    print(msg, flush=flush)

def mem_node(pid_list=None):
    if pid_list is None:
        return psutil.virtual_memory()[3]*1e-6 #- mol.mem_init
    else:
        mem_total = 0
        for pid_i in pid_list:
            mem_total += mem_from_pid(pid_i)
        return mem_total
def print_mem(rank=None, pid_list=None, log=None, left_align=False):
    gc.collect()
    mem_used = mem_node(pid_list)
    msg = ', used memory: %.2f MB'%mem_used#lib.current_memory()[0]
    if rank is not None:
        msg = rank + msg
    if left_align == False:
        msg = '    ' + msg
    if log is None:
        print(msg, flush=True)
    else:
        log.info(msg)
    
'''def get_mem_spare(mol, ratio=1, per_core=True):
    
    gc.collect()
    #mem_avail = (mol.max_memory - psutil.virtual_memory()[3]*1e-6)/comm_shm.size
    #mem_avail = (mol.max_memory/comm_shm.size - mem_from_pid(os.getpid()))
    win_mem, mem_list = get_shared(comm_shm.size)
    mem_list[irank_shm] = mem_from_pid(os.getpid())
    comm_shm.Barrier()
    mem_total = sum(mem_list)
    comm_shm.Barrier()
    free_win(win_mem)
    mem_avail = (mol.max_memory - mem_total)
    if per_core:
        mem_avail /= comm_shm.size
    return max(ratio*mem_avail, 1)'''

def ave_mem_spare(mol, ratio=1, per_core=True):
    
    mem_list = [mem_from_pid(pid) for pid in mol.pid_list]
    mem_total = sum(mem_list)
    mem_avail = (mol.max_memory - mem_total)

    if nnode > 1:
        win_mem, mem_node = get_shared(nnode, set_zeros=True, dtype=np.float32)
        if irank_shm == 0:
            mem_node[inode] = mem_avail
        Acc_and_get_GA(mem_node)
        comm.Barrier()
        mem_avail = np.mean(mem_node)
        comm.Barrier()
        free_win(win_mem)

    if per_core:
        mem_avail /= comm_shm.size
    return max(ratio*mem_avail, 1)

def get_mem_spare(mol, ratio=1, per_core=True):
    mem_list = [mem_from_pid(pid) for pid in mol.pid_list]
    mem_total = sum(mem_list)
    mem_avail = (mol.max_memory - mem_total)
    if per_core:
        mem_avail /= comm_shm.size
    return max(ratio*mem_avail, 1)


def get_buff_len(mol, size_sub, ratio, max_len, min_len=1, max_memory=None):
    if max_memory is None:
        max_memory = get_mem_spare(mol)
    if max_len is not None:
        return int(max(min(ratio*max_memory*1e6//(size_sub*8), max_len), min_len))

def getPairMat(ij, matrix, nocc):
    i, j = ij
    if i > j:
        ipair = j * nocc + i
    else:
        ipair = i * nocc + j
    mat = matrix[ipair]
    if i > j:
        mat = mat.T
    return mat

def generation_SuperMat(ijkl, matrix, blockdim, ndim): 
    i, j, k, l = ijkl
    SuperMat = np.empty((blockdim[i]+blockdim[j], blockdim[k]+blockdim[l]))
    SuperMat[:blockdim[i], :blockdim[k]] = getPairMat((i, k), matrix, ndim)
    SuperMat[blockdim[i]:, :blockdim[k]] = getPairMat((j, k), matrix, ndim) 
    SuperMat[:blockdim[i], blockdim[k]:] = getPairMat((i, l), matrix, ndim)
    SuperMat[blockdim[i]:, blockdim[k]:] = getPairMat((j, l), matrix, ndim)
    return SuperMat

def getMatFromNode(idx, matNode, addrNode, dim_list):
    midx0, midx1 = addrNode[idx]
    return matNode[midx0: midx1].reshape(dim_list[idx])

def getPairMatFromNode(ij, matNode, addrNode, dim_list, nocc):
    i, j = ij
    if i > j:
        return getMatFromNode(j*nocc+i, matNode, addrNode, dim_list).T
    else:
        return getMatFromNode(i*nocc+j, matNode, addrNode, dim_list)

def getSuperMatShared(ijkl, matrix_node, address_node, dim_list, nocc):  
    i, j, k, l = ijkl
    mat_ik = getPairMatFromNode((i, k), matrix_node, address_node, dim_list, nocc)
    mat_il = getPairMatFromNode((i, l), matrix_node, address_node, dim_list, nocc)
    mat_jk = getPairMatFromNode((j, k), matrix_node, address_node, dim_list, nocc)
    mat_jl = getPairMatFromNode((j, l), matrix_node, address_node, dim_list, nocc)
    dim_i, dim_k = mat_ik.shape
    dim_j, dim_l = mat_jl.shape
    rows = dim_i + dim_j
    cols = dim_k + dim_l
    SuperMat = np.empty((rows, cols))
    SuperMat[:dim_i, :dim_k] = mat_ik
    SuperMat[dim_i:, :dim_k] = mat_jk
    SuperMat[:dim_i, dim_k:] = mat_il
    SuperMat[dim_i:, dim_k:] = mat_jl
    return SuperMat


def getSuperMatCupy(ijkl, matrix_pair_gpu, nosv, nocc, buffer=None):  
    i, j, k, l = ijkl
    nosv_i, nosv_j, nosv_k, nosv_l = nosv[ijkl]
    
    if buffer is None:
        SuperMat = cupy.empty((nosv_i + nosv_j, nosv_k + nosv_l))
    else:
        nosv_ij = nosv_i + nosv_j
        nosv_kl = nosv_k + nosv_l
        SuperMat = buffer[:nosv_ij*nosv_kl].reshape(nosv_ij, nosv_kl)
    SuperMat[:nosv_i, :nosv_k] = matrix_pair_gpu[i*nocc+k]
    SuperMat[nosv_i:, :nosv_k] = matrix_pair_gpu[j*nocc+k]
    SuperMat[:nosv_i, nosv_k:] = matrix_pair_gpu[i*nocc+l]
    SuperMat[nosv_i:, nosv_k:] = matrix_pair_gpu[j*nocc+l]
    
    return SuperMat

def contigous_trans(a, order=None):
    if order is None:
        b = a.T
    else:
        b = a.transpose(order)
    #a = a.reshape(b.shape)
    #a[:] = b
    return np.ascontiguousarray(b)


def half_trans(mol, feri, mo_coeff, lmo_close, fit_close, slice_i, i, buf_feri=None, buf_moco=None, dot=np.dot, out=None):
    def get_ao_domains(be0, be1, lmo_close, i):
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
    al0, al1, be0, be1 = slice_i
    be_idx = [None] * (be1+1)
    for idx, be in enumerate(range(be0, be1+1)):
        be_idx[be] = idx
    cal_slice = get_ao_domains(be0, be1, lmo_close, i)
    if cal_slice is None or cal_slice == []:
        return None
    else:
        nao0 = al1 - al0
        nao1 = sum([(BE1-BE0) for BE0, BE1 in cal_slice])
        #naux = sum([(p1-p0) for p0, p1 in fit_close[i]])
        naux = feri.shape[-1]
        if buf_feri is None:
            feri_tmp = np.empty((nao1, nao0, naux))
        else:
            feri_tmp = buf_feri[:nao0*nao1*naux].reshape(nao1, nao0, naux)
        if buf_moco is None:
            moco_tmp = np.empty(nao1)
        else:
            moco_tmp = buf_moco[:nao1]
        
        idx_BE0 = 0
        for BE0, BE1 in cal_slice:
            idx_BE1 = idx_BE0 + (BE1-BE0)
            moco_tmp[idx_BE0:idx_BE1] = mo_coeff[BE0:BE1, i]
            idx_be0, idx_be1 = be_idx[BE0], be_idx[BE1]
            feri_tmp[idx_BE0:idx_BE1] = feri[idx_be0:idx_be1]
            '''idx_p0 = 0
            for p0, p1 in fit_close[i]:
                    idx_p1 = idx_p0 + (p1-p0)
                    feri_tmp[idx_BE0:idx_BE1, :, idx_p0:idx_p1] = feri[idx_be0:idx_be1, :, p0:p1]
                    idx_p0 = idx_p1'''
            idx_BE0 = idx_BE1

        try:
            if out is None:
                    return dot(moco_tmp.T, feri_tmp.reshape(nao1, -1)).reshape(nao0, naux)
            else:
                    return dot(moco_tmp.T, feri_tmp.reshape(nao1, -1), out=out).reshape(nao0, naux)
        except ValueError:
            print('DAMIT', cal_slice, flush=True)

def get_auxshell_slice(auxmol):
    aux_loc = make_loc(auxmol._bas, 'sph')
    shell_seg = []
    naux_seg = []
    for s0 in range(auxmol.nbas):
        s1 = s0 + 1
        shell_seg.append([s0, s1])
        naux_seg.append(aux_loc[s1]-aux_loc[s0])
    if len(naux_seg) < nrank:
        shell_slice = get_slice(rank_list=range(nrank), job_list=shell_seg)
        for idx, s_i in enumerate(shell_slice):
            if s_i is not None:
                shell_slice[idx] = sorted(list(set(reduce(lambda x, y :x+y, s_i))))
    else:
        shell_slice = OptPartition(nrank, shell_seg, naux_seg)[0]
        if len(shell_slice) < nrank:
            shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
            for rank_i, s_i in enumerate(shell_slice):
                if s_i is not None:
                    shell_slice[rank_i] = s_i[0]
    aux_slice = [None]*nrank
    aux_address = [None]*auxmol.nao_nr()
    for rank_i, shell_i in enumerate(shell_slice):
        if shell_i is not None:
            s0, s1 = shell_i[0], shell_i[-1]
            aux0, aux1 = aux_loc[s0], aux_loc[s1]
            aux_slice[rank_i] = []
            for idx, p in enumerate(range(aux0, aux1)):
                aux_slice[rank_i].append(p)
                aux_address[p] = [rank_i, [idx, idx+1]]
    return aux_slice, aux_address, shell_slice

def make_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError as e:
        pass

def flip_ij(i,j,mat,nosv):
    #if(i!=j):
    nosvI = nosv[i]
    nosvJ = nosv[j]
    result=np.empty_like(mat)
    result[nosvJ:,nosvJ:] = mat[:nosvI,:nosvI].T # 4 - 1.T
    result[:nosvJ,:nosvJ] = mat[nosvI:,nosvI:].T # 1 - 4.T
    result[:nosvJ,nosvJ:] = mat[:nosvI,nosvI:].T # 2 - 2.T
    result[nosvJ:,:nosvJ] = mat[nosvI:,:nosvI].T # 3 - 3.T
    #else:
    #    result = mat
    return result

def normalize_mo_orca(mol, mo_coeff):
    def fact2(k):
        """
        Compute double factorial: k!! = 1*3*5*....k
        """
        from operator import mul
        return reduce(mul, range(k, 0, -2), 1.0)

    def atom_list_converter(self):
        """
        in ORCA 's', 'p' orbitals don't require normalization.
        'g' orbitals need to be additionally scaled up by a factor of sqrt(3).
        https://orcaforum.cec.mpg.de/viewtopic.php?f=8&t=1484
        """
        ao_loc = make_loc(mol._bas, 'sph')
        for ib in range(mol.nbas):
            al0, al1 = ao_loc[ib], ao_loc[ib+1]
            l = mol.bas_angular(ib)
            
            for primitive in shell['DATA']:
                primitive[1] /= sqrt(fact2(2*l-1))
                if l == 4:
                    primitive[1] *= sqrt(3)

    def f_normalize(self, mo_coeff):
        """
        ORCA use slightly different sign conventions:
            F(+3)_ORCA = - F(+3)_MOLDEN
            F(-3)_ORCA = - F(-3)_MOLDEN
        """
        mo_coeff[5] *= -1
        mo_coeff[6] *= -1
        return super(Orca, self).f_normalize(mo_coeff)

    def g_normalize(self, mo_coeff):
        """
        ORCA use slightly different sign conventions:
            G(+3)_ORCA = - G(+3)_MOLDEN
            G(-3)_ORCA = - G(-3)_MOLDEN
            G(+4)_ORCA = - G(+4)_MOLDEN
            G(-4)_ORCA = - G(-4)_MOLDEN
        """
        mo_coeff[5] *= -1
        mo_coeff[6] *= -1
        mo_coeff[7] *= -1
        mo_coeff[8] *= -1
        return super(Orca, self).g_normalize(mo_coeff)

def get_coords_from_mol(mol, coord_only=False, info=""):
    atm_list = []
    for atm in range(mol.natm):
        atm_list.append(mol.atom_pure_symbol(atm))
    xyz_list = mol.atom_coords()*lib.param.BOHR
    coord_list = [[atm_list[atm], "%.9f"%x, "%.9f"%y, "%.9f"%z] \
                  for atm, (x, y, z) in enumerate(xyz_list)]
    coord_str = print_align(coord_list, align='lrrr', printout=False) 
    if coord_only:
        return coord_str
    else:
        coords_msg = "%d\n%s\n"%(len(atm_list), info)
        coords_msg += coord_str + "\n"
        return coords_msg

def get_ovlp(mol):
    '''Overlap matrix
    '''
    return mol.intor_symmetric('int1e_ovlp')

def str_letter(msg):
    return (''.join(x for x in msg if x.isalpha())).lower()


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


def get_omul_seg(self, slice_i, stype):
    #if stype == 'aux_mp2':
    if 'aux' in stype:
        omul_list = self.omul_aux
        dir_omul = self.dir_omul_aux
    else:
        omul_list = self.omul_mo
        dir_omul = self.dir_omul_mo
    omul_list = np.asarray(omul_list)
    rank_all = omul_list[slice_i]
    rank_list = [rank_all[0]]#list(set(rank_all.tolist()))
    seg_list = []
    seg_i = []
    for idx, rank_i in enumerate(rank_all):
        seg_i.append(slice_i[idx])
        if rank_i == rank_all[-1]:
            #rank_list.append(rank_i)
            seg_list.append(slice_i[idx:])
            break
        elif rank_all[idx+1] != rank_i:
            rank_list.append(rank_all[idx+1])
            seg_list.append(seg_i)
            seg_i = []
    file_list = ['%s/omul_%s_%d.tmp'%(dir_omul, stype, rank_i) for rank_i in rank_list]
    return file_list, seg_list

def getints3c_test(intor_name, atm, bas, env, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas)
        if 'ssc' in intor_name or 'spinor' in intor_name:
            bas = numpy.asarray(numpy.vstack((bas,bas)), dtype=numpy.int32)
            shls_slice = (0, nbas, 0, nbas, nbas, nbas*2)
            nbas = bas.shape[0]
    else:
        assert(shls_slice[1] <= nbas and
               shls_slice[3] <= nbas and
               shls_slice[5] <= nbas)

    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)
        if 'ssc' in intor_name:
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'cart')
        elif 'spinor' in intor_name:
            # The auxbasis for electron-2 is in real spherical representation
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'sph')

    naok = ao_loc[k1] - ao_loc[k0]

    if aosym in ('s1',):
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        shape = (naoi, naoj, naok, comp)
    else:
        aosym = 's2ij'
        nij = ao_loc[i1]*(ao_loc[i1]+1)//2 - ao_loc[i0]*(ao_loc[i0]+1)//2
        shape = (nij, naok, comp)
    order = 'F'
    #order = 'C'
    if 'spinor' in intor_name:
        mat = numpy.ndarray(shape, numpy.complex, out, order=order)
        drv = libcgto.GTOr3c_drv
        fill = getattr(libcgto, 'GTOr3c_fill_'+aosym)
    else:
        mat = numpy.ndarray(shape, numpy.double, out, order=order)
        drv = libcgto.GTOnr3c_drv
        fill = getattr(libcgto, 'GTOnr3c_fill_'+aosym)

    if mat.size > 0:
        # Generating opt for all indices leads to large overhead and poor OMP
        # speedup for solvent model and COSX functions. In these methods,
        # the third index of the three center integrals corresponds to a
        # large number of grids. Initializing the opt for the third index is
        # not necessary.
        if cintopt is None:
            if '3c2e' in intor_name:
                # int3c2e opt without the 3rd index.
                cintopt = make_cintopt(atm, bas[:max(i1, j1)], env, intor_name)
            else:
                cintopt = make_cintopt(atm, bas, env, intor_name)

        drv(getattr(libcgto, intor_name), fill,
            mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*6)(*(shls_slice[:6])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p))
    
    if comp == 1:
        mat = mat[:,:,:,0]
    else:
        mat = numpy.rollaxis(mat, -1, 0)
    return mat

def aux_e2(mol=None, auxmol=None, intor='int3c2e', aosym='s1', comp=None, 
              cintopt=None, shls_slice=None, hermi=0, out=None):
    
    ao_loc = None
    if '3c' in intor:
        intor = mol._add_suffix(intor)
        if shls_slice is None:
            shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                                auxmol._atm, auxmol._bas, auxmol._env)
        return getints3c(intor, atm, bas, env, shls_slice, comp, aosym, ao_loc, cintopt, out)
    elif '2c' in intor:
        if mol is None:
            mol = auxmol
        intor = mol._add_suffix(intor)
        return getints2c(intor, mol._atm, mol._bas, mol._env, shls_slice, comp, hermi, ao_loc, cintopt, out)

def aux_3c2e(auxmol, c_shls_slice, c_out):
    auxmol.c_drv(auxmol.c_intor_3c2e, auxmol.c_fill_3c2e,
                c_out, auxmol.c_comp_3c2e,
                c_shls_slice,
                auxmol.c_ao_loc_3c2e, auxmol.c_intopt,
                auxmol.c_atm, auxmol.c_natm,
                auxmol.c_bas, auxmol.c_nbas,
                auxmol.c_env)


def get_max_sum(array, n, K):
    def check(mid, array, n, K): 
        count = 0
        sum = 0
        # If individual element is greater maximum possible sum 
        for i in range(n): 
            if (array[i] > mid): 
                return False
            # Increase sum of current sub - array 
            sum += array[i] 
            # If the sum is greater than mid increase count 
            if (sum > mid): 
                count += 1
                sum = array[i] 
        count += 1
        if (count <= K): 
            return True
        return False
    start = 1
    end = 0
    #Initialise end
    for i in range(n): 
            end += array[i] 
    # Answer stores possible maximum sub array sum 
    answer = 0
    while (start <= end): 
        mid = (start + end) // 2
        # If mid is possible solution, put answer = mid; 
        if (check(mid, array, n, K)): 
            answer = mid 
            end = mid - 1
        else: 
            start = mid + 1
    return answer 

def OptPartition(n_rank, shell_seg, ao_pair_seg, match_num=True):
    def collect_slice(shell_seg, ao_pair_seg, max_sum):
        shell_i = []
        len_si = 0
        shell_slice = []
        len_list = []
        for idx, len_i in enumerate(ao_pair_seg):
            shell_i.extend(shell_seg[idx])
            len_si += len_i
            idx_next = idx+1
            if idx_next > len(ao_pair_seg)-1:
                idx_next = idx
            if (len_si + ao_pair_seg[idx_next] > max_sum) or idx==(len(ao_pair_seg)-1):
                shell_slice.append(sorted(list(set(shell_i))))
                shell_i = []
                len_list.append(len_si)
                len_si = 0
        return shell_slice, len_list
    
    #max_sum = get_max_sum(ao_pair_seg, len(ao_pair_seg), n_rank)
    max_sum = sum(ao_pair_seg)//n_rank + 10
    shell_slice, len_list = collect_slice(shell_seg, ao_pair_seg, max_sum)
    len_slice = len(len_list)
    shell_pre = shell_slice
    len_pre = len_list
    step = 0
    var_i = 10
    if (len_slice != n_rank): #and match_num:
        while len_slice != n_rank:
            shell_slice, len_list = collect_slice(shell_seg, ao_pair_seg, max_sum)
            len_slice = len(len_list)
            if len_slice < n_rank:
                    if (step > 20) and (len(len_pre)>n_rank):
                        break
                    max_sum -= var_i
                    shell_pre = shell_slice
                    len_pre = len_list
            elif len_slice > n_rank:
                    max_sum += var_i
                    shell_pre = shell_slice
                    len_pre = len_list
            step += 1
            var_i = var_i//2
            if var_i == 0:
                    var_i = 1
    return shell_slice, len_list

def check_read(fil_name, dat_name):
    with h5py.File(fil_name, 'r') as f:
        if dat_name in f.keys():
            return True
        else:
            return False

def shell_chk(fil_name='sparse.chk', dat_name='shell_slice_hf', mol=None, shell_slice=None, op='w'):
    if op == 'r':
        from osvmp2.int_prescreen import alloc_shell
        with h5py.File(fil_name, op) as f:
            shell_flat = np.asarray(f[dat_name])
        ao_loc = make_loc(mol._bas, 'sph')
        naop_shell = []
        for a0, a1, b0, b1 in shell_flat:
            naop_shell.append((ao_loc[a1]-ao_loc[a0])*(ao_loc[b1]-ao_loc[b0]))
        return alloc_shell(shell_flat, naop_shell)
    else:
        shell_flat = []
        for si in shell_slice:
            if si is not None:
                    shell_flat.extend(si)
        shell_flat = np.asarray(shell_flat)
        with h5py.File(fil_name, op) as f:
            f.create_dataset(dat_name, shape=(len(shell_flat), 4), dtype=np.int64)
            f[dat_name].write_direct(shell_flat)



def slice2seg(mol, shell_slice, max_nao=None):
    shell_check = [False]*mol.nbas
    shell_seg = []
    seg_i = []
    for idx, (a0, a1, b0, b1) in enumerate(shell_slice):
        if shell_check[a0]:
            seg_i[-1].append([b0, b1])
        else:
            if idx != 0:
                shell_seg.append(seg_i)
            seg_i = [a0, a1, [[b0, b1]]]
            shell_check[a0] = True
        if idx == (len(shell_slice)-1):
            shell_seg.append(seg_i)
    if max_nao is None:
        return shell_seg
    else:
        ao_loc = make_loc(mol._bas, 'sph')
        SHELL_SEG = []
        SEG_i = []
        nao_i = 0
        for idx, seg_i in enumerate(shell_seg):
            al0, al1 = [ao_loc[s] for s in seg_i[:2]]
            if (nao_i + al1 - al0) > max_nao:
                SHELL_SEG.append(SEG_i)
                SEG_i = [seg_i]
                nao_i = al1 - al0
            else:
                SEG_i.append(seg_i)
                nao_i += al1 - al0
            if idx == (len(shell_seg)-1):
                SHELL_SEG.append(SEG_i)
        return SHELL_SEG
        
def read_file(f_name, obj_name, idx0=None, idx1=None, buffer=None):
    var = None
    read_sucess = False; count = 0
    while (read_sucess == False) and (count<10):
        try:
            with h5py.File(f_name, 'r') as f:
                if idx0 is None:
                    if buffer is None:
                        var = np.asarray(f[obj_name])
                    else:
                        f[obj_name].read_direct(buffer)
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
            
        
def get_ncols_for_memory(max_memory_mb, num_rows, max_num_cols, min_num_cols=1, dtype="f8"):
    """
    Calculates the allowable number of columns for a 2D matrix based on available memory.

    Parameters:
        max_memory_mb (float): Maximum memory available in megabytes.
        num_rows (int): Number of rows in the matrix.
        max_num_cols (int): Upper limit for the number of columns.
        min_num_cols (int, optional): Lower limit for the number of columns. Defaults to 1.

    Returns:
        int: Calculated number of columns.
    """
    # Calculate max elements based on memory (assuming float64, 8 bytes per element)
    max_elements = max_memory_mb * 1e6 / np.dtype(dtype).itemsize
    # Max columns based on memory and rows
    mem_limit_cols = int(max_elements // num_rows)
    # Constrain result within bounds
    return max(min_num_cols, min(max_num_cols, mem_limit_cols))

def get_mo_batches(pair_slice, max_nocc, nocc):
    # Precompute MO frequencies to prioritize pairs with high-frequency MOs
    mo_counts = {}
    for pair in pair_slice:
        i = pair // nocc
        j = pair % nocc
        mo_counts[i] = mo_counts.get(i, 0) + 1
        mo_counts[j] = mo_counts.get(j, 0) + 1
    
    # Sort pairs by the sum of their MOs' frequencies (descending) to process high-frequency pairs first
    pair_slice_sorted = sorted(pair_slice, key=lambda p: (-mo_counts.get(p//nocc, 0) - mo_counts.get(p%nocc, 0), p))
    
    batches = []
    for pair in pair_slice_sorted:
        i = pair // nocc
        j = pair % nocc
        added = False
        for batch in batches:
            mo_batch = batch['mo']
            if i in mo_batch and j in mo_batch:
                batch['pairs'].append(pair)
                added = True
                break
            else:
                needed = {i, j} - mo_batch
                if len(mo_batch) + len(needed) <= max_nocc:
                    mo_batch.update(needed)
                    batch['pairs'].append(pair)
                    added = True
                    break
        if not added:
            new_mo = {i, j}
            batches.append({'mo': new_mo, 'pairs': [pair]})
    
    # Convert the MO sets to sorted lists and pairs to numpy arrays for consistency
    processed_batches = []
    for batch in batches:
        mo_list = np.sort(list(batch['mo']))
        pairs_array = np.array(batch['pairs'])
        processed_batches.append((mo_list, pairs_array))
    
    return processed_batches

def list_to_ranges(arr):
    if len(arr) == 0:
        return []
    arr = np.asarray(arr)
    # Find break points where difference > 1
    breaks = np.where(np.diff(arr) > 1)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(arr)]])

    ranges = np.column_stack([arr[starts], arr[ends - 1] + 1])
    return ranges

def get_pair_batches(pair_slice, max_npair, nocc):
    # Precompute MO frequencies to prioritize pairs with high-frequency MOs
    mo_counts = {}
    for pair in pair_slice:
        i = pair // nocc
        j = pair % nocc
        mo_counts[i] = mo_counts.get(i, 0) + 1
        mo_counts[j] = mo_counts.get(j, 0) + 1
    
    # Sort pairs by the sum of their MOs' frequencies (descending) to process high-frequency pairs first
    pair_slice_sorted = sorted(pair_slice, key=lambda p: (-mo_counts.get(p//nocc, 0) - mo_counts.get(p%nocc, 0), p))
    pair_slice_sorted = np.asarray(pair_slice_sorted, dtype=int)
    batches = []
    for pidx0 in np.arange(len(pair_slice_sorted), step=max_npair):
        pidx1 = pidx0 + max_npair
        batches.append(pair_slice_sorted[pidx0: pidx1])
    
    return batches

import numpy as np

def get_pairs_with_diag(pairs, nocc, sort_pairs=True):

    # Split pairs into i and j orbital indices
    ilist, jlist = np.divmod(pairs, nocc)
    
    # Identify unique orbitals from original pairs
    unique_mos = np.union1d(ilist, jlist)
    
    # Generate diagonal indices for unique orbitals
    diag_pairs = unique_mos * (nocc + 1)
    
    # Filter out diagonal elements from original pairs
    offdiag_pairs = pairs[ilist != jlist]
    
    # Combine off-diagonal pairs with new diagonal elements
    extended_pairs = np.concatenate([offdiag_pairs, diag_pairs])

    if sort_pairs:
        extended_pairs.sort()

    return extended_pairs


def get_extended_pairs(pairlist_close, nocc, klist_pairs, pairlist_remote=[], mos_remote=None):

    
    '''for i in mos_close:
        js = jlist_close_full[i]
        extended_pairs.extend(np.minimum(i, js) * nocc + np.maximum(i, js))'''
    extended_pairs = []
    for ipair in pairlist_close:
        i = ipair // nocc
        j = ipair % nocc
        ks = klist_pairs[ipair]
        extended_pairs.append(np.minimum(i, ks) * nocc + np.maximum(i, ks))
        extended_pairs.append(np.minimum(j, ks) * nocc + np.maximum(j, ks))

    if len(pairlist_remote) > 0:
        if mos_remote is None:
            ilist_remote, jlist_remote = np.divmod(pairlist_remote, nocc)
            mos_remote = np.unique(np.concatenate((ilist_remote, jlist_remote)))
        extended_pairs.append(mos_remote * (nocc + 1))
        extended_pairs.append(pairlist_remote)

    return np.unique(np.concatenate(extended_pairs))

def merge_intervals(intervals):
    if len(intervals) == 0:
        return []
    merged = [np.copy(intervals[0])]
    for current in intervals[1:]:
        last = merged[-1]
        current_idx0, current_idx1 = current
        if current_idx0 == last[1]:
            # Merge intervals
            merged[-1][1] = current_idx1
        else:
            merged.append([current_idx0, current_idx1])
    return merged

def node_to_local(pairlist, node_mat, local_mat, node_address):

    node_offsets = np.asarray([node_address[ipair] for ipair in pairlist])
    node_offsets = merge_intervals(node_offsets)

    local_idx0 = 0
    for node_idx0, node_idx1 in node_offsets:
        local_idx1 = local_idx0 + (node_idx1 - node_idx0)
        local_mat[local_idx0: local_idx1] = node_mat[node_idx0: node_idx1]
        local_idx0 = local_idx1

def cum_offset(sizes, return_2d=False):
    sizes = np.asarray(sizes)
    offsets = np.empty(len(sizes) + 1, dtype=sizes.dtype)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])

    if return_2d:
        offsets = np.column_stack((offsets[:-1], offsets[1:]))
    return offsets

def uniform_cum_offset(size, nelement):
    return np.arange(0, (nelement + 1) * size, size, dtype=np.int64)



libcgto = lib.load_library('libcgto')
def get_c_mol(mol, auxmol):
    '''
    intor = mol._add_suffix(intor)
    if shls_slice == None:
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                            auxmol._atm, auxmol._bas, auxmol._env)
    '''
    

    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    atm = np.asarray(atm, dtype=np.int32, order='C')
    bas = np.asarray(bas, dtype=np.int32, order='C')
    env = np.asarray(env, dtype=np.double, order='C')
    intor_name_3c2e = "int3c2e_sph"
    ao_loc_3c2e = make_loc(bas, intor_name_3c2e)

    auxmol.c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    auxmol.c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    auxmol.c_env = env.ctypes.data_as(ctypes.c_void_p)
    auxmol.c_ao_loc_3c2e = ao_loc_3c2e.ctypes.data_as(ctypes.c_void_p)
    auxmol.c_natm = ctypes.c_int(atm.shape[0])
    auxmol.c_nbas = ctypes.c_int(bas.shape[0])
    auxmol.c_comp_3c2e = ctypes.c_int(1)
    
    auxmol.c_drv = libcgto.GTOnr3c_drv
    auxmol.c_intor_3c2e = getattr(libcgto, intor_name_3c2e)
    auxmol.c_fill_3c2e = getattr(libcgto, 'GTOnr3c_fill_s1')
    auxmol.c_intopt = make_cintopt(atm, bas[:mol.nbas], env, intor_name_3c2e)

def create_h5py_datasetx(h5_file, dataset_name, shape, dtype=np.float64, chunks=None, fill_value=None):

    # Get datatype
    dtype = np.dtype(dtype)
    h5_dtype = h5py.h5t.py_create(dtype)
    
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    
    if chunks is not None:
        dcpl.set_chunk(chunks)

    if fill_value is not None:
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_ALLOC)
        fill_val = np.array(fill_value, dtype=dtype)
        dcpl.set_fill_value(fill_val)
    else:
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)

    dataspace = h5py.h5s.create_simple(shape, shape)

    datasetid = h5py.h5d.create(h5_file.id, dataset_name, h5_dtype, dataspace, dcpl)
    return h5py.Dataset(datasetid)

def create_h5py_dataset(h5_file, dataset_name, shape, dtype=np.float64, chunks=None, fill_value=None):
    """
    Create an HDF5 dataset using h5py's high-level API with custom dcpl to set FILL_TIME_NEVER.
    
    Args:
        h5_file: h5py.File object
        dataset_name: Name of the dataset (str)
        shape: Tuple specifying dataset shape
        dtype: Data type (default: np.float64)
        chunks: Chunk shape or True for auto-chunking (default: None)
        fill_value: Fill value for the dataset (default: None)
    
    Returns:
        h5py.Dataset object
    
    Raises:
        ValueError: If dataset_name exists, shape is invalid, or chunks are invalid
    """
    # Validate inputs
    if dataset_name in h5_file:
        raise ValueError(f"Dataset '{dataset_name}' already exists in file")
    if not all(isinstance(s, int) and s > 0 for s in shape):
        raise ValueError("Shape must be a tuple of positive integers")
    
    # Create dataset creation property list
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    
    # Set chunking if specified
    if chunks is not None:
        if chunks is True:
            chunks = h5py.filters.guess_chunk(shape, np.dtype(dtype).itemsize)
        elif not isinstance(chunks, tuple) or len(chunks) != len(shape):
            raise ValueError("Chunk shape must be a tuple matching dataset dimensions")
        elif any(c > s for c, s in zip(chunks, shape)):
            raise ValueError("Chunk size cannot exceed dataset size")
        dcpl.set_chunk(chunks)
    
    # Set fill time to NEVER, even if fill_value is provided
    dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    if fill_value is not None:
        fill_val = np.array(fill_value, dtype=dtype)
        dcpl.set_fill_value(fill_val)
    
    # Create dataset using high-level API with custom dcpl
    dataset = h5_file.create_dataset(
        dataset_name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        fillvalue=fill_value,
        dcpl=dcpl,
        track_times=False  # Reduce metadata overhead
    )
    return dataset

def transpose_array(array, out, axes):
    array_trans_view = array.transpose(axes)
    array_trans = out[:array.size].reshape(array_trans_view.shape)
    array_trans[:] = array_trans_view
    return array_trans