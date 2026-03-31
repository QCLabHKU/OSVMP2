import os
import itertools
import numbers
import ctypes
import numpy as np
from numpy.ctypeslib import as_ctypes
#from osvmp2.osvutil import *
import mpi4py
mpi4py.rc.thread_level = 'single'
from mpi4py.util import dtlib
from mpi4py import MPI

from osvmp2.__config__ import ngpu

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
    import cupy
    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    irank_gpu = irank % nrank_per_gpu
    cupy.cuda.runtime.setDevice(igpu_shm)
    comm_gpu = comm_shm.Split(color=igpu_shm, key=irank_shm)

MPI_CHUNK_SIZE = 2147483647
WINDOWS = []
NUM_WINDOWS = 0

def split_number(num, n):
    base_value = num // n
    remainder = num % n
    parts = np.full(n, base_value, dtype=int)
    parts[:remainder] += 1
    return parts


def equisum_partition(a, n):
    """
    Divide a list into n contiguous parts with sums as equal as possible.
    
    Args:
        a (list): List of numbers to divide.
        n (int): Number of parts to divide into.
    
    Returns:
        list: List of n sublists with sums as close as possible.
    """
    if n > len(a):
        raise ValueError("n cannot be larger than the list length")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Compute cumulative sum array
    #from itertools import accumulate
    #cumsum = list(accumulate(a, initial=0))  # [0, a[0], a[0]+a[1], ...]
    a_indices = np.arange(len(a), dtype=np.int32)
    cumsum = np.empty(len(a)+1, dtype=int)
    cumsum[0] = 0
    np.cumsum(a, out=cumsum[1:])
    total_sum = cumsum[-1]
    target = total_sum / n  # Ideal sum per part
    
    result = []
    start = 0
    
    # Find n-1 split points
    for _ in range(1, n):
        min_diff = float('inf')
        best_i = start + 1
        
        # Find the index that makes the sum closest to target
        for i in range(start + 1, len(a) + 1):
            current_sum = cumsum[i] - cumsum[start]
            diff = abs(current_sum - target)
            if diff < min_diff:
                min_diff = diff
                best_i = i
        
        # Add the part from start to best_i-1
        #result.append(a[start:best_i])
        result.append(a_indices[start:best_i])
        start = best_i
    
    # Add the remaining part
    #result.append(a[start:])
    result.append(a_indices[start:])
    return result

def get_slice(rank_list, job_size=None, job_list=None, weight_list=None, sort=False, 
              in_rank_sort=False):
    # Preprocess job inputs
    if job_list is None:
        job_list = np.arange(job_size)
    else:
        job_size = len(job_list)
    rank_list = np.asarray(rank_list)
    n_rank = len(rank_list)
    
    # For cross-node jobs
    '''if nrank == 1:
        return [job_list]'''
    if (sort and nnode > 1 and 
        len(rank_list) > nrank_shm):  
        rank_list = rank_list.reshape(nnode, -1).T.ravel()
    
    # Initialize output structure
    job_slice = [None for _ in range(n_rank)]
    
    if n_rank >= job_size:
        # Direct assignment with vectorized indexing
        for rank_i, job_i in zip(rank_list, job_list.reshape(-1, 1)):
            job_slice[rank_i] = job_i
    elif weight_list is None:
        split_indices = np.linspace(0, job_size, n_rank+1, dtype=int)
        for rank_idx, rank_i in enumerate(rank_list):
            jidx0 = split_indices[rank_idx]
            jidx1 = split_indices[rank_idx+1]
            job_slice[rank_i] = job_list[jidx0:jidx1]
    else:
        if sort:
            sorted_indices = np.argsort(-np.asarray(weight_list))
            jobs_ranks = {}
            weights_ranks = {}
            for rank_i in rank_list:
                jobs_ranks[rank_i] = []
                weights_ranks[rank_i] = []
            rank_loads = np.zeros(n_rank, dtype=int)
            for job_idx in sorted_indices:
                # Find rank with minimum current load
                lightest_rank = int(np.argmin(rank_loads))
                jobs_ranks[lightest_rank].append(int(job_idx))
                weights_ranks[lightest_rank].append(weight_list[job_idx])
                rank_loads[lightest_rank] += weight_list[job_idx]
            for rank_i in rank_list:
                if len(jobs_ranks[rank_i]) > 0:
                    if in_rank_sort:
                        sort_ids = np.argsort(-np.asarray(weights_ranks[rank_i]))
                        jids_rank = np.asarray(jobs_ranks[rank_i])[sort_ids]
                    else:
                        jids_rank = jobs_ranks[rank_i]
                    job_slice[rank_i] = job_list[jids_rank]
        else:
            job_ids_ranks = equisum_partition(weight_list, n_rank)
            idx0 = 0
            for rank_i, ids in enumerate(job_ids_ranks):
                idx1 = idx0 + len(ids)
                if in_rank_sort:
                    weight_slice = weight_list[idx0:idx1]
                    sort_ids = np.argsort(-np.asarray(weight_slice))
                    job_slice[rank_i] = job_list[idx0:idx1][sort_ids]
                else:
                    job_slice[rank_i] = job_list[idx0:idx1]
                idx0 = idx1

    return job_slice

def get_shared(shape, dtype='f8', set_zeros=False, rank_shm_buf=0, 
               comm_shared=comm_shm, use_pin=False):
    irank_shared = comm_shared.rank

    itemsize = np.dtype(dtype).itemsize
    
    if isinstance(shape, numbers.Number):
        shape = (shape,)
    buff_size = np.prod(shape)

    nbytes = buff_size * itemsize if irank_shared == rank_shm_buf else 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm_shared)
    buf, itemsize = win.Shared_query(rank_shm_buf) #MPI.PROC_NULL)
    #buf = np.array(buf, dtype='B', copy=False)
    shared_array = np.frombuffer(buf, dtype=dtype) #.reshape(shape)

    if set_zeros:
        nrank_shared = comm_shared.size
        irank_shared = irank_shared % nrank_shared

        total_bytes = buff_size * itemsize

        # Split buffer into byte-aligned chunks
        chunk_size, remainder = divmod(total_bytes, nrank_shared)
        
        # Calculate start/end offsets
        start = irank_shared * chunk_size + min(irank_shared, remainder)
        end = start + chunk_size + (1 if irank_shared < remainder else 0)
        
        # Parallel memset using ctypes
        if end > start:
            ctypes_buffer = as_ctypes(np.frombuffer(buf, dtype=np.uint8, count=nbytes))
            chunk_ptr = ctypes.byref(ctypes_buffer, start)
            ctypes.memset(chunk_ptr, 0, end - start)

        '''if irank_shared == 0:
            #shared_obj[:] = 0
            ctypes.memset(buf.ctypes.data, 0, buf.nbytes)'''
        comm_shared.Barrier()

    #shared_array = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    shared_array = shared_array.reshape(shape)

    if use_pin and irank_shared == rank_shm_buf:
        register_pinned_memory(shared_array)

    global NUM_WINDOWS
    NUM_WINDOWS += 1
    WINDOWS.append(win)
    return win, shared_array

def register_pinned_memory(host_array):
    assert ngpu > 0, "Must have GPUs to use pinned memory"
    nbytes = host_array.nbytes
    ptr = host_array.ctypes.data
    cupy.cuda.runtime.hostRegister(ptr, nbytes, 1)
    return ptr

def unregister_pinned_memory(host_array):
    ptr = host_array.ctypes.data
    cupy.cuda.runtime.hostUnregister(ptr)

def batch_get_shared_cupy(shapes, dtype=np.float64, set_zeros=False, numpy_arrays=None):
    total_size = 0
    formated_shapes = []
    for shape in shapes:
        if isinstance(shape, numbers.Number):
            total_size += shape
            shape = (shape,)
        else:
            total_size += np.prod(shape, dtype=int)
            
        formated_shapes.append(shape)

    all_shm_array, shm_array_ptr = get_shared_cupy(total_size, dtype=dtype, set_zeros=set_zeros)

    shm_arrays = []
    idx0 = 0
    for shape in formated_shapes:
        idx1 = idx0 + np.prod(shape)
        shm_arrays.append(all_shm_array[idx0:idx1].reshape(shape))
        idx0 = idx1

    if irank_gpu == 0:
        if numpy_arrays is not None:
            for cp_array, np_array in zip(shm_arrays, numpy_arrays):
                cp_array.set(np_array)
            cupy.cuda.Device().synchronize()
    comm_gpu.Barrier()

    return shm_arrays, shm_array_ptr

def get_shared_cupy(shape, dtype=np.float64, set_zeros=False, numpy_array=None):
    if isinstance(shape, numbers.Number):
        shape = (shape,)
    array_size = np.prod(shape) * np.dtype(dtype).itemsize

    if irank_gpu == 0:
        shared_cupy_array = cupy.empty(shape, dtype=dtype)
        ipc_handle = cupy.cuda.runtime.ipcGetMemHandle(shared_cupy_array.data.ptr)

        if numpy_array is not None:
            if numpy_array.shape != shape:
                raise ValueError("numpy_array shape does not match")
            if numpy_array.dtype != dtype:
                raise ValueError("numpy_array dtype does not match")
            shared_cupy_array.set(numpy_array)
        elif set_zeros:
            shared_cupy_array.fill(0)

        cupy.cuda.Device().synchronize()
    else:
        ipc_handle = None
    
    ipc_handle = comm_gpu.bcast(ipc_handle, root=0)
    
    # Non-root processes read the IPC handle and create the array
    if irank_gpu != 0:
        ipc_memptr = cupy.cuda.runtime.ipcOpenMemHandle(ipc_handle)  # Returns BaseMemory
        mem = cupy.cuda.memory.UnownedMemory(ipc_memptr, array_size, owner=None, device_id=igpu_shm)
        memptr = cupy.cuda.MemoryPointer(mem, 0)  # Create MemoryPointer
        shared_cupy_array = cupy.ndarray(shape=shape, dtype=dtype, memptr=memptr)

        cupy.cuda.Device().synchronize()
    else:
        ipc_memptr = None
 
    return shared_cupy_array, ipc_memptr


def close_ipc_handle(ipc_memptr):
    if irank_gpu != 0:
        cupy.cuda.runtime.ipcCloseMemHandle(ipc_memptr)

def copy_shm_array(array_node, array_new=None):
    assert array_node.flags['C_CONTIGUOUS']
    
    total_size = array_node.size

    if array_new is None:
        win_array_new, array_new = get_shared(total_size)

    idx_slice = get_slice(range(nrank_shm), job_size=total_size)[irank_shm]

    if idx_slice is not None:
        idx0, idx1 = idx_slice[0], idx_slice[-1]+1
        array_flatten = array_node.ravel()
        array_new[idx0:idx1] = array_flatten[idx0:idx1]
    comm_shm.Barrier()

    return win_array_new, array_new.reshape(array_node.shape)


def create_win(array, comm=comm):
    win = MPI.Win.Create(array, comm=comm)
    global NUM_WINDOWS
    NUM_WINDOWS += 1
    WINDOWS.append(win)
    return win

def free_win(win):
    win.Free()
    global NUM_WINDOWS
    NUM_WINDOWS -= 1

def fence_and_free(win):
    win.Fence()
    free_win(win)

def free_all_win():
    global NUM_WINDOWS
    global WINDOWS
    for win in WINDOWS:
        try:
            win.Free()
            NUM_WINDOWS -= 1
        except mpi4py.MPI.Exception:
            pass
    WINDOWS = []

def get_win_col(array):
    if irank_shm == 0:
        win_col = create_win(array, comm=comm)
    else:
        win_col = create_win(None, comm=comm)
    win_col.Fence()
    return win_col


def batch_acc_GA(array, win, target_rank):
    #win.Accumulate(array, target_rank=target_rank, op=MPI.SUM)
    array_size = array.size
    mpi_dtype = dtlib.from_numpy_dtype(array.dtype)
    itemsize = np.dtype(array.dtype).itemsize
    for idx0 in np.arange(array_size, step=MPI_CHUNK_SIZE):
        idx1 = min(idx0+MPI_CHUNK_SIZE, array_size)
        win.Accumulate(array[idx0:idx1], 
                       target_rank=target_rank, 
                       target=(idx0*itemsize, idx1-idx0, mpi_dtype),
                       op=MPI.SUM)

def Accumulate_GA(win=None, array=None, target_rank=0, cross_node=True):
    def acc_ga():
        if array is None: return None
        if cross_node:
            if (irank_shm == 0) and (irank != 0):
                win.Lock(target_rank)
                if array.size > MPI_CHUNK_SIZE:
                    batch_acc_GA(array, win, target_rank)
                else:
                    win.Accumulate(array, target_rank=target_rank, op=MPI.SUM)
                win.Unlock(target_rank)
        elif irank != target_rank:
            win.Lock(target_rank)
            if array.size > MPI_CHUNK_SIZE:
                    batch_acc_GA(array, win, target_rank)
            else:
                win.Accumulate(array, target_rank=target_rank, op=MPI.SUM)
            win.Unlock(target_rank)
    if nnode == 1:
        return None
    if win is None:
        win = get_win_col(array)
        acc_ga()
        fence_and_free(win)
    else:
        acc_ga()
        

def Accumulate_GA_shm(win, array_node, array):
    win.Lock(0)
    array_node += array
    win.Unlock(0)


def batch_get_GA(array, win, target_rank, target=None):
    #win.Accumulate(array, target_rank=target_rank, op=MPI.SUM)
    if target is None:
        offset = 0
    else:
        offset, _, _ = target
    array_size = array.size
    mpi_dtype = dtlib.from_numpy_dtype(array.dtype)
    for idx0 in np.arange(array_size, step=MPI_CHUNK_SIZE):
        idx1 = min(idx0+MPI_CHUNK_SIZE, array_size)
        win.Get(array[idx0:idx1], 
                target_rank=target_rank, 
                target=(idx0+offset, idx1-idx0, mpi_dtype))

def Get_GA(win, array, target_rank=0, target=None):
    win.Lock(target_rank, lock_type=MPI.LOCK_SHARED)
    if array.size > MPI_CHUNK_SIZE:
        batch_get_GA(array, win, target_rank, target)
    else:
        win.Get(array, target_rank=target_rank, target=target)
    win.Unlock(target_rank)


def Acc_and_get_GA(array):
    if nnode == 1:
        return None
    array_size = array.size
    if array.size > MPI_CHUNK_SIZE:
        wins = []
        array_segs = []
        for idx0 in np.arange(array_size, step=MPI_CHUNK_SIZE):
            idx1 = min(idx0+MPI_CHUNK_SIZE, array_size)
            array_seg = array[idx0:idx1]
            array_segs.append(array_seg)
            wins.append(get_win_col(array_seg))
        nbatch = len(wins)
        if irank_shm == 0 and irank != 0:
            for bidx in range(nbatch):
                Accumulate_GA(wins[bidx], array_segs[bidx], target_rank=0)
        for win in wins:
            win.Fence()
        if irank_shm == 0 and irank != 0:
            for bidx in range(nbatch):
                Get_GA(wins[bidx], array_segs[bidx], target_rank=0)
        for win in wins:
            fence_and_free(win)
    else:
        win_col = get_win_col(array)
        if irank_shm == 0 and irank != 0:
            Accumulate_GA(win_col, array, target_rank=0)
        win_col.Fence()
        if irank_shm == 0 and irank != 0:
            Get_GA(win_col, array, target_rank=0)
        fence_and_free(win_col)

def get_node_offsets(job_slices, job_offsets):
    node_offsets = [None for node_i in range(nnode)]
    for node_i in range(nnode):
        rank_start = node_i * nrank_shm
        rank_end = rank_start + nrank_shm - 1
        if job_slices[rank_start] is None:
            continue
        for rank_last in np.arange(rank_end, rank_start-1, step=-1):
            if job_slices[rank_last] is not None:
                
                job_start = job_offsets[job_slices[rank_start][0]][0]
                job_end = job_offsets[job_slices[rank_last][-1]][1]
                node_offsets[node_i] = [job_start, job_end]
                break
    return node_offsets

def combine_job_slices(slices_i, slices_j, sort=False):
    assert len(slices_i) == len(slices_j)
    combined_slices = [None] * len(slices_i)
    for rank_i, (slice_i, slice_j) in enumerate(zip(slices_i, slices_j)):
        if slice_i is None:
            combined_slices[rank_i] = slice_j
        elif slice_j is None:
            combined_slices[rank_i] = slice_i
        else:
            combined_slices[rank_i] = np.append(slice_i, slice_j)
        
        if sort and combined_slices[rank_i] is not None:
            combined_slices[rank_i].sort()
    return combined_slices

def Get_from_other_nodes_GA(array, node_offsets):
    
    if irank_shm == 0:
        if node_offsets[inode] is None:
            array_node = None
        else:
            idx0, idx1 = node_offsets[inode]
            array_node = array[idx0:idx1]
    else:
        array_node = None
    
    win_array = MPI.Win.Create(array_node, comm=comm)
    win_array.Fence()

    if irank_shm == 0:
        for node_now, offsets in enumerate(node_offsets):
            if node_now != inode and offsets is not None:
                idx0, idx1 = offsets
                target_rank = node_now * nrank_shm
                #print(irank, node_now, idx0, idx1, flush=True)
                win_array.Get(array[idx0:idx1], target_rank=target_rank)
        
    win_array.Fence()
    win_array.Free()

def bcast_GA(array, win_col=None):
    if nnode == 1:
        return None
    if win_col == None:
        if irank_shm == 0:
            win_col = create_win(array, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        win_col.Fence()
        new_win = True
    else:
        new_win = False
    if irank_shm == 0 and irank != 0:
        win_col.Lock(0, lock_type=MPI.LOCK_SHARED)
        win_col.Get(array, target_rank=0)
        win_col.Unlock(0)
    if new_win:
        win_col.Fence()
        free_win(win_col)
    return array

def collect_GA(array, win_col=None):
    if nnode == 1:
        return None
    if win_col == None:
        if irank_shm == 0:
            win_col = create_win(array, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        win_col.Fence()
        new_win = True
    else:
        new_win = False
    if irank_shm == 0 and irank != 0:
        win_col.Lock(0)
        win_col.Accumulate(array, target_rank=0, op=MPI.SUM)
        win_col.Unlock(0)
    if new_win:
        win_col.Fence()
        free_win(win_col)
    return array

def get_GA_slice(addr, slice_i):
    addr_slice = []
    for i in slice_i:
        addr_slice.append(addr[i])
    
    rank_all, idx_all = zip(*addr_slice)
    rank_list = []
    seg_list = []
    idx_list = []
    rank_pre = -1
    i0_pre, i1_pre = -1, -1
    idx0_pre, idx1_pre = -1, -1
    for idx, rank_i in enumerate(rank_all):
        i0 = slice_i[idx]
        i1 = i0 + 1
        idx0, idx1 = idx_all[idx]
        if (rank_i == rank_pre) and (idx0 == idx1_pre):
            i1_pre = i1
            idx1_pre = idx1
        else:
            if idx != 0:
                rank_list.append(rank_pre)
                seg_list.append([i0_pre, i1_pre])
                idx_list.append([idx0_pre, idx1_pre])
            rank_pre = rank_i
            i0_pre, i1_pre = i0, i1
            idx0_pre, idx1_pre = idx0, idx1
            
        if idx == (len(rank_all)-1):
            rank_list.append(rank_pre)
            seg_list.append([i0_pre, i1_pre])
            idx_list.append([idx0_pre, idx1_pre])
    
    return rank_list, seg_list, idx_list

    
def read_GA(addr, slice_i, buffer, win, dtype='f8', list_col=None, dim_list=None, sup_dim=1, buf_idx_start=0):
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
        
        rank_list, seg_list, idx_list = get_GA_slice(addr, slice_i)
        for idx, rank_i in enumerate(rank_list):
            recv_idx0, recv_idx1 = idx_list[idx]
            buf_idx1 = buf_idx0 + (recv_idx1-recv_idx0)
            win.Lock(rank_i, lock_type=MPI.LOCK_SHARED)
            win.Get(buffer[buf_idx0: buf_idx1], target_rank=rank_i, target=[recv_idx0*sup_dim*size_unit, (recv_idx1-recv_idx0)*sup_dim, MPI.DOUBLE])
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
                list_col[i] = buffer[buf_idx0: buf_idx1].reshape(dim_list[i])
            else:
                buf_idx1 = buf_idx0 + 1
                list_col[i] = buffer[buf_idx0: buf_idx1].reshape(sup_shape)
            buf_idx0 = buf_idx1
        return buffer, list_col
    else:
        return buffer

def get_buff_size(dim_list, slice_i):
    dim_sum = 0
    for i in slice_i:
        dim_sum += np.prod(dim_list[i])
    return dim_sum

def get_GA_node(self, slice_list, win_ga, addr_ga, dim_list, len_addr, sup_dim=1):
    if sup_dim == 1:
        buff_size = get_buff_size(dim_list, slice_list)
        win_buff, buff = get_shared(buff_size, dtype='f8')
    else:
        dim0 = len(slice_list)
        if type(sup_dim) == int:
            win_buff, buff = get_shared((dim0, sup_dim), dtype='f8')
        else:
            dim1, dim2 = sup_dim
            win_buff, buff = get_shared((dim0, dim1, dim2), dtype='f8')

    address = [None]*len_addr
    idx0 = 0
    for i in slice_list:
        idx1 = idx0 + np.prod(dim_list[i])
        address[i] = [idx0, idx1]
        idx0 = idx1
    
    slice_i = get_slice(rank_list=self.shm_ranklist, job_list=slice_list)[irank_shm]
    if slice_i is not None: 
        buff = read_GA(addr_ga, slice_i, buff, win_ga, dtype='f8', dim_list=dim_list, sup_dim=sup_dim, buf_idx_start=address[slice_i[0]][0])
    comm_shm.Barrier()
    return address, win_buff, buff

def read_GA_node(slice_i, addr_node, buf_node, dim_list, tmp_list=None, buff=None):
    if tmp_list is None:
        i = slice_i[0]
        recv_idx0, recv_idx1 = addr_node[i]
        if buff is not None:
            buff[:] = buf_node[recv_idx0: recv_idx1].reshape(dim_list[i])
            return buff
        else:
            return buf_node[recv_idx0: recv_idx1].reshape(dim_list[i])
    else:
        tmp_list = [None]*len(tmp_list)
        buf_idx0 = 0
        for i in slice_i:
            buf_idx1 = buf_idx0 + np.prod(dim_list[i])
            recv_idx0, recv_idx1 = addr_node[i]
            if buff is None:
                tmp_list[i] = buf_node[recv_idx0: recv_idx1].reshape(dim_list[i])
            else:
                buff[buf_idx0: buf_idx1] = buf_node[recv_idx0: recv_idx1]
                tmp_list[i] = buff[buf_idx0: buf_idx1].reshape(dim_list[i])
            buf_idx0 = buf_idx1
        return tmp_list