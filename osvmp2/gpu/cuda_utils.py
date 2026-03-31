import os
import mpi4py
from mpi4py import MPI
import numpy as np
import cupy
import gc
from cupy_backends.cuda.libs import cublas
#cublasH = cupy.cuda.device.get_cublas_handle()
#cublas.setStream(cublasH, 0)
from osvmp2.__config__ import ngpu
from osvmp2.mpi_addons import *

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
ngpu_shm = ngpu // nnode
nrank_per_gpu = nrank_shm // ngpu_shm
igpu = irank // nrank_per_gpu
igpu_shm = irank_shm // nrank_per_gpu
ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
irank_gpu = irank % nrank_per_gpu
cupy.cuda.runtime.setDevice(igpu_shm)

THREADS_PER_AXIS = 16

def avail_gpu_mem(max_mem=None, device_id=None, use_cupy_mem_pool=True, unit="MB"):
    assert unit in {"B", "MB", "GB"}

    B2MB = 1 / 1024**2
    B2GB = 1 / 1024**3

    if use_cupy_mem_pool:
        mem_pool = cupy.get_default_memory_pool()
        used_mem = mem_pool.used_bytes() 
        total_mem = mem_pool.get_limit() 
        if total_mem == 0:
            total_mem = cupy.cuda.runtime.memGetInfo()[1]
    else:
        if device_id is None:
            device = cupy.cuda.Device(device_id)  # Select GPU
            free_mem, total_mem = device.mem_info  # Get free and total memory
        else:
            free_mem, total_mem = cupy.cuda.runtime.memGetInfo()

        free_mem, total_mem = free_mem, total_mem
    
        used_mem = total_mem - free_mem # Calculate used memory

    if max_mem is None:
        max_mem = total_mem
    else:
        max_mem = max_mem / B2MB #MB to B

    if unit == "MB":
        scale = B2MB
    elif unit == "GB":
        scale = B2GB
    else:
        scale = 1

    return (max_mem - used_mem) * scale


def ave_gpu_memory(max_mem=None, use_cupy_mem_pool=True, unit="MB"):
    mem_gpu = avail_gpu_mem(max_mem=max_mem, 
                            use_cupy_mem_pool=use_cupy_mem_pool, 
                            unit=unit)
    if ngpu == 1:
        return mem_gpu
    win_gpu_mem, gpu_mem = get_shared(ngpu, set_zeros=True)
    if irank_gpu == 0:
        gpu_mem[igpu] = mem_gpu
    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(gpu_mem)
        comm.Barrier()
    ave_gpu_mem = np.mean(gpu_mem)
    comm_shm.Barrier()
    free_win(win_gpu_mem)
    return ave_gpu_mem

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
    from itertools import accumulate
    cumsum = list(accumulate(a, initial=0))  # [0, a[0], a[0]+a[1], ...]
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
        result.append(a[start:best_i])
        start = best_i
    
    # Add the remaining part
    result.append(a[start:])
    
    return result

def sliceJobsFor2DBlocks(indices, dimListA, dimListB, threadsX, threadsY=None):

    if threadsY is None:
        threadsY = threadsX

    # Calculate total number of elements to preallocate memory
    nSegA = (dimListA + threadsY - 1) // threadsY
    nSegB = (dimListB + threadsX - 1) // threadsX
    sizes = nSegA * nSegB

    # Generate indicesBlock and localIndicesBlock using vectorized repeat
    indicesBlock = np.repeat(indices, sizes)
    localIndicesBlock = np.repeat(np.arange(len(indices), dtype=np.int64), sizes)

    # Generate dimSegBlockA and dimSegBlockB using list comprehensions and concatenation
    dimSegBlockA = [np.repeat(np.arange(nA, dtype=np.int64) * threadsY, nB)
                    for nA, nB in zip(nSegA, nSegB)]
    dimSegBlockA = np.concatenate(dimSegBlockA)

    dimSegBlockB = [np.tile(np.arange(nB, dtype=np.int64) * threadsX, nA)
                    for nA, nB in zip(nSegA, nSegB)]
    dimSegBlockB = np.concatenate(dimSegBlockB)

    return indicesBlock, localIndicesBlock, dimSegBlockA, dimSegBlockB


def get_seg_gpu(slice_rank, max_nao_cpu):

    first_slice = slice_rank[0]
    shell_seg = [[first_slice]]
    nao_now = first_slice[0][1] - first_slice[0][0]
    for islice in slice_rank[1:]:
        nao_slice = islice[0][1] - islice[0][0]
        if nao_now + nao_slice > max_nao_cpu:
            shell_seg.append([islice])
            nao_now = nao_slice
        else:
            shell_seg[-1].append(islice)
            nao_now += nao_slice

    nao_slices_cpu = [iseg[-1][0][1] - iseg[0][0][0] for iseg in shell_seg]
    
    return shell_seg, max(nao_slices_cpu)

def dgemm_cupy(transA, transB, a, b, c, alpha, beta, cublasH=None):
    if cublasH is None:
        cublasH = cupy.cuda.device.get_cublas_handle()
    
    assert c._c_contiguous
    alpha = np.asarray(alpha, dtype=a.dtype)
    beta = np.asarray(beta, dtype=a.dtype)
    if not a._c_contiguous:
        a = cupy.ascontiguousarray(a)
    if not b._c_contiguous:
        b = cupy.ascontiguousarray(b)

    ra, ca = a.shape
    rb, cb = b.shape

    TransA =  cublas.CUBLAS_OP_N if transA == 0 else cublas.CUBLAS_OP_T
    TransB =  cublas.CUBLAS_OP_N if transB == 0 else cublas.CUBLAS_OP_T

    m = cb if transB == 0 else rb
    n = ra if TransA == 0 else ca
    k = ca if TransA == 0 else ra

    '''alpha = cupy.array(alpha, dtype=cupy.float64)
    beta = cupy.array(beta, dtype=cupy.float64)'''

    #ori_mode = cublas.getPointerMode(cublasH)
    # cublas.setPointerMode(cublasH, cublas.CUBLAS_POINTER_MODE_HOST)

    cublas.dgemm(cublasH, TransB, TransA, m, n, k, alpha.ctypes.data,
                b.data.ptr, cb, a.data.ptr, ca, beta.ctypes.data, c.data.ptr, m)
    
    # cublas.setPointerMode(cublasH, ori_mode)


def dgemv_cupy(transA, a, x, y, alpha, beta, incB=1, incC=1, cublasH=None):
    if cublasH is None:
        cublasH = cupy.cuda.device.get_cublas_handle()
    
    assert y._c_contiguous
    alpha = np.asarray(alpha, dtype=a.dtype)
    beta = np.asarray(beta, dtype=a.dtype)
    if not a._c_contiguous:
        a = cupy.ascontiguousarray(a)
    if not x._c_contiguous:
        x = cupy.ascontiguousarray(x)

    ra, ca = a.shape
    m = ca
    n = ra

    TransA =  cublas.CUBLAS_OP_N if transA == 0 else cublas.CUBLAS_OP_T

    '''alpha = cupy.array(alpha, dtype=cupy.float64)
    beta = cupy.array(beta, dtype=cupy.float64)'''

    #ori_mode = cublas.getPointerMode(cublasH)
    # cublas.setPointerMode(cublasH, cublas.CUBLAS_POINTER_MODE_HOST)

    cublas.dgemv(cublasH, TransA, m, n, alpha.ctypes.data, a.data.ptr, ca, x.data.ptr, 
                 incB, beta.ctypes.data, y.data.ptr, incC)
    
    # cublas.setPointerMode(cublasH, ori_mode)


def dger_cupy(x, y, A, alpha, incx=1, incy=1, cublasH=None):
    if cublasH is None:
        cublasH = cupy.cuda.device.get_cublas_handle()

    alpha = np.asarray(alpha, dtype=A.dtype)

    # Ensure A is C-contiguous
    assert A._c_contiguous

    # Ensure vectors are contiguous if stride is 1
    if not x._c_contiguous:
        x = cupy.ascontiguousarray(x)
    if not y._c_contiguous:
        y = cupy.ascontiguousarray(y)

    m, n = A.shape

 
    # Save and set pointer mode
    #ori_mode = cublas.getPointerMode(cublasH)
    # cublas.setPointerMode(cublasH, cublas.CUBLAS_POINTER_MODE_HOST)

    # Call CUBLAS DGER
    cublas.dger(cublasH,
                n, m,
                alpha.ctypes.data,
                y.data.ptr, incx,
                x.data.ptr, incy,
                A.data.ptr, n)

    # Restore pointer mode
    # cublas.setPointerMode(cublasH, ori_mode)

def numpy_to_cupy(numpy_array, dtype=None):
    if not isinstance(numpy_array, np.ndarray):
        numpy_array = np.asarray(numpy_array, dtype=dtype)

    if dtype is not None:
        dtype = np.dtype(dtype)
        if numpy_array.dtype != dtype:
            numpy_array = np.asarray(numpy_array, dtype=dtype)
    return cupy.asarray(numpy_array)

def free_cupy_mem():
    return None
    #cupy.cuda.Device().synchronize()
    #gc.collect()    # force garbage collection
    cupy.cuda.Device().synchronize()
    mem_before = avail_gpu_mem()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.cuda.Device().synchronize()
    #print(irank, "Freed cupy memory from %.2f MB to %.2f MB"%(mem_before, avail_gpu_mem()))
    #

def cum_offset_cupy(sizes, dtype=None):
    sizes = cupy.asarray(sizes)
    offsets = cupy.empty(len(sizes) + 1, dtype=dtype)
    offsets[0] = 0
    cupy.cumsum(sizes, out=offsets[1:], dtype=dtype)
    return offsets

def uniform_cum_offset_cupy(size, nelement, dtype=np.int64):
    return cupy.arange(0, (nelement + 1) * size, size, dtype=dtype)

def get_cupy_buffers(shapes, dtype=cupy.float64, buf=None):
    sizes = [np.prod(shape) for shape in shapes]
    total_size = np.sum(sizes)
    if buf is None:
        buffer = cupy.empty(total_size, dtype=dtype)
    else:
        if buf.size < total_size:
            raise ValueError("The buffer is too small!")
        buffer = buf[:total_size]
    buffers = []
    idx0 = 0
    for size, shape in zip(sizes, shapes):
        idx1 = idx0 + size
        buffers.append(buffer[idx0:idx1].reshape(shape))
        idx0 = idx1
    return buffers

def get_loc_buf(shm_buf_gpu, shm_size):
    #shm_size = nao * nocc
    loc_size = int((shm_buf_gpu.size - shm_size) / nrank_per_gpu)
    idx0 = irank_gpu * loc_size
    idx1 = idx0 + loc_size
    return shm_buf_gpu[shm_size:][idx0:idx1]



# CUDA C kernel: squares A[i] into Out[i]
square_kernel = cupy.RawKernel(r'''
extern "C" __global__
void square(const double* __restrict__ A,
            double* __restrict__ Out,
            const int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double a = A[i];
        Out[i] = a * a;
    }
}
''', 'square')

def compute_sq(A, out):
    threads = 256
    blocks = (A.size + threads - 1) // threads
    square_kernel((blocks,), (threads,), (A, out, A.size))



sq_sum_axis0_kernel = cupy.RawKernel(r'''
extern "C" __global__
void sq_sum_axis0(const double* __restrict__ A,
               double* __restrict__ out,
               const int M, const int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < N) {
        double acc = 0.0;
        for (int i = 0; i < M; ++i) {
            double a = A[i * N + j];
            acc += a * a;  // row-major indexing
        }
        out[j] = acc;
    }
}
''', 'sq_sum_axis0')


def sq_sum_axis0(A, out):
    threads = 256
    m, n = A.shape

    threads = 256
    blocks = (n + threads - 1) // threads
    sq_sum_axis0_kernel((blocks,), (threads,), (A, out, m, n))

# Raw kernel for finding indices where x > tolerance
find_indices_kernel = cupy.RawKernel(r'''
extern "C" __global__
void find_indices(const double* x, int* indices, const double tolerance, 
                  int size, int* kept_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        if (x[idx] > tolerance) {
            int pos = atomicAdd(kept_count, 1);
            indices[pos] = idx;
        }
    }
}
''', 'find_indices')

# Raw kernel for taking elements along axis 1
take_along_axis1_kernel = cupy.RawKernel(r'''
extern "C" __global__
void take_along_axis1(const double* A, const int* indices, double* out,
                      int rows, int cols_A, int cols_out, int num_indices) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col_out < num_indices) {
        int src_col = indices[col_out];
        if (src_col >= 0 && src_col < cols_A) {
            out[(size_t)row * cols_out + col_out] = A[(size_t)row * cols_A + src_col];
        }
    }
}
''', 'take_along_axis1')

def screen_axis1(A, x, tolerance, buffer=None, kept_indices=None, kept_count=None):
    """
    Performs cupy.take(A, cupy.where(x > tolerance)[0], axis=1) using raw kernels
    
    Args:
        A: Input array of shape (rows, cols_A)
        x: Array of shape (cols_A,) to compare against tolerance
        tolerance: Threshold value
    
    Returns:
        Output array with selected columns
    """
    # Validate input dimensions
    if A.ndim != 2 or x.ndim != 1:
        raise ValueError("A must be 2D and x must be 1D")
    if A.shape[1] != x.shape[0]:
        raise ValueError("A.shape[1] must equal x.shape[0]")
    
    rows, cols_A = A.shape
    
    # Allocate memory for indices and kept_count
    if kept_indices is None:
        kept_indices = cupy.zeros(cols_A, dtype=cupy.int32)
    if kept_count is None:
        kept_count = cupy.zeros(1, dtype=cupy.int32)
    else:
        kept_count[0] = 0
    
    # Launch kernel to find indices where x > tolerance
    threads_per_block = 512
    blocks = (cols_A + threads_per_block - 1) // threads_per_block
    find_indices_kernel((blocks,), (threads_per_block,), 
                       (x, kept_indices, tolerance, cols_A, kept_count))
    
    # Get actual number of indices found
    num_indices = int(kept_count.get()[0])  # Transfer kept_count from GPU to CPU
    
    if num_indices == 0:
        # Return empty array with correct shape if no elements satisfy condition
        if out is not None:
            return out
        else:
            return cupy.array([]).reshape(rows, 0).astype(A.dtype)
    
    # Extract only the valid indices
    indices = kept_indices[:num_indices]
    
    # Determine output shape and allocate if needed
    if buffer is None:
        out = cupy.empty((rows, num_indices), dtype=A.dtype)
    else:
        out = buffer[:rows*num_indices].reshape(rows, num_indices)
    
    # Configure grid for take operation
    threads_per_block_2d = (16, 16)
    blocks_x = (num_indices + threads_per_block_2d[0] - 1) // threads_per_block_2d[0]
    blocks_y = (rows + threads_per_block_2d[1] - 1) // threads_per_block_2d[1]
    
    # Launch kernel to perform take operation along axis 1
    take_along_axis1_kernel((blocks_x, blocks_y), threads_per_block_2d,
                           (A, indices, out, rows, cols_A, num_indices, num_indices))
    
    return out