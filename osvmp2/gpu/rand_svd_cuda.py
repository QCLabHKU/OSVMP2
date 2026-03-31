import math
import time
import numpy as np
import cupy
from osvmp2.gpu.cuda_utils import dgemm_cupy, dgemv_cupy, dger_cupy, get_cupy_buffers
from osvmp2.lib import randSvdCuda


def adaptive_randomized_range_finder(A, tolerance, r=10, max_iter=200, use_cupy=False):
    m, n = A.shape
    #scales = cupy.asarray([-1.0, 0.0, 1.0], dtype=cupy.float64)

    Qt_buff = cupy.empty((max_iter, n))
    y_tmp_buf = cupy.empty((max(r, max_iter),))

    if not use_cupy:
        yr_buff = cupy.empty((max_iter, m))
        y = cupy.empty((r, m))

    gen = cupy.random.RandomState(seed=42)
    omega = gen.randn(max_iter, n, dtype=cupy.float64)
    if use_cupy:
        yr_buff = cupy.dot(omega, A.T) 
    else:
        dgemm_cupy(0, 1, omega, A, yr_buff, 1.0, 0.0)

    omega = gen.randn(n, r, dtype=cupy.float64)
    omega, _ = cupy.linalg.qr(omega)
    if use_cupy:
        y = cupy.matmul(omega.T, A.T)
    else:
        dgemm_cupy(1, 1, omega, A, y, 1.0, 0.0)
    
    # initial range space estimate (single vector)
    thresh = tolerance / (10*cupy.sqrt(2/cupy.pi)) 
    error_approx = tolerance * 10
    pbar = range(max_iter) #tqdm(range(max_iter))


    for j in pbar:
        Qt_buff[j] = y[0]
        qj = Qt_buff[j]
        qj /= cupy.linalg.norm(qj)
        qj = qj.reshape(1, -1)

        Q = Qt_buff[:j+1]

        # draw new gaussian vec
        yr = yr_buff[j].reshape(1, -1)
        
        y_tmp = y_tmp_buf[:(j+1)].reshape(1, -1)
        if use_cupy:
            cupy.matmul(yr, Q.T, out=y_tmp)
            yr -= cupy.matmul(y_tmp, Q)
        else:
            dgemv_cupy(1, Q, yr, y_tmp, 1.0, 0.0)
            dgemv_cupy(0, Q, y_tmp, yr, -1.0, 1.0)


        # overwrite j+1:j+r-1 vecs
        y[:(r-1)] = y[1:]
        y[(r-1)] = yr.ravel()
        
        #y -= cupy.matmul(cupy.matmul(y, qj.T), qj)
        y_tmp = y_tmp_buf[:r].reshape(-1, 1)
        if use_cupy:
            cupy.matmul(y, qj.T, out=y_tmp)
            y -= cupy.matmul(y_tmp, qj)
        else:
            dgemv_cupy(1, y, qj, y_tmp, 1.0, 0.0)
            dger_cupy(y_tmp, qj, y, -1.0)

        # compute error of last r consecutive vecs
        if j % 5 == 0:
            error_approx = cupy.linalg.norm(y, axis=1).max()

            if error_approx <= thresh:
                #print("Ranges found in %d cycles"%(j+1))
                break

        # normalize yj
        if use_cupy:
            y[0] -= cupy.matmul(cupy.matmul(y[0], Q.T), Q)
        else:
            y_tmp = y_tmp_buf[:(j+1)].reshape(-1, 1)
            dgemv_cupy(1, Q, y[0], y_tmp, 1.0, 0.0)
            dgemv_cupy(0, Q, y_tmp, y[0], -1.0, 1.0)

    return Q 

def randSVD(A, tolerance, max_iter=1000, r=10):

    Q = adaptive_randomized_range_finder(A, r=r, tolerance=tolerance, 
                                                    max_iter=max_iter)

    # Stage B.
    B = Q @ A
    U_tilde, S, Vt = cupy.linalg.svd(B)
    U = Q.T @ U_tilde
    S = S[S > tolerance]
    rank = len(S)
    # Truncate.
    U, Vt = U[:, :rank], Vt[:rank, :]

    # This is useful for computing the actual error of our approximation.
    return U, S, Vt


def test_rsvd():
    def get_tensor(m, n, rank):
        """Define random, fixed-rank nxn matrix"""
        if m == n:
            full_matrices = True
        else:
            full_matrices = False
        mtx = cupy.random.randn(m, n)
        u, s, vh = cupy.linalg.svd(mtx, full_matrices=full_matrices)
        s[rank:] = 0  # fix the rank
        s = cupy.diag((s/s.max())**3)  # quickly decaying spectrum
        
        mtx = cupy.dot(cupy.dot(u, s), vh) # u @s.diag() @ v.T
        return cupy.matmul(mtx.T, mtx)

    rank_true = 50 #111  # fix rank to 111
    m, n = 200, 200
    A = get_tensor(m, n, rank_true)
    cupy.cuda.Stream.null.synchronize()

    t0 = time.time()
    U, S, Vt = randSVD(A, 1e-4)
    cupy.cuda.Stream.null.synchronize()
    print("rsvd %.2f s"%(time.time()-t0))

    A_rsvd = cupy.dot(cupy.dot(U, cupy.diag(S)), Vt)
    print(np.linalg.norm(A - A_rsvd))

def rand_svd_Cupy(A, tolerance, r=10, max_iter=200, get_U=True, get_Vt=True, 
                  workspace=None, U_buf=None, S_buf=None, Vt_buf=None, 
                  curandG=None, cublasH=None, cusolverH=None, seed=42):
    
    if curandG is None:
        rng = cupy.random.RandomState(seed=seed)
        curandG = rng._generator
    else:
        cupy.cuda.curand.setGeneratorOffset(curandG, 0)
    if cublasH is None:
        cublasH = cupy.cuda.device.get_cublas_handle()
    if cusolverH is None:
        cusolverH = cupy.cuda.device.get_cusolver_handle()

    m, n = A.shape
    if workspace is None:
        workspace_size = randSvdCuda.randomizedSvdBufferSizeCupy(cusolverH, m, n, 
                                                                 max_iter, r, get_Vt)
        workspace = cupy.empty(workspace_size, dtype=cupy.float64)

    if S_buf is None:
        S_buf = cupy.empty(max_iter, dtype=cupy.float64)

    if U_buf is None and get_U:
        U_buf = cupy.empty(m*max_iter, dtype=cupy.float64)
    
    if Vt_buf is None and get_Vt:
        Vt_buf = cupy.empty(n*max_iter, dtype=cupy.float64)
    
    ncol = randSvdCuda.randomizedSvdCupy(curandG, cublasH, cusolverH, 
                                         A, U_buf, S_buf, Vt_buf, 
                                         workspace, tolerance, max_iter, 
                                         m, n, r, get_U, get_Vt, seed)
    S = S_buf[:ncol]

    U = U_buf[:m*ncol].reshape(m, ncol) if get_U else None
    
    Vt = Vt_buf[:n*ncol].reshape(ncol, n) if get_Vt else None

    return U, S, Vt