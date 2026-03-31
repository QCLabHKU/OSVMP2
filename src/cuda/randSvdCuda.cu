#include <cstring>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cupyUtils.cuh>
#include <linalgUtils.cuh>

#define THREADS_PER_AXIS 16
#define TILE_WIDTH 16
#define THREADS_PER_BLOCK 256

// using namespace std;
namespace py = pybind11;


/*
 * Performs QR decomposition A = QR using cuSOLVER
 * Input and output are row-major matrices, matching LAPACKE behavior.
 */

void qrqCuda(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, double *A, double *Q, int m, int n) {
    // Compute rank as in the CPU function
    int rank = std::min(m, n);

    // Device memory pointers
    double *A_col, *tau, *workspace;
    int *devInfo;

    // Allocate GPU memory
    cudaMalloc(&A_col, m * n * sizeof(double)); // Column-major A
    cudaMalloc(&tau, rank * sizeof(double));    // Householder scalars
    cudaMalloc(&devInfo, sizeof(int));          // Device info for cuSOLVER

    // Define block and grid sizes for kernels
    dim3 block(THREADS_PER_AXIS, THREADS_PER_AXIS);
    dim3 gridA((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);    // For A conversion
    dim3 gridQ((rank + block.x - 1) / block.x, (m + block.y - 1) / block.y); // For Q conversion

    // Step 1: Convert A from row-major to column-major
    rowToColMajor<<<gridA, block>>>(A, A_col, m, n);

    // Step 2: Query workspace sizes for QR decomposition and Q generation
    int Lwork_geqrf, Lwork_orgqr;
    cusolverDnDgeqrf_bufferSize(cusolverH, m, n, A_col, m, &Lwork_geqrf);
    cusolverDnDorgqr_bufferSize(cusolverH, m, rank, rank, A_col, m, tau, &Lwork_orgqr);
    int Lwork = std::max(Lwork_geqrf, Lwork_orgqr);
    cudaMalloc(&workspace, Lwork * sizeof(double));

    // Step 3: Perform QR factorization on A_col
    cusolverDnDgeqrf(cusolverH, m, n, A_col, m, tau, workspace, Lwork, devInfo);

    // Step 4: Generate Q in the first rank columns of A_col
    cusolverDnDorgqr(cusolverH, m, rank, rank, A_col, m, tau, workspace, Lwork, devInfo);

    // Step 5: Convert the first rank columns of A_col to row-major Q
    colToRowMajor<<<gridQ, block>>>(A_col, Q, m, n, rank);

    // Clean up GPU memory
    cudaFree(A_col);
    cudaFree(tau);
    cudaFree(workspace);
    cudaFree(devInfo);
}

void qrqCupy(py::object A, py::object Q, int ra, int ca) {
    // Extracting CuPy array information
    double *ptrA = getCupyPtr<double>(A);
    double *ptrQ = getCupyPtr<double>(Q);

    // Create cuBLAS handle
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    // Create cuSolver handle
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    qrqCuda(cublasH, cusolverH, ptrA, ptrQ, ra, ca);

    // Destroy handles
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
}

void svdCuda(cusolverDnHandle_t cusolverH, double *A, double *U, double *S,
             double *Vt, int ra, int ca, bool getU, bool getVt, 
             double *work, int lwork_bytes, int *devInfo) {
    // Determine job parameters based on whether U and Vt are needed
    char jobu = getVt ? 'A' : 'N'; // Compute U' (maps to Vt)
    char jobvt = getU ? 'A' : 'N'; // Compute Vt' (maps to U)

    // Set dimensions (interpreting A as A^T)
    int m = ca; // Rows in column-major = columns of A
    int n = ra; // Columns in column-major = rows of A

    // Allocate device info for error checking
    //int *devInfo;
    //cudaMalloc(&devInfo, sizeof(int));

    // Call cusolverDnDgesvd
    //cusolverStatus_t status = cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, A, ca, S,
    cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, A, ca, S,
                     Vt, ca, U, ra, work, lwork_bytes, NULL, devInfo);

    // Check cuSOLVER status
    //if (status != CUSOLVER_STATUS_SUCCESS) {
    //    std::cerr << "cuSOLVER error: " << status << std::endl;
    //}

    // Check devInfo for additional errors
    /*
    int hostInfo;
    cudaMemcpy(&hostInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (hostInfo < 0) {
        std::cerr << "Parameter error: " << -hostInfo << "th parameter is wrong" << std::endl;
    } else if (hostInfo > 0) {
        std::cerr << "Not converged: " << hostInfo << " superdiagonals did not converge" << std::endl;
    }*/

    // Free device memory
    //cudaFree(work);
    //cudaFree(devInfo);
}

/*
void svdCupy(py::object A, py::object U, py::object S,
             py::object Vt, int ra, int ca, bool getU = true, bool getVt = true) {
    // Extracting CuPy array information
    double *ptrA = getCupyPtr<double>(A);
    double *ptrS = getCupyPtr<double>(S);
    double *ptrU = (getU) ? getCupyPtr<double>(U) : nullptr;
    double *ptrVt = (getVt) ? getCupyPtr<double>(Vt) : nullptr;

    // Create cuSolver handle
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    svdCuda(cusolverH, ptrA, ptrU, ptrS, ptrVt, ra, ca, getU, getVt);

    // Destroy cuSolver handle
    cusolverDnDestroy(cusolverH);
}*/

void svdCusolver(int m, int n, py::object A, int lda,
                 py::object S, py::object U, int ldu, py::object Vt, int ldvt) {
    // Extracting CuPy array information
    double *ptrA = getCupyPtr<double>(A);
    double *ptrS = getCupyPtr<double>(S);
    double *ptrU = getCupyPtr<double>(U);
    double *ptrVt = getCupyPtr<double>(Vt);

    // Create cuSolver handle
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    // Query workspace size
    int lwork;
    cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);

    // Allocate device workspace
    double *work;
    cudaMalloc(&work, lwork * sizeof(double));

    // Allocate device info for error checking
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    // Call cusolverDnDgesvd
    cusolverStatus_t status = cusolverDnDgesvd(cusolverH, 'A', 'A', m, n, ptrA, lda, ptrS,
                                               ptrU, ldu, ptrVt, ldvt, work, lwork, NULL, devInfo);

    // Check cuSOLVER status
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error: " << status << std::endl;
    }

    // Destroy cuSolver handle
    cusolverDnDestroy(cusolverH);
}

__global__ void batchedDnrm2(int m, int r, const double *y, double *errors) {
    int idx = blockIdx.x; // Each block handles one vector
    if (idx >= r)
        return;

    // Compute sum of squares for the vector y[idx * m : (idx + 1) * m - 1]
    double sum = 0.0;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        double val = y[idx * m + i];
        sum += val * val;
    }

    // Parallel reduction within the block to sum all threads' contributions
    __shared__ double partialSum[THREADS_PER_BLOCK]; // Block size assumed as 256
    partialSum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 in each block writes the final norm
    if (threadIdx.x == 0) {
        errors[idx] = sqrt(partialSum[0]);
    }
}

void printArray(double *d_arr, int size) {
    double *arr = new double[size];
    cudaMemcpy(arr, d_arr, size * sizeof(double), cudaMemcpyDeviceToHost);
    printf("[");
    for (int i = 0; i < size; ++i) {
        printf("%.2e", arr[i]); // Print with 2 decimal places
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");

    delete[] arr;
}

void print_test(double *d_array, size_t size, const std::string &lab = "") {
    double *array = new double[size];
    cudaMemcpy(array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate minimum, maximum, and mean
    double min_val = *std::min_element(array, array + size);
    double max_val = *std::max_element(array, array + size);
    double mean_val = std::accumulate(array, array + size, 0.0) / size;

    printf("%s %.6e %.6e %.6e\n", lab.c_str(), min_val, max_val, mean_val);

    delete[] array;
}

void fillWithRange(double *d_array, size_t size) {
    double *range = new double[size];
    for (int i = 0; i < size; ++i) {
        range[i] = i;
    }
    cudaMemcpy(d_array, range, size * sizeof(double), cudaMemcpyHostToDevice);
    delete[] range;
}

__global__ void findMaxValue1BlockKernel(double* data, double* result, int m) {
    extern __shared__ double sData[];
    int tid = threadIdx.x;

    double tMaxValue = -FLT_MAX;
    // Load data into shared memory
    for (int idx = tid; idx < m; idx += blockDim.x) {
        tMaxValue = max(tMaxValue, data[idx]);
    }

    sData[tid] = tMaxValue;
    
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sData[tid] = max(sData[tid], sData[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        result[0] = sData[0];
    }
}



__global__ void normalizeVec1BlockKernel(double* vec, int n) {
    __shared__ double sSquareSum[1024];
    int tid = threadIdx.x;

    double tSquareSum = 0.0;
    for (int idx = tid; idx < n; idx += blockDim.x) {
        double val = vec[idx];
        tSquareSum += val * val;
    }

    sSquareSum[tid] = tSquareSum;

    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sSquareSum[tid] += sSquareSum[tid + s];
        }
        __syncthreads();
    }

    double normYj = sqrt(sSquareSum[0]);

    for (int idx = tid; idx < n; idx += blockDim.x) {
        vec[idx] /= normYj;
    }

}


void adaptiveRandomizedRangeFinderCuda(curandGenerator_t curandG, cublasHandle_t cublasH, 
                                       cusolverDnHandle_t cusolverH,
                                       //double *A, double **qPtr, size_t *qSize,
                                       double *A, double *qBuffer, size_t *qSize,
                                       int m, int n, double tolerance, int r,
                                       int maxIter, int seed = 42) {

    // Intialize intermediate containers
    //double normYj;
    int idxYj = 0;
    int nsamp = 0;
    int mSize = m * sizeof(double);
    int restYSize = (r - 1) * mSize;
    double realTolerance = tolerance / (10 * sqrt(2 / 3.14159265358979323846));
    //double realTolerance = tolerance / 10;
    double errorApprox = tolerance * 10;

    // Preallocate temporary buffers
    /*
    double *qBuffer, *omega, *y, 
            *qjy, *errors, *vomegaPre, *yjnow, 
            *qytmp, *dErrorApprox;

    cudaMalloc(&qBuffer, m * maxIter * sizeof(double));
    cudaMalloc(&omega, n * r * sizeof(double));
    cudaMalloc(&y, r * m * sizeof(double));
    cudaMalloc(&qjy, r * sizeof(double));
    cudaMalloc(&errors, r * sizeof(double));
    cudaMalloc(&vomegaPre, maxIter * n * sizeof(double));
    cudaMalloc(&yjnow, maxIter * m * sizeof(double));
    cudaMalloc(&qytmp, maxIter * sizeof(double));
    cudaMalloc(&dErrorApprox, sizeof(double));*/
    
    //double *qBuffer, *omega, *y, 
    double *omega, *y,
            *qjy, *errors, *vomegaPre, *yjnow, 
            *qytmp, *dErrorApprox;
    
    /*
    size_t workspace_bytes = 0;
    workspace_bytes += m * maxIter * sizeof(double);           // qBuffer
    workspace_bytes += n * r * sizeof(double);                 // omega
    workspace_bytes += r * m * sizeof(double);                 // y
    workspace_bytes += r * sizeof(double);                     // qjy
    workspace_bytes += r * sizeof(double);                     // errors
    workspace_bytes += maxIter * n * sizeof(double);           // vomegaPre
    workspace_bytes += maxIter * m * sizeof(double);           // yjnow
    workspace_bytes += maxIter * sizeof(double);               // qytmp
    workspace_bytes += 1 * sizeof(double);                     // dErrorApprox
    cudaMalloc(&qBuffer, workspace_bytes);*/
    
    omega = qBuffer + m * maxIter;
    y = omega + n * r;
    qjy = y + r * m;
    errors = qjy + r;
    vomegaPre = errors + r;
    yjnow = vomegaPre + maxIter * n;
    qytmp = yjnow + maxIter * m;
    dErrorApprox = qytmp + maxIter;

    // Wrap the raw device pointer for thrust
    //thrust::device_ptr<double> errorsThrust = thrust::device_pointer_cast(errors);

    // Initialize random array with cuRAND
    //curandGenerator_t curandG;
    //curandCreateGenerator(&curandG, CURAND_RNG_PSEUDO_XORWOW);
    //curandSetPseudoRandomGeneratorSeed(curandG, seed); // or any seed you prefer

    // Assuming omega is a device pointer allocated with cudaMalloc
    curandGenerateUniformDouble(curandG, omega, n * r);

    // PQ decomposition of omega for numerical stability and convergence speed
    //qrqCuda(cublasH, cusolverH, omega, omega_q, n, r);

    dgemmSimpCuda(cublasH, 1, 1, omega, A, y, n, r, m, n);

    // Pregenerate ramdomized yj
    curandGenerateUniformDouble(curandG, vomegaPre, maxIter * n);

    dgemmSimpCuda(cublasH, 0, 1, vomegaPre, A, yjnow, maxIter, n, m, n);
    
    //while (nsamp < maxIter && errorApprox > realTolerance) {
    for (int j = 0; j < maxIter; j++) {
        /*
        cublasDnrm2_v2(cublasH, m, y, 1, &normYj);
        const double scale = 1.0 / normYj;
        cublasDscal_v2(cublasH, m, &scale, y, 1);*/


        nsamp = j + 1;

        normalizeVec1BlockKernel<<<1, 1024>>>(y, m);

        cudaMemcpy(qBuffer + idxYj, y, mSize, cudaMemcpyDeviceToDevice);
        //nsamp++;

        // ynow - (Q * Q.T * ynow)
        dgemvSimpCuda(cublasH, 0, qBuffer, yjnow + idxYj, qytmp, nsamp, m);
        dgemvSimpCuda(cublasH, 1, qBuffer, qytmp, yjnow + idxYj, nsamp, m, -1.0, 1.0);

        //cudaMemcpy(yj, y, mSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(y, y + m, restYSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(y, yjnow + idxYj, mSize, cudaMemcpyDeviceToDevice);

        dgemvSimpCuda(cublasH, 0, y, qBuffer + idxYj, qjy, r, m);
        dgerSimpCuda(cublasH, qjy, qBuffer + idxYj, y, r, m, -1.0, 1.0);

        if (j % 5 == 0 || j == maxIter - 1) {
            // compute error of last r consecutive vecs
            batchedDnrm2<<<r, THREADS_PER_BLOCK>>>(m, r, y, errors);

            // errorApprox = *std::max_element(errors, errors + r);
            //errorApprox = *thrust::max_element(errorsThrust, errorsThrust + r);
            
            findMaxValue1BlockKernel<<<1, r, r*sizeof(double)>>>(errors, dErrorApprox, r);
            cudaMemcpy(&errorApprox, dErrorApprox, sizeof(double), cudaMemcpyDeviceToHost);
            if (errorApprox <= realTolerance) {
                break;
            }
        }

        

        /*reproject onto the range orthog to Q, i.e. (y_j - Q*Q.T*y_j)
        then overwrite y_j */

        dgemvSimpCuda(cublasH, 0, qBuffer, y, qytmp, nsamp, m);
        dgemvSimpCuda(cublasH, 1, qBuffer, qytmp, y, nsamp, m, -1.0, 1.0);
        idxYj += m;

        
    }


    *qSize = nsamp * m;
    //*qPtr = new double[*qSize];
    // std::memcpy(*qPtr, qBuffer, *qSize * sizeof(double));
    //cudaMalloc((void **)qPtr, *qSize * sizeof(double));
    //cudaMemcpy(*qPtr, qBuffer, *qSize * sizeof(double), cudaMemcpyDeviceToDevice);

    // Free the memory
    //cudaFree(qBuffer);
    /*
    cudaFree(omega);
    cudaFree(y);
    cudaFree(qjy);
    cudaFree(errors);
    cudaFree(vomegaPre);
    cudaFree(yjnow);
    cudaFree(qytmp);
    cudaFree(dErrorApprox);*/

    // Destroy the generator
    //curandDestroyGenerator(curandG);
}

/*
py::object adaptiveRandomizedRangeFinderCupy(py::object A, double tolerance, int maxIter, int r, int seed = 42) {

    // Initialize cublas and cusolver handle
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    double *aPtr = getCupyPtr<double>(A);
    py::tuple aShape = A.attr("shape");
    int m = aShape[0].cast<int>();
    int n = aShape[1].cast<int>();
    double *qPtr = nullptr;
    size_t qSize;

    adaptiveRandomizedRangeFinderCuda(cublasH, cusolverH, aPtr, &qPtr, &qSize, m, n, tolerance, r, maxIter, seed);

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    int nCol = qSize / m;

    return createCupyArray<double>(qPtr, nCol, m);
}*/

size_t randomizedSvd_bufferSize(cusolverDnHandle_t cusolverH, int m, int n, int maxIter, int r, bool getVt = true) {
    int lworkSvdBytes;
    cusolverDnDgesvd_bufferSize(cusolverH, maxIter, n, &lworkSvdBytes);

    size_t workspace = 0;
    workspace += maxIter * m;           // qBuffer
    workspace += n * r;                 // omega
    workspace += r * m;                 // y
    workspace += r;                     // qjy
    workspace += r;                     // errors
    workspace += maxIter * n;           // vomegaPre
    workspace += maxIter * m;           // yjnow
    workspace += maxIter;               // qytmp
    workspace += 1;                     // dErrorApprox
    workspace += lworkSvdBytes;                          // svdBuffer
    if (getVt) workspace += n * n;                       // uTilde

    return workspace;
}

int64_t randomizedSvdBufferSizeCupy(uintptr_t cusolverH_ptr, int m, int n, 
                                    int maxIter, int r, bool getVt = true) {
    
    cusolverDnHandle_t cusolverH = reinterpret_cast<cusolverDnHandle_t>(cusolverH_ptr);

    size_t workspace = randomizedSvd_bufferSize(cusolverH, m, n, maxIter, r, getVt);

    return (int64_t)workspace;
}

void svdCudax(cusolverDnHandle_t cusolverH, double *A, double *U, double *S,
             double *Vt, int ra, int ca, bool getU, bool getVt) {
    // Determine job parameters based on whether U and Vt are needed
    char jobu = getVt ? 'A' : 'N'; // Compute U' (maps to Vt)
    char jobvt = getU ? 'A' : 'N'; // Compute Vt' (maps to U)

    // Set dimensions (interpreting A as A^T)
    int m = ca; // Rows in column-major = columns of A
    int n = ra; // Columns in column-major = rows of A

    // Query workspace size
    int lwork;
    cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);

    // Allocate device workspace
    double *work;
    cudaMalloc(&work, lwork * sizeof(double));

    // Allocate device info for error checking
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    // Set pointers based on getU and getVt
    // double *U_param = getVt ? Vt : nullptr; // Vt receives U'
    // double *VT_param = getU ? U : nullptr;  // U receives Vt'

    // Call cusolverDnDgesvd
    cusolverStatus_t status = cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, A, ca, S,
                                               Vt, ca, U, ra, work, lwork, NULL, devInfo);

    // Check cuSOLVER status
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error: " << status << std::endl;
    }

    // Check devInfo for additional errors
    int hostInfo;
    cudaMemcpy(&hostInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (hostInfo < 0) {
        std::cerr << "Parameter error: " << -hostInfo << "th parameter is wrong" << std::endl;
    } else if (hostInfo > 0) {
        std::cerr << "Not converged: " << hostInfo << " superdiagonals did not converge" << std::endl;
    }

    // Free device memory
    cudaFree(work);
    cudaFree(devInfo);
}

int randomizedSvdCupy(uintptr_t curandG_ptr, uintptr_t cublasH_ptr, uintptr_t cusolverH_ptr,
                      py::object A, py::object U, py::object S, py::object Vt, 
                      py::object workSpace, double tolerance, int maxIter, 
                      int m, int n, int r = 10, bool getU = true, bool getVt = true, 
                      int seed = 42) {

    // Initialize cublas and cusolver handle
    //cublasHandle_t cublasH;
    //cublasCreate(&cublasH);

    //cusolverDnHandle_t cusolverH;
    //cusolverDnCreate(&cusolverH);
    curandGenerator_t curandG = reinterpret_cast<curandGenerator_t>(curandG_ptr);
    cublasHandle_t cublasH = reinterpret_cast<cublasHandle_t>(cublasH_ptr); 
    cusolverDnHandle_t cusolverH = reinterpret_cast<cusolverDnHandle_t>(cusolverH_ptr);

    double *aPtr = getCupyPtr<double>(A);
    double *sPtr = getCupyPtr<double>(S);
    double *workSpacePtr = getCupyPtr<double>(workSpace);
    double *uPtr = (getU) ? getCupyPtr<double>(U) : NULL;
    double *vtPtr = (getVt) ? getCupyPtr<double>(Vt) : NULL;

    size_t qSize;

    // Find the range
    adaptiveRandomizedRangeFinderCuda(curandG, cublasH, cusolverH, aPtr, workSpacePtr, 
                                      &qSize, m, n, tolerance, r, maxIter, seed);

    int nCol = qSize / m;

    int lworkSvdBytes;
    cusolverDnDgesvd_bufferSize(cusolverH, nCol, n, &lworkSvdBytes); 

    double *Q = workSpacePtr;
    double *B = workSpacePtr + qSize; // nCol * n
    double *svdBuffer = B + nCol * n;
    size_t bufferLoc = qSize + nCol * n + lworkSvdBytes;
    int *devInfo = reinterpret_cast<int*>(workSpacePtr + bufferLoc);
    bufferLoc += 1;

    double *uTilde = (getU) ? workSpacePtr + bufferLoc : NULL;
    if (getU) bufferLoc += nCol * nCol;
    double *vtTilde = (getVt) ? workSpacePtr + bufferLoc : NULL;

    

    dgemmSimpCuda(cublasH, 0, 0, Q, aPtr, B, nCol, m, m, n);

    svdCuda(cusolverH, B, uTilde, sPtr, vtTilde, nCol, n, getU, getVt, svdBuffer, lworkSvdBytes, devInfo);
    //svdCudax(cusolverH, B, uTilde, sPtr, vtTilde, nCol, n, getU, getVt);

    if (getU) dgemmSimpCuda(cublasH, 1, 0, Q, uTilde, uPtr, nCol, m, nCol, nCol);

    if (getVt) cudaMemcpy(vtPtr, vtTilde, nCol * n * sizeof(double), cudaMemcpyDeviceToDevice);

    return nCol;
}

// Bind the functions to a Python module
PYBIND11_MODULE(randSvdCuda, m) {
    m.def("qrqCupy", &qrqCupy, "qrqCupy"),
    //m.def("svdCupy", &svdCupy, "svdCupy");
    m.def("svdCusolver", &svdCusolver, "svdCusolver");
    //m.def("adaptiveRandomizedRangeFinderCupy", &adaptiveRandomizedRangeFinderCupy, "adaptiveRandomizedRangeFinderCupy");
    m.def("randomizedSvdBufferSizeCupy", &randomizedSvdBufferSizeCupy, "randomizedSvdBufferSizeCupy");
    m.def("randomizedSvdCupy", &randomizedSvdCupy, "randomizedSvdCupy",
        py::arg("curandG_ptr"), py::arg("cublasH_ptr"), py::arg("cusolverH_ptr"),
        py::arg("A"), py::arg("U"), py::arg("S"), py::arg("Vt"), 
        py::arg("workSpace"), py::arg("tolerance"), py::arg("maxIter"),
        py::arg("m"), py::arg("n"), py::arg("r") = 10, py::arg("getU") = true, 
        py::arg("getVt") = true, py::arg("seed") = 42);
}