#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cupyUtils.cuh>
#include <linalgUtils.cuh>

#define THREADS_PER_BLOCK 256

// Kernel to convert row-major to column-major
__global__ void rowToColMajor(double *A_row, double *A_col, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    if (i < m && j < n) {
        A_col[j * m + i] = A_row[i * n + j];
    }
}

// Kernel to convert first k columns of column-major to row-major
__global__ void colToRowMajor(double *A_col, double *Q_row, int m, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    if (i < m && j < k) {
        Q_row[i * k + j] = A_col[j * m + i];
    }
}

__global__ void copyTrans3DKernel(
    double* dst, const double* src,
    size_t shapeDst0, size_t shapeDst1, size_t shapeDst2,
    size_t shapeSrc0, size_t shapeSrc1, size_t shapeSrc2,
    int permSrcX, int permSrcY, int permSrcZ,
    size_t sliceStartDst, size_t sliceEndDst,
    int sliceAxisDst) {
    
    // Compute total elements in the slice (efficiently)
    /*
    size_t sliceSize;
    if (sliceAxisDst == 0) {
        sliceSize = (sliceEndDst - sliceStartDst) * shapeDst1 * shapeDst2;
    } else if (sliceAxisDst == 1) {
        sliceSize = shapeDst0 * (sliceEndDst - sliceStartDst) * shapeDst2;
    } else {
        sliceSize = shapeDst0 * shapeDst1 * (sliceEndDst - sliceStartDst);
    }*/
    size_t sliceSize = shapeSrc0 * shapeSrc1 * shapeSrc2;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sliceSize) return;

    // Precompute destination strides (for C-contiguous layout)
    size_t strideDst0 = shapeDst1 * shapeDst2;
    size_t strideDst1 = shapeDst2;
    // strideDst2 = 1 (implicit)

    // Decompose index with last dimension (contiguous in memory) as fastest-changing
    size_t i2, i1, i0;
    size_t offsetDst;
    if (sliceAxisDst == 0) {
        i2 = i % shapeDst2;
        i1 = (i / shapeDst2) % shapeDst1;
        i0 = i / (shapeDst1 * shapeDst2);
        offsetDst = (sliceStartDst + i0) * strideDst0 + i1 * strideDst1 + i2;
    } else if (sliceAxisDst == 1) {
        i2 = i % shapeDst2;
        i1 = (i / shapeDst2) % (sliceEndDst - sliceStartDst);
        i0 = i / (shapeDst2 * (sliceEndDst - sliceStartDst));
        offsetDst = i0 * strideDst0 + (sliceStartDst + i1) * strideDst1 + i2;
    } else { // sliceAxisDst == 2
        i2 = i % (sliceEndDst - sliceStartDst);
        i1 = (i / (sliceEndDst - sliceStartDst)) % shapeDst1;
        i0 = i / (shapeDst1 * (sliceEndDst - sliceStartDst));
        offsetDst = i0 * strideDst0 + i1 * strideDst1 + (sliceStartDst + i2);
    }

    // Compute source offset using permutation
    size_t srcIndices[3] = {0, 0, 0};
    srcIndices[permSrcX] = i0;
    srcIndices[permSrcY] = i1;
    srcIndices[permSrcZ] = i2;
    
    size_t offsetSrc = srcIndices[0] * (shapeSrc1 * shapeSrc2) + 
                       srcIndices[1] * shapeSrc2 + 
                       srcIndices[2];

    dst[offsetDst] = src[offsetSrc];
}


void dgemmSimpCuda(cublasHandle_t cublasH, int transA, int transB, double *A, double *B, double *C,
                   int ra, int ca, int rb, int cb, double ALPHA, double BETA) {
    /*
    dgemm computes C = alpha * op(A) * op(B) + beta * C
    op(X) = X or X'
    M: number of rows of op(A)
    N: number of columns of op(B)
    K: number of columns of op(A) / number of rows of op(B)
    LDX: the first dimension (rowMajor: ncol, colMajor: nrow) of matrix X
    LDA: the first dimension of A
    LDB: the first dimension of B
    LDC: the first dimension of C
    A: array of dimension (LDA, ka)
    B: array of dimension (LDB, kb)
    C: array of dimension (LDC, kc)*/

    cublasOperation_t TransA = (transA == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t TransB = (transB == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int M = (transB == 0) ? cb : rb;
    int N = (transA == 0) ? ra : ca;
    int K = (transA == 0) ? ca : ra;

    // Call cuBLAS with swapped matrices (B before A) to match row-major math
    cublasDgemm(cublasH, TransB, TransA, M, N, K, &ALPHA,
                B, cb, A, ca, &BETA, C, M);
}

void dgemmCupy(int transA, int transB, py::object A, py::object B, py::object C,
               int ra, int ca, int rb, int cb, double ALPHA, double BETA) {
    // Extracting CuPy array information
    double *ptrA = getCupyPtr<double>(A);
    double *ptrB = getCupyPtr<double>(B);
    double *ptrC = getCupyPtr<double>(C);

    // Create cuBLAS handle
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    dgemmSimpCuda(cublasH, transA, transB, ptrA, ptrB, ptrC, ra, ca, rb, cb, ALPHA, BETA);

    // Destroy cuBLAS handle
    cublasDestroy(cublasH);
}

void dgemmCublas(int transA, int transB, int M, int N, int K, double ALPHA,
                 py::object A, int LDA, py::object B, int LDB, double BETA,
                 py::object C, int LDC) {
    // Extracting CuPy array information
    double *ptrA = getCupyPtr<double>(A);
    double *ptrB = getCupyPtr<double>(B);
    double *ptrC = getCupyPtr<double>(C);

    // Create cuBLAS handle
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    cublasOperation_t TransA = (transA == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t TransB = (transB == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasDgemm(cublasH, TransA, TransB, M, N, K, &ALPHA, ptrA, LDA,
                ptrB, LDB, &BETA, ptrC, LDC);

    // Destroy cuBLAS handle
    cublasDestroy(cublasH);
}

void dgemvSimpCuda(cublasHandle_t cublasH, int transA, double *A, double *x, double *y,
                   int ra, int ca, double ALPHA, double BETA,
                   int incB, int incC) {
    /*
    Computes y = alpha * A * x + beta * y
    Where:
    A - device pointer to matrix A (row-major)
    x - device pointer to input vector x
    y - device pointer to output vector y
    ra  - number of rows in A
    ca  - number of columns in A
    transA - CUBLAS_OP_N (no transpose), CUBLAS_OP_T (transpose)
    ALPHA - scalar alpha
    BETA  - scalar beta
    incB  - stride of vector x
    incC  - stride of vector y
    cublasH - cuBLAS handle
    */

    int M = ca;
    int N = ra;

    cublasOperation_t TransA = (transA == 0) ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Perform the matrix-vector multiplication
    cublasDgemv(cublasH, TransA, M, N, &ALPHA, A, ca, x, incB, &BETA, y, incC);
}

void dgemvCupy(int transA, py::object A, py::object x, py::object y,
               int ra, int ca, double ALPHA, double BETA,
               int incB, int incC) {
    // Extracting CuPy array information
    double *ptrA = getCupyPtr<double>(A);
    double *ptrX = getCupyPtr<double>(x);
    double *ptrY = getCupyPtr<double>(y);

    // Create cuBLAS handle
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    dgemvSimpCuda(cublasH, transA, ptrA, ptrX, ptrY, ra, ca, ALPHA, BETA, incB, incC);

    // Destroy cuBLAS handle
    cublasDestroy(cublasH);
}

void dgerSimpCuda(cublasHandle_t cublasH, double *A, double *B, double *C,
                  int ra, int cb, double ALPHA,
                  int incA, int incB) {
    /*
    Here, we compute C = alpha * A * B.T + C
    cublas only supports Column-major storage
    m,               // Number of rows in C
    n,               // Number of columns in C
    alpha,           // Scalar multiplier
    A, 1,            // Vector A and its increment
    B, 1,            // Vector B and its increment
    C, n             // Output matrix C and its leading dimension
    */

    // cublasDger(cublasH, cb, ra, &ALPHA, B, incB, A, incA, C, cb);
    cublasDger(cublasH, cb, ra, &ALPHA, B, incB, A, incA, C, cb);
}

void dgerSimpCupy(py::object A, py::object B, py::object C,
                  int ra, int cb, double ALPHA,
                  int incA, int incB) {
    // Extracting CuPy array information
    double *ptrA = getCupyPtr<double>(A);
    double *ptrB = getCupyPtr<double>(B);
    double *ptrC = getCupyPtr<double>(C);

    // Create cuBLAS handle
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    dgerSimpCuda(cublasH, ptrA, ptrB, ptrC, ra, cb, ALPHA, incA, incB);

    // Destroy cuBLAS handle
    cublasDestroy(cublasH);
}

void copyTrans3DCupy(py::object dst, py::object src, py::array_t<int> permSrc,
                     py::array_t<int64_t> sliceRangeDst, int sliceAxisDst) {
    // Extracting CuPy array information
    double *ptrDst = getCupyPtr<double>(dst);
    double *ptrSrc = getCupyPtr<double>(src);                 

    py::tuple dstShape = dst.attr("shape");
    size_t shapeDst0 = dstShape[0].cast<size_t>();
    size_t shapeDst1 = dstShape[1].cast<size_t>();
    size_t shapeDst2 = dstShape[2].cast<size_t>();

    py::tuple srcShape = src.attr("shape");
    size_t shapeSrc0 = srcShape[0].cast<size_t>();
    size_t shapeSrc1 = srcShape[1].cast<size_t>();
    size_t shapeSrc2 = srcShape[2].cast<size_t>();

    auto permSrcPtr = permSrc.unchecked<1>();
    int permSrcX = permSrcPtr(0);
    int permSrcY = permSrcPtr(1);
    int permSrcZ = permSrcPtr(2);

    auto sliceRangeDstPtr = sliceRangeDst.unchecked<1>();
    size_t sliceStartDst = static_cast<size_t>(sliceRangeDstPtr(0));
    size_t sliceEndDst = static_cast<size_t>(sliceRangeDstPtr(1));

    size_t totalElements = shapeSrc0 * shapeSrc1 * shapeSrc2;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    

    copyTrans3DKernel<<<blocks, threadsPerBlock>>>(ptrDst, ptrSrc, 
                     shapeDst0, shapeDst1, shapeDst2,
                     shapeSrc0, shapeSrc1, shapeSrc2, 
                     permSrcX, permSrcY, permSrcZ,
                     sliceStartDst, sliceEndDst, 
                     sliceAxisDst);
}

// Bind the functions to a Python module
PYBIND11_MODULE(linalgUtilsCuda, m) {
    m.def("dgemmCupy", &dgemmCupy, "dgemmCupy",
              py::arg("transA"), py::arg("transB"), py::arg("A"),
              py::arg("B"), py::arg("C"), py::arg("ra"),
              py::arg("ca"), py::arg("rb"), py::arg("cb"),
              py::arg("ALPHA") = 1.0, py::arg("BETA") = 0.0);
    m.def("dgemmCublas", &dgemmCublas, "dgemmCublas");
    m.def("dgemvCupy", &dgemvCupy, "dgemvCupy",
          py::arg("transA"), py::arg("A"), py::arg("x"),
          py::arg("y"), py::arg("ra"), py::arg("ca"),
          py::arg("ALPHA") = 1.0, py::arg("BETA") = 0.0,
          py::arg("incB") = 1, py::arg("incC") = 1);

    m.def("dgerSimpCupy", &dgerSimpCupy, "dgemvCupy",
          py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("ra"), py::arg("cb"), py::arg("ALPHA") = 1.0,
          py::arg("incA") = 1, py::arg("incB") = 1);
    m.def("copyTrans3DCupy", &copyTrans3DCupy, "copyTrans3DCupy");
}