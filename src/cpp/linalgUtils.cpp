// linalgUtils.cpp
#include <cblas.h>
#include <stdexcept>
#include <iostream>

CBLAS_TRANSPOSE dgemmTrans(int trans) {
    switch (trans) {
    case 0:
        return CblasNoTrans; // No transpose
    case 1:
        return CblasTrans; // Transpose
    case 2:
        return CblasConjTrans; // conjugate transpose
    default:
        throw std::invalid_argument("Invalid transpose code (0-2)");
    }
}

void dgemmCpp(int transA, int transB, double *A, double *B, double *C,
               int ra, int ca, int rb, int cb, double ALPHA = 1.0, double BETA = 0.0) {
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
    C: array of dimension (LDC, kc)
*/

    int M = (transA == 0) ? ra : ca;
    int N = (transB == 0) ? cb : rb;
    int K = (transA == 0) ? ca : ra;

    CBLAS_TRANSPOSE TransA = dgemmTrans(transA);
    CBLAS_TRANSPOSE TransB = dgemmTrans(transB);

    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, ALPHA, A, ca, B, cb, BETA, C, N);
}


void dgemvCpp(int transA, double *A, double *x, double *y,
               int ra, int ca, double ALPHA = 1.0, double BETA = 0.0,
               int incB = 1, int incC = 1) {
    /*
    Here, we compute y = alpha * A * x + beta * y
    CblasRowMajor,   // Data layout: row-major
    CblasNoTrans,    // op(A): no transpose
    M,               // Number of rows of A.
    N,               // Number of columns of A.
    alpha,           // Scalar alpha.
    A,               // Pointer to the first matrix element.
    lda,             // Leading dimension of A.
    x,               // Input vector x.
    incX,            // Increment for x.
    beta,            // Scalar beta.
    y,               // Output vector y.
    incY             // Increment for y.
    */

    CBLAS_TRANSPOSE TransA = dgemmTrans(transA);

    cblas_dgemv(CblasRowMajor, TransA, ra, ca, ALPHA, A, ca, x, 1, BETA, y, 1);
}

void dgerCpp(double *A, double *B, double *C,
              int ra, int cb, double ALPHA = 1.0,
              int incA = 1, int incB = 1) {
    /*
    Here, we compute C = alpha * A * B + C
    CblasRowMajor,   // Row-major storage
    m,               // Number of rows in C
    n,               // Number of columns in C
    alpha,           // Scalar multiplier
    A, 1,            // Vector A and its increment
    B, 1,            // Vector B and its increment
    C, n             // Output matrix C and its leading dimension
    */

    cblas_dger(CblasRowMajor, ra, cb, ALPHA, A, incA, B, incB, C, cb);
}