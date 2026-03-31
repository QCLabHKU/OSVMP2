#include <cstdint>      // for int64_t
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

extern "C" {
#include <cblas.h>
#include <lapacke.h>
}

#include <linalgUtils.h>

namespace py = pybind11;

template <typename T>
py::array_t<T> createNumpyArray(T *dataPtr, size_t m, size_t n = 0) {
    // Create capsule to handle pointer lifetime (optional)
    py::capsule freeWhenDone(dataPtr, [](void *ptr) {
        // Empty deleter - user manages memory
    });

    std::vector<size_t> shape = (n == 0) ? std::vector<size_t>{m} : std::vector<size_t>{m, n};
    std::vector<size_t> strides = (n == 0) ? std::vector<size_t>{sizeof(T)}
                                           : std::vector<size_t>{n * sizeof(T), sizeof(T)};

    return py::array_t<T>(
        shape,       // Shape (1D array)
        strides,     // Strides (C-style contiguous)
        dataPtr,     // Pointer to data buffer
        freeWhenDone // Optional capsule for memory management
    );
}

void qrqCpp(double *A, double *Q, int m, int n) {
    // Maximal rank is used by Lapacke
    int rank = std::min(m, n);

    // Tmp Array for Lapacke
    const std::unique_ptr<double[]> tau(new double[rank]);

    // Calculate QR factorisations
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau.get());

    // Create orthogonal matrix Q (in tmpA)
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, rank, rank, A, n, tau.get());

    // Copy Q (m x rank) into position
    if (m == n) {
        memcpy(Q, A, sizeof(double) * (m * n));
    } else {
        for (int row = 0; row < m; ++row) {
            memcpy(Q + row * rank, A + row * n, sizeof(double) * (rank));
        }
    }
}

void adaptiveRandomizedRangeFinder(double *A, double **qPtr, int64_t *qSize, int m, int n, 
                                   double tolerance, int r, int maxIter, int seed = 42) {
    float ta = 0.0;
    float tb = 0.0;

    // initialize a random array
    std::random_device rd;
    //std::mt19937 gen(rd());
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0, 1); // uniform distribution between 0 and 1

    // Intialize intermediate containers
    int sizeOmega = n * r;
    double normYj;
    int idxYj = 0;
    int nsamp = 0;
    int mSize = m * sizeof(double);
    int restYSize = (r - 1) * mSize;
    double realTolerance = tolerance / (10 * sqrt(2 / 3.14159265358979323846));
    double errorApprox = realTolerance * 10;

    // Preallocate temporary buffers
    double *qBuffer = new double[m * maxIter];
    double *omega = new double[sizeOmega];
    double *omega_q = new double[sizeOmega];
    double *vomega = new double[n];
    double *y = new double[r * m];
    double *ysave = new double[r * m];
    double *yj = new double[m];
    double *qjy = new double[r];
    double *errors = new double[r];
    double *vomegaPre = new double[maxIter * n];
    double *yjnow = new double[maxIter * m];
    double *qytmp = new double[maxIter];

    std::generate(omega, omega + sizeOmega, [&]() { return dis(gen); });

    // PQ decomposition of omega
    qrqCpp(omega, omega_q, n, r);

    dgemmCpp(1, 1, omega_q, A, y, n, r, m, n);

    auto ttotal = std::chrono::high_resolution_clock::now();
    auto tstart = std::chrono::high_resolution_clock::now();

    // Pregenerate ramdomized yj
    std::generate(vomegaPre, vomegaPre + maxIter * n, [&]() { return dis(gen); });
    dgemmCpp(0, 1, vomegaPre, A, yjnow, maxIter, n, m, n);

    auto tend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = tend - tstart;
    //printf("Time for preloading: %.4f\n", duration.count());

    // for (int istep = 0; istep < maxIter; ++istep) {
    while (nsamp < maxIter && errorApprox > realTolerance) {
        tstart = std::chrono::high_resolution_clock::now();
        // normalize it and append to Q
        normYj = cblas_dnrm2(m, y, 1);
        cblas_dscal(m, 1.0 / normYj, y, 1);

        memcpy(qBuffer + idxYj, y, mSize);
        nsamp++;

        tend = std::chrono::high_resolution_clock::now();
        duration = tend - tstart;
        ta += duration.count();

        tstart = std::chrono::high_resolution_clock::now();

        // ynow - (Q * Q.T * ynow)
        dgemvCpp(0, qBuffer, yjnow + idxYj, qytmp, nsamp, m);
        dgemvCpp(1, qBuffer, qytmp, yjnow + idxYj, nsamp, m, -1.0, 1.0);

        memcpy(yj, y, mSize);
        memcpy(y, y + m, restYSize);
        memcpy(y, yjnow + idxYj, mSize);

        dgemvCpp(0, y, yj, qjy, r, m);
        dgerCpp(qjy, yj, y, r, m, -1.0, 1.0);

        // compute error of last r consecutive vecs
        for (int i = 0; i < r; ++i) {
            errors[i] = cblas_dnrm2(m, y + i * m, 1);
        }

        errorApprox = *std::max_element(errors, errors + r);

        tend = std::chrono::high_resolution_clock::now();
        duration = tend - tstart;
        tb += duration.count();

        /*reproject onto the range orthog to Q, i.e. (y_j - Q*Q.T*y_j)
        then overwrite y_j */

        tstart = std::chrono::high_resolution_clock::now();
        dgemvCpp(0, qBuffer, y, qytmp, nsamp, m);
        dgemvCpp(1, qBuffer, qytmp, y, nsamp, m, -1.0, 1.0);
        idxYj += m;
        tend = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = tend - tstart;
        ta += duration.count();
    }

    tend = std::chrono::high_resolution_clock::now();
    duration = tend - ttotal;
    //printf("Time for finding range cpp: %.4f (%.4f %.4f)\n", duration.count(), ta, tb);

    *qSize = nsamp * m;
    *qPtr = new double[*qSize];
    memcpy(*qPtr, qBuffer, *qSize * sizeof(double));

    // Free the memory
    delete[] qBuffer;
    delete[] omega;
    delete[] omega_q;
    delete[] vomega;
    delete[] y;
    delete[] ysave;
    delete[] yj;
    delete[] qjy;
    delete[] errors;
    delete[] vomegaPre;
    delete[] yjnow;
    delete[] qytmp;
}

py::array_t<double> adaptiveRandomizedRangeFinderNumpy(py::array_t<double> A,
                                                       double tolerance, int r, 
                                                       int maxIter, int seed = 42) {
    py::buffer_info aBuf = A.request();
    auto aShape = aBuf.shape;
    int m = aShape[0];
    int n = aShape[1];
    double *aPtr = static_cast<double *>(aBuf.ptr);
    double *qPtr = nullptr;
    int64_t qSize;

    adaptiveRandomizedRangeFinder(aPtr, &qPtr, &qSize, m, n, tolerance, r, maxIter, seed);

    int nCol = qSize / m;

    return createNumpyArray<double>(qPtr, nCol, m);
    /*
    py::capsule free_when_done(qPtr, [](void *f) {
        delete[] static_cast<double *>(f);
    });

    return py::array_t<double>(
        {qSize},          // Shape
        {sizeof(double)}, // Stride
        qPtr,             // Data pointer
        free_when_done);
    */
}

void svd(double *A, double *U, double *S, double *Vt, int m, int n,
         bool getU = true, bool getVt = true, bool keepA = false) {

    
    // Determine job parameters based on whether U and Vt are needed
    char jobu = getU ? 'A' : 'N';   // Compute U' (maps to Vt)
    char jobvt = getVt ? 'A' : 'N'; // Compute Vt' (maps to U)

    // Allocate superb array
    lapack_int min_mn = (m < n) ? m : n;
    double *superb = new double[min_mn - 1];
    //printf("%d %d %d\n", m, n, min_mn);

    /* LAPACK routine for SVD */
    if (keepA) {
        // To prevent LAPACKE_dgesvd from changing A
        double *ACopy = new double[m * n];
        memcpy(ACopy, A, m * n * sizeof(double));

        LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, ACopy, n, S, U, m, Vt, n, superb);

        // destroy ACopy
        delete[] ACopy;
    } else {
        //printf("%d %d\n", m, n);
        LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, A, n, S, U, m, Vt, n, superb);
    }

    delete[] superb;
}

py::tuple svdNumpy(py::array A, bool getU = true, bool getVt = true) {
    py::buffer_info bufA = A.request();
    auto shape = bufA.shape;
    int m = shape[0];
    int n = shape[1];
    int min_mn = (m < n) ? m : n;
    double *ptrA = static_cast<double *>(bufA.ptr);
    double *ptrU = new double[m*m];
    double *ptrS = new double[min_mn];
    double *ptrVt = new double[n*n];
    svd(ptrA, ptrU, ptrS, ptrVt, m, n, getU, getVt, true);
    py::array_t<double> sNpArray = createNumpyArray<double>(ptrS, min_mn);
    py::array_t<double> uNpArray = createNumpyArray<double>(ptrU, m, m);
    py::array_t<double> vtNpArray = createNumpyArray<double>(ptrVt, n, n);
    return py::make_tuple(uNpArray, sNpArray, vtNpArray);
}

void randomizedSvd(double *A, double **U, double **S, double **Vt, int m, int n,
                  int *nColPtr, double tolerance, int r, int maxIter,
                  bool getU = true, bool getVt = true, int seed = 42) {

    // Find the ranger
    double *Q;
    int64_t qSize;
    adaptiveRandomizedRangeFinder(A, &Q, &qSize, m, n, tolerance, r, maxIter, seed);

    int ncol = qSize / m;

    *nColPtr = ncol;
    double *B = new double[ncol * n];
    double *uTilde = (getU) ? new double[ncol * ncol] : nullptr;
    double *vtTilde = (getVt) ? new double[n * n] : nullptr;
    *S = new double[ncol];
    *U = (getU) ? new double[m * ncol] : nullptr;
    *Vt = (getVt) ? new double[ncol * n] : nullptr;

    dgemmCpp(0, 0, Q, A, B, ncol, m, m, n);
    
    svd(B, uTilde, *S, vtTilde, ncol, n, getU, getVt);

    if (getU) {
        dgemmCpp(1, 0, Q, uTilde, *U, ncol, m, ncol, ncol);
    }

    if (getVt) {
        memcpy(*Vt, vtTilde, ncol * n * sizeof(double));
    }

    delete[] B;
    delete[] uTilde;
    delete[] vtTilde;
    delete[] Q;
}

py::tuple randomizedSvdNumpy(py::array A, double tolerance, int maxIter, int r = 10,
                            bool getU = true, bool getVt = true, int seed = 42) {

    py::buffer_info bufA = A.request();
    auto shape = bufA.shape;
    int m = shape[0];
    int n = shape[1];
    int nCol;
    double *ptrA = static_cast<double *>(bufA.ptr);
    double *S, *U, *Vt;
    
    randomizedSvd(ptrA, &U, &S, &Vt, m, n,
                 &nCol, tolerance, r, maxIter,
                 getU, getVt, seed);

    py::array_t<double> sNpArray = createNumpyArray<double>(S, nCol);
    py::object uOut = py::none();
    py::object vtOut = py::none();

    if (getU) {
        py::array_t<double> uNpArray = createNumpyArray<double>(U, m, nCol);
        uOut = uNpArray;
    }

    if (getVt) {
        py::array_t<double> vtNpArray = createNumpyArray<double>(Vt, nCol, n);
        vtOut = vtNpArray;
    }

    return py::make_tuple(uOut, sNpArray, vtOut);
}

// Bind the functions to a Python module
// double* A, int m, int n, double tolerance, int r, int maxIter
PYBIND11_MODULE(randSvd, m) {
    m.def("adaptiveRandomizedRangeFinderNumpy", &adaptiveRandomizedRangeFinderNumpy, "adaptive randomized range finder", 
        py::arg("A"), py::arg("tolerance"), py::arg("r"), py::arg("maxIter"),
        py::arg("seed") = 42);
    m.def("svdNumpy", &svdNumpy, "svdNumpy",
          py::arg("A"), py::arg("getU") = true, py::arg("getVt") = true);
    m.def("randomizedSvdNumpy", &randomizedSvdNumpy, "randomizedSvdNumpy",
          py::arg("A"), py::arg("tolerance"), py::arg("maxIter"),
          py::arg("r") = 10, py::arg("getU") = true, py::arg("getVt") = true,
          py::arg("seed") = 42);
}