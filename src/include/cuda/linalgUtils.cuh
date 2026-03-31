#pragma once

#include <pybind11/numpy.h>          // For NumPy/CuPy array handling
#include <pybind11/pybind11.h>       // For Pybind11 bindings
#include <cuda_runtime.h>

__global__ void rowToColMajor(double *A_row, double *A_col, int m, int n);

__global__ void colToRowMajor(double *A_col, double *Q_row, int m, int n, int k);

void dgemmSimpCuda(cublasHandle_t cublasH, int transA, int transB, double *A, double *B, double *C,
                   int ra, int ca, int rb, int cb, double ALPHA = 1.0, double BETA = 0.0);

void dgemmCublas(int transA, int transB, int M, int N, int K, double ALPHA,
                 py::object A, int LDA, py::object B, int LDB, double BETA,
                 py::object C, int LDC);

void dgemvSimpCuda(cublasHandle_t cublasH, int transA, double *A, double *x, double *y,
                   int ra, int ca, double ALPHA = 1.0, double BETA = 0.0,
                   int incB = 1, int incC = 1);

void dgerSimpCuda(cublasHandle_t cublasH, double *A, double *B, double *C,
                  int ra, int cb, double ALPHA = 1.0,
                  int incA = 1, int incB = 1);