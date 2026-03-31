#pragma once

#include <stdexcept>


// Matrix multiplication using DGEMM
void dgemmCpp(int transA, int transB, double *A, double *B, double *C,
              int ra, int ca, int rb, int cb, double ALPHA = 1.0, double BETA = 0.0);

// Matrix-vector multiplication using DGEMV
void dgemvCpp(int transA, double *A, double *x, double *y,
              int ra, int ca, double ALPHA = 1.0, double BETA = 0.0,
              int incB = 1, int incC = 1);

// Rank-1 update using DGER
void dgerCpp(double *A, double *B, double *C,
             int ra, int cb, double ALPHA = 1.0,
             int incA = 1, int incB = 1);

