#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cupyUtils.cuh>
#include <iostream>
#include <cmath>
#include <float.h> // For DBL_MAX
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define THREADS_PER_AXIS 16
#define THREADS_PER_BLOCK 256
#define LARGE_THREADS_PER_AXIS 32
#define LARGE_THREADS_PER_BLOCK 1024
#define NUM_STREAMS 16

namespace py = pybind11;

__device__ void copyArrayDevice(double *dest, double *src, int64_t dIdx0, int64_t sIdx0, int64_t m) {
    // Calculate the thread's unique index within the block
    int64_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    // Determine the total number of threads in the block
    int64_t stride = blockDim.x * blockDim.y;

    // Iterate over elements with a stride equal to the block's thread count
    for (int64_t idx = tid; idx < m; idx += stride) {
        dest[dIdx0 + idx] = src[sIdx0 + idx];
    }
}

__global__ void osvSFKernelx(double *qMat, double *sMat, double *fMat, double *sRatio, double *eVir,
                             int *pairs, int *nOsvOcc, int64_t *qOffsets, int64_t *sfOffsets,
                             int nVir, int nOcc) {

    __shared__ double sSmatSquare[THREADS_PER_BLOCK];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Get the indices for input and out put matrices
    int pairIdx = blockIdx.x;
    int i = pairs[pairIdx] / nOcc;
    int j = pairs[pairIdx] % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];

    int64_t qIIdx0 = qOffsets[i];
    int64_t qJIdx0 = qOffsets[j];
    int64_t sfIdx0 = sfOffsets[pairIdx];

    double tSmatSquare = 0.0;
    for (uint32_t osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
            double sValue = 0.0;
            double fValue = 0.0;

            for (uint32_t k = 0; k < nVir; ++k) {
                double qq = qMat[qIIdx0 + k * nOsvI + osvIdxY] * qMat[qJIdx0 + k * nOsvJ + osvIdxX];
                sValue += qq;
                fValue += qq * eVir[k];
            }

            int64_t idxSF = sfIdx0 + osvIdxY * nOsvJ + osvIdxX;
            sMat[idxSF] = sValue;
            fMat[idxSF] = fValue;
            tSmatSquare += sValue * sValue;
        }
    }

    sSmatSquare[tid] = tSmatSquare;

    __syncthreads();

    // Perform reduction
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sSmatSquare[tid] += sSmatSquare[tid + stride];
        }
        __syncthreads();
    }

    // Write block result to output array
    if (tid == 0) {
        sRatio[pairIdx] = sSmatSquare[0] / ((nOsvI + nOsvJ) * 0.5);
    }
}

// #define MAX_NUM_VIR 32
// #define SHARED_Q_SIZE 1536
#define SF_SMEM_SIZE 5888

__global__ void osvSFKernel(double *qMat, double *sMat, double *fMat,
                            double *sRatio, double *eVir, int *pairs, int *nOsvOcc,
                            int64_t *qOffsets, int64_t *sfOffsets, int nVir, int nOcc) {

    // Shared memory for qI and qJ values
    //__shared__ double sEVir[MAX_NUM_VIR];
    //__shared__ double sQMatI[SHARED_Q_SIZE];
    //__shared__ double sQMatJ[SHARED_Q_SIZE];
    __shared__ double sharedMem[SF_SMEM_SIZE];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    // Get the indices for input and out put matrices
    int pairIdx = blockIdx.x;
    int i = pairs[pairIdx] / nOcc;
    int j = pairs[pairIdx] % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];

    int sEVirSize = SF_SMEM_SIZE / (1 + nOsvI + nOsvJ);
    int sQiSize = sEVirSize * nOsvI;
    int sQjSize = sEVirSize * nOsvJ;

    double *sEVir = sharedMem;
    double *sQMatI = sEVir + sEVirSize;
    double *sQMatJ = sQMatI + sQiSize;

    double *qMatI = qMat + qOffsets[i];
    double *qMatJ = qMat + qOffsets[j];

    int kChunks = sEVirSize;

    int64_t sfIdx0 = sfOffsets[pairIdx];
    double *sMatIj = sMat + sfIdx0;
    double *fMatIj = fMat + sfIdx0;

    for (int osvIdx = tid; osvIdx < (nOsvI * nOsvJ); osvIdx += threadsPerBlock) {
        int osvIdxY = osvIdx / nOsvJ;
        int osvIdxX = osvIdx % nOsvJ;

        int idxSF = osvIdxY * nOsvJ + osvIdxX;
        sMatIj[idxSF] = 0.0;
        fMatIj[idxSF] = 0.0;
    }

    for (int kStart = 0; kStart < nVir; kStart += kChunks) {
        int kEnd = min(kStart + kChunks, nVir);
        int nVirBatch = kEnd - kStart;
        int qMatSizeI = nVirBatch * nOsvI;
        int qMatSizeJ = nVirBatch * nOsvJ;

        for (int kIdx = tid; kIdx < nVirBatch; kIdx += threadsPerBlock) {
            sEVir[kIdx] = eVir[kIdx + kStart];
        }

        for (int qIdx = tid; qIdx < qMatSizeI; qIdx += threadsPerBlock) {
            int k = kStart + qIdx / nOsvI;
            int osvIdxI = qIdx % nOsvI;
            sQMatI[qIdx] = qMatI[k * nOsvI + osvIdxI];
        }

        for (int qIdx = tid; qIdx < qMatSizeJ; qIdx += threadsPerBlock) {
            int k = kStart + qIdx / nOsvJ;
            int osvIdxJ = qIdx % nOsvJ;
            sQMatJ[qIdx] = qMatJ[k * nOsvJ + osvIdxJ];
        }

        __syncthreads();

        // int rIdx = 0;
        for (int osvIdx = tid; osvIdx < (nOsvI * nOsvJ); osvIdx += threadsPerBlock) {
            int osvIdxY = osvIdx / nOsvJ;
            int osvIdxX = osvIdx % nOsvJ;

            double sValue = 0.0;
            double fValue = 0.0;
            for (int kIdx = 0; kIdx < nVirBatch; ++kIdx) {
                double qq = sQMatI[kIdx * nOsvI + osvIdxY] *
                            sQMatJ[kIdx * nOsvJ + osvIdxX];
                sValue += qq;
                fValue += qq * sEVir[kIdx];
            }

            int idxSF = osvIdxY * nOsvJ + osvIdxX;
            sMatIj[idxSF] += sValue;
            fMatIj[idxSF] += fValue;
        }

        __syncthreads();
    }

    // int rIdx = 0;
    double tSmatSquare = 0.0;
    for (int osvIdx = tid; osvIdx < (nOsvI * nOsvJ); osvIdx += threadsPerBlock) {
        int osvIdxY = osvIdx / nOsvJ;
        int osvIdxX = osvIdx % nOsvJ;
        double sIj = sMatIj[osvIdxY * nOsvJ + osvIdxX];
        tSmatSquare += sIj * sIj;
    }

    double *sSmatSquare = sQMatI;
    sSmatSquare[tid] = tSmatSquare;

    __syncthreads();

    // Perform reduction
    for (int stride = threadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sSmatSquare[tid] += sSmatSquare[tid + stride];
        }
        __syncthreads();
    }

    // Write block result to output array
    if (tid == 0) {
        sRatio[pairIdx] = sSmatSquare[0] / ((nOsvI + nOsvJ) * 0.5);
    }
}

__global__ void imujpKernelx(double *ialp, double *qao, double *imujp, int *fitPair,
                             int *nFitPair, int *nOsvOcc, int64_t *qaoOffsets,
                             int64_t *imujpOffsets, int *fitLocalOffsets,
                             int *occJsBlock, int *occJIndices, int *osvIndices,
                             int *fitLocalIndices, int nAo, int nAux) {

    int occJ = occJsBlock[blockIdx.x];
    int occJIdx = occJIndices[blockIdx.x];
    int nOsvJ = nOsvOcc[occJ];

    int nFitIj = nFitPair[occJIdx];

    double *qaoJ = qao + qaoOffsets[occJ];
    size_t imujpIdx0 = imujpOffsets[occJIdx];

    int fitLocalIdx = fitLocalIndices[blockIdx.x] + threadIdx.y;
    int osvIdx = osvIndices[blockIdx.x] + threadIdx.x;
    int fitIdx = fitPair[fitLocalOffsets[occJIdx] + fitLocalIdx];

    double value = 0.0;
    if (fitLocalIdx < nFitIj && osvIdx < nOsvJ) {
        for (int al = 0; al < nAo; ++al) {
            value += ialp[fitIdx * nAo + al] * qaoJ[al * nOsvJ + osvIdx];
        }
        imujp[imujpIdx0 + fitLocalIdx * nOsvJ + osvIdx] = value;
    }
}

//#define MAX_NUM_NOSV 32
#define SHARED_INT_IMUJP 3072*2

__global__ void imujpKernel(double *ialp, double *qao, double *imujp, int *fitPair,
                            int *nFitPair, int *nOsvOcc, int64_t *qaoOffsets,
                            int64_t *imujpOffsets, int *fitLocalOffsets,
                            int *occJsBlock, int *occJIndices, int *osvIndices,
                            int *fitLocalIndices, int nAo, int nAux) {

    int bid = blockIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int occJ = occJsBlock[bid];
    int occJIdx = occJIndices[bid];
    int nOsvJ = nOsvOcc[occJ];
    int nFitIj = nFitPair[occJIdx];

    int fitStartBlock = fitLocalIndices[bid];
    int nFitIjBlock = min(blockDim.y, nFitIj - fitStartBlock);
    size_t fitLocalIdx = fitStartBlock + threadIdx.y;
    int fitLocalIdxStart = fitLocalOffsets[occJIdx];

    int osvStartBlock = osvIndices[bid];
    int nOsvJBlock = min(blockDim.x, nOsvJ - osvStartBlock);
    int osvIdx = osvStartBlock + threadIdx.x;

    // Shared memory for Qao and ialp
    //__shared__ int sFitIndices[MAX_NUM_NOSV];
    //__shared__ double inputArray[SHARED_INPUT_SIZE];
    __shared__ int sFitIndices[SHARED_INT_IMUJP];
    int shmIntSize = (blockDim.y % 2) ? blockDim.y + 1 : blockDim.y;
    int shmDoubleSize = (SHARED_INT_IMUJP - shmIntSize) / 2;
    double *shmDouble = (double*)(sFitIndices + shmIntSize);
    

    int sQaoSize = shmDoubleSize / (nFitIjBlock + nOsvJBlock) * nOsvJBlock;
    int sIalpSize = shmDoubleSize - sQaoSize;

    //double *sQao = inputArray;
    double *sQao = shmDouble;
    double *sIalp = sQao + sQaoSize;

    double *qaoJ = qao + qaoOffsets[occJ];
    double *imujpPtr = imujp + imujpOffsets[occJIdx];

    // Load fitting indices to shared memory
    if (fitLocalIdx < nFitIj) {
        int fitIdx = fitPair[fitLocalIdxStart + fitLocalIdx];
        sFitIndices[threadIdx.y] = fitIdx;
    } else {
        sFitIndices[threadIdx.y] = 0.0;
    }

    __syncthreads();

    // Double buffering
    int alChunks = min(sIalpSize / nFitIjBlock, sQaoSize / nOsvJBlock) / 2;
    int nAoBatch = min(alChunks, nAo);
    int qaoSize = nAoBatch * nOsvJBlock;
    int ialpSize = nAoBatch * nFitIjBlock;

    for (int qaoIdx = tid; qaoIdx < qaoSize; qaoIdx += threadsPerBlock) {
        int al = qaoIdx / nOsvJBlock;
        size_t osvIdxNow = osvStartBlock + qaoIdx % nOsvJBlock;
        sQao[qaoIdx] = qaoJ[al * nOsvJ + osvIdxNow];
    }

    for (int ialpIdx = tid; ialpIdx < ialpSize; ialpIdx += threadsPerBlock) {
        int al = ialpIdx % nAoBatch;
        int fitIdxNow = sFitIndices[ialpIdx / nAoBatch];
        sIalp[ialpIdx] = ialp[(size_t)fitIdxNow * nAo + al];
    }

    __syncthreads();

    double value = 0.0;
    bool toCompute = (fitLocalIdx < nFitIj && osvIdx < nOsvJ);

    int alBatch = 0;
    for (int alStart = 0; alStart < nAo; alStart += alChunks) {
        int alEnd = min(alStart + alChunks, nAo);
        nAoBatch = alEnd - alStart;
        qaoSize = nAoBatch * nOsvJBlock;
        ialpSize = nAoBatch * nFitIjBlock;

        bool isEvenBatch = (alBatch % 2 == 0);
        int sQaoWriteStart = (isEvenBatch) ? sQaoSize / 2 : 0;
        int sQaoReadStart = (isEvenBatch) ? 0 : sQaoSize / 2;
        int sIalpWriteStart = (isEvenBatch) ? sIalpSize / 2 : 0;
        int sIalpReadStart = (isEvenBatch) ? 0 : sIalpSize / 2;

        // Load data for the next batch
        int alStartNext = alEnd;
        int nAoNext = min(alChunks, nAo - alStartNext);
        int qaoSizeNext = nAoNext * nOsvJBlock;
        int ialpSizeNext = nAoNext * nFitIjBlock;

        for (int qaoIdx = tid; qaoIdx < qaoSizeNext; qaoIdx += threadsPerBlock) {
            int al = alStartNext + qaoIdx / nOsvJBlock;
            size_t osvIdxNow = osvStartBlock + qaoIdx % nOsvJBlock;
            sQao[sQaoWriteStart + qaoIdx] = qaoJ[al * nOsvJ + osvIdxNow];
        }

        for (int ialpIdx = tid; ialpIdx < ialpSizeNext; ialpIdx += threadsPerBlock) {
            int al = alStartNext + ialpIdx % nAoNext;
            size_t fitIdxNow = sFitIndices[ialpIdx / nAoNext];
            sIalp[sIalpWriteStart + ialpIdx] = ialp[fitIdxNow * nAo + al];
        }

        if (toCompute) {
            for (size_t alIdx = 0; alIdx < nAoBatch; ++alIdx) {
                // value += sIalp[alIdx * nFitIjBlock + threadIdx.y] * sQao[alIdx * nOsvJBlock + threadIdx.x];
                value += sIalp[sIalpReadStart + threadIdx.y * nAoBatch + alIdx] * sQao[sQaoReadStart + alIdx * nOsvJBlock + threadIdx.x];
            }
        }

        alBatch++;
        __syncthreads();
    }

    if (toCompute) {
        imujpPtr[fitLocalIdx * nOsvJ + osvIdx] = value;
    }
}

#define MAX_KMAT_SMEM 1504

__global__ void closeOsvKmatKernel(double *imujp, double *kMat, int *pairs, int *fitPair,
                                   int *nOsvOcc, int64_t *fitLocalOffsets,
                                   int64_t *imujpOffsets, int64_t *kOffsets,
                                   int nOcc, int nAux) {

    int threadsPerBlock = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int pairIdx = blockIdx.x;
    int iPair = pairs[pairIdx];
    int i = iPair / nOcc;
    int j = iPair % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];
    int nOsvIj = nOsvI + nOsvJ;
    int64_t fitLocalIdx0 = fitLocalOffsets[pairIdx];
    int64_t fitLocalIdx1 = fitLocalOffsets[pairIdx + 1];

    double *kMatIj = kMat + kOffsets[pairIdx];

    __shared__ double sMem[MAX_KMAT_SMEM];

    for (int oIdxY = threadIdx.y; oIdxY < nOsvIj; oIdxY += blockDim.y) {
        for (int oIdxX = threadIdx.x; oIdxX < nOsvIj; oIdxX += blockDim.x) {
            kMatIj[oIdxY * nOsvIj + oIdxX] = 0.0;
        }
    }

    if (i == j) {
        double *gImuip = imujp + imujpOffsets[i * nOcc + i];
        int pChunks = MAX_KMAT_SMEM / nOsvI;

        for (int64_t pLocalStart = fitLocalIdx0; pLocalStart < fitLocalIdx1; pLocalStart += pChunks) {
            int nAuxBatch = min(pChunks, (int)(fitLocalIdx1 - pLocalStart));
            int imuipSize = nAuxBatch * nOsvI;

            for (int imuipIdx = tid; imuipIdx < imuipSize; imuipIdx += threadsPerBlock) {
                int pIdx = fitPair[pLocalStart + imuipIdx / nOsvI];
                int osvIdx = imuipIdx % nOsvI;
                sMem[imuipIdx] = gImuip[pIdx * nOsvI + osvIdx];
            }

            __syncthreads();

            for (int oIdxY = threadIdx.y; oIdxY < nOsvIj; oIdxY += blockDim.y) {
                int osvIdxY = oIdxY % nOsvI;
                for (int oIdxX = threadIdx.x; oIdxX < nOsvIj; oIdxX += blockDim.x) {
                    int osvIdxX = oIdxX % nOsvI;
                    double value = 0.0;
                    for (int bPIdx = 0; bPIdx < nAuxBatch; ++bPIdx) {
                        double imup = sMem[bPIdx * nOsvI + osvIdxY];
                        double jnup = sMem[bPIdx * nOsvI + osvIdxX];
                        value += imup * jnup;
                    }
                    kMatIj[oIdxY * nOsvIj + oIdxX] += value;
                }
            }

            __syncthreads();
        }

    } else {
        double *gImuip = imujp + imujpOffsets[i * nOcc + i];
        double *gImujp = imujp + imujpOffsets[i * nOcc + j];
        double *gJnujp = imujp + imujpOffsets[j * nOcc + j];
        double *gJnuip = imujp + imujpOffsets[j * nOcc + i];

        // II block
        {
            int pChunks = MAX_KMAT_SMEM / (2 * nOsvI);
            double *sImup = sMem;
            double *sJnup = sMem + MAX_KMAT_SMEM / 2;

            for (int64_t pLocalStart = fitLocalIdx0; pLocalStart < fitLocalIdx1; pLocalStart += pChunks) {
                int nAuxBatch = min(pChunks, (int)(fitLocalIdx1 - pLocalStart));
                int imuipSize = nAuxBatch * nOsvI;

                for (int imuipIdx = tid; imuipIdx < imuipSize; imuipIdx += threadsPerBlock) {
                    int pLocalIdx = pLocalStart + imuipIdx / nOsvI;
                    int pIdx = fitPair[pLocalIdx];
                    int osvIdx = imuipIdx % nOsvI;
                    sImup[imuipIdx] = gImuip[pIdx * nOsvI + osvIdx];
                    sJnup[imuipIdx] = gJnuip[(pLocalIdx - fitLocalIdx0) * nOsvI + osvIdx];
                }

                __syncthreads();

                for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                    for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                        double value = 0.0;
                        for (int bPIdx = 0; bPIdx < nAuxBatch; ++bPIdx) {
                            double imup = sImup[bPIdx * nOsvI + oIdxY];
                            double jnup = sJnup[bPIdx * nOsvI + oIdxX];
                            value += imup * jnup;
                        }
                        kMatIj[oIdxY * nOsvIj + oIdxX] += value;
                    }
                }

                __syncthreads();
            }
        }

        // IJ block
        {
            int pChunks = MAX_KMAT_SMEM / (nOsvI + nOsvJ);
            double *sImup = sMem;
            double *sJnup = sMem + pChunks * nOsvI;

            for (int64_t pLocalStart = fitLocalIdx0; pLocalStart < fitLocalIdx1; pLocalStart += pChunks) {
                int nAuxBatch = min(pChunks, (int)(fitLocalIdx1 - pLocalStart));
                int imupSize = nAuxBatch * nOsvI;
                int jnupSize = nAuxBatch * nOsvJ;

                for (int imupIdx = tid; imupIdx < imupSize; imupIdx += threadsPerBlock) {
                    int pLocalIdx = pLocalStart + imupIdx / nOsvI;
                    int pIdx = fitPair[pLocalIdx];
                    int osvIdx = imupIdx % nOsvI;
                    sImup[imupIdx] = gImuip[pIdx * nOsvI + osvIdx];
                }

                for (int jnupIdx = tid; jnupIdx < jnupSize; jnupIdx += threadsPerBlock) {
                    int pLocalIdx = pLocalStart + jnupIdx / nOsvJ;
                    int pIdx = fitPair[pLocalIdx];
                    int osvIdx = jnupIdx % nOsvJ;
                    sJnup[jnupIdx] = gJnujp[pIdx * nOsvJ + osvIdx];
                }

                __syncthreads();

                for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                    for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                        double value = 0.0;
                        for (int bPIdx = 0; bPIdx < nAuxBatch; ++bPIdx) {
                            double imup = sImup[bPIdx * nOsvI + oIdxY];
                            double jnup = sJnup[bPIdx * nOsvJ + oIdxX];
                            value += imup * jnup;
                        }
                        kMatIj[oIdxY * nOsvIj + oIdxX + nOsvI] += value;
                    }
                }

                __syncthreads();
            }
        }

        // JI block
        {
            int pChunks = MAX_KMAT_SMEM / (nOsvJ + nOsvI);
            double *sImup = sMem;
            double *sJnup = sMem + pChunks * nOsvJ;

            for (int64_t pLocalStart = fitLocalIdx0; pLocalStart < fitLocalIdx1; pLocalStart += pChunks) {
                int nAuxBatch = min(pChunks, (int)(fitLocalIdx1 - pLocalStart));
                int imupSize = nAuxBatch * nOsvJ;
                int jnupSize = nAuxBatch * nOsvI;

                for (int imupIdx = tid; imupIdx < imupSize; imupIdx += threadsPerBlock) {
                    int pLocalIdx = pLocalStart + imupIdx / nOsvJ;
                    int osvIdx = imupIdx % nOsvJ;
                    sImup[imupIdx] = gImujp[(pLocalIdx - fitLocalIdx0) * nOsvJ + osvIdx];
                }

                for (int jnupIdx = tid; jnupIdx < jnupSize; jnupIdx += threadsPerBlock) {
                    int pLocalIdx = pLocalStart + jnupIdx / nOsvI;
                    int osvIdx = jnupIdx % nOsvI;
                    sJnup[jnupIdx] = gJnuip[(pLocalIdx - fitLocalIdx0) * nOsvI + osvIdx];
                }

                __syncthreads();

                for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                    int osvIdxY = oIdxY + nOsvI;
                    for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                        double value = 0.0;
                        for (int bPIdx = 0; bPIdx < nAuxBatch; ++bPIdx) {
                            double imup = sImup[bPIdx * nOsvJ + oIdxY];
                            double jnup = sJnup[bPIdx * nOsvI + oIdxX];
                            value += imup * jnup;
                        }
                        kMatIj[osvIdxY * nOsvIj + oIdxX] += value;
                    }
                }

                __syncthreads();
            }
        }

        // JJ block
        {
            int pChunks = MAX_KMAT_SMEM / (2 * nOsvJ);
            double *sImup = sMem;
            double *sJnup = sMem + MAX_KMAT_SMEM / 2;

            for (int64_t pLocalStart = fitLocalIdx0; pLocalStart < fitLocalIdx1; pLocalStart += pChunks) {
                int nAuxBatch = min(pChunks, (int)(fitLocalIdx1 - pLocalStart));
                int imupSize = nAuxBatch * nOsvJ;

                for (int imupIdx = tid; imupIdx < imupSize; imupIdx += threadsPerBlock) {
                    int pLocalIdx = pLocalStart + imupIdx / nOsvJ;
                    int pIdx = fitPair[pLocalIdx];
                    int osvIdx = imupIdx % nOsvJ;
                    sImup[imupIdx] = gImujp[(pLocalIdx - fitLocalIdx0) * nOsvJ + osvIdx];
                    sJnup[imupIdx] = gJnujp[pIdx * nOsvJ + osvIdx];
                }

                __syncthreads();

                for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                    int osvIdxY = oIdxY + nOsvI;
                    for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                        double value = 0.0;
                        for (int bPIdx = 0; bPIdx < nAuxBatch; ++bPIdx) {
                            double imup = sImup[bPIdx * nOsvJ + oIdxY];
                            double jnup = sJnup[bPIdx * nOsvJ + oIdxX];
                            value += imup * jnup;
                        }
                        kMatIj[osvIdxY * nOsvIj + oIdxX + nOsvI] += value;
                    }
                }

                __syncthreads();
            }
        }
    }
}

__global__ void remoteOsvKmatKernel(double *imujp, double *kMat, int *pairs,
                                    int *fitPair, int *nOsvOcc, int64_t *fitLocalOffsets,
                                    int64_t *imujpOffsets, int64_t *kOffsets,
                                    int nOcc, int nAux) {

    int threadsPerBlock = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int pairIdx = blockIdx.x;
    int iPair = pairs[pairIdx];
    int i = iPair / nOcc;
    int j = iPair % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];
    int64_t fitLocalIdx0 = fitLocalOffsets[pairIdx];
    int64_t fitLocalIdx1 = fitLocalOffsets[pairIdx + 1];

    double *gImuip = imujp + imujpOffsets[i * nOcc + i];
    double *gJnujp = imujp + imujpOffsets[j * nOcc + j];
    double *kMatIj = kMat + kOffsets[pairIdx];

    __shared__ double sMem[MAX_KMAT_SMEM];

    for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
        for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
            kMatIj[oIdxY * nOsvJ + oIdxX] = 0.0;
        }
    }

    int pChunks = MAX_KMAT_SMEM / (nOsvI + nOsvJ);
    double *sImup = sMem;
    double *sJnup = sMem + pChunks * nOsvI;

    for (int64_t pLocalStart = fitLocalIdx0; pLocalStart < fitLocalIdx1; pLocalStart += pChunks) {
        int nAuxBatch = min(pChunks, (int)(fitLocalIdx1 - pLocalStart));
        int imupSize = nAuxBatch * nOsvI;
        int jnupSize = nAuxBatch * nOsvJ;

        for (int imupIdx = tid; imupIdx < imupSize; imupIdx += threadsPerBlock) {
            int pIdx = fitPair[pLocalStart + imupIdx / nOsvI];
            int osvIdx = imupIdx % nOsvI;
            sImup[imupIdx] = gImuip[pIdx * nOsvI + osvIdx];
        }

        for (int jnupIdx = tid; jnupIdx < jnupSize; jnupIdx += threadsPerBlock) {
            int pIdx = fitPair[pLocalStart + jnupIdx / nOsvJ];
            int osvIdx = jnupIdx % nOsvJ;
            sJnup[jnupIdx] = gJnujp[pIdx * nOsvJ + osvIdx];
        }

        __syncthreads();

        for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
            for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                double value = 0.0;
                for (int bPIdx = 0; bPIdx < nAuxBatch; ++bPIdx) {
                    double imup = sImup[bPIdx * nOsvI + oIdxY];
                    double jnup = sJnup[bPIdx * nOsvJ + oIdxX];
                    value += imup * jnup;
                }
                kMatIj[oIdxY * nOsvJ + oIdxX] += value;
            }
        }

        __syncthreads();
    }
}

__device__ void eigh1BlockDevice(const double *A, double *eigVal, double *eigVec, double *tempEigVal, double *tempEigVec, int n) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.y * blockDim.x;

    const double EPS = 1e-12;

    // Initialize tempEigVal from input (row-major)
    for (int i = threadIdx.y; i < n; i += blockDim.y) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            tempEigVal[i * n + j] = A[i * n + j];
        }
    }

    // Initialize tempEigVec to identity (eigenvectors as columns)
    for (int j = threadIdx.y; j < n; j += blockDim.y) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            tempEigVec[j * n + i] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Jacobi iterations (max 100 sweeps)
    for (int sweep = 0; sweep < 100; ++sweep) {
        int pairsKept = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                __syncthreads();

                double val_ij = tempEigVal[i * n + j];
                if (val_ij * val_ij < EPS)
                    continue;
                pairsKept += 1;

                double val_ii = tempEigVal[i * n + i];
                double val_jj = tempEigVal[j * n + j];
                double tau = (val_jj - val_ii) / (2.0 * val_ij);
                double t = (tau >= 0)
                               ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                               : -1.0 / (-tau + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                // Save 2x2 block before rotation
                double val_ii_old = val_ii;
                double val_ij_old = val_ij;
                double val_jj_old = val_jj;

                // Update all columns EXCEPT i,j
                for (int k = tid; k < n; k += threadsPerBlock) {
                    if (k != i && k != j) {
                        double val_ik = tempEigVal[i * n + k];
                        double val_jk = tempEigVal[j * n + k];
                        tempEigVal[i * n + k] = c * val_ik - s * val_jk;
                        tempEigVal[j * n + k] = s * val_ik + c * val_jk;
                        tempEigVal[k * n + i] = tempEigVal[i * n + k]; // Maintain symmetry
                        tempEigVal[k * n + j] = tempEigVal[j * n + k];
                    }
                }

                __syncthreads();

                // Update 2x2 block (i,j)
                if (tid == 0)
                    tempEigVal[i * n + i] = c * c * val_ii_old - 2.0 * c * s * val_ij_old + s * s * val_jj_old;
                if (tid == 1)
                    tempEigVal[j * n + j] = s * s * val_ii_old + 2.0 * c * s * val_ij_old + c * c * val_jj_old;
                if (tid == 2)
                    tempEigVal[i * n + j] = tempEigVal[j * n + i] = c * s * (val_ii_old - val_jj_old) + (c * c - s * s) * val_ij_old;

                // Update eigenvectors (CRITICAL: rotate columns)
                for (int k = tid; k < n; k += threadsPerBlock) {
                    double vec_ki = tempEigVec[k * n + i];
                    double vec_kj = tempEigVec[k * n + j];
                    tempEigVec[k * n + i] = c * vec_ki - s * vec_kj;
                    tempEigVec[k * n + j] = s * vec_ki + c * vec_kj;
                }
            }
        }
        if (pairsKept == 0) {
            break;
        }
    }

    __syncthreads();

    // Sort eigenvalues and eigenvectors (ascending)
    __shared__ int sSortedIndices[1024];
    __shared__ double sSortedValues[1024];

    if (tid < n) {
        sSortedValues[tid] = tempEigVal[tid * n + tid];
        sSortedIndices[tid] = tid;
    }

    __syncthreads();

    for (int k = 0; k < n; ++k) {
        __syncthreads();
        bool do_compare = ((k % 2 == 0) && (tid % 2 == 0)) || ((k % 2 == 1) && (tid % 2 == 1));
        if (do_compare && tid < n - 1) {
            int partner = tid + 1;
            if (sSortedValues[tid] > sSortedValues[partner]) {
                // Swap values
                double tmp_val = sSortedValues[tid];
                sSortedValues[tid] = sSortedValues[partner];
                sSortedValues[partner] = tmp_val;
                // Swap indices
                int tmp_idx = sSortedIndices[tid];
                sSortedIndices[tid] = sSortedIndices[partner];
                sSortedIndices[partner] = tmp_idx;
            }
        }
    }

    __syncthreads();

    if (tid < n) {
        eigVal[tid] = sSortedValues[tid];
        int newIdx = sSortedIndices[tid];
        for (int k = 0; k < n; ++k) {
            eigVec[k * n + tid] = tempEigVec[k * n + newIdx];
        }
    }
}


__device__ void printTest(double *array, int n, int index) {
    double aveValue = 0.0;
    double maxValue = -DBL_MAX;
    double minValue = DBL_MAX;

    for (int i = 0; i < n; ++i) {
        double value = array[i];
        aveValue += value;
        maxValue = (maxValue < value)? value : maxValue;
        minValue = (minValue > value)? value : maxValue;
    }

    printf("(%d) AVE: %.4e, MAX: %.4e, MIN: %.4e\n", index, aveValue / n, maxValue, minValue);
}

__device__ void printArray(double *array, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d %.4e\n", i, array[i]);
    }
}

__device__ int cutoffEighLoaded1BlockDevice(double *sMemDouble, int *sMemInt, double *eigVal, double *eigVec,
                                            double *tempEigVal, double *tempEigVec, int n, 
                                            double cutoff) {
    
    const double EPS = 1e-12;

    double *sEigVal_i = sMemDouble;
    double *sEigVal_jj = sEigVal_i + n;
    double *sEigVal_ij = sEigVal_jj + n;
    double *sEigVec_i = sEigVal_ij + n;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.y * blockDim.x;

    // Initialize tempEigVec to identity (eigenvectors as columns)
    for (int ji = tid; ji < n * n; ji += threadsPerBlock) {
        int j = ji / n;
        int i = ji % n;
        tempEigVec[j * n + i] = (i == j) ? 1.0 : 0.0;
    }

    __syncthreads();

    for (int k = tid; k < n; k += threadsPerBlock) {
        sEigVal_jj[k] = tempEigVal[k * n + k];
    }

    // Jacobi iterations (max 100 sweeps)
    for (int sweep = 0; sweep < 100; ++sweep) {
        int pairsKept = 0;
        for (int i = 0; i < n; ++i) {
            for (int k = tid; k < n; k += threadsPerBlock) {
                int ik = (i < k)? i * n + k : k * n + i;
                double val_ik = tempEigVal[ik];
                sEigVal_i[k] = val_ik;
                //sEigVec_i[k] = tempEigVec[k * n + i];
                sEigVec_i[k] = tempEigVec[i * n + k];

                if (k > i) {
                    sEigVal_ij[k] = val_ik;
                }

            }

            __syncthreads();


            for (int j = i + 1; j < n; ++j) {
                //double val_ij = tempEigVal[i * n + j];
                double val_ij = sEigVal_ij[j];//sEigVal_i[j];

                if (val_ij * val_ij < EPS) continue;
                pairsKept += 1;

                //double val_ii = sEigVal_i[i];
                double val_ii = sEigVal_jj[i];
                double val_jj = sEigVal_jj[j];//tempEigVal[j * n + j];
                
                double tau = (val_jj - val_ii) / (2.0 * val_ij);
                double t = (tau >= 0)
                            ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                            : -1.0 / (-tau + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                __syncthreads();
                
                // Update all columns EXCEPT i,j n < threadsPerBlock
                for (int k = tid; k < n; k += threadsPerBlock) {
                    if (k != i && k != j) {
                        int jk = (j < k)? j * n + k : k * n + j;
                        double val_jk = tempEigVal[jk];
                        double val_ik = sEigVal_i[k];
                        double new_val_ik = c * val_ik - s * val_jk;
                        sEigVal_i[k] = new_val_ik;
                        tempEigVal[jk] = s * val_ik + c * val_jk;
                        //tempEigVal[k * n + j] = new_val_jk;

                        if (k > j) {
                            sEigVal_ij[k] = new_val_ik;
                        }
                    }
                }

                // Update 2x2 block (i,j)
                if (tid == i % threadsPerBlock) {
                    sEigVal_jj[i] = c * c * val_ii - 2.0 * c * s * val_ij + s * s * val_jj;
                    sEigVal_i[j] = c * s * (val_ii - val_jj) + (c * c - s * s) * val_ij;
                }
                if (tid == j % threadsPerBlock) {
                    sEigVal_jj[j] = s * s * val_ii + 2.0 * c * s * val_ij + c * c * val_jj;
                }

                // Update eigenvectors (CRITICAL: rotate columns)
                for (int k = tid; k < n; k += threadsPerBlock) {
                    double vec_ki = sEigVec_i[k];
                    //double vec_kj = tempEigVec[k * n + j];
                    double vec_kj = tempEigVec[j * n + k];
                    sEigVec_i[k] = c * vec_ki - s * vec_kj;
                    //tempEigVec[k * n + j] = s * vec_ki + c * vec_kj;
                    tempEigVec[j * n + k] = s * vec_ki + c * vec_kj;
                }

                __syncthreads();

            }

            for (int k = tid; k < n; k += threadsPerBlock) {
                double val_ik = sEigVal_i[k];
                int ik = (i < k)? i * n + k : k * n + i;
                tempEigVal[ik] = val_ik;
                //tempEigVal[i * n + k] = val_ik;
                //tempEigVal[k * n + i] = val_ik;
                //tempEigVec[k * n + i] = sEigVec_i[k];
                tempEigVec[i * n + k] = sEigVec_i[k];
            }

            __syncthreads();

        }
        if (pairsKept == 0) break;
    }

    for (int k = tid; k < n; k += threadsPerBlock) {
        tempEigVal[k * n + k] = sEigVal_jj[k];
    }

    __syncthreads();

    // Sort eigenvalues and eigenvectors (ascending)
    int *sSortedIndices = sMemInt;
    double *sSortedValues = sMemDouble;

    for (int k = tid; k < n; k += threadsPerBlock) {
        sSortedValues[k] = tempEigVal[k * n + k];
        sSortedIndices[k] = k;
    }

    __syncthreads(); 
    
    if (tid == 0) sSortedIndices[n+1] = 0;

    // Perform odd-even transposition sort in shared memory (ascending order)
    for (int phase = 0; phase < n; ++phase) {
        int step = (phase % 2 == 0) ? 0 : 1;
        for (int i = tid * 2 + step; i < n - 1; i += threadsPerBlock * 2) {
            if (sSortedValues[i] > sSortedValues[i + 1]) {
                // Swap values
                double temp_val = sSortedValues[i];
                sSortedValues[i] = sSortedValues[i + 1];
                sSortedValues[i + 1] = temp_val;
                // Swap indices
                int temp_idx = sSortedIndices[i];
                sSortedIndices[i] = sSortedIndices[i + 1];
                sSortedIndices[i + 1] = temp_idx;
            }
        }
        __syncthreads();
    }

    for (int k = tid; k < n; k += threadsPerBlock) {
        if (sSortedValues[k] > cutoff) {
            if (k == 0 || sSortedValues[k - 1] <= cutoff) {
                sSortedIndices[n+1] = k;
            }
        }
    }

    __syncthreads();

    int idxCutoff = sSortedIndices[n+1];
    int nKept = n - idxCutoff;

    for (int idxKept = tid; idxKept < nKept; idxKept += threadsPerBlock) {

        int oriIdx = idxKept + idxCutoff;
        int sortedIdx = sSortedIndices[oriIdx];

        eigVal[idxKept] = sSortedValues[oriIdx];
        for (int k = 0; k < n; ++k) {
            //eigVec[k * nKept + idxKept] = tempEigVec[k * n + sortedIdx];
            eigVec[k * nKept + idxKept] = tempEigVec[sortedIdx * n + k];
        }
    }

    return nKept;
}

__device__ void cholDecompDevice(const double* A, double* L, int n, double *sMemDouble) {


    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    double *sSum = sMemDouble;
    double *sLik = sSum + max(n, threadsPerBlock);
    double *sAij = sLik + n;

    // Initialize error flag and L matrix
    for (int idx = tid; idx < n * n; idx += threadsPerBlock) {
        L[idx] = 0.0;
    }
    __syncthreads();

    for (int i = 0; i < n; i++) {
        // Load i-th row of L into shared memory
        for (int k = tid; k < i; k += threadsPerBlock) {
            sLik[k] = L[i * n + k];
        }
                               
        for (int j = tid + i; j < n; j += threadsPerBlock) {
            sAij[j] = A[i * n + j];
        }
                               
        __syncthreads();

        // Compute sum of squares for diagonal element (L[i][i])
        double sum = 0.0;
        for (int k = tid; k < i; k += threadsPerBlock) {
            double lik = sLik[k];
            sum += lik * lik;
        }
        sSum[tid] = sum;
        __syncthreads();

        // Reduce sum across all active threads
        for (int stride = threadsPerBlock / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sSum[tid] += sSum[tid + stride];
            }
            __syncthreads();
        }

        // Compute diagonal element 
        if (tid == 0) {
            double diag_val = sAij[i] - sSum[0];
            double new_lii = sqrt(diag_val);
            sLik[i] = new_lii; // Update shared memory
        }
        
        double *sLji = sSum;

        for (int j = tid + i + 1; j < n; j += threadsPerBlock) {
            sLji[j] = 0.0;
        }
        __syncthreads();

        // Off-diagonal elements

        for (int j = threadIdx.y + i + 1; j < n; j += blockDim.y) {
            double sum_off = 0.0;
            for (int k = threadIdx.x; k < i; k += blockDim.x) {
                sum_off += L[j * n + k] * sLik[k]; 
            }
            atomicAdd(&sLji[j], sum_off);
        }
        
        if (tid == 0) {
            L[i * n + i] = sLik[i];
        }
        
        __syncthreads();
        

        for (int j = tid + i + 1; j < n; j += threadsPerBlock) {
            L[j * n + i] = (sAij[j] - sLji[j]) / sLik[i];
        }
    
        __syncthreads();
        
    }
}

// Solve xA = B
__device__ void solveUppTri(const double* A, const double* B, double* x, 
                                     int n, int nrhs, double *sMemDouble, 
                                     int sMemDoubleSize) {
    
                                             
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    // Solve for each RHS column using chunks
    int iChunks = sMemDoubleSize / (2 * nrhs + n);
    double *sA = sMemDouble;
    double *sB = sA + iChunks * n;
    double *sX = sB + iChunks * nrhs;

    // Process in reverse order for upper triangular (backward substitution)
    for (int iEnd = n; iEnd > 0; iEnd -= iChunks) {
        int nIBatch = min(iChunks, iEnd);
        int iStart = iEnd - nIBatch; 
        
        int aSize = nIBatch * n;
        int bSize = nIBatch * nrhs;
        int xSize = nIBatch * nrhs;
                                             
        // Load A matrix chunk
        for (int aIdx = tid; aIdx < aSize; aIdx += threadsPerBlock) {
            int i = iStart + aIdx / n;
            int j = aIdx % n;
            sA[aIdx] = A[i * n + j];
        }
                                             
        // Load B matrix chunk
        for (int bIdx = tid; bIdx < bSize; bIdx += threadsPerBlock) {
            int i = iStart + bIdx / nrhs;
            int col = bIdx % nrhs;
            sB[bIdx] = B[i * nrhs + col];
        }
        
        // Load existing x values (for elements we haven't computed yet)
        for (int xIdx = tid; xIdx < xSize; xIdx += threadsPerBlock) {
            int i = iStart + xIdx / nrhs;
            int col = xIdx % nrhs;
            sX[xIdx] = x[i * nrhs + col];
        }
        
        __syncthreads();

        // Backward substitution (process in reverse order)
        for (int iIdx = nIBatch - 1; iIdx >= 0; iIdx--) {
            int i = iStart + iIdx;

            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                // Solve U * x = B (backward substitution)
                double sum = 0.0;
                for (int j = i + 1; j < n; j++) {
                    double xj = (j < iEnd) ? sX[(j - iStart) * nrhs + col] 
                                           : x[j * nrhs + col];
                    sum += sA[iIdx * n + j] * xj;
                }
                sX[iIdx * nrhs + col] = (sB[iIdx * nrhs + col] - sum) / sA[iIdx * n + i];
            }  
            __syncthreads(); // Wait for all threads to finish this row
        }
                                             
        __syncthreads();
        
        // Write results back to global memory
        for (int iIdx = 0; iIdx < nIBatch; iIdx++) {
            int i = iStart + iIdx;
            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                x[i * nrhs + col] = sX[iIdx * nrhs + col];
            }
        }
        __syncthreads();
    }
}

__device__ void solveUppTriATransDevice(const double* A, const double* B, double* x, 
                                     int n, int nrhs, double *sMemDouble, 
                                     int sMemDoubleSize) {
    
                                             
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    // Solve for each RHS column using chunks
    int iChunks = sMemDoubleSize / (2 * nrhs + n);
    double *sA = sMemDouble;
    double *sB = sA + iChunks * n;
    double *sX = sB + iChunks * nrhs;

    // Process in reverse order for upper triangular (backward substitution)
    for (int iEnd = n; iEnd > 0; iEnd -= iChunks) {
        int nIBatch = min(iChunks, iEnd);
        int iStart = iEnd - nIBatch; 
        
        int aSize = nIBatch * n;
        int bSize = nIBatch * nrhs;
        int xSize = nIBatch * nrhs;
                                             
        // Load A matrix chunk
        for (int aIdx = tid; aIdx < aSize; aIdx += threadsPerBlock) {
            int i = iStart + aIdx % nIBatch;
            int j = aIdx / nIBatch;
            sA[aIdx] = A[j * n + i];
        }
                                             
        // Load B matrix chunk
        for (int bIdx = tid; bIdx < bSize; bIdx += threadsPerBlock) {
            int i = iStart + bIdx / nrhs;
            int col = bIdx % nrhs;
            sB[bIdx] = B[i * nrhs + col];
        }
        
        // Load existing x values (for elements we haven't computed yet)
        for (int xIdx = tid; xIdx < xSize; xIdx += threadsPerBlock) {
            int i = iStart + xIdx / nrhs;
            int col = xIdx % nrhs;
            sX[xIdx] = x[i * nrhs + col];
        }
        
        __syncthreads();

        // Backward substitution (process in reverse order)
        for (int iIdx = nIBatch - 1; iIdx >= 0; iIdx--) {
            int i = iStart + iIdx;

            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                // Solve U * x = B (backward substitution)
                double sum = 0.0;
                for (int j = i + 1; j < n; j++) {
                    double xj = (j < iEnd) ? sX[(j - iStart) * nrhs + col] 
                                           : x[j * nrhs + col];
                    sum += sA[j * nIBatch + iIdx] * xj;
                }
                sX[iIdx * nrhs + col] = (sB[iIdx * nrhs + col] - sum) / sA[i * nIBatch + iIdx];
            }  
            __syncthreads(); // Wait for all threads to finish this row
        }
                                             
        __syncthreads();
        
        // Write results back to global memory
        for (int iIdx = 0; iIdx < nIBatch; iIdx++) {
            int i = iStart + iIdx;
            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                x[i * nrhs + col] = sX[iIdx * nrhs + col];
            }
        }
        __syncthreads();
    }
}

// Solve xA = B
__device__ void solveLowTriDevice(const double* A, const double* B, double* x, 
                                     int n, int nrhs, double *sMemDouble, 
                                     int sMemDoubleSize) {
    
                                             
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    // Solve for each RHS column
    int iChunks = sMemDoubleSize / (2 * nrhs + n);
    double *sA = sMemDouble;
    double *sB = sA + iChunks * n;
    double *sX = sB + iChunks * nrhs;

    for (int iStart = 0; iStart < n; iStart += iChunks) {
        int nIBatch = min(iChunks, n - iStart);
        int aSize = nIBatch * n;
        int bSize = nIBatch * nrhs;
        int xSize = nIBatch * nrhs;
                                             
        for (int aIdx = tid; aIdx < aSize; aIdx += threadsPerBlock) {
            int i = iStart + aIdx / n;
            int j = aIdx % n;
            sA[aIdx] = A[i * n + j];
        }
                                             
        for (int bIdx = tid; bIdx < bSize; bIdx += threadsPerBlock) {
            int i = iStart + bIdx / nrhs;
            int col = bIdx % nrhs;
            sB[bIdx] = B[i * nrhs + col];
        }
        
        for (int xIdx = tid; xIdx < xSize; xIdx += threadsPerBlock) {
            int i = iStart + xIdx / nrhs;
            int col = xIdx % nrhs;
            sX[xIdx] = x[i * nrhs + col];
        }
        
        __syncthreads();

        for (int iIdx = 0; iIdx < nIBatch; iIdx++) {
            int i = iStart + iIdx;

            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                // Solve L * x = B (forward substitution)
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    double xj = (j >= iStart) ? sX[(j - iStart) * nrhs + col] 
                                              : x[j * nrhs + col];
                    sum += sA[iIdx * n + j] * xj;
                }
                sX[iIdx * nrhs + col] = (sB[iIdx * nrhs + col] - sum) / sA[iIdx * n + i];
            }  
            
        }
                                             
        __syncthreads();
        
        for (int iIdx = 0; iIdx < nIBatch; iIdx++) {
            int i = iStart + iIdx;
            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                x[i * nrhs + col] = sX[iIdx * nrhs + col];
            }
        }
        __syncthreads();
    }
}

__device__ void solveLowTriBTransDevice(const double* A, const double* B, double* x, 
                                     int n, int nrhs, double *sMemDouble, 
                                     int sMemDoubleSize) {
    
                                             
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    // Solve for each RHS column
    int iChunks = sMemDoubleSize / (2 * nrhs + n);
    double *sA = sMemDouble;
    double *sB = sA + iChunks * n;
    double *sX = sB + iChunks * nrhs;

    for (int iStart = 0; iStart < n; iStart += iChunks) {
        int nIBatch = min(iChunks, n - iStart);
        int aSize = nIBatch * n;
        int bSize = nIBatch * nrhs;
        int xSize = nIBatch * nrhs;
                                             
        for (int aIdx = tid; aIdx < aSize; aIdx += threadsPerBlock) {
            int i = iStart + aIdx / n;
            int j = aIdx % n;
            sA[aIdx] = A[i * n + j];
        }
                                             
        for (int bIdx = tid; bIdx < bSize; bIdx += threadsPerBlock) {
            int i = iStart + bIdx % nIBatch;
            int col = bIdx / nIBatch;
            sB[bIdx] = B[col * n + i];
        }
        
        for (int xIdx = tid; xIdx < xSize; xIdx += threadsPerBlock) {
            int i = iStart + xIdx / nrhs;
            int col = xIdx % nrhs;
            sX[xIdx] = x[i * nrhs + col];
        }
        
        __syncthreads();

        for (int iIdx = 0; iIdx < nIBatch; iIdx++) {
            int i = iStart + iIdx;

            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                // Solve L * x = B (forward substitution)
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    double xj = (j >= iStart) ? sX[(j - iStart) * nrhs + col] 
                                              : x[j * nrhs + col];
                    sum += sA[iIdx * n + j] * xj;
                }
                sX[iIdx * nrhs + col] = (sB[col * nIBatch + iIdx] - sum) / sA[iIdx * n + i];
            }  
            
        }
                                             
        __syncthreads();
        
        for (int iIdx = 0; iIdx < nIBatch; iIdx++) {
            int i = iStart + iIdx;
            for (int col = tid; col < nrhs; col += threadsPerBlock) {
                x[i * nrhs + col] = sX[iIdx * nrhs + col];
            }
        }
        __syncthreads();
    }
}


__device__ void load4Blocks(double *dst, double *src, int64_t *startIds, int nOsvI, int nOsvJ) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.y * blockDim.x;

    int nOsvIj = nOsvI + nOsvJ;

    // ii block
    for (int oIdxYX = tid; oIdxYX < nOsvI * nOsvI; oIdxYX += threadsPerBlock) {
        int oIdxY = oIdxYX / nOsvI;
        int oIdxX = oIdxYX % nOsvI;
        dst[oIdxY * nOsvIj + oIdxX] = src[startIds[0] + oIdxY * nOsvI + oIdxX];
    }

    // ij block
    for (int oIdxYX = tid; oIdxYX < nOsvI * nOsvJ; oIdxYX += threadsPerBlock) {
        int oIdxY = oIdxYX / nOsvJ;
        int oIdxX = oIdxYX % nOsvJ;
        dst[oIdxY * nOsvIj + oIdxX + nOsvI] = src[startIds[1] + oIdxY * nOsvJ + oIdxX];
    }

    // ji block
    for (int oIdxYX = tid; oIdxYX < nOsvJ * nOsvI; oIdxYX += threadsPerBlock) {
        int oIdxY = oIdxYX / nOsvI;
        int oIdxX = oIdxYX % nOsvI;
        dst[(oIdxY + nOsvI) * nOsvIj + oIdxX] = src[startIds[1] + oIdxX * nOsvJ + oIdxY];
    }

    // jj block
    for (int oIdxYX = tid; oIdxYX < nOsvJ * nOsvJ; oIdxYX += threadsPerBlock) {
        int oIdxY = oIdxYX / nOsvJ;
        int oIdxX = oIdxYX % nOsvJ;
        dst[(oIdxY + nOsvI) * nOsvIj + oIdxX + nOsvI] = src[startIds[2] + oIdxX * nOsvJ + oIdxY];
    }
}

//#define MAX_NOSV 320
//#define SMEM_DOUBLE_CLOSE_PRECON 3840
#define SMEM_CLOSE_PRECON 32000

__global__ void closeOsvPreconKernel(double *sMat, double *fMat, double *eOcc, double *xMat,
                                     double *emuij, double *tempA, double *tempB, double *tempC,
                                     int *pairs, int *extPairIndices, int *nOsvOcc, int *nColXmat,
                                     int64_t *sfOffsets, int64_t *xOffsets, int nOcc, int shmIntSize) {
                                    

    //__shared__ double sMemDouble[SMEM_DOUBLE_CLOSE_PRECON];
    //__shared__ int sMemInt[2*MAX_NOSV];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.y * blockDim.x;

    int iPair = pairs[blockIdx.x];
    int i = iPair / nOcc;
    int j = iPair % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];
    int nOsvIj = nOsvI + nOsvJ;


    //__shared__ int sMemInt[SMEM_INT_CLOSE_PRECON];
    extern __shared__ int sMemInt[];
    int sMemIntSize = (nOsvIj % 2)? nOsvIj + 1 : nOsvIj + 2;
    int sMemDoubleSize = (shmIntSize - sMemIntSize) / 2;
    double *sMemDouble = (double*)(sMemInt + sMemIntSize);

    nColXmat[blockIdx.x] = nOsvIj;

    int64_t xMatStart = xOffsets[blockIdx.x];
    double *xMatIj = xMat + xMatStart;
    double *emuijPair = emuij + xMatStart;
    double *temAIj = tempA + xMatStart;
    double *temBIj = tempB + xMatStart;
    double *temCIj = tempC + xMatStart;

    // Construct 4-block sMat
    int64_t sfStarts[3]; // ii, ij, jj
    sfStarts[0] = sfOffsets[extPairIndices[i * nOcc + i]];
    sfStarts[1] = sfOffsets[extPairIndices[i * nOcc + j]];
    sfStarts[2] = sfOffsets[extPairIndices[j * nOcc + j]];
    
    load4Blocks(temAIj, sMat, sfStarts, nOsvI, nOsvJ);

    __syncthreads();

    // Enforce S to be positive definite
    for (int osvIdx = tid; osvIdx < nOsvIj; osvIdx += threadsPerBlock) {
        temAIj[osvIdx * nOsvIj + osvIdx] += 1e-10;
    }

    //L = np.linalg.cholesky(S)
    // emuijPair = np.linalg.cholesky(temAIj)
    cholDecompDevice(temAIj, emuijPair, nOsvIj, sMemDouble);

    load4Blocks(temAIj, fMat, sfStarts, nOsvI, nOsvJ);

    __syncthreads();
    
    //B = scipy.linalg.solve_triangular(L, F, lower=True)
    // temBIj = scipy.linalg.solve_triangular(emuijPair, temAIj)
    solveLowTriDevice(emuijPair, temAIj, temBIj, nOsvIj, nOsvIj, 
                         sMemDouble, sMemDoubleSize);
    
    //C = scipy.linalg.solve_triangular(L, B.conj().T, lower=True)
    // temAIj = scipy.linalg.solve_triangular(emuijPair, temBIj.conj().T)
    solveLowTriBTransDevice(emuijPair, temBIj, temAIj, nOsvIj, nOsvIj, 
                                    sMemDouble, sMemDoubleSize);
    
    // Solve standard Hermitian eigenvalue problem
    //eigenvalues, Y = np.linalg.eigh(C)
    // temBIj, temCIj = np.linalg.eigh(temAIj)
    cutoffEighLoaded1BlockDevice(sMemDouble, sMemInt, temBIj, temCIj, temAIj, xMatIj, nOsvIj, -DBL_MAX);

    __syncthreads();
    
    //X = scipy.linalg.solve_triangular(L.conj().T, Y, lower=False)
    // xMatIj = scipy.linalg.solve_triangular(emuijPair.conj().T, temCIj, lower=False)
    solveUppTriATransDevice(emuijPair, temCIj, xMatIj, nOsvIj, nOsvIj,
                                    sMemDouble, sMemDoubleSize);

    // eij = eo_gpu[i] + eo_gpu[j]
    // eab = eigval + eigval.reshape(-1, 1)
    // effective_d = 1.0 / (eij - eab)
    double eij = eOcc[i] + eOcc[j];

    for (int oIdxYX = tid; oIdxYX < nOsvIj * nOsvIj; oIdxYX += threadsPerBlock) {
        int oIdxY = oIdxYX / nOsvIj;
        int oIdxX = oIdxYX % nOsvIj;

        double eab = temBIj[oIdxY] + temBIj[oIdxX];
        emuijPair[oIdxY * nOsvIj + oIdxX] = 1.0 / (eij - eab);
    }
}
#define SMEM_REMOTE_PRECON 19200
__global__ void remoteOsvPreconKernel(double *fMat, double *xMat, double *emui, 
                                      double *tempMat, int *nOsvBlock, 
                                      int64_t *fiiOffsets, int64_t *eOffsets) {
    
    int nOsvI = nOsvBlock[blockIdx.x];
    
    //__shared__ double sMemDouble[4*MAX_NOSV];
    //__shared__ int sMemInt[MAX_NOSV + 1];

    extern __shared__ int sMemInt[];
    int intSize = (nOsvI % 2)? nOsvI + 1 : nOsvI;
    double *sMemDouble = (double*)(sMemInt + intSize);

    int64_t fiiStart = fiiOffsets[blockIdx.x];
    double *fMatIi = fMat + fiiStart;
    double *xMatIi = xMat + fiiStart;
    double *emuIi = emui + eOffsets[blockIdx.x];
    double *tempIi = tempMat + fiiStart;

    // emui, xii = cupy.linalg.eigh(f_ii)
    cutoffEighLoaded1BlockDevice(sMemDouble, sMemInt, emuIi, xMatIi, fMatIi, tempIi, nOsvI, 0.0);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define CLOSE_RES_SMEM 4000

__device__ void closeRmatDevice(double *sMat, double *fMat, double *tMat, double *rMat,
                                double *tempMatA, double *tempMatB, double *locFock,
                                int *nOsvOcc, int *kOccPair,
                                int64_t *rOffsets, int64_t *tOffsets,
                                int64_t *sfOffsets, int64_t *tempOffsets,
                                int64_t *kOccOffsets, int nOcc, double kOccTol, 
                                int pairIdx, int iPair, double *sMem) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int i = iPair / nOcc;
    int j = iPair % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];
    int nOsvIj = nOsvI + nOsvJ;

    //int64_t rIdx0 = rOffsets[pairIdx];
    double *rMatIj = rMat + rOffsets[pairIdx];
    int64_t tempIdx0 = tempOffsets[pairIdx];

    int64_t kOccIdx0 = kOccOffsets[pairIdx];
    int64_t kOccIdx1 = kOccOffsets[pairIdx + 1];

    int64_t sfStartIi = sfOffsets[i * nOcc + i];
    int64_t sfStartIj = sfOffsets[i * nOcc + j];
    int64_t sfStartJj = sfOffsets[j * nOcc + j];

    for (int64_t kOccIdx = kOccIdx0; kOccIdx < kOccIdx1; ++kOccIdx) {
        int k = kOccPair[kOccIdx];
        double locFockKj = locFock[k * nOcc + j];
        double locFockIk = locFock[i * nOcc + k];
        bool kEqualsJ = (k == j);
        bool kExceedsJ = (k > j);
        bool iEqualsK = (i == k);
        bool iExceedsK = (i > k);
        int nOsvK = nOsvOcc[k];

        int pairIk = (iExceedsK) ? k * nOcc + i : i * nOcc + k;
        int pairKj = (kExceedsJ) ? j * nOcc + k : k * nOcc + j;

        int64_t sfStartIk = sfOffsets[pairIk];
        int64_t sfStartKj = sfOffsets[pairKj];

        if (fabsf(locFockKj) > kOccTol) {
            int nOsvIk = nOsvI + nOsvK;
            //int64_t tIkIdx0 = (iExceedsK) ? tOffsets[k * nOcc + i] : tOffsets[i * nOcc + k];
            double *gTMatIk = tMat + tOffsets[pairIk];

            // Compute tempMatA: bMat = -locFockKj * sMatIkij + deltaKj * fMatIjij
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIk; osvIdxY += blockDim.y) {
                int oIdxY, occY, nOsvY;
                if (osvIdxY < nOsvI) {
                    oIdxY = osvIdxY;
                    occY = i;
                    nOsvY = nOsvI;
                } else {
                    oIdxY = osvIdxY - nOsvI;
                    occY = k;
                    nOsvY = nOsvK;
                }

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    int oIdxX, occX, nOsvX;
                    if (osvIdxX < nOsvI) {
                        oIdxX = osvIdxX;
                        occX = i;
                        nOsvX = nOsvI;
                    } else {
                        oIdxX = osvIdxX - nOsvI;
                        occX = j;
                        nOsvX = nOsvJ;
                    }

                    int64_t sIdx0;
                    double sMatIkij;
                    if (occY > occX) {
                        sIdx0 = sfOffsets[occX * nOcc + occY];
                        sMatIkij = sMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                    } else {
                        sIdx0 = sfOffsets[occY * nOcc + occX];
                        sMatIkij = sMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                    }

                    if (kEqualsJ) {
                        double fMatIjij;
                        if (occY > occX) {
                            fMatIjij = fMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                        } else {
                            fMatIjij = fMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                        }
                        tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = -locFockKj * sMatIkij + fMatIjij;
                    } else {
                        tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = -locFockKj * sMatIkij;
                    }
                }
            }*/

            // Quadrant 1: oIdxY = i, oIdxX = i
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartIi + oIdxY * nOsvI + oIdxX;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 2: oIdxY = i, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartIj + oIdxY * nOsvJ + oIdxX;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 3: oIdxY = k, oIdxX = i
            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    int64_t sfIdx = (iExceedsK) ? sfStartIk + oIdxY * nOsvI + oIdxX
                                                : sfStartIk + oIdxX * nOsvK + oIdxY;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 4: oIdxY = k, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = (kExceedsJ) ? sfStartKj + oIdxX * nOsvK + oIdxY
                                                : sfStartKj + oIdxY * nOsvJ + oIdxX;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx];
                    }
                }
            }



            // Compute tempMatB: sMatIjik * tMatIk
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                int sMatOIdxY, occY, nOsvY;
                if (osvIdxY < nOsvI) {
                    sMatOIdxY = osvIdxY;
                    occY = i;
                    nOsvY = nOsvI;
                } else {
                    sMatOIdxY = osvIdxY - nOsvI;
                    occY = j;
                    nOsvY = nOsvJ;
                }

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIk; osvIdxX += blockDim.x) {
                    int tMatOIdxX = (osvIdxX < nOsvI) ? osvIdxX + nOsvK : osvIdxX - nOsvI;
                    double value = 0.0;

                    for (int osvIdxZ = 0; osvIdxZ < nOsvIk; ++osvIdxZ) {
                        int sMatOIdxZ, tMatOIdxZ, occZ, nOsvZ;
                        if (osvIdxZ < nOsvI) {
                            sMatOIdxZ = osvIdxZ;
                            tMatOIdxZ = osvIdxZ + nOsvK;
                            occZ = i;
                            nOsvZ = nOsvI;
                        } else {
                            sMatOIdxZ = osvIdxZ - nOsvI;
                            tMatOIdxZ = osvIdxZ - nOsvI;
                            occZ = k;
                            nOsvZ = nOsvK;
                        }

                        double sMatIjik;
                        if (occY > occZ) {
                            int64_t sIdx0 = sfOffsets[occZ * nOcc + occY];
                            sMatIjik = sMat[sIdx0 + sMatOIdxZ * nOsvY + sMatOIdxY];
                        } else {
                            int64_t sIdx0 = sfOffsets[occY * nOcc + occZ];
                            sMatIjik = sMat[sIdx0 + sMatOIdxY * nOsvZ + sMatOIdxZ];
                        }

                        double tMatIk = (iExceedsK) ? gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ]
                                                    : gTMatIk[osvIdxZ * nOsvIk + osvIdxX];
                        value += sMatIjik * tMatIk;
                    }

                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] = value;
                }
            }

            __syncthreads();*/

            for (int oIdxY = threadIdx.y; oIdxY < nOsvIj; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvIk; oIdxX += blockDim.x) {
                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] = 0.0;
                }
            }

            __syncthreads();
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 1           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            {
                int maxOsvZBatch = CLOSE_RES_SMEM / nOsvI / 2;
                double *sSMat = sMem;
                double *sTMat = sSMat + CLOSE_RES_SMEM / 2;
                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bSMatSize;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIi + oIdxY * nOsvI + oIdxZ];
                        }

                        if (iExceedsK) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {

                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bSMatSize;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxY];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvI + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxY * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 2           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (nOsvI + nOsvK);
                int sSMatSize = maxOsvZBatch * nOsvI;
                double *sSMat = sMem;
                double *sTMat = sSMat + sSMatSize;

                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        // sMat[sfStartIi + oIdxY * nOsvI + oIdxZ];
                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIi + oIdxY * nOsvI + oIdxZ];
                        }

                        if (iExceedsK) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxY];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvI + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxY * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 3           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (nOsvJ + nOsvI);
                int sSMatSize = maxOsvZBatch * nOsvJ;

                double *sSMat = sMem;
                double *sTMat = sSMat + sSMatSize;

                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvI;

                        // sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        }

                        if (iExceedsK) {
                            // tIkIdx0 + (oIdxX + nOsvK) * nOsvIk + oIdxZ + nOsvK
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + oIdxZ * nOsvIk + oIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvI;

                        if (kExceedsJ) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxY * nOsvK + oIdxZ];
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxY];
                            }
                        }

                        if (iExceedsK) {
                            // tIkIdx0 + tMatOIdxX * nOsvIk + oIdxZ
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + (oIdxZ + nOsvI) * nOsvIk + oIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 4           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (nOsvJ + nOsvK);
                int sSMatSize = maxOsvZBatch * nOsvJ;

                double *sSMat = sMem;
                double *sTMat = sSMat + sSMatSize;

                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;

                        // sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        }

                        if (iExceedsK) {
                            // tIkIdx0 + oIdxX * nOsvIk + oIdxZ + nOsvK
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + oIdxZ * nOsvIk + osvIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;

                        if (kExceedsJ) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxY * nOsvK + oIdxZ];
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxY];
                            }
                        }

                        if (iExceedsK) {
                            
                            // tIkIdx0 + oIdxX * nOsvIk + oIdxZ
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + (oIdxZ + nOsvI) * nOsvIk + osvIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            // Accumulate to rMat: tempMatB * tempMatA
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < nOsvIk; ++osvIdxZ) {
                        value += tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxZ] * tempMatA[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }
                    rMat[rIdx0 + osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }
            __syncthreads();*/

            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (2 * nOsvIj);
                int sAMatSize = maxOsvZBatch * nOsvIj;

                double *sAMat = sMem;
                double *sBMat = sAMat + sAMatSize;

                // Z part 1: occZ = i
                for (int oIdxZStart = 0; oIdxZStart < nOsvIk; oIdxZStart += maxOsvZBatch) {
                    int bNOsvZ = min(maxOsvZBatch, nOsvIk - oIdxZStart);
                    int bAMatSize = bNOsvZ * nOsvIj;
                    int bBMatSize = bAMatSize;

                    for (int bBMatIdx = tid; bBMatIdx < bBMatSize; bBMatIdx += threadsPerBlock) {
                        int osvIdxY = bBMatIdx / bNOsvZ;
                        int osvIdxZ = oIdxZStart + bBMatIdx % bNOsvZ;
                        sBMat[bBMatIdx] = tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxZ];
                    }

                    for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                        int osvIdxZ = oIdxZStart + bAMatIdx / nOsvIj;
                        int osvIdxX = bAMatIdx % nOsvIj;
                        sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }

                    __syncthreads();

                    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                            double value = 0.0;
                            for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                                value += sBMat[osvIdxY * bNOsvZ + osvIdxZ] * sAMat[osvIdxZ * nOsvIj + osvIdxX];
                            }
                            rMatIj[osvIdxY * nOsvIj + osvIdxX] += value;
                        }
                    }

                    __syncthreads();
                }
            }

        }

        

        if (fabsf(locFockIk) > kOccTol) {
            int nOsvKj = nOsvK + nOsvJ;
            //int64_t tKjIdx0 = (kExceedsJ) ? tOffsets[j * nOcc + k] : tOffsets[k * nOcc + j];
            double *gTMatKj = tMat + tOffsets[pairKj];

            // Compute tempMatA: bMat = -locFockIk * sMatIjkj + deltaIk * fMatIjij
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                int oIdxY, occY, nOsvY;
                if (osvIdxY < nOsvI) {
                    oIdxY = osvIdxY;
                    occY = i;
                    nOsvY = nOsvI;
                } else {
                    oIdxY = osvIdxY - nOsvI;
                    occY = j;
                    nOsvY = nOsvJ;
                }

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvKj; osvIdxX += blockDim.x) {
                    int oIdxX, occX, nOsvX;
                    if (osvIdxX < nOsvK) {
                        oIdxX = osvIdxX;
                        occX = k;
                        nOsvX = nOsvK;
                    } else {
                        oIdxX = osvIdxX - nOsvK;
                        occX = j;
                        nOsvX = nOsvJ;
                    }

                    int64_t sIdx0;
                    double sMatIjkj;
                    if (occY > occX) {
                        sIdx0 = sfOffsets[occX * nOcc + occY];
                        sMatIjkj = sMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                    } else {
                        sIdx0 = sfOffsets[occY * nOcc + occX];
                        sMatIjkj = sMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                    }

                    if (iEqualsK) {
                        double fMatIjij;
                        if (occY > occX) {
                            fMatIjij = fMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                        } else {
                            fMatIjij = fMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                        }
                        tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxX] = -locFockIk * sMatIjkj + fMatIjij;
                    } else {
                        tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxX] = -locFockIk * sMatIjkj;
                    }
                }
            }*/

            // Quadrant 1: oIdxY = i, oIdxX = k
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                    
                    int64_t sfIdx = (iExceedsK) ? sfStartIk + oIdxX * nOsvI + oIdxY
                                                : sfStartIk + oIdxY * nOsvK + oIdxX;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 2: oIdxY = i, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartIj + oIdxY * nOsvJ + oIdxX;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 3: oIdxY = j, oIdxX = k
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                    int64_t sfIdx = (kExceedsJ) ? sfStartKj + oIdxY * nOsvK + oIdxX
                                                : sfStartKj + oIdxX * nOsvJ + oIdxY;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 4: oIdxY = j, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartJj + oIdxY * nOsvJ + oIdxX;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvKj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    tempMatB[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = 0.0;
                }
            }

            __syncthreads();

            // Compute tempMatB: tMatKj * sMatKjij
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvKj; osvIdxY += blockDim.y) {
                int tMatOIdxY = (osvIdxY < nOsvK) ? osvIdxY + nOsvJ : osvIdxY - nOsvK;

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    int sMatOIdxX, occX, nOsvX;
                    if (osvIdxX < nOsvI) {
                        sMatOIdxX = osvIdxX;
                        occX = i;
                        nOsvX = nOsvI;
                    } else {
                        sMatOIdxX = osvIdxX - nOsvI;
                        occX = j;
                        nOsvX = nOsvJ;
                    }

                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < nOsvKj; ++osvIdxZ) {
                        int sMatOIdxZ, tMatOIdxZ, occZ, nOsvZ;
                        if (osvIdxZ < nOsvK) {
                            sMatOIdxZ = osvIdxZ;
                            tMatOIdxZ = osvIdxZ + nOsvJ;
                            occZ = k;
                            nOsvZ = nOsvK;
                        } else {
                            sMatOIdxZ = osvIdxZ - nOsvK;
                            tMatOIdxZ = osvIdxZ - nOsvK;
                            occZ = j;
                            nOsvZ = nOsvJ;
                        }

                        double tMatKj = (kExceedsJ) ? gTMatKj[tMatOIdxZ * nOsvKj + tMatOIdxY]
                                                    : gTMatKj[osvIdxY * nOsvKj + osvIdxZ];

                        double sMatKjij;
                        if (occZ > occX) {
                            int64_t sIdx0 = sfOffsets[occX * nOcc + occZ];
                            sMatKjij = sMat[sIdx0 + sMatOIdxX * nOsvZ + sMatOIdxZ];
                        } else {
                            int64_t sIdx0 = sfOffsets[occZ * nOcc + occX];
                            sMatKjij = sMat[sIdx0 + sMatOIdxZ * nOsvX + sMatOIdxX];
                        }

                        value += tMatKj * sMatKjij;
                    }

                    tempMatB[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = value;
                }
            }

            __syncthreads();*/

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 1           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (nOsvK + nOsvI);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvI;
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxX];
                            }
                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxX * nOsvK + oIdxZ];
                            }
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }

                        

                        __syncthreads();
                    }
                }

                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxX * nOsvJ + oIdxZ];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }
            
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 2           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (nOsvK + nOsvJ);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvJ;
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;


                        if (kExceedsJ) {

                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxX * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {

                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxX];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {

                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }

                        

                        __syncthreads();
                    }
                }



                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartJj + oIdxZ * nOsvJ + oIdxX];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 3           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (nOsvJ + nOsvI);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvI;

                
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvJ;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxX];
                            }
                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxX * nOsvK + oIdxZ];
                            }
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }
                        
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvJ;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxX * nOsvJ + oIdxZ];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 4           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (2 * nOsvJ);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvJ;

                
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bSMatSize;


                        if (kExceedsJ) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxX * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {

                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxX];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }
                        
                        __syncthreads();
                    }
                }


                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bSMatSize;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartJj + oIdxZ * nOsvJ + oIdxX];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }

            // Accumulate to rMat: tempMatA * tempMatB
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < nOsvKj; ++osvIdxZ) {
                        value += tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxZ] * tempMatB[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }
                    rMatIj[osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }

            __syncthreads();*/

            {
                int maxOsvZBatch = CLOSE_RES_SMEM / (2 * nOsvIj);
                int sAMatSize = maxOsvZBatch * nOsvIj;

                double *sAMat = sMem;
                double *sBMat = sAMat + sAMatSize;

                // Z part 1: occZ = i
                for (int oIdxZStart = 0; oIdxZStart < nOsvKj; oIdxZStart += maxOsvZBatch) {
                    int bNOsvZ = min(maxOsvZBatch, nOsvKj - oIdxZStart);
                    int bAMatSize = bNOsvZ * nOsvIj;
                    int bBMatSize = bAMatSize;

                    for (int bBMatIdx = tid; bBMatIdx < bBMatSize; bBMatIdx += threadsPerBlock) {
                        int osvIdxX = bBMatIdx % nOsvIj;
                        int osvIdxZ = oIdxZStart + bBMatIdx / nOsvIj;
                        sBMat[bBMatIdx] = tempMatB[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }

                    for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                        int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                        int osvIdxY = bAMatIdx / bNOsvZ;
                        sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxZ];
                    }

                    __syncthreads();

                    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                            double value = 0.0;
                            for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                                value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sBMat[osvIdxZ * nOsvIj + osvIdxX];
                            }
                            rMatIj[osvIdxY * nOsvIj + osvIdxX] += value;
                        }
                    }

                    __syncthreads();
                }
            }
        }
    }
}

__device__ void updateCloseTMatDevice(double *tMat, double *rMat, double *xMat, double *emuij, double *tempMatA,
                                      int64_t *tempOffsets, int64_t *tOffsets, int64_t *rOffsets, int64_t *xOffsetsFull,
                                      int64_t *eOffsets, int *nOsvOcc, int nOcc,
                                      int pairIdx, int iPair, double *sMem) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int i = iPair / nOcc;
    int j = iPair % nOcc;
    int nOsvIj = nOsvOcc[i] + nOsvOcc[j];

    int64_t tIdx0 = tOffsets[iPair];
    int64_t rIdx0 = rOffsets[pairIdx];
    int64_t xIdx0 = xOffsetsFull[iPair * 2];
    int64_t xIdx1 = xOffsetsFull[iPair * 2 + 1];
    int nColXMat = (xIdx1 - xIdx0) / nOsvIj;
    int64_t eIdx0 = eOffsets[iPair];

    int64_t tempIdx0 = tempOffsets[pairIdx];

    // aMat = xij.T * rij
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvIj; ++osvIdxZ) {
                value += xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxY] * rMat[rIdx0 + osvIdxZ * nOsvIj + osvIdxX];
            }
            tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = value;
        }
    }

    __syncthreads();*/

    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = 0.0;
            }
        }

        int maxOsvZBatch = CLOSE_RES_SMEM / (nColXMat + nOsvIj);
        int sXMatSize = maxOsvZBatch * nColXMat;

        double *sXMat = sMem;
        double *sRMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvIj; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvIj - oIdxZStart);
            int bXMatSize = bNOsvZ * nColXMat;
            int bRMatSize = bNOsvZ * nOsvIj;

            for (int bRMatIdx = tid; bRMatIdx < bRMatSize; bRMatIdx += threadsPerBlock) {
                int osvIdxX = bRMatIdx % nOsvIj;
                int osvIdxZ = oIdxZStart + bRMatIdx / nOsvIj;
                sRMat[bRMatIdx] = rMat[rIdx0 + osvIdxZ * nOsvIj + osvIdxX];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx / nColXMat;
                int osvIdxY = bXMatIdx % nColXMat;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxY];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sXMat[osvIdxZ * nColXMat + osvIdxY] * sRMat[osvIdxZ * nOsvIj + osvIdxX];
                    }
                    tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    // r'ij = emuij * (aMat * xij)
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvIj; ++osvIdxZ) {
                value += tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxZ] * xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }

            int idx = osvIdxY * nColXMat + osvIdxX;
            rMat[rIdx0 + idx] = emuij[eIdx0 + idx] * value;
        }
    }

    __syncthreads();*/

    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                rMat[rIdx0 + osvIdxY * nColXMat + osvIdxX] = 0.0;
            }
        }

        int maxOsvZBatch = CLOSE_RES_SMEM / (2 * nColXMat);
        int sXMatSize = maxOsvZBatch * nColXMat;

        double *sXMat = sMem;
        double *sAMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvIj; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvIj - oIdxZStart);
            int bXMatSize = bNOsvZ * nColXMat;
            int bAMatSize = bXMatSize;

            for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                int osvIdxY = bAMatIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxZ];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx / nColXMat;
                int osvIdxX = bXMatIdx % nColXMat;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sXMat[osvIdxZ * nColXMat + osvIdxX];
                    }
                    rMat[rIdx0 + osvIdxY * nColXMat + osvIdxX] += value;
                }
            }

            __syncthreads();
        }

        for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                int idx = osvIdxY * nColXMat + osvIdxX;
                rMat[rIdx0 + idx] *= emuij[eIdx0 + idx];
            }
        }
    }


    __syncthreads();

    // aMat = xij * r'ij
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nColXMat; ++osvIdxZ) {
                value += xMat[xIdx0 + osvIdxY * nColXMat + osvIdxZ] * rMat[rIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }
            tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxX] = value;
        }
    }

    __syncthreads();*/

    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxX] = 0.0;
            }
        }
        int maxOsvZBatch = CLOSE_RES_SMEM / (nColXMat + nOsvIj);
        int sXMatSize = maxOsvZBatch * nOsvIj;

        double *sXMat = sMem;
        double *sRMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nColXMat; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nColXMat - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvIj;
            int bRMatSize = bNOsvZ * nColXMat;

            for (int bRMatIdx = tid; bRMatIdx < bRMatSize; bRMatIdx += threadsPerBlock) {
                int osvIdxX = bRMatIdx % nColXMat;
                int osvIdxZ = oIdxZStart + bRMatIdx / nColXMat;
                sRMat[bRMatIdx] = rMat[rIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx % bNOsvZ;
                int osvIdxY = bXMatIdx / bNOsvZ;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxY * nColXMat + osvIdxZ];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sXMat[osvIdxY * bNOsvZ + osvIdxZ] * sRMat[osvIdxZ * nColXMat + osvIdxX];
                    }
                    tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    // tMat += (aMat * xij.T)
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nColXMat; ++osvIdxZ) {
                value += tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxZ] * xMat[xIdx0 + osvIdxX * nColXMat + osvIdxZ];
            }

            tMat[tIdx0 + osvIdxY * nOsvIj + osvIdxX] += value;
        }
    }

    __syncthreads();*/

    {
        int maxOsvZBatch = CLOSE_RES_SMEM / (2 * nOsvIj);
        int sXMatSize = maxOsvZBatch * nOsvIj;

        double *sXMat = sMem;
        double *sAMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nColXMat; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nColXMat - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvIj;
            int bAMatSize = bXMatSize;

            for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                int osvIdxY = bAMatIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxZ];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx % bNOsvZ;
                int osvIdxX = bXMatIdx / bNOsvZ;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxX * nColXMat + osvIdxZ];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sXMat[osvIdxX * bNOsvZ + osvIdxZ];
                    }
                    tMat[tIdx0 + osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }
}


__device__ void closeMp2EneDevice(double *tMat, double *kMat, double *pairEnergies, int64_t *tOffsets,
                                  int64_t *kOffsets, int *nOsvOcc, int nOcc, int pairIdx,
                                  int iPair, double *sMem) {

    //__shared__ double sPairEne[THREADS_PER_BLOCK];
    double *sPairEne = sMem;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int i = iPair / nOcc;
    int j = iPair % nOcc;
    int nOsvIj = nOsvOcc[i] + nOsvOcc[j];

    int64_t tIdx0 = tOffsets[iPair];
    int64_t kIdx0 = kOffsets[iPair];

    double tPairEne = 0.0;

    // aMat = kMat * (2 * tMat - tMat.T)
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
            int idx = osvIdxY * nOsvIj + osvIdxX;
            double kij = kMat[kIdx0 + idx];
            double tij = tMat[tIdx0 + idx];
            double tji = tMat[tIdx0 + osvIdxX * nOsvIj + osvIdxY];
            tPairEne += kij * (2 * tij - tji);
        }
    }

    sPairEne[tid] = tPairEne;

    __syncthreads();

    // Parallel reduction
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sPairEne[tid] += sPairEne[tid + stride];
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        pairEnergies[pairIdx] = (i == j) ? sPairEne[0] : 2 * sPairEne[0];
    }
}

__global__ void closeResidualKernel(double *sMat, double *fMat, double *tMatInit, double *tMatNew, double *kMat, double *rMat,
                                    double *xMat, double *emuij, double *tempMatA, double *tempMatB,
                                    double *locFock, double *pairEnergies,
                                    int *pairs, int *nOsvOcc, int *kOccPair, int64_t *rOffsets,
                                    int64_t *sfOffsets, int64_t *tInitOffsets, int64_t *tNewOffsets,
                                    int64_t *kOffsets, int64_t *xOffsetsFull, int64_t *eOffsets, int64_t *tempOffsets,
                                    int64_t *kOccOffsets, // int64_t *extPairIndices,
                                    int nOcc, double kOccTol) {
    __shared__ double sMem[CLOSE_RES_SMEM];
    int pairIdx = blockIdx.x;
    int iPair = pairs[pairIdx];

    // Initialize R matrix
    copyArrayDevice(rMat, kMat, rOffsets[pairIdx], kOffsets[iPair], rOffsets[pairIdx+1] - rOffsets[pairIdx]);

    // Compute R matrix
    closeRmatDevice(sMat, fMat, tMatInit, rMat,
                    tempMatA, tempMatB, locFock,
                    nOsvOcc, kOccPair,
                    rOffsets, tInitOffsets,
                    sfOffsets, tempOffsets,
                    kOccOffsets, // extPairIndices,
                    nOcc, kOccTol, pairIdx, iPair, sMem);

    // Update MP2 amplitudes
    updateCloseTMatDevice(tMatNew, rMat, xMat, emuij, tempMatA,
                          tempOffsets, tNewOffsets, rOffsets, xOffsetsFull,
                          eOffsets, nOsvOcc, nOcc,
                          pairIdx, iPair, sMem);

    // Compute pair energies
    closeMp2EneDevice(tMatNew, kMat, pairEnergies, tNewOffsets,
                      kOffsets, nOsvOcc, nOcc, pairIdx,
                      iPair, sMem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// Remote //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#define REMOTE_RES_SMEM 4000

__device__ void remoteRmatDevice(double *sMat, double *fMat, double *tMat, double *rMat,
                                 double *locFock, int *nOsvOcc,
                                 int64_t *rOffsets, int64_t *tOffsets, int64_t *sfOffsets,
                                 int nOcc, int pairIdx, int iPair, double *sMem) {

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    int i = iPair / nOcc;
    int j = iPair % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];

    int pairIi = i * nOcc + i;
    int pairJj = j * nOcc + j;

    int64_t rIdx0 = rOffsets[pairIdx];

    int64_t tIjIdx0 = tOffsets[iPair];
    int64_t tIiIdx0 = tOffsets[pairIi];
    int64_t tJjIdx0 = tOffsets[pairJj];

    int64_t sfIjIdx0 = sfOffsets[iPair];
    int64_t sfIiIdx0 = sfOffsets[pairIi];
    int64_t sfJjIdx0 = sfOffsets[pairJj];

    double locFockIj = locFock[iPair];
    double locFockIi = locFock[pairIi];
    double locFockJj = locFock[pairJj];

    
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
            double value = 0.0;
            // First term: tMatIj*(fMatJj - locFockJj) - locFockIj*sMatIj*tMatJj
            for (int osvIdxZ = 0; osvIdxZ < nOsvJ; ++osvIdxZ) {
                double tMatIj = tMat[tIjIdx0 + osvIdxY * nOsvJ + osvIdxZ];
                double fMatJj = fMat[sfJjIdx0 + osvIdxZ * nOsvJ + osvIdxX];
                double sMatIj = sMat[sfIjIdx0 + osvIdxY * nOsvJ + osvIdxZ];
                double tMatJj = tMat[tJjIdx0 + osvIdxZ * 2 * nOsvJ + nOsvJ + osvIdxX];

                value += tMatIj * (fMatJj - locFockJj) - locFockIj * sMatIj * tMatJj;
            }

            // Second term: (fMatIi - locFockIi)*tMatIj - locFockIj*tMatIi*sMatIj
            for (int osvIdxZ = 0; osvIdxZ < nOsvI; ++osvIdxZ) {
                double fMatIi = fMat[sfIiIdx0 + osvIdxY * nOsvI + osvIdxZ];
                double tMatIj = tMat[tIjIdx0 + osvIdxZ * nOsvJ + osvIdxX];
                double tMatIi = tMat[tIiIdx0 + osvIdxY * 2 * nOsvI + nOsvI + osvIdxZ];
                double sMatIj = sMat[sfIjIdx0 + osvIdxZ * nOsvJ + osvIdxX];

                value += (fMatIi - locFockIi) * tMatIj - locFockIj * tMatIi * sMatIj;
            }

            rMat[rIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
        }
    }

    __syncthreads();
    */

    // First term: tMatIj*(fMatJj - locFockJj) - locFockIj*sMatIj*tMatJj
    {
        int maxOsvZBatch = REMOTE_RES_SMEM / (2 * nOsvJ + 2 * nOsvI);
        int sIjSize = maxOsvZBatch * nOsvI;
        int sJjSize = maxOsvZBatch * nOsvJ;

        double *sTij = sMem;
        double *sSij = sTij + sIjSize;
        double *sTjj = sSij + sIjSize;
        double *sFjj = sTjj + sJjSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
            int bYzSize = bNOsvZ * nOsvI;
            int bXzSize = bNOsvZ * nOsvJ;

            for (int bYzIdx = tid; bYzIdx < bYzSize; bYzIdx += threadsPerBlock) {
                int osvIdxY = bYzIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bYzIdx % bNOsvZ;
                sTij[bYzIdx] = tMat[tIjIdx0 + osvIdxY * nOsvJ + osvIdxZ];
                sSij[bYzIdx] = sMat[sfIjIdx0 + osvIdxY * nOsvJ + osvIdxZ];
            }

            for (int bXzIdx = tid; bXzIdx < bXzSize; bXzIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXzIdx / nOsvJ;
                int osvIdxX = bXzIdx % nOsvJ;
                sTjj[bXzIdx] = tMat[tJjIdx0 + osvIdxZ * 2 * nOsvJ + nOsvJ + osvIdxX];
                sFjj[bXzIdx] = fMat[sfJjIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        int idxIj = osvIdxY * bNOsvZ + osvIdxZ;
                        int idxJj = osvIdxZ * nOsvJ + osvIdxX;

                        value += sTij[idxIj] * (sFjj[idxJj] - locFockJj) 
                                - locFockIj * sSij[idxIj] * sTjj[idxJj];
                    }
                    rMat[rIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    // Second term: (fMatIi - locFockIi)*tMatIj - locFockIj*tMatIi*sMatIj
    {
        int maxOsvZBatch = REMOTE_RES_SMEM / (2 * nOsvJ + 2 * nOsvI);
        int sIjSize = maxOsvZBatch * nOsvJ;
        int sIiSize = maxOsvZBatch * nOsvI;

        double *sTij = sMem;
        double *sSij = sTij + sIjSize;
        double *sTii = sSij + sIjSize;
        double *sFii = sTii + sIiSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
            int bYzSize = bNOsvZ * nOsvI;
            int bXzSize = bNOsvZ * nOsvJ;

            for (int bYzIdx = tid; bYzIdx < bYzSize; bYzIdx += threadsPerBlock) {
                int osvIdxY = bYzIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bYzIdx % bNOsvZ;
                sTii[bYzIdx] = tMat[tIiIdx0 + osvIdxY * 2 * nOsvI + nOsvI + osvIdxZ];
                sFii[bYzIdx] = fMat[sfIiIdx0 + osvIdxY * nOsvI + osvIdxZ];
            }

            for (int bXzIdx = tid; bXzIdx < bXzSize; bXzIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXzIdx / nOsvJ;
                int osvIdxX = bXzIdx % nOsvJ;
                sTij[bXzIdx] = tMat[tIjIdx0 + osvIdxZ * nOsvJ + osvIdxX];
                sSij[bXzIdx] = sMat[sfIjIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        int idxIi = osvIdxY * bNOsvZ + osvIdxZ;
                        int idxIj = osvIdxZ * nOsvJ + osvIdxX;

                        value += (sFii[idxIi] - locFockIi) * sTij[idxIj] 
                                - locFockIj * sTii[idxIi] * sSij[idxIj];
                    }
                    rMat[rIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }
}

__device__ void updateRemoteTMatDevice(double *tMat, double *rMat, double *xMat, double *emui, double *eOcc, double *tempMatA,
                                       int64_t *tempOffsets, int64_t *tOffsets, int64_t *rOffsets, int64_t *xOffsets,
                                       int64_t *eOffsets, int *nOsvOcc, int nOcc,
                                       // int64_t pairIdx, int64_t extPairIdx, int64_t iPair) {
                                       int pairIdx, int iPair, double *sMem) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int i = iPair / nOcc;
    int j = iPair % nOcc;
    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];

    /*
    int64_t tIdx0 = tOffsets[extPairIdx];
    int64_t rIdx0 = rOffsets[pairIdx];*/
    int64_t tIdx0 = tOffsets[iPair];
    int64_t rIdx0 = rOffsets[pairIdx];
    int64_t xiiIdx0 = xOffsets[i];
    int64_t xjjIdx0 = xOffsets[j];
    int64_t emuiIdx0 = eOffsets[i];
    int64_t emujIdx0 = eOffsets[j];
    int64_t tempIdx0 = tempOffsets[pairIdx];

    double eOccIj = eOcc[i] + eOcc[j];

    // aMat = xii.T * rij
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvI; ++osvIdxZ) {
                value += xMat[xiiIdx0 + osvIdxZ * nOsvI + osvIdxY] * rMat[rIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }

            tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxX] = value;
        }
    }*/

    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxX] = 0.0;
            }
        }

        int maxOsvZBatch = REMOTE_RES_SMEM / (nOsvI + nOsvJ);
        int sXMatSize = maxOsvZBatch * nOsvI;

        double *sXMat = sMem;
        double *sRMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvI;
            int bRMatSize = bNOsvZ * nOsvJ;

            for (int bRMatIdx = tid; bRMatIdx < bRMatSize; bRMatIdx += threadsPerBlock) {
                int osvIdxX = bRMatIdx % nOsvJ;
                int osvIdxZ = oIdxZStart + bRMatIdx / nOsvJ;
                sRMat[bRMatIdx] = rMat[rIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx / nOsvI;
                int osvIdxY = bXMatIdx % nOsvI;
                sXMat[bXMatIdx] = xMat[xiiIdx0 + osvIdxZ * nOsvI + osvIdxY];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sXMat[osvIdxZ * nOsvI + osvIdxY] * sRMat[osvIdxZ * nOsvJ + osvIdxX];
                    }
                    tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    __syncthreads();

    // r'ij = (aMat * xjj) / (ei + ej - emui - emuj)
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvJ; ++osvIdxZ) {
                value += tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxZ] * xMat[xjjIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }
            double eijmunu = eOccIj - emui[emuiIdx0 + osvIdxY] - emui[emujIdx0 + osvIdxX];
            rMat[rIdx0 + osvIdxY * nOsvJ + osvIdxX] = value / eijmunu;
        }
    }*/

    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                rMat[rIdx0 + osvIdxY * nOsvJ + osvIdxX] = 0.0;
            }
        }

        int maxOsvZBatch = REMOTE_RES_SMEM / (nOsvI + nOsvJ);
        int sXMatSize = maxOsvZBatch * nOsvJ;

        double *sXMat = sMem;
        double *sAMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvJ;
            int bAMatSize = bNOsvZ * nOsvI;

            for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                int osvIdxY = bAMatIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxZ];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx / nOsvJ;
                int osvIdxX = bXMatIdx % nOsvJ;
                sXMat[bXMatIdx] = xMat[xjjIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sXMat[osvIdxZ * nOsvJ + osvIdxX];
                    }
                    rMat[rIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
                }
            }

            __syncthreads();
        }

        for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                double eijmunu = eOccIj - emui[emuiIdx0 + osvIdxY] - emui[emujIdx0 + osvIdxX];
                rMat[rIdx0 + osvIdxY * nOsvJ + osvIdxX] /= eijmunu;
            }
        }
    }

    __syncthreads();

    // aMat = xii * r'ij
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvI; ++osvIdxZ) {
                value += xMat[xiiIdx0 + osvIdxY * nOsvI + osvIdxZ] * rMat[rIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }
            tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxX] = value;
        }
    }
        
    __syncthreads();*/

    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxX] = 0.0;
            }
        }
        int maxOsvZBatch = REMOTE_RES_SMEM / (nOsvI + nOsvJ);
        int sXMatSize = maxOsvZBatch * nOsvI;

        double *sXMat = sMem;
        double *sRMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvI;
            int bRMatSize = bNOsvZ * nOsvJ;

            for (int bRMatIdx = tid; bRMatIdx < bRMatSize; bRMatIdx += threadsPerBlock) {
                int osvIdxX = bRMatIdx % nOsvJ;
                int osvIdxZ = oIdxZStart + bRMatIdx / nOsvJ;
                sRMat[bRMatIdx] = rMat[rIdx0 + osvIdxZ * nOsvJ + osvIdxX];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx % bNOsvZ;
                int osvIdxY = bXMatIdx / bNOsvZ;
                sXMat[bXMatIdx] = xMat[xiiIdx0 + osvIdxY * nOsvI + osvIdxZ];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sXMat[osvIdxY * bNOsvZ + osvIdxZ] * sRMat[osvIdxZ * nOsvJ + osvIdxX];
                    }
                    tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    

    // tMat += (aMat * xjj.T)
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvJ; ++osvIdxZ) {
                value += tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxZ] * xMat[xjjIdx0 + osvIdxX * nOsvJ + osvIdxZ];
            }

            tMat[tIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
        }
    }*/

    {
        int maxOsvZBatch = REMOTE_RES_SMEM / (nOsvI + nOsvJ);
        int sXMatSize = maxOsvZBatch * nOsvJ;

        double *sXMat = sMem;
        double *sAMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvJ;
            int bAMatSize = bNOsvZ * nOsvI;

            for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                int osvIdxY = bAMatIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nOsvJ + osvIdxZ];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx % bNOsvZ;
                int osvIdxX = bXMatIdx / bNOsvZ;
                sXMat[bXMatIdx] = xMat[xjjIdx0 + osvIdxX * nOsvJ + osvIdxZ];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sXMat[osvIdxX * bNOsvZ + osvIdxZ];
                    }
                    tMat[tIdx0 + osvIdxY * nOsvJ + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    __syncthreads();
}

__device__ void remoteMp2EneDevice(double *tMat, double *kMat, double *pairEnergies, int64_t *tOffsets,
                                   int64_t *kOffsets, int *nOsvOcc, int nOcc, int pairIdx,
                                   int iPair, double *sMem) {

    //__shared__ double sPairEne[THREADS_PER_BLOCK];
    double *sPairEne = sMem;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int i = iPair / nOcc;
    int j = iPair % nOcc;
    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];

    /*int64_t tIdx0 = tOffsets[extPairIdx];
    int64_t kIdx0 = kOffsets[pairIdx];*/
    int64_t tIdx0 = tOffsets[iPair];
    int64_t kIdx0 = kOffsets[iPair];

    double tPairEne = 0.0;

    // aMat = kMat * (2 * tMat - tMat.T)
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvI; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvJ; osvIdxX += blockDim.x) {
            int idx = osvIdxY * nOsvJ + osvIdxX;
            double kij = kMat[kIdx0 + idx];
            double tij = tMat[tIdx0 + idx];
            tPairEne += 2 * kij * tij;
        }
    }

    sPairEne[tid] = tPairEne;

    __syncthreads();

    // Parallel reduction
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sPairEne[tid] += sPairEne[tid + stride];
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        pairEnergies[pairIdx] = 2 * sPairEne[0]; // Remote pairs must be off-diagonal
    }
}


__global__ void remoteResidualKernel(double *sMat, double *fMat, double *tMat, double *kMat, double *rMat,
                                     double *xMat, double *emui, double *eOcc, double *tempMatA,
                                     double *locFock, double *pairEnergies, int *pairs, int *nOsvOcc,
                                     int64_t *rOffsets, int64_t *sfOffsets, int64_t *tOffsets, int64_t *kOffsets,
                                     int64_t *xOffsets, int64_t *eOffsets, int64_t *tempOffsets,
                                     // int64_t *extPairIndices, int64_t nOcc) {
                                     int nOcc) {
    __shared__ double sMem[4000];

    int pairIdx = blockIdx.x;
    int iPair = pairs[pairIdx];
    // int64_t extPairIdx = extPairIndices[iPair];

    int64_t rIdx0 = rOffsets[pairIdx];
    int64_t rIdx1 = rOffsets[pairIdx + 1];
    int64_t kIdx0 = kOffsets[iPair];
    // int64_t kIdx1 = rIdx0 + nOsvOcc[iPair / nOcc] * nOsvOcc[iPair % nOcc];

    // Initialize R matrix
    copyArrayDevice(rMat, kMat, rIdx0, kIdx0, rIdx1 - rIdx0);

    // Compute R matrix
    remoteRmatDevice(sMat, fMat, tMat, rMat,
                     locFock, nOsvOcc,
                     rOffsets, tOffsets, sfOffsets,
                     // extPairIndices,
                     nOcc, pairIdx, iPair, sMem);

    // Update MP2 amplitudes
    updateRemoteTMatDevice(tMat, rMat, xMat, emui, eOcc, tempMatA,
                           tempOffsets, tOffsets, rOffsets, xOffsets,
                           eOffsets, nOsvOcc, nOcc,
                           // pairIdx, extPairIdx, iPair);
                           pairIdx, iPair, sMem);

    // Compute pair energies
    remoteMp2EneDevice(tMat, kMat, pairEnergies, tOffsets,
                       kOffsets, nOsvOcc, nOcc, pairIdx,
                       // extPairIdx, iPair);
                       iPair, sMem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// CLUSTER /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CLUSTER_RES_SMEM 5888

__device__ void clusterRmatDevice(double *sMat, double *fMat, double *tMat, double *rMat,
                                  double *tempMatA, double *tempMatB, double *locFock, int *pairs,
                                  int *nOsvOcc, int *kOccPair, int64_t *tOffsets,
                                  int64_t *rOffsets, int64_t *sfOffsets, int64_t *tempOffsets,
                                  int64_t *kOccOffsets, int *extPairIndices,
                                  int nOcc, double kOccTol, int64_t pairIdx, int64_t iPair,
                                  int64_t blockPairIdx0, int64_t blockPairIdx1, bool useDynAmp, double *sMem) {


    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    int i = iPair / nOcc;
    int j = iPair % nOcc;

    int nOsvI = nOsvOcc[i];
    int nOsvJ = nOsvOcc[j];
    int nOsvIj = nOsvI + nOsvJ;

    //int64_t rIdx0 = (useDynAmp) ? rOffsets[blockIdx.x] : rOffsets[pairIdx];
    double *rMatIj = rMat + ((useDynAmp) ? rOffsets[blockIdx.x] : rOffsets[pairIdx]);
    int64_t tempIdx0 = tempOffsets[blockIdx.x];

    int64_t kOccIdx0 = kOccOffsets[pairIdx];
    int64_t kOccIdx1 = kOccOffsets[pairIdx + 1];

    int64_t sfStartIi = sfOffsets[extPairIndices[i * nOcc + i]];
    int64_t sfStartIj = sfOffsets[extPairIndices[i * nOcc + j]];
    int64_t sfStartJj = sfOffsets[extPairIndices[j * nOcc + j]];

    for (int64_t kOccIdx = kOccIdx0; kOccIdx < kOccIdx1; kOccIdx++) {
        int k = kOccPair[kOccIdx];
        double locFockKj = locFock[k * nOcc + j];
        double locFockIk = locFock[i * nOcc + k];
        bool kEqualsJ = (k == j);
        bool kExceedsJ = (k > j);
        bool iEqualsK = (i == k);
        bool iExceedsK = (i > k);
        int nOsvK = nOsvOcc[k];

        int pairIk = (iExceedsK) ? k * nOcc + i : i * nOcc + k;
        int pairKj = (kExceedsJ) ? j * nOcc + k : k * nOcc + j;
        int pairIdxIk = -1, pairIdxKj = -1;
        for (int pidx = blockPairIdx0; pidx < blockPairIdx1; ++pidx) {
            int currentPair = pairs[pidx];
            pairIdxIk = (currentPair == pairIk) ? pidx : pairIdxIk;
            pairIdxKj = (currentPair == pairKj) ? pidx : pairIdxKj;
        }

        int64_t sfStartIk = sfOffsets[extPairIndices[pairIk]];
        int64_t sfStartKj = sfOffsets[extPairIndices[pairKj]];

        if (fabsf(locFockKj) > kOccTol) {
            int nOsvIk = nOsvI + nOsvK;
            //int64_t tIkIdx0 = tOffsets[pairIdxIk];
            double *gTMatIk = tMat + tOffsets[pairIdxIk];

            // Compute tempMatA: bMat = -locFockKj * sMatIkij + deltaKj * fMatIjij
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIk; osvIdxY += blockDim.y) {
                int oIdxY, occY, nOsvY;
                if (osvIdxY < nOsvI) {
                    oIdxY = osvIdxY;
                    occY = i;
                    nOsvY = nOsvI;
                } else {
                    oIdxY = osvIdxY - nOsvI;
                    occY = k;
                    nOsvY = nOsvK;
                }

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    int oIdxX, occX, nOsvX;
                    if (osvIdxX < nOsvI) {
                        oIdxX = osvIdxX;
                        occX = i;
                        nOsvX = nOsvI;
                    } else {
                        oIdxX = osvIdxX - nOsvI;
                        occX = j;
                        nOsvX = nOsvJ;
                    }

                    int64_t sIdx0;
                    double sMatIkij;
                    if (occY > occX) {
                        int64_t sPairIdx = extPairIndices[occX * nOcc + occY];
                        sIdx0 = sfOffsets[sPairIdx];
                        sMatIkij = sMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                    } else {
                        int64_t sPairIdx = extPairIndices[occY * nOcc + occX];
                        sIdx0 = sfOffsets[sPairIdx];
                        sMatIkij = sMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                    }

                    if (kEqualsJ) {
                        double fMatIjij;
                        if (occY > occX) {
                            fMatIjij = fMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                        } else {
                            fMatIjij = fMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                        }
                        tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = -locFockKj * sMatIkij + fMatIjij;
                    } else {
                        tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = -locFockKj * sMatIkij;
                    }
                }
            }*/

            // Quadrant 1: oIdxY = i, oIdxX = i
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartIi + oIdxY * nOsvI + oIdxX;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 2: oIdxY = i, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartIj + oIdxY * nOsvJ + oIdxX;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 3: oIdxY = k, oIdxX = i
            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    int64_t sfIdx = (iExceedsK) ? sfStartIk + oIdxY * nOsvI + oIdxX
                                                : sfStartIk + oIdxX * nOsvK + oIdxY;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + oIdxX] = -locFockKj * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 4: oIdxY = k, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = (kExceedsJ) ? sfStartKj + oIdxX * nOsvK + oIdxY
                                                : sfStartKj + oIdxY * nOsvJ + oIdxX;
                    if (kEqualsJ) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvIj + (oIdxX + nOsvI)] = -locFockKj * sMat[sfIdx];
                    }
                }
            }

            // Compute tempMatB: sMatIjik * tMatIk
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                int sMatOIdxY, occY, nOsvY;
                if (osvIdxY < nOsvI) {
                    sMatOIdxY = osvIdxY;
                    occY = i;
                    nOsvY = nOsvI;
                } else {
                    sMatOIdxY = osvIdxY - nOsvI;
                    occY = j;
                    nOsvY = nOsvJ;
                }

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIk; osvIdxX += blockDim.x) {
                    int tMatOIdxX = (osvIdxX < nOsvI) ? osvIdxX + nOsvK : osvIdxX - nOsvI;
                    double value = 0.0;

                    for (int osvIdxZ = 0; osvIdxZ < nOsvIk; osvIdxZ++) {
                        int sMatOIdxZ, tMatOIdxZ, occZ, nOsvZ;
                        if (osvIdxZ < nOsvI) {
                            sMatOIdxZ = osvIdxZ;
                            tMatOIdxZ = osvIdxZ + nOsvK;
                            occZ = i;
                            nOsvZ = nOsvI;
                        } else {
                            sMatOIdxZ = osvIdxZ - nOsvI;
                            tMatOIdxZ = osvIdxZ - nOsvI;
                            occZ = k;
                            nOsvZ = nOsvK;
                        }

                        double sMatIjik;
                        if (occY > occZ) {
                            int sPairIdx = extPairIndices[occZ * nOcc + occY];
                            int64_t sIdx0 = sfOffsets[sPairIdx];
                            sMatIjik = sMat[sIdx0 + sMatOIdxZ * nOsvY + sMatOIdxY];
                        } else {
                            int sPairIdx = extPairIndices[occY * nOcc + occZ];
                            int64_t sIdx0 = sfOffsets[sPairIdx];
                            sMatIjik = sMat[sIdx0 + sMatOIdxY * nOsvZ + sMatOIdxZ];
                        }

                        double tMatIk = (iExceedsK) ? gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ]
                                                    : gTMatIk[osvIdxZ * nOsvIk + osvIdxX];
                        value += sMatIjik * tMatIk;
                    }

                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] = value;
                }
            }

            __syncthreads();*/

            

            /*
            // Quadrant 1: oIdxY = i, oIdxX = i
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = i
                    for (int oIdxZ = 0; oIdxZ < nOsvI; oIdxZ++) {
                        double sMatIjik = sMat[sfStartIi + oIdxY * nOsvI + oIdxZ]; 
                        double tMatIk = (iExceedsK) ? gTMatIk[(oIdxX + nOsvK) * nOsvIk + (oIdxZ + nOsvK)] // tii.T
                                                    : gTMatIk[oIdxZ * nOsvIk + oIdxX]; // tii
                        value += sMatIjik * tMatIk;
                    }

                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatIjik = (iExceedsK) ? sMat[sfStartIk + oIdxZ * nOsvI + oIdxY]
                                                      : sMat[sfStartIk + oIdxY * nOsvK + oIdxZ];
                        double tMatIk = (iExceedsK) ? gTMatIk[(oIdxX + nOsvK) * nOsvIk + oIdxZ] // tik.T
                                                    : gTMatIk[(oIdxZ + nOsvI) * nOsvIk + oIdxX]; // tki
                        value += sMatIjik * tMatIk;
                    }

                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] = value;
                }
            }

            
            // Quadrant 2: oIdxY = i, oIdxX = k
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = i
                    for (int oIdxZ = 0; oIdxZ < nOsvI; oIdxZ++) {
                        double sMatIjik = sMat[sfStartIi + oIdxY * nOsvI + oIdxZ]; 
                        double tMatIk = (iExceedsK) ? gTMatIk[oIdxX * nOsvIk + (oIdxZ + nOsvK)] // tki.T
                                                    : gTMatIk[oIdxZ * nOsvIk + (oIdxX + nOsvI)]; // tik
                        value += sMatIjik * tMatIk;
                    }

                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatIjik = (iExceedsK) ? sMat[sfStartIk + oIdxZ * nOsvI + oIdxY] // ski.T
                                                      : sMat[sfStartIk + oIdxY * nOsvK + oIdxZ]; // sik
                        double tMatIk = (iExceedsK) ? gTMatIk[oIdxX * nOsvIk + oIdxZ] // tkk.T
                                                    : gTMatIk[(oIdxZ + nOsvI) * nOsvIk + (oIdxX + nOsvI)]; // tkk
                        value += sMatIjik * tMatIk;
                    }
                    tempMatB[tempIdx0 + oIdxY * nOsvIk + (oIdxX + nOsvI)] = value;
                }
            }

            
            // Quadrant 3: oIdxY = j, oIdxX = i
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = i
                    for (int oIdxZ = 0; oIdxZ < nOsvI; oIdxZ++) {
                        double sMatIjik = sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY]; 
                        double tMatIk = (iExceedsK) ? gTMatIk[(oIdxX + nOsvK) * nOsvIk + (oIdxZ + nOsvK)] // tii.T
                                                    : gTMatIk[oIdxZ * nOsvIk + oIdxX]; // tii
                        value += sMatIjik * tMatIk;
                    }

                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatIjik = (kExceedsJ) ? sMat[sfStartKj + oIdxY * nOsvK + oIdxZ]  // sjk
                                                      : sMat[sfStartKj + oIdxZ * nOsvJ + oIdxY]; // skj.T
                        double tMatIk = (iExceedsK) ? gTMatIk[(oIdxX + nOsvK) * nOsvIk + oIdxZ] // tik.T
                                                    : gTMatIk[(oIdxZ + nOsvI) * nOsvIk + oIdxX]; // tki
                        value += sMatIjik * tMatIk;
                    }
                    tempMatB[tempIdx0 + (oIdxY + nOsvI) * nOsvIk + oIdxX] = value;
                }
            }*/


            /*
            // Quadrant 4: oIdxY = j, oIdxX = k
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = i
                    for (int oIdxZ = 0; oIdxZ < nOsvI; oIdxZ++) {
                        double sMatIjik = sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY]; // tij.T
                        double tMatIk = (iExceedsK) ? gTMatIk[oIdxX * nOsvIk + (oIdxZ + nOsvK)] // tki.T
                                                    : gTMatIk[oIdxZ * nOsvIk + (oIdxX + nOsvI)]; // tik
                        value += sMatIjik * tMatIk;
                    }

                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatIjik = (kExceedsJ) ? sMat[sfStartKj + oIdxY * nOsvK + oIdxZ]  // sjk
                                                      : sMat[sfStartKj + oIdxZ * nOsvJ + oIdxY]; // skj.T
                        double tMatIk = (iExceedsK) ? gTMatIk[oIdxX * nOsvIk + oIdxZ] // tkk.T
                                                    : gTMatIk[(oIdxZ + nOsvI) * nOsvIk + (oIdxX + nOsvI)]; // tkk
                        value += sMatIjik * tMatIk;
                    }
                    tempMatB[tempIdx0 + (oIdxY + nOsvI) * nOsvIk + (oIdxX + nOsvI)] = value;
                }
            }

            __syncthreads();*/

            for (int oIdxY = threadIdx.y; oIdxY < nOsvIj; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvIk; oIdxX += blockDim.x) {
                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] = 0.0;
                }
            }

            __syncthreads();
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 1           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / nOsvI / 2;
                double *sSMat = sMem;
                double *sTMat = sSMat + CLUSTER_RES_SMEM / 2;
                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bSMatSize;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIi + oIdxY * nOsvI + oIdxZ];
                        }

                        if (iExceedsK) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {

                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bSMatSize;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxY];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvI + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxY * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 2           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (nOsvI + nOsvK);
                int sSMatSize = maxOsvZBatch * nOsvI;
                double *sSMat = sMem;
                double *sTMat = sSMat + sSMatSize;

                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        // sMat[sfStartIi + oIdxY * nOsvI + oIdxZ];
                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIi + oIdxY * nOsvI + oIdxZ];
                        }

                        if (iExceedsK) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxY];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvI + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxY * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxY * bNOsvZ + oIdxZ] * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIk + oIdxX + nOsvI] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 3           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (nOsvJ + nOsvI);
                int sSMatSize = maxOsvZBatch * nOsvJ;

                double *sSMat = sMem;
                double *sTMat = sSMat + sSMatSize;

                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvI;

                        // sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        }

                        if (iExceedsK) {
                            // tIkIdx0 + (oIdxX + nOsvK) * nOsvIk + oIdxZ + nOsvK
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + oIdxZ * nOsvIk + oIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvI;

                        if (kExceedsJ) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxY * nOsvK + oIdxZ];
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxY];
                            }
                        }

                        if (iExceedsK) {
                            // tIkIdx0 + tMatOIdxX * nOsvIk + oIdxZ
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ + nOsvK;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + (oIdxZ + nOsvI) * nOsvIk + oIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvI + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxZ * nOsvI + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + oIdxX] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 4           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (nOsvJ + nOsvK);
                int sSMatSize = maxOsvZBatch * nOsvJ;

                double *sSMat = sMem;
                double *sTMat = sSMat + sSMatSize;

                // Z part 1: occZ = i
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvI; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvI - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;

                        // sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxY = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxZ * nOsvJ + oIdxY];
                        }

                        if (iExceedsK) {
                            // tIkIdx0 + oIdxX * nOsvIk + oIdxZ + nOsvK
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ + nOsvK;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + oIdxZ * nOsvIk + osvIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sSMat[oIdxZ * nOsvJ + oIdxY] * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;

                        if (kExceedsJ) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxY * nOsvK + oIdxZ];
                            }

                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxY = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxY];
                            }
                        }

                        if (iExceedsK) {
                            
                            // tIkIdx0 + oIdxX * nOsvIk + oIdxZ
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int tMatOIdxX = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxX * nOsvIk + tMatOIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }

                        } else {
                            // tIkIdx0 + (oIdxZ + nOsvI) * nOsvIk + osvIdxX
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int tMatOIdxZ = oIdxZStart + bTMatIdx / nOsvK + nOsvI;
                                int tMatOIdxX = bTMatIdx % nOsvK + nOsvI;
                                sTMat[bTMatIdx] = gTMatIk[tMatOIdxZ * nOsvIk + tMatOIdxX];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                int osvIdxY = oIdxY + nOsvI;
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                                    int osvIdxX = oIdxX + nOsvI;
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sJk = (kExceedsJ) ? sSMat[oIdxY * bNOsvZ + oIdxZ]
                                                                 : sSMat[oIdxZ * nOsvJ + oIdxY];
                                        value += sJk * sTMat[oIdxZ * nOsvK + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxX] += value;
                                }
                            }
                        }

                        __syncthreads();
                    }
                }
            }

            // Accumulate to rMat: tempMatB * tempMatA
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < nOsvIk; osvIdxZ++) {
                        value += tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxZ] * tempMatA[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }
                    rMatIj[osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }*/

            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (2 * nOsvIj);
                int sAMatSize = maxOsvZBatch * nOsvIj;

                double *sAMat = sMem;
                double *sBMat = sAMat + sAMatSize;

                // Z part 1: occZ = i
                for (int oIdxZStart = 0; oIdxZStart < nOsvIk; oIdxZStart += maxOsvZBatch) {
                    int bNOsvZ = min(maxOsvZBatch, nOsvIk - oIdxZStart);
                    int bAMatSize = bNOsvZ * nOsvIj;
                    int bBMatSize = bAMatSize;

                    for (int bBMatIdx = tid; bBMatIdx < bBMatSize; bBMatIdx += threadsPerBlock) {
                        int osvIdxY = bBMatIdx / bNOsvZ;
                        int osvIdxZ = oIdxZStart + bBMatIdx % bNOsvZ;
                        sBMat[bBMatIdx] = tempMatB[tempIdx0 + osvIdxY * nOsvIk + osvIdxZ];
                    }

                    for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                        int osvIdxZ = oIdxZStart + bAMatIdx / nOsvIj;
                        int osvIdxX = bAMatIdx % nOsvIj;
                        sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }

                    __syncthreads();

                    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                            double value = 0.0;
                            for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                                value += sBMat[osvIdxY * bNOsvZ + osvIdxZ] * sAMat[osvIdxZ * nOsvIj + osvIdxX];
                            }
                            rMatIj[osvIdxY * nOsvIj + osvIdxX] += value;
                        }
                    }

                    __syncthreads();
                }
            }

        }


        if (fabsf(locFockIk) > kOccTol) {
            int nOsvKj = nOsvK + nOsvJ;
            //int64_t tKjIdx0 = tOffsets[pairIdxKj];
            double *gTMatKj = tMat + tOffsets[pairIdxKj];

            // Compute tempMatA: bMat = -locFockIk * sMatIjkj + deltaIk * fMatIjij
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                int oIdxY, occY, nOsvY;
                if (osvIdxY < nOsvI) {
                    oIdxY = osvIdxY;
                    occY = i;
                    nOsvY = nOsvI;
                } else {
                    oIdxY = osvIdxY - nOsvI;
                    occY = j;
                    nOsvY = nOsvJ;
                }

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvKj; osvIdxX += blockDim.x) {
                    int oIdxX, occX, nOsvX;
                    if (osvIdxX < nOsvK) {
                        oIdxX = osvIdxX;
                        occX = k;
                        nOsvX = nOsvK;
                    } else {
                        oIdxX = osvIdxX - nOsvK;
                        occX = j;
                        nOsvX = nOsvJ;
                    }

                    int64_t sIdx0;
                    double sMatIjkj;
                    if (occY > occX) {
                        int sPairIdx = extPairIndices[occX * nOcc + occY];
                        sIdx0 = sfOffsets[sPairIdx];
                        sMatIjkj = sMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                    } else {
                        int sPairIdx = extPairIndices[occY * nOcc + occX];
                        sIdx0 = sfOffsets[sPairIdx];
                        sMatIjkj = sMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                    }

                    if (iEqualsK) {
                        double fMatIjij;
                        if (occY > occX) {
                            fMatIjij = fMat[sIdx0 + oIdxX * nOsvY + oIdxY];
                        } else {
                            fMatIjij = fMat[sIdx0 + oIdxY * nOsvX + oIdxX];
                        }
                        tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxX] = -locFockIk * sMatIjkj + fMatIjij;
                    } else {
                        tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxX] = -locFockIk * sMatIjkj;
                    }
                }
            }*/

            // Quadrant 1: oIdxY = i, oIdxX = k
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                    
                    int64_t sfIdx = (iExceedsK) ? sfStartIk + oIdxX * nOsvI + oIdxY
                                                : sfStartIk + oIdxY * nOsvK + oIdxX;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 2: oIdxY = i, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvI; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartIj + oIdxY * nOsvJ + oIdxX;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + oIdxY * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 3: oIdxY = j, oIdxX = k
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvK; oIdxX += blockDim.x) {
                    int64_t sfIdx = (kExceedsJ) ? sfStartKj + oIdxY * nOsvK + oIdxX
                                                : sfStartKj + oIdxX * nOsvJ + oIdxY;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + oIdxX] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            // Quadrant 4: oIdxY = j, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    int64_t sfIdx = sfStartJj + oIdxY * nOsvJ + oIdxX;
                    if (iEqualsK) {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx] + fMat[sfIdx];
                    } else {
                        tempMatA[tempIdx0 + (oIdxY + nOsvI) * nOsvKj + (oIdxX + nOsvK)] = -locFockIk * sMat[sfIdx];
                    }
                }
            }

            // Compute tempMatB: tMatKj * sMatKjij
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvKj; osvIdxY += blockDim.y) {
                int tMatOIdxY = (osvIdxY < nOsvK) ? osvIdxY + nOsvJ : osvIdxY - nOsvK;

                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    int sMatOIdxX, occX, nOsvX;
                    if (osvIdxX < nOsvI) {
                        sMatOIdxX = osvIdxX;
                        occX = i;
                        nOsvX = nOsvI;
                    } else {
                        sMatOIdxX = osvIdxX - nOsvI;
                        occX = j;
                        nOsvX = nOsvJ;
                    }

                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < nOsvKj; osvIdxZ++) {
                        int sMatOIdxZ, tMatOIdxZ, occZ, nOsvZ;
                        if (osvIdxZ < nOsvK) {
                            sMatOIdxZ = osvIdxZ;
                            tMatOIdxZ = osvIdxZ + nOsvJ;
                            occZ = k;
                            nOsvZ = nOsvK;
                        } else {
                            sMatOIdxZ = osvIdxZ - nOsvK;
                            tMatOIdxZ = osvIdxZ - nOsvK;
                            occZ = j;
                            nOsvZ = nOsvJ;
                        }

                        double tMatKj;
                        if (kExceedsJ) {
                            tMatKj = gTMatKj[tMatOIdxZ * nOsvKj + tMatOIdxY];
                        } else {
                            tMatKj = gTMatKj[osvIdxY * nOsvKj + osvIdxZ];
                        }

                        double sMatKjij;
                        if (occZ > occX) {
                            int sPairIdx = extPairIndices[occX * nOcc + occZ];
                            int64_t sIdx0 = sfOffsets[sPairIdx];
                            sMatKjij = sMat[sIdx0 + sMatOIdxX * nOsvZ + sMatOIdxZ];
                        } else {
                            int sPairIdx = extPairIndices[occZ * nOcc + occX];
                            int64_t sIdx0 = sfOffsets[sPairIdx];
                            sMatKjij = sMat[sIdx0 + sMatOIdxZ * nOsvX + sMatOIdxX];
                        }

                        value += tMatKj * sMatKjij;
                    }

                    tempMatB[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = value;
                }
            }

            __syncthreads();*/


            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvKj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    tempMatB[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = 0.0;
                }
            }

            __syncthreads();

            // Quadrant 1: oIdxY = k, oIdxX = i
            /*
            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatKjij = (iExceedsK) ? sMat[sfStartIk + oIdxZ * nOsvI + oIdxX]
                                                      : sMat[sfStartIk + oIdxX * nOsvK + oIdxZ];

                        double tMatKj = (kExceedsJ) ? gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + (oIdxY + nOsvJ)] // tkk.T
                                                    : gTMatKj[oIdxY * nOsvKj + oIdxZ]; // tkk
                        value += tMatKj * sMatKjij;
                    }

                    // oIdxZ = j
                    for (int oIdxZ = 0; oIdxZ < nOsvJ; oIdxZ++) {
                        double sMatKjij = sMat[sfStartIj + oIdxX * nOsvJ + oIdxZ];

                        double tMatKj = (kExceedsJ) ? gTMatKj[oIdxZ * nOsvKj + (oIdxY + nOsvJ)] // tjk.T
                                                    : gTMatKj[oIdxY * nOsvKj + (oIdxZ + nOsvK)]; // tkj
                        value += tMatKj * sMatKjij;
                    }

                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] = value;
                }
            }

            

            
            // Quadrant 2: oIdxY = k, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatKjij = (kExceedsJ) ? sMat[sfStartKj + oIdxX * nOsvK + oIdxZ] // skj
                                                      : sMat[sfStartKj + oIdxZ * nOsvJ + oIdxX];

                        double tMatKj = (kExceedsJ) ? gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + (oIdxY + nOsvJ)] // tkk.T
                                                    : gTMatKj[oIdxY * nOsvKj + oIdxZ]; // tkk
                        value += tMatKj * sMatKjij;
                    }

                    // oIdxZ = j
                    for (int oIdxZ = 0; oIdxZ < nOsvJ; oIdxZ++) {
                        double sMatKjij = sMat[sfStartJj + oIdxZ * nOsvJ + oIdxX]; // sjj

                        double tMatKj = (kExceedsJ) ? gTMatKj[oIdxZ * nOsvKj + (oIdxY + nOsvJ)] // tjk.T
                                                    : gTMatKj[oIdxY * nOsvKj + (oIdxZ + nOsvK)]; // tkj
                        value += tMatKj * sMatKjij;
                    }

                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] = value;
                }
            }

            // Quadrant 3: oIdxY = j, oIdxX = i
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatKjij = (iExceedsK) ? sMat[sfStartIk + oIdxZ * nOsvI + oIdxX]
                                                      : sMat[sfStartIk + oIdxX * nOsvK + oIdxZ];

                        double tMatKj = (kExceedsJ) ? gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + oIdxY] // tkj.T
                                                    : gTMatKj[(oIdxY + nOsvK) * nOsvKj + oIdxZ]; // tjk
                        value += tMatKj * sMatKjij;
                    }

                    // oIdxZ = j
                    for (int oIdxZ = 0; oIdxZ < nOsvJ; oIdxZ++) {
                        double sMatKjij = sMat[sfStartIj + oIdxX * nOsvJ + oIdxZ]; // sji

                        double tMatKj = (kExceedsJ) ? gTMatKj[oIdxZ * nOsvKj + oIdxY] // tjj.T
                                                    : gTMatKj[(oIdxY + nOsvK) * nOsvKj + (oIdxZ + nOsvK)]; // tjj
                        value += tMatKj * sMatKjij;
                    }

                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] = value;
                }
            }

            // Quadrant 4: oIdxY = j, oIdxX = j
            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                    double value = 0.0;
                    // oIdxZ = k
                    for (int oIdxZ = 0; oIdxZ < nOsvK; oIdxZ++) {
                        double sMatKjij = (kExceedsJ) ? sMat[sfStartKj + oIdxX * nOsvK + oIdxZ] // skj
                                                      : sMat[sfStartKj + oIdxZ * nOsvJ + oIdxX];

                        double tMatKj = (kExceedsJ) ? gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + oIdxY] // tkj.T
                                                    : gTMatKj[(oIdxY + nOsvK) * nOsvKj + oIdxZ]; // tjk
                        value += tMatKj * sMatKjij;
                    }

                    // oIdxZ = j
                    for (int oIdxZ = 0; oIdxZ < nOsvJ; oIdxZ++) {
                        double sMatKjij = sMat[sfStartJj + oIdxZ * nOsvJ + oIdxX]; // sjj

                        double tMatKj = (kExceedsJ) ? gTMatKj[oIdxZ * nOsvKj + oIdxY] // tjj.T
                                                    : gTMatKj[(oIdxY + nOsvK) * nOsvKj + (oIdxZ + nOsvK)]; // tjj
                        value += tMatKj * sMatKjij;
                    }

                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] = value;
                }
            }*/


            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 1           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (nOsvK + nOsvI);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvI;
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxX];
                            }
                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxX * nOsvK + oIdxZ];
                            }
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }

                        

                        __syncthreads();
                    }
                }

                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvK;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxX * nOsvJ + oIdxZ];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }
            
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 2           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (nOsvK + nOsvJ);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvJ;
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;


                        if (kExceedsJ) {

                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxX * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {

                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxX];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {

                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }

                        

                        __syncthreads();
                    }
                }



                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bNOsvZ * nOsvK;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartJj + oIdxZ * nOsvJ + oIdxX];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvK;
                                int oIdxY = bTMatIdx % nOsvK;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + (oIdxY + nOsvJ)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvK + oIdxY] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxY * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvK; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + oIdxY * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 3           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (nOsvJ + nOsvI);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvI;

                
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvJ;

                        if (iExceedsK) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvI;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvI;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxZ * nOsvI + oIdxX];
                            }
                        } else {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartIk + oIdxX * nOsvK + oIdxZ];
                            }
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        double sMatKjij = (iExceedsK) ? sSMat[oIdxZ * nOsvI + oIdxX]
                                                                    : sSMat[oIdxX * bNOsvZ + oIdxZ];

                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sMatKjij;
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }
                        
                        __syncthreads();
                    }
                }

                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvI;
                        int bTMatSize = bNOsvZ * nOsvJ;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx / bNOsvZ;
                            int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                            sSMat[bSMatIdx] = sMat[sfStartIj + oIdxX * nOsvJ + oIdxZ];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvI; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + oIdxX] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////          QUADRANT 4           //////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (2 * nOsvJ);
                double *sSMat = sMem;
                double *sTMat = sSMat + maxOsvZBatch * nOsvJ;

                
                // Z part 1: occZ = k
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvK; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvK - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bSMatSize;


                        if (kExceedsJ) {
                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx / bNOsvZ;
                                int oIdxZ = oIdxZStart + bSMatIdx % bNOsvZ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxX * nOsvK + oIdxZ];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxZ + nOsvJ) * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sSMat[oIdxX * bNOsvZ + oIdxZ];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {

                            for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                                int oIdxX = bSMatIdx % nOsvJ;
                                int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                                sSMat[bSMatIdx] = sMat[sfStartKj + oIdxZ * nOsvJ + oIdxX];
                            }

                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + oIdxZ];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }
                        
                        __syncthreads();
                    }
                }


                // Z part 2: occZ = j
                {
                    for (int oIdxZStart = 0; oIdxZStart < nOsvJ; oIdxZStart += maxOsvZBatch) {
                        int bNOsvZ = min(maxOsvZBatch, nOsvJ - oIdxZStart);
                        int bSMatSize = bNOsvZ * nOsvJ;
                        int bTMatSize = bSMatSize;

                        for (int bSMatIdx = tid; bSMatIdx < bSMatSize; bSMatIdx += threadsPerBlock) {
                            int oIdxX = bSMatIdx % nOsvJ;
                            int oIdxZ = oIdxZStart + bSMatIdx / nOsvJ;
                            sSMat[bSMatIdx] = sMat[sfStartJj + oIdxZ * nOsvJ + oIdxX];
                        }

                        if (kExceedsJ) {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx / nOsvJ;
                                int oIdxY = bTMatIdx % nOsvJ;
                                sTMat[bTMatIdx] = gTMatKj[oIdxZ * nOsvKj + oIdxY];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxZ * nOsvJ + oIdxY] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }

                        } else {
                            for (int bTMatIdx = tid; bTMatIdx < bTMatSize; bTMatIdx += threadsPerBlock) {
                                int oIdxZ = oIdxZStart + bTMatIdx % bNOsvZ;
                                int oIdxY = bTMatIdx / bNOsvZ;
                                sTMat[bTMatIdx] = gTMatKj[(oIdxY + nOsvK) * nOsvKj + (oIdxZ + nOsvK)];
                            }

                            __syncthreads();

                            for (int oIdxY = threadIdx.y; oIdxY < nOsvJ; oIdxY += blockDim.y) {
                                for (int oIdxX = threadIdx.x; oIdxX < nOsvJ; oIdxX += blockDim.x) {
                                    double value = 0.0;
                                    for (int oIdxZ = 0; oIdxZ < bNOsvZ; ++oIdxZ) {
                                        value += sTMat[oIdxY * bNOsvZ + oIdxZ] * sSMat[oIdxZ * nOsvJ + oIdxX];
                                    }
                                    tempMatB[tempIdx0 + (oIdxY + nOsvK) * nOsvIj + (oIdxX + nOsvI)] += value;
                                }
                            }
                            
                        }

                        __syncthreads();
                    }
                }
                
            }


            // Accumulate to rMat: tempMatA * tempMatB
            /*
            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < nOsvKj; osvIdxZ++) {
                        value += tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxZ] * tempMatB[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }
                    rMatIj[osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }*/

            {
                int maxOsvZBatch = CLUSTER_RES_SMEM / (2 * nOsvIj);
                int sAMatSize = maxOsvZBatch * nOsvIj;

                double *sAMat = sMem;
                double *sBMat = sAMat + sAMatSize;

                // Z part 1: occZ = i
                for (int oIdxZStart = 0; oIdxZStart < nOsvKj; oIdxZStart += maxOsvZBatch) {
                    int bNOsvZ = min(maxOsvZBatch, nOsvKj - oIdxZStart);
                    int bAMatSize = bNOsvZ * nOsvIj;
                    int bBMatSize = bAMatSize;

                    for (int bBMatIdx = tid; bBMatIdx < bBMatSize; bBMatIdx += threadsPerBlock) {
                        int osvIdxX = bBMatIdx % nOsvIj;
                        int osvIdxZ = oIdxZStart + bBMatIdx / nOsvIj;
                        sBMat[bBMatIdx] = tempMatB[tempIdx0 + osvIdxZ * nOsvIj + osvIdxX];
                    }

                    for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                        int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                        int osvIdxY = bAMatIdx / bNOsvZ;
                        sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nOsvKj + osvIdxZ];
                    }

                    __syncthreads();

                    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                            double value = 0.0;
                            for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                                value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sBMat[osvIdxZ * nOsvIj + osvIdxX];
                            }
                            rMatIj[osvIdxY * nOsvIj + osvIdxX] += value;
                        }
                    }

                    __syncthreads();
                }
            }

        }
    }
}


__device__ void updateClusterTMatDevice(double *tMat, double *rMat, double *xMat, double *emuij, double *tempMatA,
                                        int64_t *tempOffsets, int64_t *tOffsets, int64_t *rOffsets, int64_t *xOffsets,
                                        int64_t *eOffsets, int *nOsvOcc, int nOcc,
                                        int64_t pairIdx, int extPairIdx, int64_t iPair, bool useDynAmp, double *sMem) {
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int i = iPair / nOcc;
    int j = iPair % nOcc;
    int nOsvIj = nOsvOcc[i] + nOsvOcc[j];

    //int64_t rIdx0 = rOffsets[blockIdx.x];
    int64_t rIdx0 = (useDynAmp) ? rOffsets[blockIdx.x] : rOffsets[pairIdx];
    int64_t tIdx0 = tOffsets[pairIdx];

    int64_t xIdx0 = xOffsets[extPairIdx];
    int nColXMat = (xOffsets[extPairIdx + 1] - xIdx0) / nOsvIj;
    int64_t eIdx0 = eOffsets[extPairIdx];
    int64_t tempIdx0 = tempOffsets[blockIdx.x];

    // aMat = xij.T * rij
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvIj; ++osvIdxZ) {
                value += xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxY] * rMat[rIdx0 + osvIdxZ * nOsvIj + osvIdxX];
            }
            tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = value;
        }
    }

    __syncthreads();*/


    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] = 0.0;
            }
        }

        int maxOsvZBatch = CLUSTER_RES_SMEM / (nColXMat + nOsvIj);
        int sXMatSize = maxOsvZBatch * nColXMat;

        double *sXMat = sMem;
        double *sRMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvIj; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvIj - oIdxZStart);
            int bXMatSize = bNOsvZ * nColXMat;
            int bRMatSize = bNOsvZ * nOsvIj;

            for (int bRMatIdx = tid; bRMatIdx < bRMatSize; bRMatIdx += threadsPerBlock) {
                int osvIdxX = bRMatIdx % nOsvIj;
                int osvIdxZ = oIdxZStart + bRMatIdx / nOsvIj;
                sRMat[bRMatIdx] = rMat[rIdx0 + osvIdxZ * nOsvIj + osvIdxX];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx / nColXMat;
                int osvIdxY = bXMatIdx % nColXMat;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxY];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sXMat[osvIdxZ * nColXMat + osvIdxY] * sRMat[osvIdxZ * nOsvIj + osvIdxX];
                    }
                    tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    
    // r'ij = emuij * (aMat * xij)
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nOsvIj; ++osvIdxZ) {
                value += tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxZ] * xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }

            int idx = osvIdxY * nColXMat + osvIdxX;
            rMat[rIdx0 + idx] = emuij[eIdx0 + idx] * value;
        }
    }*/

    
    
    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                rMat[rIdx0 + osvIdxY * nColXMat + osvIdxX] = 0.0;
            }
        }

        int maxOsvZBatch = CLUSTER_RES_SMEM / (2 * nColXMat);
        int sXMatSize = maxOsvZBatch * nColXMat;

        double *sXMat = sMem;
        double *sAMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nOsvIj; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nOsvIj - oIdxZStart);
            int bXMatSize = bNOsvZ * nColXMat;
            int bAMatSize = bXMatSize;

            for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                int osvIdxY = bAMatIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nOsvIj + osvIdxZ];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx / nColXMat;
                int osvIdxX = bXMatIdx % nColXMat;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sXMat[osvIdxZ * nColXMat + osvIdxX];
                    }
                    rMat[rIdx0 + osvIdxY * nColXMat + osvIdxX] += value;
                }
            }

            __syncthreads();
        }

        for (int osvIdxY = threadIdx.y; osvIdxY < nColXMat; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                int idx = osvIdxY * nColXMat + osvIdxX;
                rMat[rIdx0 + idx] *= emuij[eIdx0 + idx];
            }
        }
    }


    __syncthreads();

    
    // aMat = xij * r'ij
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nColXMat; ++osvIdxZ) {
                value += xMat[xIdx0 + osvIdxY * nColXMat + osvIdxZ] * rMat[rIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }
            tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxX] = value;
        }
    }
    __syncthreads();   */ 
    

    
    {
        for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
            for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxX] = 0.0;
            }
        }
        int maxOsvZBatch = CLUSTER_RES_SMEM / (nColXMat + nOsvIj);
        int sXMatSize = maxOsvZBatch * nOsvIj;

        double *sXMat = sMem;
        double *sRMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nColXMat; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nColXMat - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvIj;
            int bRMatSize = bNOsvZ * nColXMat;

            for (int bRMatIdx = tid; bRMatIdx < bRMatSize; bRMatIdx += threadsPerBlock) {
                int osvIdxX = bRMatIdx % nColXMat;
                int osvIdxZ = oIdxZStart + bRMatIdx / nColXMat;
                sRMat[bRMatIdx] = rMat[rIdx0 + osvIdxZ * nColXMat + osvIdxX];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx % bNOsvZ;
                int osvIdxY = bXMatIdx / bNOsvZ;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxY * nColXMat + osvIdxZ];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nColXMat; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sXMat[osvIdxY * bNOsvZ + osvIdxZ] * sRMat[osvIdxZ * nColXMat + osvIdxX];
                    }
                    tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }


    

    // tMat += (aMat * xij.T)
    /*
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
            double value = 0.0;
            for (int osvIdxZ = 0; osvIdxZ < nColXMat; ++osvIdxZ) {
                value += tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxZ] * xMat[xIdx0 + osvIdxX * nColXMat + osvIdxZ];
            }

            tMat[tIdx0 + osvIdxY * nOsvIj + osvIdxX] += value;
        }
    }
    __syncthreads();    
    */
    
    {
        int maxOsvZBatch = CLUSTER_RES_SMEM / (2 * nOsvIj);
        int sXMatSize = maxOsvZBatch * nOsvIj;

        double *sXMat = sMem;
        double *sAMat = sXMat + sXMatSize;

        for (int oIdxZStart = 0; oIdxZStart < nColXMat; oIdxZStart += maxOsvZBatch) {
            int bNOsvZ = min(maxOsvZBatch, nColXMat - oIdxZStart);
            int bXMatSize = bNOsvZ * nOsvIj;
            int bAMatSize = bXMatSize;

            for (int bAMatIdx = tid; bAMatIdx < bAMatSize; bAMatIdx += threadsPerBlock) {
                int osvIdxY = bAMatIdx / bNOsvZ;
                int osvIdxZ = oIdxZStart + bAMatIdx % bNOsvZ;
                sAMat[bAMatIdx] = tempMatA[tempIdx0 + osvIdxY * nColXMat + osvIdxZ];
            }

            for (int bXMatIdx = tid; bXMatIdx < bXMatSize; bXMatIdx += threadsPerBlock) {
                int osvIdxZ = oIdxZStart + bXMatIdx % bNOsvZ;
                int osvIdxX = bXMatIdx / bNOsvZ;
                sXMat[bXMatIdx] = xMat[xIdx0 + osvIdxX * nColXMat + osvIdxZ];
            }

            __syncthreads();

            for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
                for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
                    double value = 0.0;
                    for (int osvIdxZ = 0; osvIdxZ < bNOsvZ; ++osvIdxZ) {
                        value += sAMat[osvIdxY * bNOsvZ + osvIdxZ] * sXMat[osvIdxX * bNOsvZ + osvIdxZ];
                    }
                    tMat[tIdx0 + osvIdxY * nOsvIj + osvIdxX] += value;
                }
            }

            __syncthreads();
        }
    }

    
}

__device__ void clusterMp2EneDevice(double *tMat, double *kMat, double *pairEnergies, int64_t *kOffsets,
                                    int64_t *tOffsets, int *nOsvOcc, int64_t nOcc,
                                    int pairIdx, int extPairIdx, int iPair, double *sMem) {

    //__shared__ double sPairEne[THREADS_PER_BLOCK];
    double *sPairEne = sMem;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int i = iPair / nOcc;
    int j = iPair % nOcc;
    int nOsvIj = nOsvOcc[i] + nOsvOcc[j];

    //int64_t kIdx0 = kOffsets[extPairIdx];
    //int64_t tIdx0 = tOffsets[pairIdx];
    double *kMatIj = kMat + kOffsets[extPairIdx];
    double *tMatIj = tMat + tOffsets[pairIdx];

    double tPairEne = 0.0;

    // aMat = kMat * (2 * tMat - tMat.T)
    for (int osvIdxY = threadIdx.y; osvIdxY < nOsvIj; osvIdxY += blockDim.y) {
        for (int osvIdxX = threadIdx.x; osvIdxX < nOsvIj; osvIdxX += blockDim.x) {
            int idxIj = osvIdxY * nOsvIj + osvIdxX;
            //double kij = kMat[kIdx0 + idx];
            //double tij = tMat[tIdx0 + idx];
            //double tji = tMat[tIdx0 + osvIdxX * nOsvIj + osvIdxY];
            double kij = kMatIj[idxIj];
            double tij = tMatIj[idxIj];
            double tji = tMatIj[osvIdxX * nOsvIj + osvIdxY];
            tPairEne += kij * (2 * tij - tji);
        }
    }

    sPairEne[tid] = tPairEne;

    __syncthreads();

    // Parallel reduction
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sPairEne[tid] += sPairEne[tid + stride];
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        pairEnergies[pairIdx] = (i == j) ? sPairEne[0] : 2 * sPairEne[0];
    }
}

__global__ void clusterResidualKernel(double *sMat, double *fMat, double *tMat, double *tInit, double *kMat, double *rMat,
                                      double *xMat, double *emuij, double *tempMatA, double *tempMatB,
                                      double *locFock, double *pairEnergies, int *resCycles,
                                      int *pairs, int *nOsvOcc, int *kOccPair,
                                      int64_t *sfOffsets, int64_t *kOffsets, int64_t *tOffsets,
                                      int64_t *rOffsets, int64_t *xOffsets,
                                      int64_t *eOffsets, int64_t *tempOffsets,
                                      int64_t *kOccOffsets, int *extPairIndices,
                                      int *pairOffsets, int nOcc, double kOccTol,
                                      double mp2EneTol, int maxCycle, bool useDynAmp) {

    __shared__ double sMem[CLUSTER_RES_SMEM];
    
    int clusIdx = blockIdx.x;
    int blockPairIdx0 = pairOffsets[clusIdx];
    int blockPairIdx1 = pairOffsets[clusIdx + 1];

    // Copy tInit -> tMat
    for (int pairIdx = blockPairIdx0; pairIdx < blockPairIdx1; ++pairIdx) {
        int iPair = pairs[pairIdx];
        int extPairIdx = extPairIndices[iPair];

        int tInitIdx0 = kOffsets[extPairIdx];
        int tInitSize = kOffsets[extPairIdx + 1] - tInitIdx0;

        // Initialize R matrix
        copyArrayDevice(tMat, tInit, tOffsets[pairIdx], tInitIdx0, tInitSize);
    }

    __syncthreads();

    double mp2Ene = 0.0, lastMp2Ene = 0.0;
    for (int iCycle = 0; iCycle < maxCycle; ++iCycle) {
        mp2Ene = 0.0;

        for (int pairIdx = blockPairIdx0; pairIdx < blockPairIdx1; ++pairIdx) {
            int iPair = pairs[pairIdx];
            int extPairIdx = extPairIndices[iPair];

            int kIdx0 = kOffsets[extPairIdx];
            int kMatSize = kOffsets[extPairIdx + 1] - kIdx0;

            // Initialize R matrix
            int64_t rIdx0 = (useDynAmp) ? rOffsets[clusIdx] : rOffsets[pairIdx];
            copyArrayDevice(rMat, kMat, rIdx0, kIdx0, kMatSize);

            // Compute R matrix
            clusterRmatDevice(sMat, fMat, tMat, rMat,
                              tempMatA, tempMatB, locFock, pairs,
                              nOsvOcc, kOccPair, tOffsets, rOffsets, sfOffsets, tempOffsets,
                              kOccOffsets, extPairIndices,
                              nOcc, kOccTol, pairIdx, iPair,
                              blockPairIdx0, blockPairIdx1, useDynAmp, sMem);
            
            if (useDynAmp) {
                // Update MP2 amplitudes
                updateClusterTMatDevice(tMat, rMat, xMat, emuij, tempMatA,
                                        tempOffsets, tOffsets, rOffsets, xOffsets,
                                        eOffsets, nOsvOcc, nOcc,
                                        pairIdx, extPairIdx, iPair, useDynAmp, sMem);

                // Compute pair energies
                clusterMp2EneDevice(tMat, kMat, pairEnergies, kOffsets, tOffsets, nOsvOcc,
                                    nOcc, pairIdx, extPairIdx, iPair, sMem);

                __syncthreads();

                mp2Ene += pairEnergies[pairIdx];
            }
        }

        if (!useDynAmp) {
            __syncthreads();

            for (int pairIdx = blockPairIdx0; pairIdx < blockPairIdx1; ++pairIdx) {
                int iPair = pairs[pairIdx];
                int extPairIdx = extPairIndices[iPair];

                // Update MP2 amplitudes
                updateClusterTMatDevice(tMat, rMat, xMat, emuij, tempMatA,
                                        tempOffsets, tOffsets, rOffsets, xOffsets,
                                        eOffsets, nOsvOcc, nOcc,
                                        pairIdx, extPairIdx, iPair, useDynAmp, sMem);

                // Compute pair energies
                clusterMp2EneDevice(tMat, kMat, pairEnergies, kOffsets, tOffsets, nOsvOcc,
                                    nOcc, pairIdx, extPairIdx, iPair, sMem);

                __syncthreads();

                mp2Ene += pairEnergies[pairIdx];
            }
        }

        if (fabsf(mp2Ene - lastMp2Ene) < mp2EneTol) {
            if (threadIdx.y * blockDim.x + threadIdx.x == 0) {
                resCycles[clusIdx] = iCycle + 1;
            }
            break;
        }

        lastMp2Ene = mp2Ene;
    }
}

void osvSFCupy(py::object qMat, py::object sMat, py::object fMat, py::object sRatio, py::object eneVir,
               py::object pairs, py::object nOsvOcc, py::object qOffsets, py::object sfOffsets,
               int nVir, int nOcc, int nPair) {
    // Extracting buffer info from cupy arrays
    double *ptrQ = getCupyPtr<double>(qMat);
    double *ptrS = getCupyPtr<double>(sMat);
    double *ptrF = getCupyPtr<double>(fMat);
    double *ptrSRatio = getCupyPtr<double>(sRatio);
    double *ptrEVir = getCupyPtr<double>(eneVir);
    int *ptrPairs = getCupyPtr<int>(pairs);
    int *ptrNosvOcc = getCupyPtr<int>(nOsvOcc);
    int64_t *ptrQOffsets = getCupyPtr<int64_t>(qOffsets);
    int64_t *ptrSfOffsets = getCupyPtr<int64_t>(sfOffsets);

    //  Defining block and grid sizes
    //   Assign one pair to one block

    // dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);
    int threadsPerBlock = LARGE_THREADS_PER_BLOCK;

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int sharedMemSize = prop.sharedMemPerBlock; // To avoid low occupancy


    // Request max sharedMem
    cudaFuncSetAttribute(osvSFKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // osvSFKernel<<<nPair, threadsPerBlock, sharedMemSize>>>(ptrQ, ptrS, ptrF, ptrSRatio, ptrEVir,
    osvSFKernel<<<nPair, threadsPerBlock>>>(ptrQ, ptrS, ptrF, ptrSRatio, ptrEVir,
                                            ptrPairs, ptrNosvOcc, ptrQOffsets, ptrSfOffsets,
                                            nVir, nOcc); //, sharedMemSize);
}

void imujpCupy(py::object ialp, py::object qao, py::object imujp, py::object fitPair,
               py::object nFitPair, py::object nOsvOcc, py::object qaoOffsets,
               py::object imujpOffsets, py::object fitLocalOffsets,
               py::object occJsBlock, py::object occJIndices, py::object osvIndices,
               py::object fitLocalIndices, int nAo, int nAux, int blocks, int threadsX, int threadsY) {

    // Extracting buffer info from cupy arrays
    double *ptrIalp = getCupyPtr<double>(ialp);
    double *ptrQao = getCupyPtr<double>(qao);
    double *ptrImujp = getCupyPtr<double>(imujp);
    int *ptrFitPair = getCupyPtr<int>(fitPair);
    int *ptrNFitPair = getCupyPtr<int>(nFitPair);
    int *ptrNosvOcc = getCupyPtr<int>(nOsvOcc);
    int64_t *ptrQaoOffsets = getCupyPtr<int64_t>(qaoOffsets);
    int64_t *ptrImujpOffsets = getCupyPtr<int64_t>(imujpOffsets);
    int *ptrFitLocalOffsets = getCupyPtr<int>(fitLocalOffsets);
    int *ptrOccJsBlock = getCupyPtr<int>(occJsBlock);
    int *ptrOccJIndices = getCupyPtr<int>(occJIndices);
    int *ptrOsvIndices = getCupyPtr<int>(osvIndices);
    int *ptrFitLocalIndices = getCupyPtr<int>(fitLocalIndices);

    //  Defining block and grid sizes
    dim3 threads(threadsX, threadsY);
    // dim3 blocks((cb + threadsPerBlock.x - 1) / threadsPerBlock.x, (ra + threadsPerBlock.y - 1) / threadsPerBlock.y);


    // Request max sharedMem
    cudaFuncSetAttribute(imujpKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // Launching CUDA kernel
    // imujpKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(ptrIalp, ptrQao, ptrImujp, ptrFitPair,
    imujpKernel<<<blocks, threads>>>(ptrIalp, ptrQao, ptrImujp, ptrFitPair,
                                     ptrNFitPair, ptrNosvOcc, ptrQaoOffsets,
                                     ptrImujpOffsets, ptrFitLocalOffsets,
                                     ptrOccJsBlock, ptrOccJIndices, ptrOsvIndices,
                                     ptrFitLocalIndices, nAo, nAux);
}

void closeOsvKmatCupy(py::object imujp, py::object kMat, py::object pairs,
                      py::object fitPair, py::object nOsvOcc, py::object fitLocalOffsets,
                      py::object imujpOffsets, py::object kOffsets,
                      int nOcc, int nAux, int nPair) {

    // Extracting buffer info from cupy arrays
    double *ptrImujp = getCupyPtr<double>(imujp);
    double *ptrKMat = getCupyPtr<double>(kMat);
    int *ptrPair = getCupyPtr<int>(pairs);
    int *ptrFitPair = getCupyPtr<int>(fitPair);
    int *ptrNosvOcc = getCupyPtr<int>(nOsvOcc);
    int64_t *ptrFitLocalOffsets = getCupyPtr<int64_t>(fitLocalOffsets);
    int64_t *ptrImujpOffsets = getCupyPtr<int64_t>(imujpOffsets);
    int64_t *ptrKOffsets = getCupyPtr<int64_t>(kOffsets);

    //  Defining block and grid sizes
    dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);
    // int threadsPerBlock = THREADS_PER_BLOCK;
    //  dim3 blocks((cb + threadsPerBlock.x - 1) / threadsPerBlock.x, (ra + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launching CUDA kernel

    closeOsvKmatKernel<<<nPair, threadsPerBlock>>>(ptrImujp, ptrKMat, ptrPair,
                                                   ptrFitPair, ptrNosvOcc, ptrFitLocalOffsets,
                                                   ptrImujpOffsets, ptrKOffsets, // ptrImujpPairIndices,
                                                   nOcc, nAux);
}

void remoteOsvKmatCupy(py::object imujp, py::object kMat, py::object pairs,
                       py::object fitPair, py::object nOsvOcc, py::object fitLocalOffsets,
                       py::object imujpOffsets, py::object kOffsets,
                       int64_t nOcc, int64_t nAux, int64_t nPair) {

    // Extracting buffer info from cupy arrays
    double *ptrImujp = getCupyPtr<double>(imujp);
    double *ptrKMat = getCupyPtr<double>(kMat);
    int *ptrPair = getCupyPtr<int>(pairs);
    int *ptrFitPair = getCupyPtr<int>(fitPair);
    int *ptrNosvOcc = getCupyPtr<int>(nOsvOcc);
    int64_t *ptrFitLocalOffsets = getCupyPtr<int64_t>(fitLocalOffsets);
    int64_t *ptrImujpOffsets = getCupyPtr<int64_t>(imujpOffsets);
    int64_t *ptrKOffsets = getCupyPtr<int64_t>(kOffsets);
    // int64_t *ptrImujpPairIndices = getCupyPtr<int64_t>(imujpPairIndices);

    //  Defining block and grid sizes
    dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);
    // int threadsPerBlock = THREADS_PER_BLOCK;
    //  dim3 blocks((cb + threadsPerBlock.x - 1) / threadsPerBlock.x, (ra + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Request max sharedMem
    cudaFuncSetAttribute(remoteOsvKmatKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // Launching CUDA kernel
    remoteOsvKmatKernel<<<nPair, threadsPerBlock>>>(ptrImujp, ptrKMat, ptrPair,
                                                    ptrFitPair, ptrNosvOcc, ptrFitLocalOffsets,
                                                    ptrImujpOffsets, ptrKOffsets, // ptrImujpPairIndices,
                                                    nOcc, nAux);
}

void closeOsvPreconCupy(py::object sMat, py::object fMat, py::object eOcc, 
                        py::object xMat, py::object emuij, py::object tempA, 
                        py::object tempB, py::object tempC, py::object pairs, 
                        py::object extPairIndices, py::object nOsvOcc, py::object nColXmat,
                        py::object sfOffsets, py::object xOffsets, int nOcc, int nPair,
                        int threads, int maxNosvIj) {

    // Extracting buffer info from cupy arrays
    double *ptrSMat = getCupyPtr<double>(sMat);
    double *ptrFMat = getCupyPtr<double>(fMat);
    double *ptrEOcc = getCupyPtr<double>(eOcc);
    double *ptrXMat = getCupyPtr<double>(xMat);
    double *ptrEmuij = getCupyPtr<double>(emuij);
    double *ptrTempA = getCupyPtr<double>(tempA);
    double *ptrTempB = getCupyPtr<double>(tempB);
    double *ptrTempC = getCupyPtr<double>(tempC);

    int *ptrPairs = getCupyPtr<int>(pairs);
    int *ptrExtPairIndices = getCupyPtr<int>(extPairIndices);
    int *ptrNosvOcc = getCupyPtr<int>(nOsvOcc);
    int *ptrNColXmat = getCupyPtr<int>(nColXmat);

    int64_t *ptrSfOffsets = getCupyPtr<int64_t>(sfOffsets);
    int64_t *ptrXOffsets = getCupyPtr<int64_t>(xOffsets);

    //  Defining block and grid sizes
    //dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);
    int threadsPerAxis = std::sqrt(threads);
    dim3 threadsPerBlock(threadsPerAxis, threadsPerAxis);
    //dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);
    //int threadsPerBlock = threads;


    cudaDeviceProp deviceProp; 
    int device; 
    cudaGetDevice(&device); 
    cudaGetDeviceProperties(&deviceProp, device);

    //int minShmSize = SMEM_INT_CLOSE_PRECON * 4; // bytes
    int memRequired = 4 * maxNosvIj * sizeof(double) + (maxNosvIj+2) * sizeof(int); // bytes
    memRequired = std::max(SMEM_CLOSE_PRECON, memRequired);

    if (memRequired > deviceProp.sharedMemPerBlockOptin) {
        throw std::runtime_error("Requested shared memory exceeds limit");
    }

    int blocksPerSM = (deviceProp.sharedMemPerMultiprocessor + memRequired - 1) / memRequired;
    memRequired = std::max((int)deviceProp.sharedMemPerMultiprocessor / blocksPerSM, memRequired);
    memRequired = std::min((int)deviceProp.sharedMemPerBlockOptin, memRequired);

    // printf("Requesting SHM %d bytes\n", memRequired);

    // Request max sharedMem
    cudaFuncSetAttribute(closeOsvPreconKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // Launching CUDA kernel
    closeOsvPreconKernel<<<nPair, threadsPerBlock, memRequired>>>(ptrSMat, ptrFMat, ptrEOcc, ptrXMat,
                                            ptrEmuij, ptrTempA, ptrTempB, ptrTempC,
                                            ptrPairs, ptrExtPairIndices, ptrNosvOcc, ptrNColXmat,
                                            ptrSfOffsets, ptrXOffsets, nOcc, memRequired/sizeof(int));
}

void remoteOsvPreconCupy(py::object fMat, py::object xMat, py::object emui,
                         py::object tempMat, py::object nOsvBlock,
                         py::object fiiOffsets, py::object eOffsets,
                         int blocks, int threads, int maxNosv) {
    
    // Extract raw pointers from CuPy arrays
    double* ptrFMat = getCupyPtr<double>(fMat);
    double* ptrXMat = getCupyPtr<double>(xMat);
    double* ptrEmui = getCupyPtr<double>(emui);
    double* ptrTempMat = getCupyPtr<double>(tempMat);

    int* ptrNBlock = getCupyPtr<int>(nOsvBlock);
    int64_t* ptrFiiOffsets = getCupyPtr<int64_t>(fiiOffsets);
    int64_t* ptrEOffsets = getCupyPtr<int64_t>(eOffsets);


    cudaDeviceProp deviceProp; 
    int device; 
    cudaGetDevice(&device); 
    cudaGetDeviceProperties(&deviceProp, device);

    //int minShmSize = SMEM_INT_CLOSE_PRECON * 4; // bytes
    int memRequired = 4 * maxNosv * sizeof(double) + (maxNosv+2) * sizeof(int); // bytes
    memRequired = std::max(SMEM_REMOTE_PRECON, memRequired);

    if (memRequired > deviceProp.sharedMemPerBlockOptin) {
        throw std::runtime_error("Requested shared memory exceeds limit");
    }

    int blocksPerSM = (deviceProp.sharedMemPerMultiprocessor + memRequired - 1) / memRequired;
    memRequired = std::max((int)deviceProp.sharedMemPerMultiprocessor / blocksPerSM, memRequired);
    memRequired = std::min((int)deviceProp.sharedMemPerBlockOptin, memRequired);

    // printf("Requesting SHM %d bytes\n", memRequired);

    // Request max sharedMem
    cudaFuncSetAttribute(remoteOsvPreconKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // Call the kernel
    remoteOsvPreconKernel<<<blocks, threads, memRequired>>>(ptrFMat, ptrXMat, ptrEmui, ptrTempMat,
                                               ptrNBlock, ptrFiiOffsets, ptrEOffsets);
}

void clusterResidualCupy(py::object sMat, py::object fMat, py::object tMat, py::object tInit, py::object kMat,
                         py::object rMat, py::object xMat, py::object emuij, py::object tempMatA,
                         py::object tempMatB, py::object locFock, py::object pairEnergies, py::object resCycles,
                         py::object pairs, py::object nOsvOcc, py::object kOccPair,
                         py::object sfOffsets, py::object kOffsets, py::object tOffsets, py::object rOffsets,
                         py::object xOffsets, py::object eOffsets,
                         py::object tempOffsets, py::object kOccOffsets,
                         py::object extPairIndices, py::object pairOffsets,
                         int nOcc, double kOccTol, double mp2EneTol, int maxCycle, bool useDynAmp) {

    // Extract raw pointers from CuPy arrays
    double *ptrS = getCupyPtr<double>(sMat);
    double *ptrF = getCupyPtr<double>(fMat);
    double *ptrT = getCupyPtr<double>(tMat);
    double *ptrTinit = getCupyPtr<double>(tInit);
    double *ptrK = getCupyPtr<double>(kMat);
    double *ptrR = getCupyPtr<double>(rMat);
    double *ptrX = getCupyPtr<double>(xMat);
    double *ptrEmuij = getCupyPtr<double>(emuij);
    double *ptrTempA = getCupyPtr<double>(tempMatA);
    double *ptrTempB = getCupyPtr<double>(tempMatB);
    double *ptrLocFock = getCupyPtr<double>(locFock);
    double *ptrPairEnergies = getCupyPtr<double>(pairEnergies);

    int *ptrResCycles = getCupyPtr<int>(resCycles);
    int *ptrPairs = getCupyPtr<int>(pairs);
    int *ptrNOsvOcc = getCupyPtr<int>(nOsvOcc);
    int *ptrKOccPair = getCupyPtr<int>(kOccPair);

    int64_t *ptrSfOffsets = getCupyPtr<int64_t>(sfOffsets);
    int64_t *ptrKOffsets = getCupyPtr<int64_t>(kOffsets);
    int64_t *ptrTOffsets = getCupyPtr<int64_t>(tOffsets);
    int64_t *ptrROffsets = getCupyPtr<int64_t>(rOffsets);
    int64_t *ptrXOffsets = getCupyPtr<int64_t>(xOffsets);
    int64_t *ptrEOffsets = getCupyPtr<int64_t>(eOffsets);
    int64_t *ptrTempOffsets = getCupyPtr<int64_t>(tempOffsets);

    int64_t *ptrKOccOffsets = getCupyPtr<int64_t>(kOccOffsets);
    int *ptrPairExtIdx = getCupyPtr<int>(extPairIndices);
    int *ptrPairOffsets = getCupyPtr<int>(pairOffsets);

    // Determine grid dimensions from pairOffsets size
    int blocks = pairOffsets.attr("size").cast<int>() - 1;

    // Configure kernel launch parameters
    dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);

    // Request max sharedMem
    cudaFuncSetAttribute(clusterResidualKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // Launch kernel with extended grid
    clusterResidualKernel<<<blocks, threadsPerBlock>>>(
        ptrS, ptrF, ptrT, ptrTinit, ptrK, ptrR, ptrX, ptrEmuij,
        ptrTempA, ptrTempB, ptrLocFock, ptrPairEnergies, ptrResCycles,
        ptrPairs, ptrNOsvOcc, ptrKOccPair,
        ptrSfOffsets, ptrKOffsets, ptrTOffsets, ptrROffsets,
        ptrXOffsets,
        ptrEOffsets, ptrTempOffsets, ptrKOccOffsets,
        ptrPairExtIdx, ptrPairOffsets,
        nOcc, kOccTol, mp2EneTol, maxCycle, useDynAmp);
}



void closeResidualCupy(py::object sMat, py::object fMat, py::object tMatInit, py::object tMatNew, py::object kMat,
                       py::object rMat, py::object xMat, py::object emuij, py::object tempMatA,
                       py::object tempMatB, py::object locFock, py::object pairEnergies,
                       py::object pairs, py::object nOsvOcc, py::object kOccPair, py::object rOffsets,
                       py::object sfOffsets, py::object tInitOffsets, py::object tNewOffsets, py::object kOffsets,
                       py::object xOffsetsFull, py::object eOffsets, py::object tempOffsets,
                       py::object kOccOffsets, 
                       int nOcc, double kOccTol) {
    // Extract buffer info from CuPy arrays
    double *ptrS = getCupyPtr<double>(sMat);
    double *ptrF = getCupyPtr<double>(fMat);
    double *ptrTInit = getCupyPtr<double>(tMatInit);
    double *ptrTNew = getCupyPtr<double>(tMatNew);
    double *ptrK = getCupyPtr<double>(kMat);
    double *ptrR = getCupyPtr<double>(rMat);
    double *ptrX = getCupyPtr<double>(xMat);
    double *ptrEmuij = getCupyPtr<double>(emuij);
    double *ptrTempMatA = getCupyPtr<double>(tempMatA);
    double *ptrTempMatB = getCupyPtr<double>(tempMatB);
    double *ptrLocFock = getCupyPtr<double>(locFock);
    double *ptrPairEnergies = getCupyPtr<double>(pairEnergies);

    int *ptrPairs = getCupyPtr<int>(pairs);
    int *ptrNOsvOcc = getCupyPtr<int>(nOsvOcc);
    int *ptrKOccPair = getCupyPtr<int>(kOccPair);
    int64_t *ptrROffsets = getCupyPtr<int64_t>(rOffsets);
    int64_t *ptrSfOffsets = getCupyPtr<int64_t>(sfOffsets);
    int64_t *ptrTInitOffsets = getCupyPtr<int64_t>(tInitOffsets);
    int64_t *ptrTNewOffsets = getCupyPtr<int64_t>(tNewOffsets);
    int64_t *ptrKOffsets = getCupyPtr<int64_t>(kOffsets);
    int64_t *ptrXOffsetsFull = getCupyPtr<int64_t>(xOffsetsFull);
    int64_t *ptrEOffsets = getCupyPtr<int64_t>(eOffsets);
    int64_t *ptrTempOffsets = getCupyPtr<int64_t>(tempOffsets);
    int64_t *ptrKOccOffsets = getCupyPtr<int64_t>(kOccOffsets);
    // int64_t *ptrExtPairIndices = getCupyPtr<int64_t>(extPairIndices);

    // Determine grid dimensions from pairs size
    int blocks = pairs.attr("size").cast<int>();

    // Define block and grid sizes (adjust THREADS_PER_AXIS as needed)
    dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);

    // Request max sharedMem
    cudaFuncSetAttribute(closeResidualKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // Launch the CUDA kernel
    closeResidualKernel<<<blocks, threadsPerBlock>>>(
        ptrS, ptrF, ptrTInit, ptrTNew, ptrK, ptrR, ptrX, ptrEmuij, ptrTempMatA, ptrTempMatB,
        ptrLocFock, ptrPairEnergies, ptrPairs, ptrNOsvOcc, ptrKOccPair, ptrROffsets, ptrSfOffsets,
        ptrTInitOffsets, ptrTNewOffsets, ptrKOffsets, ptrXOffsetsFull, ptrEOffsets,
        ptrTempOffsets, ptrKOccOffsets, nOcc, kOccTol);
}

void remoteResidualCupy(py::object sMat, py::object fMat, py::object tMat, py::object kMat,
                        py::object rMat, py::object xMat, py::object emui, py::object eOcc, py::object tempMatA,
                        py::object locFock, py::object pairEnergies,
                        py::object pairs, py::object nOsvOcc, py::object rOffsets,
                        py::object sfOffsets, py::object tOffsets, py::object kOffsets,
                        py::object xOffsets, py::object eOffsets, py::object tempOffsets,
                        // py::object extPairIndices, int64_t nOcc) {
                        int nOcc) {
    // Extract device pointers from CuPy arrays
    double *ptrS = getCupyPtr<double>(sMat);
    double *ptrF = getCupyPtr<double>(fMat);
    double *ptrT = getCupyPtr<double>(tMat);
    double *ptrK = getCupyPtr<double>(kMat);
    double *ptrR = getCupyPtr<double>(rMat);
    double *ptrX = getCupyPtr<double>(xMat);
    double *ptrEmui = getCupyPtr<double>(emui);
    double *ptrEOcc = getCupyPtr<double>(eOcc);
    double *ptrTempMatA = getCupyPtr<double>(tempMatA);
    double *ptrLocFock = getCupyPtr<double>(locFock);
    double *ptrPairEnergies = getCupyPtr<double>(pairEnergies);

    int *ptrPairs = getCupyPtr<int>(pairs);
    int *ptrNOsvOcc = getCupyPtr<int>(nOsvOcc);
    int64_t *ptrROffsets = getCupyPtr<int64_t>(rOffsets);
    int64_t *ptrSfOffsets = getCupyPtr<int64_t>(sfOffsets);
    int64_t *ptrTOffsets = getCupyPtr<int64_t>(tOffsets);
    int64_t *ptrKOffsets = getCupyPtr<int64_t>(kOffsets);
    int64_t *ptrXOffsets = getCupyPtr<int64_t>(xOffsets);
    int64_t *ptrEOffsets = getCupyPtr<int64_t>(eOffsets);
    int64_t *ptrTempOffsets = getCupyPtr<int64_t>(tempOffsets);
    // int64_t *ptrExtPairIndices = getCupyPtr<int64_t>(extPairIndices);

    // Determine grid dimensions from pairs size
    int blocks = pairs.attr("size").cast<int>();

    // Configure kernel launch parameters
    dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);

    // Request max sharedMem
    cudaFuncSetAttribute(remoteResidualKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
                            
    // Launch kernel with 1D grid of blocks (one block per pair)
    remoteResidualKernel<<<blocks, threadsPerBlock>>>(
        ptrS, ptrF, ptrT, ptrK, ptrR, ptrX, ptrEmui, ptrEOcc, ptrTempMatA,
        ptrLocFock, ptrPairEnergies, ptrPairs, ptrNOsvOcc, ptrROffsets,
        ptrSfOffsets, ptrTOffsets, ptrKOffsets, ptrXOffsets,
        // ptrEOffsets, ptrTempOffsets, ptrExtPairIndices, nOcc);
        ptrEOffsets, ptrTempOffsets, nOcc);
}

// Binding the function for Python
PYBIND11_MODULE(osvMp2Cuda, m) {
    m.def("osvSFCupy", &osvSFCupy, "OSV sMat/fMat");
    m.def("imujpCupy", &imujpCupy, "OSV K: imujp");
    m.def("closeOsvKmatCupy", &closeOsvKmatCupy, "OSV K: kmat close");
    m.def("remoteOsvKmatCupy", &remoteOsvKmatCupy, "OSV K: kmat remote");
    m.def("closeOsvPreconCupy", &closeOsvPreconCupy, "closeOsvPreconCupy");
    m.def("remoteOsvPreconCupy", &remoteOsvPreconCupy, "remoteOsvPreconCupy");
    m.def("clusterResidualCupy", &clusterResidualCupy, "clusterResidualCupy");
    m.def("closeResidualCupy", &closeResidualCupy, "closeResidualCupy");
    m.def("remoteResidualCupy", &remoteResidualCupy, "remoteResidualCupy");
}
