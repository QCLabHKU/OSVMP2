#include <cstdint>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cupyUtils.cuh>

#define THREADS_PER_AXIS 16
#define THREADS_PER_BLOCK 256
#define LARGE_THREADS_PER_AXIS 32
#define LARGE_THREADS_PER_BLOCK 1024
#define NUM_STREAMS 16

#define SHD_ARRAY_SIZE 1024
#define STRIDE_OCC 4
#define STRIDE_AUX 4
namespace py = pybind11;

__global__ void sparseHalfTransKernel(
    const double* occCoeff, const double* int3c, double* ialp,
    const int* sortedPairOffsets, const int* otherAos, 
    const int* sortedPairIndices, const int* activeAos,
    const int nOcc, const int nAo, const int nAux, const int nAuxFull, 
    const int nSparsePair, const int nActiveAo, const int pIdx0){

    const int iStartBlk = blockIdx.x * blockDim.x * STRIDE_OCC;
    const int pIdxStartBlk = blockIdx.y * blockDim.y * STRIDE_AUX;

    int local_tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    
    __shared__ double sOccCoeff[SHD_ARRAY_SIZE];
    __shared__ double sInt3c[SHD_ARRAY_SIZE];

    int idx_ao = blockIdx.z * blockDim.z + threadIdx.z;
    int this_ao = activeAos[idx_ao];  
    int start = sortedPairOffsets[idx_ao];    
    int end = sortedPairOffsets[idx_ao + 1];

    int nOccBlock = min(nOcc - iStartBlk, blockDim.x * STRIDE_OCC);
    int nAuxBlock = min(nAux - pIdxStartBlk, blockDim.y * STRIDE_AUX);

    int batchSize = SHD_ARRAY_SIZE / max(nOccBlock, nAuxBlock);

    for (int beIdx0 = start; beIdx0 < end; beIdx0 += batchSize) {
        int nBeBatch = min(end - beIdx0, batchSize);

        for (int sOccCoeffIdx = local_tid; sOccCoeffIdx < (nBeBatch * nOccBlock); sOccCoeffIdx += threadsPerBlock) {
            size_t iNow = iStartBlk + sOccCoeffIdx / nBeBatch;
            int beIdx = beIdx0 + sOccCoeffIdx % nBeBatch;
            sOccCoeff[sOccCoeffIdx] = occCoeff[iNow * nAo + otherAos[beIdx]];
        }

        for (int sInt3cIdx = local_tid; sInt3cIdx < (nBeBatch * nAuxBlock); sInt3cIdx += threadsPerBlock) {
            size_t pairIdx = sortedPairIndices[beIdx0 + sInt3cIdx / nAuxBlock];
            int pIdxNow = pIdxStartBlk + sInt3cIdx % nAuxBlock;
            sInt3c[sInt3cIdx] = int3c[pairIdx * nAux + pIdxNow];
        }

        __syncthreads();

        
        for (int ipIdx = local_tid; ipIdx < (nOccBlock * nAuxBlock); ipIdx += threadsPerBlock) {
            int iBlock = ipIdx / nAuxBlock;
            int pIdxBlock = ipIdx % nAuxBlock; 

            double sum = 0.0;
            for (int batchBeIdx = 0; batchBeIdx < nBeBatch; ++batchBeIdx) {
                sum += sOccCoeff[iBlock * nBeBatch + batchBeIdx] * 
                        sInt3c[batchBeIdx * nAuxBlock + pIdxBlock];
            }

            ialp[(size_t)(iStartBlk+iBlock) * (nAo * nAuxFull) + this_ao * nAuxFull + (pIdxStartBlk+pIdxBlock) + pIdx0] += sum;
        }

        __syncthreads();
    }
    
}

void sparseHalfTransCupy(py::object occCoeff, py::object int3c, py::object ialp,
                         py::object sortedPairOffsets, py::object otherAos, 
                         py::object sortedPairIndices, py::object activeAos,
                         int nOcc, int nAo, int nAux, int nAuxFull, 
                         int nSparsePair, int nActiveAo, int pIdx0) {
    // Extract device pointers from CuPy arrays
    double* ptrOccCoeff         = getCupyPtr<double>(occCoeff);
    double* ptrInt3c            = getCupyPtr<double>(int3c);
    double* ptrIalp             = getCupyPtr<double>(ialp);
    int*    ptrSortedPairOffsets = getCupyPtr<int>(sortedPairOffsets);  // change to int64_t* if needed
    int*    ptrOtherAos         = getCupyPtr<int>(otherAos);
    int*    ptrSortedPairIndices = getCupyPtr<int>(sortedPairIndices);
    int*    ptrActiveAos       = getCupyPtr<int>(activeAos);

    constexpr dim3 threadsPerBlock(16, 16, 1);

    const int tasksPerThreadX = threadsPerBlock.x * STRIDE_OCC;  // 64
    const int tasksPerThreadY = threadsPerBlock.y * STRIDE_AUX;  // 64

    dim3 blocks(
        (nOcc  + tasksPerThreadX - 1) / tasksPerThreadX,
        (nAux  + tasksPerThreadY - 1) / tasksPerThreadY,
        (nActiveAo  + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    cudaFuncSetAttribute(sparseHalfTransKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    sparseHalfTransKernel<<<blocks, threadsPerBlock>>>(
        ptrOccCoeff, ptrInt3c, ptrIalp,
        ptrSortedPairOffsets, ptrOtherAos,
        ptrSortedPairIndices, ptrActiveAos,
        nOcc, nAo, nAux, nAuxFull,
        nSparsePair, nActiveAo, pIdx0
    );

}


// Bind the functions to a Python module
PYBIND11_MODULE(int3c2eCuda, m) {
    m.def("sparseHalfTransCupy", &sparseHalfTransCupy, "sparseHalfTransCupy");
}