#include <algorithm> // For std::min
#include <cmath>     // For std::abs, std::hypot, std::atan2, std::cos, std::sin, std::sqrt
#include <cstdlib>
#include <cublas_v2.h>               // For CUBLAS functions
#include <cuda_runtime.h>            // For CUDA runtime (kernel launches, memory management)
#include <iostream>                  // For std::cout
#include <pybind11/numpy.h>          // For NumPy/CuPy array handling
#include <pybind11/pybind11.h>       // For Pybind11 bindings
#include <pybind11/stl.h>            // For STL conversions in Pybind11
#include <stdexcept>                 // For std::invalid_argument
#include <thrust/execution_policy.h> // For thrust::device
#include <thrust/sort.h>             // For thrust::sort_by_key
#include <cupyUtils.cuh>

#define THREADS_PER_AXIS 16
#define TILE_WIDTH 16
#define THREADS_PER_BLOCK 256

// using namespace std;
namespace py = pybind11;

#define PIJA_SMEM_SIZE 2432

// atomsBlock, aoSizeSum, mosRowBlock, mosColBlock
__global__ void pijaLowdinKernel(double *cs, double *pija, int *atomsBlock,
                               int *aoSizeSum, int *mosRowBlock, int *mosColBlock,
                               int nmo, int natom, int nao) {

    __shared__ double sCsRow[PIJA_SMEM_SIZE];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    int ia = atomsBlock[blockIdx.x];

    int aoIdx0 = (ia == 0) ? 0 : aoSizeSum[ia - 1];
    int aoIdx1 = aoSizeSum[ia];

    // Each thread computes multiple elements of the output matrix
    int rowStartBlock = mosRowBlock[blockIdx.x];
    size_t row = rowStartBlock + threadIdx.y;
    int nRowBlock = min(blockDim.y, nmo - rowStartBlock);
    

    int colStartBlock = mosColBlock[blockIdx.x];
    size_t col = colStartBlock + threadIdx.x;
    int nColBlock = min(blockDim.x, nmo - colStartBlock);
    

    int alChunks = PIJA_SMEM_SIZE / (nRowBlock + nColBlock);
    double *sCsCol = sCsRow + alChunks * nRowBlock;

    double value = 0.0;
    for (int alStart = aoIdx0; alStart < aoIdx1; alStart += alChunks) {
        int nAlBatch = min(alChunks, aoIdx1 - alStart);
        int csRowSize = nAlBatch * nRowBlock;
        int csColSize = nAlBatch * nColBlock;

        for (int csRowIdx = tid; csRowIdx < csRowSize; csRowIdx += threadsPerBlock) {
            int i = rowStartBlock + csRowIdx / nAlBatch;
            int al = alStart + csRowIdx % nAlBatch;
            sCsRow[csRowIdx] = cs[i * nao + al];
        }

        for (int csColIdx = tid; csColIdx < csColSize; csColIdx += threadsPerBlock) {
            int i = colStartBlock + csColIdx / nAlBatch;
            int al = alStart + csColIdx % nAlBatch;
            sCsCol[csColIdx] = cs[i * nao + al];
        }

        __syncthreads();

        if (row < nmo && col < nmo) { 
            for (int alIdx = 0; alIdx < nAlBatch; alIdx++) {
                //value += cs[al * nmo + row] * cs[al * nmo + col];
                value += sCsRow[threadIdx.y * nAlBatch + alIdx] * sCsCol[threadIdx.x * nAlBatch + alIdx];
            }
        }
        
        __syncthreads();
            
    }

    if (row < nmo && col < nmo) {
        pija[(row * nmo + col) * natom + ia] = value;
    }
}

__global__ void pijaLowdinKernels(double *cs, double *pija, int *atomsBlock,
                               int *aoSizeSum, int *mosRowBlock, int *mosColBlock,
                               int nmo, int natom, int nao) {
    int ia = atomsBlock[blockIdx.x];

    int aoIdx0 = (ia == 0) ? 0 : aoSizeSum[ia - 1];
    int aoIdx1 = aoSizeSum[ia];

    // Each thread computes multiple elements of the output matrix
    int row = mosRowBlock[blockIdx.x] + threadIdx.y;
    int col = mosColBlock[blockIdx.x] + threadIdx.x;


    if (row < nmo && col < nmo) {
        double value = 0.0;
        for (int al = aoIdx0; al < aoIdx1; ++al) {
            //value += cs[al * nmo + row] * cs[al * nmo + col];
            value += cs[row * nao + al] * cs[col * nao + al];
        }

        pija[(row * nmo + col) * natom + ia] = value;
    }
}

void pijaLowdinCupy(py::object sc, py::object pija, py::object atomsBlock,
                    py::object aoSizeSum, py::object mosRowBlock, py::object mosColBlock,
                    int nmo, int natom, int nao, int blocks) {

    // Extracting buffer info from cupy arrays
    double *ptrSc = getCupyPtr<double>(sc);
    double *ptrPija = getCupyPtr<double>(pija);
    int *ptrAtomsBlock = getCupyPtr<int>(atomsBlock);
    int *ptrAoSizeSum = getCupyPtr<int>(aoSizeSum);
    int *ptrRMosRowBlock = getCupyPtr<int>(mosRowBlock);
    int *ptrMosColBlock = getCupyPtr<int>(mosColBlock);

    //  Defining block and grid sizes
    dim3 threadsPerBlock(THREADS_PER_AXIS, THREADS_PER_AXIS);

    // Request max sharedMem
    cudaFuncSetAttribute(pijaLowdinKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    // Launching CUDA kernel
    pijaLowdinKernel<<<blocks, threadsPerBlock>>>(ptrSc, ptrPija, ptrAtomsBlock, ptrAoSizeSum,
                                                ptrRMosRowBlock, ptrMosColBlock, nmo, natom, nao);
}

__global__ void initIdentityMatrixKernel(double* matrix, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int index = row * n + col;
        matrix[index] = (row == col) ? 1.0 : 0.0;
    }
}


__global__ void computeBijKernel(int nmo, int natom, int npair, double *pija,
                               int *pairs, double *bij) {
    const int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= npair)
        return;

    // Generate upper triangle indices (i,j)

    // Map idx to (i, j) where i < j
    // Compute i using the quadratic formula
    double discriminant = static_cast<double>(2 * nmo - 1) * (2 * nmo - 1) - 8 * pairIdx;
    int i = static_cast<int>(std::floor((2 * nmo - 1 - std::sqrt(discriminant)) / 2));

    // Compute number of pairs before i
    int pairsBeforeI = i * (2 * nmo - i - 1) / 2;

    // Compute j
    int j = (i + 1) + (pairIdx - pairsBeforeI);

    int iPair = i * nmo + j;

    // Compute bij = |sum(pija_ij * (pija_ii - pija_jj))|
    const double *pija_ij = pija + iPair * natom;
    const double *pija_ii = pija + (i * nmo + i) * natom;
    const double *pija_jj = pija + (j * nmo + j) * natom;

    double result = 0.0;
    for (int ia = 0; ia < natom; ++ia) {
        result += pija_ij[ia] * (pija_ii[ia] - pija_jj[ia]);
    }

    pairs[pairIdx] = iPair;
    bij[pairIdx] = fabs(result);
}

void sortPairsCuda(int nmo, int natom, int nPair, double *pija,
                   int *sortedPairs) {
    const int blocks = (nPair + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Device memory allocations
    double *bij;
    int *unsortedPairs;
    cudaMalloc(&bij, nPair * sizeof(double));
    cudaMalloc(&unsortedPairs, nPair * sizeof(int));

    // 1. Compute initial pairs and bij values
    computeBijKernel<<<blocks, THREADS_PER_BLOCK>>>(nmo, natom, nPair, pija, unsortedPairs, bij);

    // 2. Sort pairs in-place using bij values (descending order)
    thrust::sort_by_key(thrust::device,
                        bij, bij + nPair,           // Keys
                        unsortedPairs,              // Values
                        thrust::greater<double>()); // Reverse order

    // 3. Copy sorted pairs to output
    cudaMemcpy(sortedPairs, unsortedPairs, nPair * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(bij);
    cudaFree(unsortedPairs);

}

__global__ void computeRotAnglesKernel(double *pija_ij, double *pija_ii, double *pija_jj,
                                     double *aij, double *bij, int natom) {

    __shared__ double sAij[THREADS_PER_BLOCK];
    __shared__ double sBij[THREADS_PER_BLOCK];

    const int ia = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform element-wise multiplication
    bool isInBound = ia < natom;
    double pij = (isInBound) ? pija_ij[ia] : 0.0;
    double pii = (isInBound) ? pija_ii[ia] : 0.0;
    double pjj = (isInBound) ? pija_jj[ia] : 0.0;
    double vij =  pii - pjj;
    double tAij = pij * pij - 0.25 * vij * vij;
    double tBij = pij * vij;

    // Store result in shared memory
    sAij[threadIdx.x] = tAij;
    sBij[threadIdx.x] = tBij;

    __syncthreads();

    // Block-wise reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sAij[threadIdx.x] += sAij[threadIdx.x + stride];
            sBij[threadIdx.x] += sBij[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (threadIdx.x == 0) {
        aij[blockIdx.x] = sAij[0];
        bij[blockIdx.x] = sBij[0];
    }
}

__global__ void reduceAijBij1Kernel(double *aij, double *bij, double *sumAB, int blocksAtom) {
    __shared__ double sAij[THREADS_PER_BLOCK];
    __shared__ double sBij[THREADS_PER_BLOCK];

    // Store result in shared memory
    sAij[threadIdx.x] = (threadIdx.x < blocksAtom)? aij[threadIdx.x] : 0.0;
    sBij[threadIdx.x] = (threadIdx.x < blocksAtom)? bij[threadIdx.x] : 0.0;

    __syncthreads();

    // Block-wise reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sAij[threadIdx.x] += sAij[threadIdx.x + stride];
            sBij[threadIdx.x] += sBij[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (threadIdx.x == 0) {
        sumAB[0] = sAij[0];
        sumAB[1] = sBij[0];
    }
}

__global__ void rotateUKernel(double *ut, int nmo, int i, int j, double cosa, double sina) {

    // Each thread handles one row of the matrix
    const int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < nmo) {
        // Calculate memory offsets for this row
        //const int offset_i = k * nmo + i;
        //const int offset_j = k * nmo + j;
        const int offset_i = i * nmo + k;
        const int offset_j = j * nmo + k;

        // Load current values
        const double ut_ik = ut[offset_i];
        const double ut_jk = ut[offset_j];

        // Apply Givens rotation
        ut[offset_i] = ut_ik * cosa + ut_jk * sina;
        ut[offset_j] = -ut_ik * sina + ut_jk * cosa;
    }
}


__global__ void rotatePijaRowKernelx(double *pija, int nmo, int natom,
                                  int i, int j, double cosa, double sina) {
    // Each thread handles one row of the matrix
    const int ka = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = ka / natom;
    const int ia = ka % natom;

    if (k < nmo && ia < natom) {
        // Calculate memory offsets for this row
        const size_t offset_i = (size_t)(i * nmo + k) * natom + ia;
        const size_t offset_j = (size_t)(j * nmo + k) * natom + ia;

        // Load current values
        const double pika = pija[offset_i];
        const double pjka = pija[offset_j];

        // Apply Givens rotation
        pija[offset_i] = pika * cosa + pjka * sina;
        pija[offset_j] = -pika * sina + pjka * cosa;
    }
}

__global__ void rotatePijaColKernelx(double *pija, int nmo, int natom,
                                  int i, int j, double cosa, double sina) {
    // Each thread handles one row of the matrix
    const int ka = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = ka / natom;
    const int ia = ka % natom;

    if (k < nmo && ia < natom) {
        // Calculate memory offsets for this row
        const size_t offset_i = (size_t)(k * nmo + i) * natom + ia;
        const size_t offset_j = (size_t)(k * nmo + j) * natom + ia;

        // Load current values
        const double pkia = pija[offset_i];
        const double pkja = pija[offset_j];

        // Apply Givens rotation
        pija[offset_i] = pkia * cosa + pkja * sina;
        pija[offset_j] = -pkia * sina + pkja * cosa;
    }
}

__global__ void rotatePijaRowKernel(double* pija, int nmo, int natom,
                                    int i, int j, double cosa, double sina) {
    // Precompute column bases
    double* col_i = pija + i * nmo * natom;
    double* col_j = pija + j * nmo * natom;
    
    // Grid-striding loop over all elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < nmo * natom; 
         idx += blockDim.x * gridDim.x) {
        const int k = idx / natom;  // Row index
        const int ia = idx % natom; // Atom index

        // Direct pointer arithmetic
        double* elem_i = col_i + k * natom + ia;
        double* elem_j = col_j + k * natom + ia;
        
        const double pika = *elem_i;
        const double pjka = *elem_j;
        
        *elem_i = pika * cosa + pjka * sina;
        *elem_j = -pika * sina + pjka * cosa;
    }
}

__global__ void rotatePijaColKernel(double* pija, int nmo, int natom,
                                    int i, int j, double cosa, double sina) {
    // Precompute slice size per k
    const int slice_size = nmo * natom;
    
    // Grid-striding loop over all elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < nmo * natom; 
         idx += blockDim.x * gridDim.x) {
        const int k = idx / natom;   // Row index
        const int ia = idx % natom;  // Atom index
        
        // Calculate base address for k-slice
        double* slice_base = pija + k * slice_size;
        
        // Direct pointer arithmetic
        double* elem_i = slice_base + i * natom + ia;
        double* elem_j = slice_base + j * natom + ia;
        
        const double pkia = *elem_i;
        const double pkja = *elem_j;
        
        *elem_i = pkia * cosa + pkja * sina;
        *elem_j = -pkia * sina + pkja * cosa;
    }
}


// Main optimization routine
void locJacobiCuda(double *pija, double *ut, int nmo, int natom,
                   int maxCycle, double locTol) {

    size_t pijaSize = nmo * nmo * natom;
    int nPair = nmo * (nmo - 1) / 2;

    const int blocksAtom = (natom + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int blocksMo = (nmo + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int blocksPija = (nmo * natom + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /*
    int *pairs = new int[nPair];
    */
    
    int* pairs;
    double *h_sumAB;
    cudaMallocHost((void**)&pairs, nPair*sizeof(int));
    cudaMallocHost((void**)&h_sumAB, 2*sizeof(double));

    double *d_aij, *d_bij, *d_sumAB;
    cudaMalloc(&d_aij, blocksAtom * sizeof(double));
    cudaMalloc(&d_bij, blocksAtom * sizeof(double));
    cudaMalloc(&d_sumAB, 2*sizeof(double));

    // Initilize the U matrix as a unitary matrix
    dim3 blocks((nmo + THREADS_PER_AXIS - 1) / THREADS_PER_AXIS, 
                (nmo + THREADS_PER_AXIS - 1) / THREADS_PER_AXIS);
    dim3 threads(THREADS_PER_AXIS, THREADS_PER_AXIS);
    initIdentityMatrixKernel<<<blocks, threads>>>(ut, nmo);

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    double fun = 0.0; // Variable to store the result
    cublasDdot(cublasH, pijaSize, pija, 1, pija, 1, &fun);
    std::cout << "Initial funval = " << fun << std::endl;

    cublasDestroy(cublasH);

    float delta;
    for (int icycle = 0; icycle < maxCycle; ++icycle) {
        delta = 0.0;

        sortPairsCuda(nmo, natom, nPair, pija, pairs);

        for (int pairIdx = 0; pairIdx < nPair; pairIdx++) {
            int iPair = pairs[pairIdx];
            int i = iPair / nmo;
            int j = iPair % nmo;

            // Calculate rotation parameters
            double *pija_ij = pija + iPair * natom;         // pija[i,j,:]
            double *pija_ii = pija + (i * nmo + i) * natom; // pija[i,i,:]
            double *pija_jj = pija + (j * nmo + j) * natom; // pija[j,j,:]

            computeRotAnglesKernel<<<blocksAtom, THREADS_PER_BLOCK>>>(pija_ij, pija_ii, pija_jj,
                                                                            d_aij, d_bij, natom);

            reduceAijBij1Kernel<<<1, blocksAtom>>>(d_aij, d_bij, d_sumAB, blocksAtom);
            cudaMemcpy(h_sumAB, d_sumAB, 2*sizeof(double), cudaMemcpyDeviceToHost);
            
            double h_sumAij = h_sumAB[0];
            double h_sumBij = h_sumAB[1];
            
            if (std::abs(h_sumAij) < 1e-10 && std::abs(h_sumBij) < 1e-10) continue;

            double theta = std::atan2(h_sumBij, -h_sumAij); // theta = 4a
            double a = theta / 4.0;
            double cosa = std::cos(a);
            double sina = std::sin(a);

            if (std::abs(sina) < 1e-10) // Skip if rotation is negligible
                continue;

            double p1 = std::hypot(h_sumAij, h_sumBij);
            double cos4a = std::cos(theta);

            delta += p1 * (1 - cos4a);
            
            // Update transformation matrix U
            rotateUKernel<<<blocksMo, THREADS_PER_BLOCK>>>(ut, nmo, i, j, cosa, sina);

            // Bra-transformation (rows)
            rotatePijaRowKernel<<<blocksPija, THREADS_PER_BLOCK>>>(pija, nmo, natom, i, j, cosa, sina);

            // Ket-transformation (columns)
            rotatePijaColKernel<<<blocksPija, THREADS_PER_BLOCK>>>(pija, nmo, natom, i, j, cosa, sina);
        }

        fun += delta;
        printf("    Iter. %d, fun=%.8f, delta=%.4e\n", icycle, fun, delta);

        if (delta < locTol)
            break;
    }

    cudaFree(d_aij);
    cudaFree(d_bij);
    cudaFree(d_sumAB);
    cudaFreeHost(pairs);
    cudaFreeHost(h_sumAB);

    std::cout << (delta < locTol ? "Converged!" : "Not converged") << std::endl;
}

void locJacobiCupy(py::object pija, py::object ut, int maxCycle = 1000, double locTol = 1e-6) {
    py::tuple pijaShape = pija.attr("shape");
    int nmo = pijaShape[0].cast<int>();
    int natom = pijaShape[2].cast<int>();

    double *ptrPija = getCupyPtr<double>(pija);
    double *ptrUt = getCupyPtr<double>(ut);

    locJacobiCuda(ptrPija, ptrUt, nmo, natom, maxCycle, locTol);
}

// Binding the function for Python
PYBIND11_MODULE(localizationCuda, m) {
    m.def("pijaLowdinCupy", &pijaLowdinCupy, "pijaLowdinCupy");
    m.def("locJacobiCupy", &locJacobiCupy, "locJacobiCupy");
}