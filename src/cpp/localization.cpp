#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

void createIdentity(double *A, int m) {
    // Zero out entire matrix first
    std::memset(A, 0, m * m * sizeof(double));

    // Set diagonal elements to 1.0
    for (int i = 0; i < m; ++i) {
        A[i * m + i] = 1.0;
    }
}

void sortPairs(int nmo, int natom, int nPair, double *pija, int64_t *sortedPairs) {
    /*
    ijdx = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            bij = abs(np.sum(pija[i, j] * (pija[i, i] - pija[j, j])))
            ijdx.append((i, j, bij))
    ijdx = sorted(ijdx, key=lambda x: x[2], reverse=True)
    */

    // Allocate temporary arrays
    int64_t *unsortedPairs = new int64_t[nPair];
    int64_t *indices = new int64_t[nPair];
    double *bij = new double[nPair];
    double *vij = new double[natom];

    // Compute bij for each pair
    int64_t pairIdx = 0;
    for (int i = 0; i < nmo; ++i) {
        for (int j = i + 1; j < nmo; ++j) {
            int64_t iPair = i * nmo + j;
            indices[pairIdx] = pairIdx;
            unsortedPairs[pairIdx] = iPair;

            // Access pija slices
            double *pija_ij = pija + iPair * natom;         // pija[i,j,:]
            double *pija_ii = pija + (i * nmo + i) * natom; // pija[i,i,:]
            double *pija_jj = pija + (j * nmo + j) * natom; // pija[j,j,:]

            cblas_dcopy(natom, pija_ii, 1, vij, 1);       // vij = pija_ii
            cblas_daxpy(natom, -1.0, pija_jj, 1, vij, 1); // vij = vij - pija_jj

            // Compute the dot product between pija_ij and sub
            bij[pairIdx] = std::abs(cblas_ddot(natom, pija_ij, 1, vij, 1));

            pairIdx++;
        }
    }

    // Sort indices by bij in descending order
    std::sort(indices, indices + nPair, [&bij](int64_t a, int64_t b) {
        return bij[a] > bij[b];
    });

    // Step 5: Fill sorted_pairs with sorted (i, j) pairs
    for (int64_t pairIdx = 0; pairIdx < nPair; ++pairIdx) {
        sortedPairs[pairIdx] = unsortedPairs[indices[pairIdx]];
    }

    // Clean up
    delete[] unsortedPairs;
    delete[] indices;
    delete[] bij;
    delete[] vij;
}

void rotateU(double *u, int64_t nmo, int64_t i, int64_t j, double cosa, double sina) {
    for (int64_t k = 0; k < nmo; ++k) {
        const double u_ki = u[k * nmo + i];
        const double u_kj = u[k * nmo + j];
        u[k * nmo + i] = u_ki * cosa + u_kj * sina;
        u[k * nmo + j] = -u_ki * sina + u_kj * cosa;
    }
}

void rotatePijaRowx(double *pija, double *bufIp, double *bufJp,
                    int64_t nmo, int64_t natom, int64_t i, int64_t j,
                    double cosa, double sina) {
    int64_t sliceSize = nmo * natom;

    double *pijaIrow = pija + i * sliceSize;
    double *pijaJrow = pija + j * sliceSize;

    // bufIp = cosa * pijaIrow + sina * pijaJrow
    cblas_dcopy(sliceSize, pijaIrow, 1, bufIp, 1);
    cblas_dscal(sliceSize, cosa, bufIp, 1);
    cblas_daxpy(sliceSize, sina, pijaJrow, 1, bufIp, 1);

    // bufJp = -sina * pijaIrow + cosa * pijaJrow
    cblas_dcopy(sliceSize, pijaJrow, 1, bufJp, 1);
    cblas_dscal(sliceSize, cosa, bufJp, 1);
    cblas_daxpy(sliceSize, -sina, pijaIrow, 1, bufJp, 1);

    // Write results back to original array
    cblas_dcopy(sliceSize, bufIp, 1, pijaIrow, 1);
    cblas_dcopy(sliceSize, bufJp, 1, pijaJrow, 1);
}

void rotatePijaColx(double *pija, double *bufIp, double *bufJp,
                    int64_t nmo, int64_t natom, int64_t i, int64_t j,
                    double cosa, double sina) {

    int64_t sliceSize = nmo * natom;

    for (int k = 0; k < nmo; ++k) {
        double *pijaIcol = pija + k * sliceSize + i * natom;
        double *pijaJcol = pija + k * sliceSize + j * natom;

        // Preserve original values
        cblas_dcopy(natom, pijaIcol, 1, bufIp, 1);
        cblas_dcopy(natom, pijaJcol, 1, bufJp, 1);

        // Compute and store new pijaIcol
        cblas_dcopy(natom, bufIp, 1, pijaIcol, 1);
        cblas_dscal(natom, cosa, pijaIcol, 1);
        cblas_daxpy(natom, sina, bufJp, 1, pijaIcol, 1);

        // Compute and store new pijaJcol
        cblas_dcopy(natom, bufJp, 1, pijaJcol, 1);
        cblas_dscal(natom, cosa, pijaJcol, 1);
        cblas_daxpy(natom, -sina, bufIp, 1, pijaJcol, 1);
    }
}

void rotatePijaRow(double *pija, int64_t nmo, int64_t natom,
                   int64_t i, int64_t j, double cosa, double sina) {

    for (int64_t k = 0; k < nmo; ++k) {
        for (int64_t ia = 0; ia < natom; ++ia) {
            // Calculate memory offsets for this row
            const int64_t offset_i = (i * nmo + k) * natom + ia;
            const int64_t offset_j = (j * nmo + k) * natom + ia;

            // Load current values
            const double pika = pija[offset_i];
            const double pjka = pija[offset_j];

            // Apply Givens rotation
            pija[offset_i] = pika * cosa + pjka * sina;
            pija[offset_j] = -pika * sina + pjka * cosa;
        }
    }
}

void rotatePijaCol(double *pija, int64_t nmo, int64_t natom,
                   int64_t i, int64_t j, double cosa, double sina) {

    for (int64_t k = 0; k < nmo; ++k) {
        for (int64_t ia = 0; ia < natom; ++ia) {
            // Calculate memory offsets for this column
            const int64_t offset_i = (k * nmo + i) * natom + ia;
            const int64_t offset_j = (k * nmo + j) * natom + ia;

            // Load current values
            const double pkia = pija[offset_i];
            const double pkja = pija[offset_j];

            // Apply Givens rotation
            pija[offset_i] = pkia * cosa + pkja * sina;
            pija[offset_j] = -pkia * sina + pkja * cosa;
        }
    }
}

// Main optimization routine
void locJacobi(double *pija, double *u, int nmo, int natom,
               int maxCycle, double locTol, int iRank) {
    int64_t pijaSize = nmo * nmo * natom;
    int64_t nPair = nmo * (nmo - 1) / 2;
    int64_t *pairs = new int64_t[nPair];
    double *vij = new double[natom];
    int64_t sliceSize = nmo * natom;
    double *bufIp = new double[sliceSize];
    double *bufJp = new double[sliceSize];
    double delta = locTol * 10;

    double fun = cblas_ddot(pijaSize, pija, 1, pija, 1);
    
    if (iRank == 0) {
        printf("Initial funval = %.8f\n", fun);
    }

    int iCycle = 0;

    while (delta > locTol) {
        delta = 0.0;
        sortPairs(nmo, natom, nPair, pija, pairs);

        for (int64_t pairIdx = 0; pairIdx < nPair; pairIdx++) {
            int64_t iPair = pairs[pairIdx];
            int i = iPair / nmo;
            int j = iPair % nmo;

            // Calculate rotation parameters

            double *pija_ij = pija + iPair * natom;         // pija[i,j,:]
            double *pija_ii = pija + (i * nmo + i) * natom; // pija[i,i,:]
            double *pija_jj = pija + (j * nmo + j) * natom; // pija[j,j,:]

            cblas_dcopy(natom, pija_ii, 1, vij, 1);       // vij = pija_ii
            cblas_daxpy(natom, -1.0, pija_jj, 1, vij, 1); // vij = vij - pija_jj

            double aij = cblas_ddot(natom, pija_ij, 1, pija_ij, 1) - 0.25 * cblas_ddot(natom, vij, 1, vij, 1);
            double bij = cblas_ddot(natom, pija_ij, 1, vij, 1);

            if (std::abs(aij) < 1e-10 && std::abs(bij) < 1e-10)
                continue;

            double theta = std::atan2(bij, -aij); // theta = 4a
            double a = theta / 4.0;
            double cosa = std::cos(a);
            double sina = std::sin(a);

            if (std::abs(sina) < 1e-10) // Skip if rotation is negligible
                continue;

            double p1 = std::hypot(aij, bij);
            double cos4a = std::cos(theta);

            delta += p1 * (1 - cos4a);

            // Update transformation matrix U
            rotateU(u, nmo, i, j, cosa, sina);

            // Bra-transformation (rows)
            rotatePijaRow(pija, nmo, natom, i, j, cosa, sina);

            // Ket-transformation (columns)
            rotatePijaCol(pija, nmo, natom, i, j, cosa, sina);
        }

        fun += delta;
        ++iCycle;

        if (iRank == 0) {
            printf("    Cycle %d, delta=%.4e, fun=%.8f\n", iCycle, delta, fun);
        }
        

        if (iCycle >= maxCycle)
            break;
    }

    delete[] pairs;
    delete[] vij;
    delete[] bufIp;
    delete[] bufJp;

    if (iRank == 0) {
        printf("%s after %d cycles.\n", delta < locTol ? "Localization converged" : "Localization did not converged", iCycle);
    }
}

void locJacobiNumpy(py::array pija, py::array u, int maxCycle = 1000, double locTol = 1e-6, int iRank = 0) {
    py::buffer_info bufPija = pija.request();
    auto shape = bufPija.shape;
    int nmo = shape[0];
    int natom = shape[2];

    double *ptrPija = static_cast<double *>(bufPija.ptr);
    double *ptrU = static_cast<double *>(u.request().ptr);

    locJacobi(ptrPija, ptrU, nmo, natom, maxCycle, locTol, iRank);
}

PYBIND11_MODULE(localization, m) {
    m.def("locJacobiNumpy", &locJacobiNumpy, "locJacobiNumpy");
}