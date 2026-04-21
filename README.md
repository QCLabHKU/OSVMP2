# OSVMP2

A high-performance quantum chemistry package designed for parallel OSV-MP2 and MBE(3)-OSV-MP2 calculations on CPU and GPU architectures.

---

## Features
* Hartree Fock
* OSV-MP2, MBE(3)-OSV-MP2, and MBE(3)-OSV-MP2 with global corrections
* Density fitting: incore, outcore and direct
* Without density fitting (for HF on GPU only): direct
* Energy


### To-do
* Gradients for HF and MP2

## Dependent Packages

### Basic
* **numpy**
* **scipy**
* **psutil**
* **pyscf**
* **mpi4py**
* **h5py** (mpi-io)

### GPU Support
* **cuda-toolkit**
* **cupy**
* **gpu4pyscf**

### QM/MM
* **openMM**
* **ParmEd**

Verify that all required dependencies are installed by running:

```bash
python check_dependency.py
```
---

## Build Instructions

### Standard Build
From your project root (where the `./src` directory is located), execute the following:

```bash
mkdir build && cd build
cmake ../src
cmake --build . -j8
```

### Specific Configurations
* **Manual BLAS Path:** Use this if auto-detection fails.
  ```bash
  cmake ../src -DBLAS_ROOT=/PATH_TO_BLAS/include
  ```
* **Multiple CUDA Architectures:**
  ```bash
  cmake ../src -DCUDA_ARCHITECTURES="75;80;86"
  ```


### Environment Setup
You can set up the **OSVMP2** environment in one of two ways:
#### Option 1: Using the provided setup script (recommended)
```bash
# Replace PATH_TO_OSVMP2 with your actual installation path
source PATH_TO_OSVMP2/env_osvmp2.sh
```

#### Option 2: Manual export

```bash
export OSVPATH=PATH_TO_OSVMP2
export PYTHONPATH=$OSVPATH:$PYTHONPATH
export osvmp2=$OSVPATH/osvmp2/opt_df.py
```

**NOTE:** Optionally, you can the lines from your chosen option to your shell configuration file (e.g., ~/.bashrc or ~/.zshrc) for a permanent setup.

---

## Run an OSVMP2 Calculation

For detailed configuration examples, please refer to the `examples/` directory.

### CPU Execution
To run a calculation using `ncore` CPU cores:
```bash
mpirun -np ncore python $osvmp2 xxx.inp
```

### GPU Execution
1. Set the `ngpu` parameter within your `.inp` file.
2. The number of MPI processes (`ncore`) **must be equal** to the number of GPUs (`ngpu`).
```bash
mpirun -np ngpu python $osvmp2 xxx.inp
```

## Citation
If **OSVMP2** contributes to your research, please cite the following papers:

* CPU platform:
Liang, Q.; Yang, J. Third-Order Many-Body Expansion of OSV-MP2 Wave Function for Low-Order Scaling Analytical Gradient Computation. J. Chem. Theory Comput. 2021, 17, 6841–6860. doi:10.1021/acs.jctc.1c00561

* GPU platform:
Liang, Q.; Yang, J. Multi-GPU MBE(3)-OSV-MP2 for performant large-scale ab initio calculations. arXiv: 2603.16575, 2026. https://arxiv.org/abs/2603.16575

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. 
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
