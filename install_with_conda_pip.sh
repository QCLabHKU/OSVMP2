# Basic
pip install numpy scipy psutil pyscf
conda install -c conda-forge mpi4py h5py=*=mpi_mpich_*

# GPU
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install cupy-cuda11x gpu4pyscf-cuda11x cutensor-cu11