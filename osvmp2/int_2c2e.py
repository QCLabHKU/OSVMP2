import os
import scipy
import numpy as np
import h5py
from osvmp2.pbc.gamma_int_2c2e import get_2c2e_gamma
from osvmp2.osvutil import read_file
from osvmp2.mpi_addons import get_shared, free_win
from mpi4py import MPI


#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
#inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//nrank_shm
inode = irank // nrank_shm

def get_j2c_low(self, file_name, save_file_only=False, use_lowT=False):

    if save_file_only:
        win_low, low_node = None, None
    else:
        win_low, low_node = get_shared((self.naoaux, self.naoaux))
    
    exist_file_j2c = os.path.isfile(file_name)
    comm.Barrier()

    win_j2c = None
    if self.mol.pbc and not exist_file_j2c:
        win_j2c, j2c = get_2c2e_gamma(self.with_df.df_builder, self.with_df.kpts)

    if irank_shm == 0:
        #Get j2c
        if exist_file_j2c and not save_file_only:
            if use_lowT:
                low_node[:] = read_file(file_name, 'low').T #transpose
            else:
                read_file(file_name, 'low', buffer=low_node)
        else:
            if irank == 0:
                with h5py.File(file_name, 'w') as f:
                    if self.mol.pbc:
                        if self.mol.gamma_only:
                            #j2c = j2c[0]
                            f.create_dataset('j2c', data=j2c)
                            low = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
                            f.create_dataset('low', data=low)
                        else:
                            raise NotImplementedError
                    else:
                    
                        j2c = self.with_df.auxmol.intor('int2c2e', hermi=1)
                        f.create_dataset('j2c', data=j2c, dtype=np.float64)
                        low = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
                        f.create_dataset('low', data=low)
            else:
                if self.mol.pbc:
                    if self.mol.gamma_only:
                        #j2c = j2c[0]
                        low = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
                    else:
                        raise NotImplementedError
                else:
                    j2c = self.with_df.auxmol.intor('int2c2e', hermi=1)
                    low = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
            if not save_file_only:
                if use_lowT:
                    low_node[:] = low.T #transpose
                else:
                    low_node[:] = low
    comm_shm.Barrier()

    if win_j2c is not None:
        free_win(win_j2c)

    return win_low, low_node