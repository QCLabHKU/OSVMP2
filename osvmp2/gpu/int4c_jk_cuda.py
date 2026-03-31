# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Compute J/K matrices
'''

import os
import sys
import ctypes
import warnings
import math
import numpy as np
import cupy
import scipy.linalg
from collections import Counter
from pyscf.gto import ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD, PTR_COEFF
from pyscf import lib, gto
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import (
    load_library, condense, transpose_sum, reduce_to_device, hermi_triu,
    asarray)
from gpu4pyscf.__config__ import shm_size
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import basis_seg_contraction, _split_l_ctr_groups

import osvmp2
from osvmp2.__config__ import ngpu
from osvmp2.osvutil import *
from osvmp2.mpi_addons import get_shared, Acc_and_get_GA, free_win, Accumulate_GA_shm
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

#ngpu = min(int(os.environ.get("ngpu", nrank)), nrank)
ngpu_shm = ngpu // nnode
nrank_per_gpu = nrank_shm // ngpu_shm
igpu = irank // nrank_per_gpu
igpu_shm = irank_shm // nrank_per_gpu
irank_gpu = irank % nrank_per_gpu
cupy.cuda.runtime.setDevice(igpu_shm)


#libvhf_rys = load_library('libgvhf_rys')
libvhf_rys = np.ctypeslib.load_library('int4cJKCuda', f"{osvmp2.lib.__path__[0]}")
libvhf_rys.RYS_build_jk.restype = ctypes.c_int
libvhf_rys.RYS_init_constant.restype = ctypes.c_int
libvhf_rys.cuda_version.restype = ctypes.c_int
CUDA_VERSION = libvhf_rys.cuda_version()
libgint = load_library('libgint')

PTR_BAS_COORD = 7
LMAX = 4
TILE = 2
QUEUE_DEPTH = 262144
SHM_SIZE = shm_size - 1024
del shm_size
GOUT_WIDTH = 42
THREADS = 256
GROUP_SIZE = 256

def group_basis(mol, tile=1, group_size=None, return_bas_mapping=False,
                sparse_coeff=False):
  
    original_mol = mol

    # When sparse_coeff is enabled, an array of AO mapping indices will be
    # returned which can facilitate the transformation of the integral matrix
    # between sorted_mol and mol using fancy-indexing, without applying the
    # expensive C.T.dot(mat).dot(C). This fast transformation assumes one-one
    # mapping between the basis shells of the two types of mol instatnce,
    # ignoring general contraction. Enabling `allow_replica` will produce
    # replicated segment-contracted shells for general contracted shells.
    if sparse_coeff:
        mol, coeff = basis_seg_contraction(
            mol, allow_replica=True, sparse_coeff=sparse_coeff)
    else:
        mol, coeff = basis_seg_contraction(mol, sparse_coeff=sparse_coeff)

    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    l_ctrs = mol._bas[:,[ANG_OF, NPRIM_OF]]
    # Ensure the more contracted Gaussians being accessed first
    l_ctrs_descend = l_ctrs.copy()
    l_ctrs_descend[:,1] = -l_ctrs[:,1]
    uniq_l_ctr, where, inv_idx, l_ctr_counts = np.unique(
        l_ctrs_descend, return_index=True, return_inverse=True, return_counts=True, axis=0)
    uniq_l_ctr[:,1] = -uniq_l_ctr[:,1]

    if not sparse_coeff:
        nao_orig = coeff.shape[1]
        ao_loc = mol.ao_loc
        coeff = cupy.split(coeff, ao_loc[1:-1], axis=0)
    else:
        ao_loc = mol.ao_loc_nr(cart=original_mol.cart)
        ao_idx = np.array_split(np.arange(original_mol.nao), ao_loc[1:-1])
        sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)
        ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])

    pad_bas = []
    if tile > 1:
        assert not return_bas_mapping, 'bas_mapping requires tile=1'
        l_ctr_counts_orig = l_ctr_counts.copy()
        pad_inv_idx = []
        env_ptr = mol._env.size
        # for each pattern, padding basis to the end of mol._bas, ensure alignment to tile
        for n, (l_ctr, m, counts) in enumerate(zip(uniq_l_ctr, where, l_ctr_counts)):
            if counts % tile == 0: continue
            n_alined = (counts+tile-1) & (0x100000-tile)
            padding = n_alined - counts
            l_ctr_counts[n] = n_alined

            bas = mol._bas[m].copy()
            bas[PTR_COEFF] = env_ptr
            pad_bas.extend([bas] * padding)
            pad_inv_idx.extend([n] * padding)

            l = l_ctr[0]
            nf = (l + 1) * (l + 2) // 2
            if not sparse_coeff:
                coeff.extend([cupy.zeros((nf, nao_orig))] * padding)

        inv_idx = np.hstack([inv_idx.ravel(), pad_inv_idx])

    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    if_pad_bas = np.array([False] * mol.nbas + [True] * len(pad_bas))[sorted_idx]

    max_nprims = uniq_l_ctr[:,1].max()
    mol._env = np.append(mol._env, np.zeros(max_nprims))
    if pad_bas:
        mol._bas = np.vstack([mol._bas, pad_bas])[sorted_idx]
    else:
        mol._bas = mol._bas[sorted_idx]
    assert mol._bas.dtype == np.int32

    ## Limit the number of AOs in each group
    if group_size is not None:
        uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
            uniq_l_ctr, l_ctr_counts, group_size, tile)

    # PTR_BAS_COORD is required by various CUDA kernels
    mol._bas[:,PTR_BAS_COORD] = mol._atm[mol._bas[:,ATOM_OF],PTR_COORD]

    if not sparse_coeff:
        coeff = cupy.vstack([coeff[i] for i in sorted_idx])
        assert coeff.shape[0] < 32768
        if return_bas_mapping:
            return mol, coeff, uniq_l_ctr, l_ctr_counts, sorted_idx.argsort()
        else:
            return mol, coeff, uniq_l_ctr, l_ctr_counts
    else:
        l_ctr_offsets = np.cumsum(l_ctr_counts)[:-1]
        if_pad_bas_per_l_ctr = np.split(if_pad_bas, l_ctr_offsets)
        l_ctr_pad_counts = np.array([np.sum(if_pad) for if_pad in if_pad_bas_per_l_ctr])
        l_ctr_pad_counts = np.asarray(l_ctr_pad_counts, dtype=np.int32)
        if return_bas_mapping:
            return mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts, sorted_idx.argsort()
        else:
            return mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts

def equisum_partition(a, n):
    """
    Divide a list into n contiguous parts with sums as equal as possible.
    
    Args:
        a (list): List of numbers to divide.
        n (int): Number of parts to divide into.
    
    Returns:
        list: List of n sublists with sums as close as possible.
    """
    if n > len(a):
        raise ValueError("n cannot be larger than the list length")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Compute cumulative sum array
    #from itertools import accumulate
    #cumsum = list(accumulate(a, initial=0))  # [0, a[0], a[0]+a[1], ...]
    a_indices = np.arange(len(a), dtype=np.int32)
    cumsum = np.empty(len(a)+1, dtype=int)
    cumsum[0] = 0
    np.cumsum(a, out=cumsum[1:])
    total_sum = cumsum[-1]
    target = total_sum / n  # Ideal sum per part
    
    result = []
    start = 0
    
    # Find n-1 split points
    for _ in range(1, n):
        min_diff = float('inf')
        best_i = start + 1
        
        # Find the index that makes the sum closest to target
        for i in range(start + 1, len(a) + 1):
            current_sum = cumsum[i] - cumsum[start]
            diff = abs(current_sum - target)
            if diff < min_diff:
                min_diff = diff
                best_i = i
        
        # Add the part from start to best_i-1
        #result.append(a[start:best_i])
        result.append(a_indices[start:best_i])
        start = best_i
    
    # Add the remaining part
    #result.append(a[start:])
    result.append(a_indices[start:])
    return result

def equisum_partition_sort(sizes, n_parts):
    """
    Partition 1D array into n_parts with sums as balanced as possible.
    
    Parameters:
    -----------
    sizes : np.ndarray
        1D array of values to partition
    n_parts : int
        Number of partitions
        
    Returns:
    --------
    list of lists : Original indices grouped by partition
    """
    sizes = np.asarray(sizes)
    n = len(sizes)
    
    # Sort indices by size descending (largest first)
    sorted_indices = np.argsort(sizes)[::-1]
    
    # Track sum of each partition and their indices
    partition_sums = np.zeros(n_parts)
    partition_indices = [[] for _ in range(n_parts)]
    
    # Assign each item to the partition with smallest current sum
    for idx in sorted_indices:
        # Find partition with minimum sum
        min_part = np.argmin(partition_sums)
        
        # Add current item to that partition
        partition_indices[min_part].append(idx)
        partition_sums[min_part] += sizes[idx]
    
    return partition_indices

def get_eri_tasks_ranks(tasks, eri_sizes_tasks, npart=ngpu, sort=True):

    #sizes_ranks = equisum_partition(eri_sizes_tasks, npart)
    if sort:
        task_ids_ranks = equisum_partition_sort(eri_sizes_tasks, npart)
    else:
        task_ids_ranks = equisum_partition(eri_sizes_tasks, npart)

    tasks_ranks = []
    for task_ids_rank in task_ids_ranks:
        tasks_rank = [tasks[task_id] for task_id in task_ids_rank]
        tasks_ranks.append(tasks_rank)
    return tasks_ranks

def compute_jk(mol, dm, hermi=0, vhfopt=None, with_j=True, with_k=True, verbose=None):
    '''Compute J, K matrices
    '''
    assert with_j or with_k
    log = logger.new_logger(mol, verbose)
    #cput0 = log.init_timer()

    if vhfopt is None:
        vhfopt = JKOpt(mol).build()

    mol = vhfopt.sorted_mol
    nao_orig = vhfopt.mol.nao

    dm = cupy.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    #:dms = cupy.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    dms = cupy.asarray(dms, order='C')

    vj, vk = vhfopt.get_jk(dms, hermi, with_j, with_k, log)
    if with_k and irank_shm == 0:
        #:vk = cupy.einsum('pi,npq,qj->nij', vhfopt.coeff, vk, vhfopt.coeff)
        vk = vhfopt.apply_coeff_CT_mat_C(vk)
        vk = vk.reshape(dm.shape)

    if with_j and irank_shm == 0:
        #:vj = cupy.einsum('pi,npq,qj->nij', vhfopt.coeff, vj, vhfopt.coeff)
        vj = vhfopt.apply_coeff_CT_mat_C(vj)
        vj = vj.reshape(dm.shape)
    #log.timer('vj and vk', *cput0)
    return vj, vk


class JKOpt:
    def __init__(self, mol, cutoff=1e-13):
        self.mol = mol
        self.direct_scf_tol = cutoff
        self.uniq_l_ctr = None
        self.l_ctr_offsets = None
        self.h_shls = None
        self.tile = TILE

        # Hold cache on GPU devices
        self._rys_envs = {}
        self._q_cond = {}
        self._tile_q_cond = {}
        self._s_estimator = {}
        self._cupy_ao_idx = {}

    def build(self, group_size=None, verbose=None):
        mol = self.mol
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        #group_size = 200
        (mol, ao_idx, l_ctr_pad_counts, 
         uniq_l_ctr, l_ctr_counts) = group_basis(mol, self.tile, group_size, 
                                                 sparse_coeff = True)
        self.sorted_mol = mol
        self.ao_idx = ao_idx
        self.l_ctr_pad_counts = l_ctr_pad_counts
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))

        # very high angular momentum basis are processed on CPU
        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax > LMAX:
            self.h_shls = l_slices[LMAX+1:].tolist()
        else:
            self.h_shls = []

        nbas = mol.nbas
        ao_loc = mol.ao_loc
        q_cond = np.empty((nbas,nbas))
        intor = mol._add_suffix('int2e')
        _vhf.libcvhf.CVHFnr_int2e_q_cond(
            getattr(_vhf.libcvhf, intor), lib.c_null_ptr(),
            q_cond.ctypes, ao_loc.ctypes,
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
        q_cond = np.log(q_cond + 1e-300).astype(np.float32)
        self.q_cond_cpu = q_cond

        tile = self.tile
        if tile > 1:
            ntiles = nbas // tile
            self._tile_q_cond_cpu = q_cond.reshape(ntiles,tile,ntiles,tile).max(axis=(1,3))
        else:
            self._tile_q_cond_cpu = q_cond

        if mol.omega < 0:
            # CVHFnr_sr_int2e_q_cond in pyscf has bugs in upper bound estimator.
            # Use the local version of s_estimator instead
            s_estimator = np.empty((nbas,nbas), dtype=np.float32)
            libvhf_rys.sr_eri_s_estimator(
                s_estimator.ctypes, ctypes.c_float(mol.omega),
                mol._atm.ctypes, ctypes.c_int(mol.natm),
                mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
            self.s_estimator_cpu = s_estimator
        log.timer('Initialize q_cond', *cput0)
        return self

    def sort_orbitals(self, mat, axis=[]):
        ''' 
        Transform given axis of a matrix into sorted AO 
        '''
        idx = self.ao_idx
        shape_ones = (1,) * mat.ndim
        fancy_index = []
        for dim, n in enumerate(mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        return mat[tuple(fancy_index)]

    def unsort_orbitals(self, sorted_mat, axis=[]):
        ''' 
        Transform given axis of a matrix into sorted AO 
        '''
        idx = self.ao_idx
        shape_ones = (1,) * sorted_mat.ndim
        fancy_index = []
        for dim, n in enumerate(sorted_mat.shape):
            if dim in axis:
                assert n == len(idx)
                indices = idx
            else:
                indices = np.arange(n)
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
        mat = cupy.empty_like(sorted_mat)
        mat[tuple(fancy_index)] = sorted_mat
        return mat

    def apply_coeff_C_mat_CT(self, spherical_matrix):
        '''
        Unsort AO and perform sph2cart transformation (if needed) for the last 2 axes
        Fused kernel to perform 'ip,npq,qj->nij' 
        '''
        spherical_matrix = cupy.asarray(spherical_matrix)
        spherical_matrix_ndim = spherical_matrix.ndim
        if spherical_matrix_ndim == 2:
            spherical_matrix = spherical_matrix[None]
        counts = spherical_matrix.shape[0]
        n_spherical = self.mol.nao
        assert spherical_matrix.shape[1] == n_spherical
        assert spherical_matrix.shape[2] == n_spherical
        n_cartesian = self.sorted_mol.nao

        l_ctr_count = np.asarray(self.l_ctr_offsets[1:] - self.l_ctr_offsets[:-1], dtype = np.int32)
        l_ctr_l = np.asarray(self.uniq_l_ctr[:,0].copy(), dtype = np.int32)
        self.l_ctr_pad_counts = np.asarray(self.l_ctr_pad_counts, dtype = np.int32)
        cupy_ao_idx = self.cupy_ao_idx
        stream = cupy.cuda.get_current_stream()

        # ref = cupy.einsum("ij,qjk,kl->qil", self.coeff, spherical_matrix, self.coeff.T)

        out = cupy.zeros((counts, n_cartesian, n_cartesian), order = "C")
        for i_dm in range(counts):
            libgint.cart2sph_C_mat_CT_with_padding(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                ctypes.cast(out[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.cast(spherical_matrix[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_cartesian),
                ctypes.c_int(n_spherical),
                ctypes.c_int(l_ctr_l.shape[0]),
                l_ctr_l.ctypes.data_as(ctypes.c_void_p),
                l_ctr_count.ctypes.data_as(ctypes.c_void_p),
                self.l_ctr_pad_counts.ctypes.data_as(ctypes.c_void_p),
                ctypes.cast(cupy_ao_idx.data.ptr, ctypes.c_void_p),
                ctypes.c_bool(self.mol.cart),
            )

        if spherical_matrix_ndim == 2:
            out = out[0]
        return out

    def apply_coeff_CT_mat_C(self, cartesian_matrix):
        '''
        Sort AO and perform cart2sph transformation (if needed) for the last 2 axes
        Fused kernel to perform 'ip,npq,qj->nij' 
        '''
        cartesian_matrix = cupy.asarray(cartesian_matrix)
        cartesian_matrix_ndim = cartesian_matrix.ndim
        if cartesian_matrix_ndim == 2:
            cartesian_matrix = cartesian_matrix[None]
        counts = cartesian_matrix.shape[0]
        n_cartesian = self.sorted_mol.nao
        assert cartesian_matrix.shape[1] == n_cartesian
        assert cartesian_matrix.shape[2] == n_cartesian
        n_spherical = self.mol.nao

        l_ctr_count = np.asarray(self.l_ctr_offsets[1:] - self.l_ctr_offsets[:-1], dtype = np.int32)
        l_ctr_l = np.asarray(self.uniq_l_ctr[:,0].copy(), dtype = np.int32)
        self.l_ctr_pad_counts = np.asarray(self.l_ctr_pad_counts, dtype = np.int32)
        cupy_ao_idx = self.cupy_ao_idx
        stream = cupy.cuda.get_current_stream()

        # ref = cupy.einsum("ij,qjk,kl->qil", self.coeff.T, cartesian_matrix, self.coeff)

        out = cupy.empty((counts, n_spherical, n_spherical), order = "C")
        for i_dm in range(counts):
            libgint.cart2sph_CT_mat_C_with_padding(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                ctypes.cast(cartesian_matrix[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.cast(out[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_cartesian),
                ctypes.c_int(n_spherical),
                ctypes.c_int(l_ctr_l.shape[0]),
                l_ctr_l.ctypes.data_as(ctypes.c_void_p),
                l_ctr_count.ctypes.data_as(ctypes.c_void_p),
                self.l_ctr_pad_counts.ctypes.data_as(ctypes.c_void_p),
                ctypes.cast(cupy_ao_idx.data.ptr, ctypes.c_void_p),
                ctypes.c_bool(self.mol.cart),
            )

        if cartesian_matrix_ndim == 2:
            out = out[0]
        return out

    def apply_coeff_C_mat(self, right_matrix):
        '''
        Sort AO and perform cart2sph transformation (if needed) for the second last axis
        Fused kernel to perform 'ip,npq->niq' 
        '''
        right_matrix = cupy.asarray(right_matrix)
        assert right_matrix.ndim == 2
        assert right_matrix.shape[0] == self.mol.nao
        n_cartesian = self.sorted_mol.nao
        n_second = right_matrix.shape[1]

        l_ctr_count = np.asarray(self.l_ctr_offsets[1:] - self.l_ctr_offsets[:-1], dtype = np.int32)
        l_ctr_l = np.asarray(self.uniq_l_ctr[:,0].copy(), dtype = np.int32)
        self.l_ctr_pad_counts = np.asarray(self.l_ctr_pad_counts, dtype = np.int32)
        cupy_ao_idx = self.cupy_ao_idx
        stream = cupy.cuda.get_current_stream()

        # ref = self.coeff @ right_matrix

        right_matrix = cupy.ascontiguousarray(right_matrix)

        out = cupy.zeros((n_cartesian, n_second), order = "C")
        libgint.cart2sph_C_mat_with_padding(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(right_matrix.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_second),
            ctypes.c_int(l_ctr_l.shape[0]),
            l_ctr_l.ctypes.data_as(ctypes.c_void_p),
            l_ctr_count.ctypes.data_as(ctypes.c_void_p),
            self.l_ctr_pad_counts.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(cupy_ao_idx.data.ptr, ctypes.c_void_p),
            ctypes.c_bool(self.mol.cart),
        )

        return out

    @property
    def q_cond(self):
        device_id = cupy.cuda.Device().id
        if device_id not in self._q_cond:
            with cupy.cuda.Device(device_id):
                self._q_cond[device_id] = asarray(self.q_cond_cpu)
        return self._q_cond[device_id]

    @property
    def tile_q_cond(self):
        device_id = cupy.cuda.Device().id
        if device_id not in self._tile_q_cond:
            with cupy.cuda.Device(device_id):
                q_cpu = self._tile_q_cond_cpu
                self._tile_q_cond[device_id] = asarray(q_cpu)
        return self._tile_q_cond[device_id]

    @property
    def s_estimator(self):
        if not self.mol.omega < 0:
            return None
        device_id = cupy.cuda.Device().id
        if device_id not in self._rys_envs:
            with cupy.cuda.Device(device_id):
                s_cpu = self.s_estimator_cpu
                self._s_estimator[device_id] = asarray(s_cpu)
        return self._s_estimator[device_id]

    @property
    def cupy_ao_idx(self):
        device_id = cupy.cuda.Device().id
        if device_id not in self._cupy_ao_idx:
            with cupy.cuda.Device(device_id):
                ao_idx_cpu = self.ao_idx
                self._cupy_ao_idx[device_id] = cupy.asarray(ao_idx_cpu, dtype = cupy.int32)
        return self._cupy_ao_idx[device_id]

    @property
    def rys_envs(self):
        device_id = cupy.cuda.Device().id
        if device_id not in self._rys_envs:
            with cupy.cuda.Device(device_id):
                mol = self.sorted_mol
                _atm = cupy.array(mol._atm)
                _bas = cupy.array(mol._bas)
                _env = cupy.array(_scale_sp_ctr_coeff(mol))
                ao_loc = cupy.array(mol.ao_loc)
                self._rys_envs[device_id] = rys_envs = RysIntEnvVars(
                    mol.natm, mol.nbas,
                    _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
                    ao_loc.data.ptr)
                rys_envs._env_ref_holder = (_atm, _bas, _env, ao_loc)
        return self._rys_envs[device_id]

    @property
    def coeff(self):
        coeff = np.zeros((self.sorted_mol.nao, self.mol.nao))

        l_max = max([l_ctr[0] for l_ctr in self.uniq_l_ctr])
        if self.mol.cart:
            cart2sph_per_l = [np.eye((l+1)*(l+2)//2) for l in range(l_max + 1)]
        else:
            cart2sph_per_l = [gto.mole.cart2sph(l, normalized = "sp") for l in range(l_max + 1)]
        i_spherical_offset = 0
        i_cartesian_offset = 0
        for i, (l, _) in enumerate(self.uniq_l_ctr):
            cart2sph = cart2sph_per_l[l]
            l_ctr_count = self.l_ctr_offsets[i + 1] - self.l_ctr_offsets[i]
            l_ctr_pad_count = self.l_ctr_pad_counts[i]
            for _ in range(l_ctr_count - l_ctr_pad_count):
                coeff[i_cartesian_offset : i_cartesian_offset + cart2sph.shape[0],
                      i_spherical_offset : i_spherical_offset + cart2sph.shape[1]] = cart2sph
                i_cartesian_offset += cart2sph.shape[0]
                i_spherical_offset += cart2sph.shape[1]
            for _ in range(l_ctr_pad_count):
                i_cartesian_offset += cart2sph.shape[0]
        assert len(self.ao_idx) == self.mol.nao
        coeff = self.unsort_orbitals(coeff, axis = [1])
        return asarray(coeff)

    def get_jk(self, dms, hermi, with_j, with_k, verbose):
        '''
        Build JK for the sorted_mol. Density matrices dms and the return JK
        matrices are all corresponding to the sorted_mol
        '''
        if callable(dms):
            dms = dms()
        mol = self.sorted_mol
        log = logger.new_logger(mol, verbose)
        ao_loc = mol.ao_loc
        uniq_l_ctr = self.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = self.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        assert dms.ndim == 3 and dms.shape[-1] == ao_loc[-1]
        dm_cond = condense('absmax', dms, ao_loc)
        if hermi == 0:
            # Wrap the triu contribution to tril
            dm_cond = dm_cond + dm_cond.T
        dm_cond = cupy.log(dm_cond + 1e-300).astype(np.float32)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)

        tasks = [(i,j,k,l)
                 for i in range(n_groups)
                 for j in range(i+1)
                 for k in range(i+1)
                 for l in range(k+1)]

        schemes = {t: quartets_scheme(mol, uniq_l_ctr[list(t)], with_j, with_k) for t in tasks}

        #def proc(dms, dm_cond):
        device_id = cupy.cuda.device.get_device_id()
        stream = cupy.cuda.stream.get_current_stream()
        log = logger.new_logger(mol, verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms) # transfer to current device
        dm_cond = cupy.asarray(dm_cond)

        if hermi == 0:
            # Contract the tril and triu parts separately
            dms = cupy.vstack([dms, dms.transpose(0,2,1)])
        n_dm, nao = dms.shape[:2]
        tile_q_cond = self.tile_q_cond
        tile_q_ptr = ctypes.cast(tile_q_cond.data.ptr, ctypes.c_void_p)
        q_ptr = ctypes.cast(self.q_cond.data.ptr, ctypes.c_void_p)
        s_ptr = lib.c_null_ptr()
        if mol.omega < 0:
            s_ptr = ctypes.cast(self.s_estimator.data.ptr, ctypes.c_void_p)

        vj = vk = None
        vj_ptr = vk_ptr = lib.c_null_ptr()
        assert with_j or with_k
        if with_k:
            win_vk, vk_node = get_shared(dms.shape, set_zeros=True)
            vk = cupy.zeros(dms.shape)
            vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
        if with_j:
            win_vj, vj_node = get_shared(dms.shape, set_zeros=True)
            vj = cupy.zeros(dms.shape)
            vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)

        tile_mappings = _make_tril_tile_mappings(l_ctr_bas_loc, tile_q_cond,
                                                    log_cutoff-log_max_dm)
        workers = gpu_specs['multiProcessorCount']
        pool = cupy.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
        info = cupy.empty(2, dtype=np.uint32)

        init_constant(mol)
        timing_counter = Counter()
        kern_counts = 0
        kern = libvhf_rys.RYS_build_jk

        eri_sizes_tasks = []
        for i, j, k, l in tasks:
            eri_sizes_tasks.append(tile_mappings[i,j].size * tile_mappings[k,l].size)
        tasks_ranks = get_eri_tasks_ranks(tasks, eri_sizes_tasks, npart=nrank)

        if irank < len(tasks_ranks):
            tasks = tasks_ranks[irank]
            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                tile_ij_mapping = tile_mappings[i,j]
                tile_kl_mapping = tile_mappings[k,l]
                if len(tile_ij_mapping) == 0 or len(tile_kl_mapping) == 0:
                    continue
                scheme = schemes[task]
                err = kern(
                    vj_ptr, vk_ptr, ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    self.rys_envs, (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(tile_ij_mapping.size),
                    ctypes.c_int(tile_kl_mapping.size),
                    ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                    tile_q_ptr, q_ptr, s_ptr,
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(info.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(workers),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                if err != 0:
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    msg = f'processing {llll}, tasks = {info[1].get()} on Device {device_id}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                    kern_counts += 1

            if with_j:
                Accumulate_GA_shm(win_vj, vj_node, vj.get())
            if with_k:
                Accumulate_GA_shm(win_vk, vk_node, vk.get())

        comm.Barrier()
        if nnode > 1:
            if with_j:
                Acc_and_get_GA(vj_node)
            if with_k:
                Acc_and_get_GA(vk_node)
            comm.Barrier()

        if irank_shm == 0:
            if with_j:
                vj.set(vj_node)
                if hermi == 1:
                    vj *= 2.
                else:
                    vj, vjT = vj[:n_dm//2], vj[n_dm//2:]
                    vj += vjT.transpose(0,2,1)
                vj = transpose_sum(vj)
            if with_k:
                vk.set(vk_node)
                if hermi == 1:
                    vk = transpose_sum(vk)
                else:
                    vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                    vk += vkT.transpose(0,2,1)
        else:
            vj = vk = None
        #return vj, vk, kern_counts, timing_counter

        #results = multi_gpu.run(proc, args=(dms, dm_cond), non_blocking=True)
        if self.h_shls:
            dms = dms.get()
            dm_cond = None
        else:
            dms = dm_cond = None


        h_shls = self.h_shls
        if h_shls:
            raise NotImplementedError
            log.debug3('Integrals for %s functions on CPU',
                       lib.param.ANGULAR[LMAX+1])
            scripts = []
            if with_j:
                scripts.append('ji->s2kl')
            if with_k:
                if hermi == 1:
                    scripts.append('jk->s2il')
                else:
                    scripts.append('jk->s1il')
            shls_excludes = [0, h_shls[0]] * 4
            vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                     dms, 1, mol._atm, mol._bas, mol._env,
                                     shls_excludes=shls_excludes)
            if with_j and with_k:
                vj1 = asarray(vs_h[0])
                vk1 = asarray(vs_h[1])
            elif with_j:
                vj1 = asarray(vs_h[0])
            else:
                vk1 = asarray(vs_h[0])
            if with_j:
                vj += hermi_triu(vj1)
            if with_k:
                if hermi:
                    vk1 = hermi_triu(vk1)
                vk += vk1
        if with_j: free_win(win_vj)
        if with_k: free_win(win_vk)
        return vj, vk

    

class RysIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
    ]

def _scale_sp_ctr_coeff(mol):
    # Match normalization factors of s, p functions in libcint
    _env = mol._env.copy()
    ls = mol._bas[:,ANG_OF]
    ptr, idx = np.unique(mol._bas[:,PTR_COEFF], return_index=True)
    ptr = ptr[ls[idx] < 2]
    idx = idx[ls[idx] < 2]
    fac = ((ls[idx]*2+1) / (4*np.pi)) ** .5
    nprim = mol._bas[idx,NPRIM_OF]
    nctr = mol._bas[idx,NCTR_OF]
    for p, n, f in zip(ptr, nprim*nctr, fac):
        _env[p:p+n] *= f
    return _env

def iter_cart_xyz(n):
    return [(x, y, n-x-y)
            for x in reversed(range(n+1))
            for y in reversed(range(n+1-x))]

def g_pair_idx(ij_inc=None):
    dat = []
    xyz = [np.array(iter_cart_xyz(li)) for li in range(LMAX+1)]
    for li in range(LMAX+1):
        for lj in range(LMAX+1):
            li1 = li + 1
            idx = (xyz[lj][:,None] * li1 + xyz[li]).transpose(2,0,1)
            dat.append(idx.ravel())
    g_idx = np.hstack(dat).astype(np.int32)
    offsets = np.cumsum([0] + [x.size for x in dat]).astype(np.int32)
    return g_idx, offsets

def init_constant(mol):
    g_idx, offsets = g_pair_idx()
    err = libvhf_rys.RYS_init_constant(
        g_idx.ctypes, offsets.ctypes, mol._env.ctypes,
        ctypes.c_int(mol._env.size), ctypes.c_int(SHM_SIZE))
    if err != 0:
        device_id = cupy.cuda.device.get_device_id()
        raise RuntimeError(f'CUDA kernel initialization on device {device_id}')

def _make_tril_tile_mappings(l_ctr_bas_loc, tile_q_cond, cutoff, tile=TILE):
    n_groups = len(l_ctr_bas_loc) - 1
    ntiles = tile_q_cond.shape[0]
    tile_mappings = {}
    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            i0 = ish0 // tile
            i1 = ish1 // tile
            j0 = jsh0 // tile
            j1 = jsh1 // tile
            sub_tile_q = tile_q_cond[i0:i1,j0:j1]
            mask = sub_tile_q > cutoff
            if i == j:
                mask = cupy.tril(mask)
            t_ij = (cupy.arange(i0, i1, dtype=np.int32)[:,None] * ntiles +
                    cupy.arange(j0, j1, dtype=np.int32))
            idx = cupy.argsort(sub_tile_q[mask])[::-1]
            tile_mappings[i,j] = t_ij[mask][idx]
    return tile_mappings

def _make_j_engine_pair_locs(mol):
    ls = mol._bas[:,ANG_OF]
    ll = (ls[:,None]+ls).ravel()
    pair_loc = np.append(0, np.cumsum((ll+1)*(ll+2)*(ll+3)//6))
    return np.asarray(pair_loc, dtype=np.int32)

def quartets_scheme(mol, l_ctr_pattern, with_j, with_k, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    nfl = (ll + 1) * (ll + 2) // 2
    gout_size = nfi * nfj * nfk * nfl
    g_size = (li+1)*(lj+1)*(lk+1)*(ll+1)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = order // 2 + 1
    jk_cache_size = 0
    if with_j: jk_cache_size += nfi*nfj + nfk*nfl
    if with_k: jk_cache_size += nfi*nfk + nfi*nfl + nfj*nfk + nfj*nfl
    root_g_jk_cache_shared = max(nroots*2 + g_size*3, jk_cache_size)
    unit = root_g_jk_cache_shared + ij_prims + 9
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    while gout_stride < 16 and gout_size / (gout_stride*GOUT_WIDTH) > 1:
        n //= 2
        gout_stride *= 2
    return n, gout_stride

def _j_engine_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nmax = max(lij, lkl)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    g_size = (lij+1)*(lkl+1)
    nf3_ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    nroots = order // 2 + 1

    unit = nroots*2 + g_size*3 + ij_prims + 9
    dm_cache_size = nf3_ij + nf3_kl*2 + (lij+1)*(lkl+1)*(nmax+2)
    gout_size = nf3_ij * nf3_kl
    if dm_cache_size < gout_size:
        unit += dm_cache_size
        shm_size -= nf3_ij * TILE*TILE * 8
        with_gout = False
    else:
        unit += gout_size
        with_gout = True

    if mol.omega < 0:
        unit += nroots*2
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride, with_gout

def _nearest_power2(n, return_leq=True):
    '''nearest 2**x that is leq or geq than n.

    Kwargs:
        return_leq specifies that the return is less or equal than n.
        Otherwise, the return is greater or equal than n.
    '''
    n = int(n)
    assert n > 0
    if return_leq:
        return 1 << (n.bit_length() - 1)
    else:
        return 1 << ((n-1).bit_length())