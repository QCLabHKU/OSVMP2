import os
import sys
import time
import ctypes
import math
import cupy
import cupy as cp
import cupyx
import scipy
import numpy as np
import h5py
import psutil
from cupy_backends.cuda.libs import cublas
from pyscf import gto
from gpu4pyscf.lib.cupy_helper import load_library
from pyscf.lib.parameters import ANGULAR
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, conc_env
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.mole import group_basis, basis_seg_contraction, PTR_BAS_COORD
from gpu4pyscf.scf.jk import _scale_sp_ctr_coeff, _nearest_power2
from gpu4pyscf.scf.int4c2e import libgint
from gpu4pyscf.df.int3c2e import VHFOpt
from gpu4pyscf.df.int3c2e_bdiv import (_conc_locs, estimate_shl_ovlp, init_constant, 
                                      libgint_rys, int3c2e_scheme, Int3c2eEnvVars) #Int3c2eOpt
from osvmp2.__config__ import ngpu
from osvmp2.lib import int3c2eCuda
from osvmp2.osvutil import *
from osvmp2.gpu.cuda_utils import avail_gpu_mem
from osvmp2.mpi_addons import get_shared, register_pinned_memory, unregister_pinned_memory, Acc_and_get_GA, free_win
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

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)
ngpu_shm = ngpu // nnode
nrank_per_gpu = nrank_shm // ngpu_shm
igpu = irank // nrank_per_gpu
igpu_shm = irank_shm // nrank_per_gpu
irank_gpu = irank % nrank_per_gpu
cupy.cuda.runtime.setDevice(igpu_shm)

# Set up process pairs for double buffering
ipair_rank = irank // 2
irank_pair = irank % 2
comm_pair = comm.Split(color=ipair_rank, key=irank)

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 54
THREADS = 256


class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = mol
        self.auxmol = auxmol
        self.sorted_mol = None

    def build(self, cutoff=1e-14, group_size=None, aux_group_size=None):
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        # allow_replica=True to transform the general contracted basis sets into
        # segment contracted sets
        mol, c2s = basis_seg_contraction(self.mol, allow_replica=True)
        mol, coeff, uniq_l_ctr, l_ctr_counts, bas_mapping = group_basis(
            mol, tile=1, return_bas_mapping=True, group_size=group_size)
        self.sorted_mol = mol
        self.uniq_l_ctr = uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.coeff = coeff.dot(c2s).get()
        # Sorted AO indices, allow using the fancyindices to transform tensors
        # between sorted_mol and mol (see function sort_orbitals)
        ao_loc = mol.ao_loc_nr(cart=self.mol.cart)
        ao_idx = np.array_split(np.arange(self.mol.nao), ao_loc[1:-1])
        self.ao_idx = np.hstack([ao_idx[i] for i in bas_mapping]).argsort()

        auxmol, aux_idx, l_ctr_aux_pad_counts, uniq_l_ctr_aux, l_ctr_aux_counts = group_basis(
            self.auxmol, tile=1, sparse_coeff=True, group_size=aux_group_size)
        self.sorted_auxmol = auxmol
        self.uniq_l_ctr_aux = uniq_l_ctr_aux
        l_ctr_aux_offsets = self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_aux_counts))
        self.l_ctr_aux_pad_counts = l_ctr_aux_pad_counts
        self.aux_idx = aux_idx

        self.aux_coeff = self.get_aux_coeff(fitting=True)

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas_cpu[mol.nbas:,PTR_BAS_COORD] += off
        self._atm = _atm_cpu
        self._bas = _bas_cpu
        self._env = _env_cpu
        self.cutoff = cutoff
        '''
        ao_loc_cpu = self.sorted_mol.ao_loc
        aux_loc = self.sorted_auxmol.ao_loc
        _atm = cupy.array(_atm_cpu, dtype=np.int32)
        _bas = cupy.array(_bas_cpu, dtype=np.int32)
        _env = cupy.array(_env_cpu, dtype=np.float64)
        ao_loc = cupy.asarray(_conc_locs(ao_loc_cpu, aux_loc), dtype=np.int32)
        self.int3c2e_envs = Int3c2eEnvVars.new(
            self.sorted_mol.natm, self.sorted_mol.nbas, _atm, _bas, _env, ao_loc, math.log(self.cutoff))'''

        nksh_per_block = 16
        # the auxiliary function offset (address) in the output tensor for each blockIdx.y
        ksh_offsets = []
        for ksh0, ksh1 in zip(l_ctr_aux_offsets[:-1], l_ctr_aux_offsets[1:]):
            ksh_offsets.append(np.arange(ksh0, ksh1, nksh_per_block, dtype=np.int32))
        ksh_offsets.append(l_ctr_aux_offsets[-1])
        ksh_offsets = np.hstack(ksh_offsets)
        ksh_offsets += mol.nbas
        self.ksh_offsets = ksh_offsets

        uniq_l = uniq_l_ctr[:,0]
        assert uniq_l.max() <= LMAX
        n_groups = len(uniq_l)
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))

        ovlp = estimate_shl_ovlp(mol)
        mask = np.tril(ovlp > cutoff)
        # The effective shell pair = ish*nbas+jsh
        shl_pair_idx = []
        # the bas_ij_idx offset for each blockIdx.x
        shl_pair_offsets = []
        # the AO-pair offset (address) in the output tensor for each blockIdx.x
        ao_pair_loc = []
        nao_pair0 = nao_pair = 0
        sp0 = sp1 = 0
        nbas = mol.nbas
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            t_ij = (cupy.arange(ish0, ish1, dtype=np.int32)[:,None] * nbas +
                    cupy.arange(jsh0, jsh1, dtype=np.int32))
            idx = t_ij[mask[ish0:ish1,jsh0:jsh1]]
            nshl_pair = idx.size
            shl_pair_idx.append(idx)
            nfi = (li + 1) * (li + 2) // 2
            nfj = (lj + 1) * (lj + 2) // 2
            nfij = nfi * nfj
            nao_pair0, nao_pair = nao_pair, nao_pair + nfij * nshl_pair

            sp0, sp1 = sp1, sp1 + nshl_pair
            nsp_per_block = THREADS*2
            shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))
            ao_pair_loc.append(
                np.arange(nao_pair0, nao_pair, nsp_per_block*nfij, dtype=np.int32))
            if log.verbose >= logger.DEBUG2:
                log.debug2('group=(%d,%d), li,lj=(%d,%d), sp range(%d,%d,%d), '
                           'nao_pair offset=%d',
                           i, j, li, lj, sp0, sp1, nsp_per_block, nao_pair0)

        self.shl_pair_idx = shl_pair_idx
        shl_pair_offsets.append([sp1])
        self.shl_pair_offsets = np.hstack(shl_pair_offsets)
        ao_pair_loc.append(nao_pair)
        self.ao_pair_loc = np.hstack(ao_pair_loc)
        if log.verbose >= logger.DEBUG1:
            log.timer_debug1('initialize int3c2e_kernel', *t0)
        return self

    def get_aux_coeff(self, fitting=True):
        coeff = np.zeros((self.sorted_auxmol.nao, self.auxmol.nao))

        l_max = max([l_ctr[0] for l_ctr in self.uniq_l_ctr_aux])
        if self.mol.cart:
            cart2sph_per_l = [np.eye((l+1)*(l+2)//2) for l in range(l_max + 1)]
        else:
            cart2sph_per_l = [gto.mole.cart2sph(l, normalized = "sp") for l in range(l_max + 1)]
        i_spherical_offset = 0
        i_cartesian_offset = 0
        for i, l in enumerate(self.uniq_l_ctr_aux[:,0]):
            cart2sph = cart2sph_per_l[l]
            ncart, nsph = cart2sph.shape
            l_ctr_count = self.l_ctr_aux_offsets[i + 1] - self.l_ctr_aux_offsets[i]
            cart_offs = i_cartesian_offset + np.arange(l_ctr_count) * ncart
            sph_offs = i_spherical_offset + np.arange(l_ctr_count) * nsph
            cart_idx = cart_offs[:,None] + np.arange(ncart)
            sph_idx = sph_offs[:,None] + np.arange(nsph)
            coeff[cart_idx[:,:,None],sph_idx[:,None,:]] = cart2sph
            l_ctr_pad_count = self.l_ctr_aux_pad_counts[i]
            i_cartesian_offset += (l_ctr_count + l_ctr_pad_count) * ncart
            i_spherical_offset += l_ctr_count * nsph

        out_gpu = cupy.empty_like(coeff)
        coeff_gpu = cupy.asarray(coeff)
        aux_idx_gpu = cupy.asarray(self.aux_idx, dtype=cupy.int32)
        #out_gpu[:,aux_idx_gpu] = coeff_gpu
        sorted_ids = cupy.empty_like(aux_idx_gpu, dtype=cupy.int32)
        sorted_ids[aux_idx_gpu] = cupy.arange(len(aux_idx_gpu), dtype=cupy.int32)
        cupy.take(coeff_gpu, sorted_ids, axis=1, out=out_gpu)
        coeff = coeff_gpu = None
        
        if fitting:
            j2c_cpu = self.auxmol.intor('int2c2e', hermi=1)
            j2c_gpu = cupy.asarray(j2c_cpu)

            low_gpu = cupy.linalg.cholesky(j2c_gpu)
            cupyx.scipy.linalg.solve_triangular(low_gpu, out_gpu.T, lower=True, 
                                                overwrite_b=True)
        out = cupy.asnumpy(out_gpu)
        return out

    def create_ao_pair_mapping(self, cart=True, return_offsets=False):
        '''ao_pair_mapping stores AO-pair addresses in the nao x nao matrix,
        which allows the decompression for the CUDA kernel generated compressed_int3c:
        sparse_int3c[ao_pair_mapping] = compressed_int3c

        int3c2e CUDA kernel stores intgrals as [ij_shl,j,i,k,ksh].
        ao_pair_mapping indicates the ij addresses in int3c[k,i,j];
        '''
        mol = self.sorted_mol
        ao_loc = cupy.asarray(mol.ao_loc_nr(cart))
        nao = ao_loc[-1]
        uniq_l = self.uniq_l_ctr[:,0]
        if cart:
            nf = (uniq_l + 1) * (uniq_l + 2) // 2
        else:
            nf = uniq_l * 2 + 1
        carts = [cupy.arange(n) for n in nf]
        n_groups = len(uniq_l)
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        nbas = mol.nbas
        ao_pair_mapping = []
        for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
            ish, jsh = divmod(bas_ij_idx, nbas)
            iaddr = ao_loc[ish,None] + carts[i]
            jaddr = ao_loc[jsh,None] + carts[j]
            ao_pair_mapping.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
        if return_offsets:
            npairs = [len(imap) for imap in ao_pair_mapping]
            offsets = np.empty(len(npairs)+1, dtype=np.int32)
            offsets[0] = 0
            np.cumsum(npairs, out=offsets[1:])
            return cupy.hstack(ao_pair_mapping), offsets
        else:
            return cupy.hstack(ao_pair_mapping)
    
    def int3c2e_generator(self, cutoff=1e-14, verbose=None, buf=None):
        '''Generator that yields the 3c2e integral tensor in multiple batches.

        Each batch is a two-dimensional tensor. The first dimension corresponds
        to compressed orbital pairs, which can be indexed using the row and cols
        returned by the .orbital_pair_nonzero_indices() method. The second
        dimension is a slice along the auxiliary basis dimension.
        '''
        if self.sorted_mol is None:
            self.build(cutoff)
        log = logger.new_logger(self.mol, verbose)
        t0 = t1 = log.init_timer()
        l_ctr_offsets = self.l_ctr_offsets
        l_ctr_aux_offsets = self.l_ctr_aux_offsets
        #int3c2e_envs = self.int3c2e_envs
        
        _atm_cpu = self._atm
        _bas_cpu = self._bas
        _env_cpu = self._env
        mol = self.sorted_mol
        omega = mol.omega
        ao_loc_cpu = self.sorted_mol.ao_loc
        aux_loc = self.sorted_auxmol.ao_loc
        naux = aux_loc[-1]

        _atm = cupy.array(_atm_cpu, dtype=np.int32)
        _bas = cupy.array(_bas_cpu, dtype=np.int32)
        _env = cupy.array(_env_cpu, dtype=np.float64)
        ao_loc = cupy.asarray(_conc_locs(ao_loc_cpu, aux_loc), dtype=np.int32)
        int3c2e_envs = Int3c2eEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc, math.log(self.cutoff))

        uniq_l = self.uniq_l_ctr[:,0]
        nfcart = (uniq_l + 1) * (uniq_l + 2) // 2
        n_groups = len(uniq_l)
        ij_tasks = [(i, j) for i in range(n_groups) for j in range(i+1)]
        if buf is None:
            npair_ij = 0
            for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
                nfij = nfcart[i] * nfcart[j]
                npair_ij = max(npair_ij, len(bas_ij_idx) * nfij)
            buf = cupy.empty((npair_ij, naux))

        init_constant(mol)
        kern = libgint_rys.fill_int3c2e
        timing_collection = {}
        kern_counts = 0

        for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
            npair_ij = len(bas_ij_idx)
            if npair_ij == 0:
                continue
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            bas_ij_idx = cupy.asarray(bas_ij_idx, dtype=np.int32)
            li = uniq_l[i]
            lj = uniq_l[j]
            nfij = nfcart[i] * nfcart[j]
            int3c = cupy.ndarray((npair_ij*nfij, naux), dtype=np.float64, memptr=buf.data)
            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
                shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(li, lj, lk, omega)
                log.debug2('int3c2e_scheme for %s: %s', lll, scheme)
                err = kern(
                    ctypes.cast(int3c.data.ptr, ctypes.c_void_p),
                    ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(*shls_slice), aux_loc.ctypes,
                    ctypes.c_int(naux), ctypes.c_int(npair_ij),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(mol.natm),
                    _bas_cpu.ctypes, ctypes.c_int(mol.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
                if log.verbose >= logger.DEBUG1:
                    t1, t1p = log.timer_debug1(f'processing {lll}', *t1), t1
                    if lll not in timing_collection:
                        timing_collection[lll] = 0
                    timing_collection[lll] += t1[1] - t1p[1]
                    kern_counts += 1

            ij_shls = ish0, ish1, jsh0, jsh1
            yield ij_shls, int3c

        if log.verbose >= logger.DEBUG1:
            cupy.cuda.Stream.null.synchronize()
            log.timer('int3c2e', *t0)
            log.debug1('kernel launches %d', kern_counts)
            for lll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', lll, t)
    

    def int3c2e_slice_generator(self, cutoff=1e-14, verbose=None, buf=None):
        '''Generator that yields the 3c2e integral tensor in multiple batches.

        Each batch is a two-dimensional tensor. The first dimension corresponds
        to compressed orbital pairs, which can be indexed using the row and cols
        returned by the .orbital_pair_nonzero_indices() method. The second
        dimension is a slice along the auxiliary basis dimension.
        '''
        if self.sorted_mol is None:
            self.build(cutoff)
        log = logger.new_logger(self.mol, verbose)
        t0 = t1 = log.init_timer()
        l_ctr_offsets = self.l_ctr_offsets
        l_ctr_aux_offsets = self.l_ctr_aux_offsets
        #int3c2e_envs = self.int3c2e_envs
        _atm_cpu = self._atm
        _bas_cpu = self._bas
        _env_cpu = self._env
        mol = self.sorted_mol
        omega = mol.omega
        ao_loc_cpu = self.sorted_mol.ao_loc
        aux_loc = self.sorted_auxmol.ao_loc
        naux = aux_loc[-1]

        _atm = cupy.array(_atm_cpu, dtype=np.int32)
        _bas = cupy.array(_bas_cpu, dtype=np.int32)
        _env = cupy.array(_env_cpu, dtype=np.float64)
        ao_loc = cupy.asarray(_conc_locs(ao_loc_cpu, aux_loc), dtype=np.int32)
        int3c2e_envs = Int3c2eEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc, math.log(self.cutoff))

        uniq_l = self.uniq_l_ctr[:,0]
        nfcart = (uniq_l + 1) * (uniq_l + 2) // 2
        n_groups = len(uniq_l)
        ij_tasks = [(i, j) for i in range(n_groups) for j in range(i+1)]
        if buf is None:
            npair_ij = 0
            for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
                nfij = nfcart[i] * nfcart[j]
                npair_ij = max(npair_ij, len(bas_ij_idx) * nfij)
            buf = cupy.empty((npair_ij, naux))

        init_constant(mol)
        kern = libgint_rys.fill_int3c2e
        all_bas_ij_idx = [cupy.asarray(bas_ij_idx, dtype=np.int32) for bas_ij_idx in self.shl_pair_idx]

        #for (i, j), bas_ij_idx in zip(ij_tasks, self.shl_pair_idx):
        def generate_int3c2e_slice(cpij):
            i, j = ij_tasks[cpij]
            bas_ij_idx = all_bas_ij_idx[cpij] #self.shl_pair_idx[cpij]

            npair_ij = len(bas_ij_idx)
            if npair_ij == 0:
                return None, None
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            #bas_ij_idx = cupy.asarray(bas_ij_idx, dtype=np.int32)
            li = uniq_l[i]
            lj = uniq_l[j]
            nfij = nfcart[i] * nfcart[j]
            #int3c = cupy.ndarray((npair_ij*nfij, naux), dtype=np.float64, memptr=buf.data)
            int3c = buf[:npair_ij*nfij*naux].reshape(npair_ij*nfij, naux)

            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
                shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(li, lj, lk, omega)
                
                err = kern(
                    ctypes.cast(int3c.data.ptr, ctypes.c_void_p),
                    ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(*shls_slice), aux_loc.ctypes,
                    ctypes.c_int(naux), ctypes.c_int(npair_ij),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(mol.natm),
                    _bas_cpu.ctypes, ctypes.c_int(mol.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
                
            ij_shls = ish0, ish1, jsh0, jsh1
            return ij_shls, int3c
        return generate_int3c2e_slice

    def int3c2e_aux_slice_generator(self, buf=None, cutoff=1e-14, verbose=None):
        if self.sorted_mol is None:
            self.build(cutoff)
        log = logger.new_logger(self.mol, verbose)
        t0 = t1 = log.init_timer()
        l_ctr_offsets = self.l_ctr_offsets
        l_ctr_aux_offsets = self.l_ctr_aux_offsets
        #int3c2e_envs = self.int3c2e_envs
        _atm_cpu = self._atm
        _bas_cpu = self._bas
        _env_cpu = self._env
        mol = self.sorted_mol
        omega = mol.omega
        ao_loc_cpu = self.sorted_mol.ao_loc
        aux_loc = self.sorted_auxmol.ao_loc
        naux = aux_loc[-1]

        _atm = cupy.array(_atm_cpu, dtype=np.int32)
        _bas = cupy.array(_bas_cpu, dtype=np.int32)
        _env = cupy.array(_env_cpu, dtype=np.float64)
        ao_loc = cupy.asarray(_conc_locs(ao_loc_cpu, aux_loc), dtype=np.int32)
        int3c2e_envs = Int3c2eEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc, math.log(self.cutoff))
        
        lks = self.uniq_l_ctr_aux[:,0]
        aux_loc_slice = aux_loc.copy()
        for cpk in range(len(lks)):
            ksh0, ksh1 = l_ctr_aux_offsets[cpk:cpk+2]
            aux_loc_slice[ksh0:ksh1] -= aux_loc_slice[ksh0]

        uniq_l = self.uniq_l_ctr[:,0]
        nfcart = (uniq_l + 1) * (uniq_l + 2) // 2
        n_groups = len(uniq_l)
        ij_tasks = np.asarray([(i, j) for i in range(n_groups) for j in range(i+1)], dtype=np.int32)

        init_constant(mol)
        kern = libgint_rys.fill_int3c2e
        

        if buf is None:
            max_naux = np.max(aux_loc[l_ctr_aux_offsets[1:]] - aux_loc[l_ctr_aux_offsets[:-1]])
            nbas_ij = np.asarray([len(bas_ij_idx) for bas_ij_idx in self.shl_pair_idx], dtype=np.int32)
            nfij = nfcart[ij_tasks[:, 0]] * nfcart[ij_tasks[:, 1]]
            npair_ij = np.max(nbas_ij * nfij)

            buf = cupy.empty((npair_ij*max_naux))
        
        bas_ij_indices = [cupy.asarray(bas_ij_idx) for bas_ij_idx in self.shl_pair_idx]

        def compute_int3c_aux_slice(cpij, cpk):

            lk = lks[cpk]
            ksh0, ksh1 = l_ctr_aux_offsets[cpk:cpk+2]
            naux_slice = aux_loc[ksh1] - aux_loc[ksh0]
            #aux_loc_slice[ksh0:ksh1] -= aux_loc_slice[ksh0]

            i, j = ij_tasks[cpij]
            bas_ij_idx = bas_ij_indices[cpij]

            npair_ij = len(bas_ij_idx)
            if npair_ij == 0:
                return None, None
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            li = uniq_l[i]
            lj = uniq_l[j]
            nfij = nfcart[i] * nfcart[j]
                
            shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
            lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
            scheme = int3c2e_scheme(li, lj, lk, omega)
            log.debug2('int3c2e_scheme for %s: %s', lll, scheme)

            #int3c = cupy.ndarray((npair_ij*nfij, naux_slice), dtype=np.float64, memptr=buf.data)
            int3c = buf[:npair_ij*nfij*naux_slice].reshape(npair_ij*nfij, naux_slice)
            err = kern(
                ctypes.cast(int3c.data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                (ctypes.c_int*6)(*shls_slice), aux_loc_slice.ctypes,
                ctypes.c_int(naux_slice), ctypes.c_int(npair_ij),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                _atm_cpu.ctypes, ctypes.c_int(mol.natm),
                _bas_cpu.ctypes, ctypes.c_int(mol.nbas), _env_cpu.ctypes)
            if err != 0:
                raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')

            shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
            return shls_slice, int3c
        
        return compute_int3c_aux_slice
    

def preprocess_half_trans(ao_pair_mapping, nao_cart, enforce_tri=True):

    # Compute index arrays once
    n_pairs = len(ao_pair_mapping)
    al_indices = (ao_pair_mapping // nao_cart).astype(cupy.int32)
    be_indices = (ao_pair_mapping % nao_cart).astype(cupy.int32)
    
    # Create pair indices array for tracking
    pair_ids = cupy.arange(n_pairs, dtype=cupy.int32)

    if enforce_tri:
        tri_mask = cupy.where(al_indices >= be_indices)[0]
        al_indices = al_indices[tri_mask]
        be_indices = be_indices[tri_mask]
        pair_ids = pair_ids[tri_mask]

    # Diagonal mask to exclude from 'be' position processing
    diag_mask = (al_indices == be_indices)
    
    # Build unified arrays for both positions:
    # - 'al' position includes all pairs
    # - 'be' position excludes diagonal pairs
    combined_ao_ids = cupy.concatenate([al_indices, be_indices[~diag_mask]])
    
    # Single sort operation to group by AO ID
    sort_order = cupy.argsort(combined_ao_ids)
    
    # Apply sorting to create final grouped arrays
    sorted_ao_ids = combined_ao_ids[sort_order]
    sorted_pair_indices = cupy.concatenate([pair_ids, pair_ids[~diag_mask]])[sort_order]
    other_aos = cupy.concatenate([be_indices, al_indices[~diag_mask]])[sort_order]
    
    # Extract unique AOs and compute segment sizes for offsets
    all_uniq_aos, uniq_counts = cupy.unique(sorted_ao_ids, return_counts=True)
    
    # Compute cumulative offsets (faster than incremental append)
    sortedPairOffsets = cupy.empty(len(all_uniq_aos) + 1, dtype=cupy.int32)
    sortedPairOffsets[0] = 0
    cupy.cumsum(uniq_counts, out=sortedPairOffsets[1:])
    
    return all_uniq_aos, sortedPairOffsets, other_aos, sorted_pair_indices

def sparse_half_trans(int3c, occCoeff, nao_cart, ialp=None, 
                      pidx0=0, naux_full=None, enforce_tri=True, preproc_info=None, 
                      ao_pair_mapping=None):

    assert preproc_info is not None or ao_pair_mapping is not None

    nSparsePair, naux = int3c.shape
    #nocc = occCoeff.shape[1]
    nocc = occCoeff.shape[0]
    if naux_full is None:
        naux_full = naux

    if ialp is None:
        ialp = cupy.zeros((nocc, nao_cart * naux), dtype=cupy.float64)
    
    if preproc_info is None:
        (all_uniq_aos, sortedPairOffsets, 
        other_aos, sorted_pair_indices) = preprocess_half_trans(ao_pair_mapping, nao_cart, enforce_tri)
    else:
        (all_uniq_aos, sortedPairOffsets, 
        other_aos, sorted_pair_indices) = preproc_info
    nao0 = len(all_uniq_aos)

    int3c2eCuda.sparseHalfTransCupy(occCoeff, int3c, ialp,
                                    sortedPairOffsets, other_aos,
                                    sorted_pair_indices, all_uniq_aos,
                                    nocc, nao_cart, naux, naux_full,
                                    nSparsePair, nao0, pidx0)
    
    return ialp

PIN_SPEED = 3186.07

def generate_ialp_cuda(self, intopt, occCoeff_full, buf_size=None, cpu_buf_size=None, 
                       events=None, recorders=None, log=None):
    
    if log is None:
        log = logger.Logger(sys.stdout, self.verbose)
    
    if self.fully_direct == 2:
        assert events is not None
    if recorders is None:
        recorders = [[], [], []]
    recorder_feri, recorder_ialp, recorder_data = recorders

    nao_cart = intopt.mol.nao_nr(cart=True)
    naux_cart = intopt.auxmol.nao_nr(cart=True)

    assert occCoeff_full.shape[1] == nao_cart

    aux_loc = intopt.sorted_auxmol.ao_loc
    uniq_l = intopt.uniq_l_ctr[:,0]
    nfcart = (uniq_l + 1) * (uniq_l + 2) // 2
    n_groups = len(uniq_l)
    ij_tasks = np.asarray([(i, j) for i in range(n_groups) for j in range(i+1)], dtype=np.int32)
    
    ao_pair_mapping_cart, ao_pair_offsets = intopt.create_ao_pair_mapping(cart=True, return_offsets=True)
    
    
    full_nocc, nao_cart = occCoeff_full.shape
    nbas_ij = np.asarray([len(bas_ij_idx) for bas_ij_idx in intopt.shl_pair_idx], dtype=np.int32)
    nfij = nfcart[ij_tasks[:, 0]] * nfcart[ij_tasks[:, 1]]
    max_naopair = np.max(nbas_ij * nfij)
    max_naux = np.max(aux_loc[intopt.l_ctr_aux_offsets[1:]] - aux_loc[intopt.l_ctr_aux_offsets[:-1]])
    int3c_buf_size = max_naopair * max_naux
    alp_size = 2*nao_cart*max_naux if self.fully_direct == 2 else nao_cart*naux_cart

    if buf_size is None:
        buf_size = int(0.8 * avail_gpu_mem(self.gpu_memory, unit="B") / 8)
    if self.fully_direct == 2:
        if cpu_buf_size is None:
            cpu_buf_size = int(get_mem_spare(self.mol, ratio=0.8) * 1e6 / 8)
        nocc_limit = min(cpu_buf_size // (nao_cart*naux_cart), full_nocc)
        if nocc_limit == 0:
            raise MemoryError("%d GB is insufficient"%(cpu_buf_size*8*1e-9))
        max_size = max(nocc_limit*alp_size+int3c_buf_size, 2*nocc_limit*nao_cart*naux_cart)
    else:
        nocc_limit = full_nocc
        max_size = full_nocc*alp_size+int3c_buf_size
    buf_size = min(max_size, buf_size)
    
    max_nocc = min(nocc_limit, (buf_size-int3c_buf_size) // alp_size)
    nbatch = full_nocc // max_nocc 
    if full_nocc % max_nocc != 0:
        nbatch += 1
    log.info("    max_nocc=%d, batch_nocc=%d, nbatch=%d, GPU buffer %.2f GB"%(full_nocc, max_nocc, nbatch, 
                                                                              buf_size*8 / (1024**3)))
    buf_gpu = cupy.empty(buf_size)

    int3c_buf_gpu = buf_gpu[:int3c_buf_size]
    ialp_buf_gpu = buf_gpu[int3c_buf_size:]
    
    comp_stream = cupy.cuda.Stream.null
    if self.fully_direct == 2:
        if ialp_buf_gpu.size % 2 != 0:
            ialp_buf_gpu = ialp_buf_gpu[:-1]
        ialp_buf_gpu = ialp_buf_gpu.reshape(2, -1)

        t0 = get_current_time()
        ialp_buf_cpu = np.empty(max_nocc*nao_cart*naux_cart, dtype=np.float64)
        register_pinned_memory(ialp_buf_cpu)
        print_time(["pinned memory %.2f MB"%(ialp_buf_cpu.size*8*1e-6), get_elapsed_time(t0)], log)

        data_stream = cupy.cuda.Stream(non_blocking=True)

        comp_events, data_events = events

    preproc_batches = []
    for cpij in range(len(ij_tasks)):
        pair_idx0, pair_idx1 = ao_pair_offsets[[cpij, cpij+1]]
        preproc_batches.append(preprocess_half_trans(ao_pair_mapping_cart[pair_idx0:pair_idx1], 
                                                     nao_cart))

    aux_batches = []
    for cpk in range(len(intopt.uniq_l_ctr_aux)):
        ksh0, ksh1 = intopt.l_ctr_aux_offsets[cpk:cpk+2]
        p0, p1 = aux_loc[[ksh0, ksh1]]
        aux_batches.append([p0, p1])

    int3c2e_generator = intopt.int3c2e_aux_slice_generator(buf=int3c_buf_gpu)

    tt = get_current_time()
    for i0 in np.arange(full_nocc, step=max_nocc):
        i1 = min(full_nocc, i0+max_nocc)
        nocc_batch = i1 - i0

        #ialp = cupy.zeros((nocc_batch, nao_cart * naux_cart))
        if self.fully_direct == 2:
            ialp_cpu = ialp_buf_cpu[:nocc_batch*nao_cart*naux_cart]
        else:
            ialp_shape = (nocc_batch, nao_cart, naux_cart)
            ialp = ialp_buf_gpu[:np.prod(ialp_shape)].reshape(ialp_shape)
            ialp.fill(0.0)
            ialp_comp_gpu = ialp

        occCoeff = occCoeff_full[i0:i1]

        for cpk, (p0, p1) in enumerate(aux_batches):

            naux_slice = p1 - p0
            data_idx = 1 - cpk % 2
            comp_idx = cpk % 2

            with comp_stream:
                if self.fully_direct == 2:
                    if cpk > 1:
                        comp_stream.wait_event(data_events[comp_idx])
                    ialp_comp_shape = (nocc_batch, nao_cart, naux_slice)
                    ialp_comp_gpu = ialp_buf_gpu[comp_idx][:np.prod(ialp_comp_shape)].reshape(ialp_comp_shape)
                    ialp_comp_gpu.fill(0.0)

                    pidx0 = 0
                    naux_full = naux_slice
                else:
                    pidx0 = p0
                    naux_full = naux_cart

                for cpij in range(len(ij_tasks)):
                    
                    t0 = get_current_time()
                    #print(cpij, cpk, flush=True)
                    _, int3c = int3c2e_generator(cpij, cpk)
                    recorder_feri.append(record_elapsed_time(t0))

                    if int3c is not None:
                        t0 = get_current_time()
                        sparse_half_trans(int3c, occCoeff, nao_cart, ialp=ialp_comp_gpu, 
                                        pidx0=pidx0, naux_full=naux_full, 
                                        preproc_info=preproc_batches[cpij])
                        recorder_ialp.append(record_elapsed_time(t0))

                if self.fully_direct == 2:
                    comp_events[comp_idx].record()
        
            if self.fully_direct == 2:
                with data_stream:
                    data_stream.wait_event(comp_events[data_idx])

                    prev_cpk = cpk - 1
                    prev_p0, prev_p1 = aux_batches[prev_cpk]
                    prev_naux_slice = prev_p1 - prev_p0
                    ialp_data_gpu = ialp_buf_gpu[data_idx][:nocc_batch*nao_cart*prev_naux_slice]
                    
                    cpu_idx0 = prev_p0 * nocc_batch * nao_cart
                    cpu_idx1 = prev_p1 * nocc_batch * nao_cart
                    t0 = get_current_time()

                    cupy.asnumpy(ialp_data_gpu, out=ialp_cpu[cpu_idx0:cpu_idx1], 
                                stream=data_stream)
                    
                    recorder_data.append(record_elapsed_time(t0))
                    data_events[data_idx].record()

        if self.fully_direct == 2:
            with data_stream:
                cpk += 1
                data_idx = 1 - cpk % 2
                data_stream.wait_event(comp_events[data_idx])
                
                prev_cpk = cpk - 1
                prev_p0, prev_p1 = aux_batches[prev_cpk]
                prev_naux_slice = prev_p1 - prev_p0
                ialp_data_gpu = ialp_buf_gpu[data_idx][:nocc_batch*nao_cart*prev_naux_slice]
                
                cpu_idx0 = prev_p0 * nocc_batch * nao_cart
                cpu_idx1 = prev_p1 * nocc_batch * nao_cart
                t0 = get_current_time()
                
                cupy.asnumpy(ialp_data_gpu, out=ialp_cpu[cpu_idx0:cpu_idx1], 
                            stream=data_stream)

                
                recorder_data.append(record_elapsed_time(t0))
                data_events[data_idx].record()
            
            comp_stream.synchronize()
            data_stream.synchronize()

            max_nocc_load = min(nocc_batch, buf_size // (nao_cart * (2*naux_cart + max_naux)))
            #print("nocc_load: %d"%max_nocc_load)
            print(max_nocc_load, nocc_batch)

            occ_batches = [[iidx0, min(nocc_batch, iidx0+max_nocc_load)] for iidx0 
                            in np.arange(nocc_batch, step=max_nocc_load)]
            
            ialp_load_buf_gpu = buf_gpu[:2*max_nocc_load*nao_cart*naux_cart].reshape(2, -1)

            # Pre-load
            with data_stream:
                data_idx = 0
                iidx0, iidx1 = occ_batches[0]
                nocc_slice = iidx1 - iidx0
                ialp_data_gpu = ialp_load_buf_gpu[data_idx][:nocc_slice*nao_cart*naux_cart].reshape(nocc_slice, nao_cart, naux_cart)
                
                t0 = get_current_time()
                for p0, p1 in aux_batches:
                    naux_slice = p1 - p0
                    cpu_idx0 = p0 * nocc_batch * nao_cart
                    cpu_idx1 = p1 * nocc_batch * nao_cart
                    ialp_seg_cpu = ialp_cpu[cpu_idx0:cpu_idx1].reshape(nocc_batch, nao_cart, naux_slice)
                    ialp_load_gpu = buf_gpu[-nocc_slice*nao_cart*naux_slice:].reshape(nocc_slice, nao_cart, naux_slice)
                    ialp_load_gpu.set(ialp_seg_cpu[iidx0:iidx1])
                    ialp_data_gpu[:, :, p0:p1] = ialp_load_gpu
                recorder_data.append(record_elapsed_time(t0))

                data_events[data_idx].record()


            for bidx, (iidx0, iidx1) in enumerate(occ_batches):
                nocc_slice = iidx1 - iidx0
                data_idx = 1 - bidx % 2
                comp_idx = bidx % 2
                
                if bidx + 1 < len(occ_batches):
                    with data_stream:
                        if bidx > 0:
                            data_stream.wait_event(comp_events[data_idx])

                        next_iidx0, next_iidx1 = occ_batches[bidx + 1]
                        next_nocc_slice = next_iidx1 - next_iidx0
                        ialp_data_gpu = ialp_load_buf_gpu[data_idx][:next_nocc_slice*nao_cart*naux_cart].reshape(next_nocc_slice, nao_cart, naux_cart)

                        t0 = get_current_time()
                        for p0, p1 in aux_batches:
                            naux_slice = p1 - p0
                            cpu_idx0 = p0 * nocc_batch * nao_cart
                            cpu_idx1 = p1 * nocc_batch * nao_cart
                            ialp_seg_cpu = ialp_cpu[cpu_idx0:cpu_idx1].reshape(nocc_batch, nao_cart, naux_slice)
                            ialp_load_gpu = buf_gpu[-next_nocc_slice*nao_cart*naux_slice:].reshape(next_nocc_slice, nao_cart, naux_slice)
                            ialp_load_gpu.set(ialp_seg_cpu[next_iidx0:next_iidx1])
                            ialp_data_gpu[:, :, p0:p1] = ialp_load_gpu

                        recorder_data.append(record_elapsed_time(t0))

                        data_events[data_idx].record()

                with comp_stream:
                    comp_stream.wait_event(data_events[comp_idx])
                    if iidx1+i0 == full_nocc:
                        #cupy.cuda.runtime.deviceSynchronize()
                        unregister_pinned_memory(ialp_buf_cpu)
                    ialp_comp_gpu = ialp_load_buf_gpu[comp_idx][:nocc_slice*nao_cart*naux_cart] 
                    yield (comp_idx, iidx0+i0, iidx1+i0), ialp_comp_gpu.reshape(nocc_slice, nao_cart, naux_cart)
 
        else:
            yield (i0, i1), ialp
