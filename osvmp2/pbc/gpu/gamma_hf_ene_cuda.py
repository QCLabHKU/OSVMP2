import os
import sys
import ctypes
import h5py
import numpy as np
import cupy
from pyscf.lib import logger
from pyscf import lib, gto
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.rsdf_builder import (estimate_ke_cutoff_for_omega)
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.tools import k2gamma
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, asarray, sandwich_dot)
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.pbc.df.int3c2e import libpbc#, sr_aux_e2#, fill_triu_bvk_conj)
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_LR
from osvmp2.__config__ import ngpu
from osvmp2.osvutil import *
from osvmp2.mpi_addons import *
from osvmp2.pbc.gpu.gamma_int_3c2e_cuda import (get_pbc_slice_ranks, get_pbc_nao_range, 
                                                SRInt3c2eOpt, gamma_aux_2e_cuda, sr_aux_e2, sr_aux_e2_nuc)
from osvmp2.gpu.cuda_utils import (avail_gpu_mem, ave_gpu_memory, get_seg_gpu, dgemm_cupy, dgemv_cupy)
from mpi4py import MPI
#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
#inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm= comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//comm_shm.size
inode = irank // nrank_shm

#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)
ngpu_shm = ngpu // nnode
nrank_per_gpu = nrank_shm // ngpu_shm
igpu = irank // nrank_per_gpu
igpu_shm = irank_shm // nrank_per_gpu
irank_gpu = irank % nrank_per_gpu
cupy.cuda.runtime.setDevice(igpu_shm)

THREADS = 256

def get_gamma_ialp_bp_cuda(self, dm):
    cupy.get_default_memory_pool().free_all_blocks()

    nocc = self.occ_coeff.shape[1]
    cell = self.mol
    auxcell = self.auxmol
    nao = cell.nao_nr()
    naoaux = auxcell.nao_nr()
    
    omega = self.with_df.df_builder.omega
    int3c2e_opt = self.intopt #SRInt3c2eOpt(cell, auxcell, -omega, fitting=True)

    self.ao_idx = int3c2e_opt.ao_idx

    ncaux = auxcell.nao_nr(cart=True)
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    ngrid = np.prod(cell.symmetrize_mesh(cell.cutoff_to_mesh(ke_cutoff)))
    shared_buf_size = nao*nocc + naoaux*ncaux + ngrid*(naoaux+ncaux)
    gpu_memory_f8 = ave_gpu_memory(max_mem=self.gpu_memory) * 1e6 // 8 - shared_buf_size
    gpu_memory_f8 /= nrank_per_gpu
    min_nao, max_nao = get_pbc_nao_range(int3c2e_opt)
    max_l = max(cell._bas[:, gto.ANG_OF])
    max_c2s = ((max_l+1)*(max_l+2)//2) / (2*max_l + 1)

    max_nao1 = min(max_nao, max(min_nao, int(0.2*gpu_memory_f8 / (nocc+nocc*naoaux))))
    max_nao0 = min(max_nao, max(min_nao, int(0.6*gpu_memory_f8 / (3*(max_c2s**2)*max_nao1*naoaux))))
    max_nao1 = min(max_nao, int(0.8 * gpu_memory_f8 / (3*(max_c2s**2)*max_nao0*naoaux + nocc*(naoaux+1))))

    feri_mem_f8 = int(0.8*gpu_memory_f8 - nocc*max_nao1*(1+naoaux))

    int3c2e_opt.build(group_size_aoi=max_nao0, group_size_aoj=max_nao1)

    ao_slice, shell_slice_ranks = get_pbc_slice_ranks(int3c2e_opt, ncore=nrank, group_i=False)

    if igpu is None:
        shell_slice_gpu = None
    else:
        shell_slice_gpu = shell_slice_ranks[igpu]

    nao = self.nao
    naoaux = self.naoaux
    
    win_bp, bp_node = get_shared(naoaux, set_zeros=True)

    if nocc != self.nocc_pre:
        file_ialp = h5py.File(self.file_ialp, 'w', driver='mpio', comm=comm)
        ialp_data = file_ialp.create_dataset('ialp', (nocc, nao, naoaux), dtype='f8')
    else:
        file_ialp = h5py.File(self.file_ialp, 'r+', driver='mpio', comm=comm)
        ialp_data = file_ialp['ialp']

    max_memory = get_mem_spare(cell, 0.9)
    if shell_slice_gpu is not None:
        ij_tasks = []
        for _, ijs in shell_slice_gpu:
            ij_tasks.extend(ijs)
    else:
        ij_tasks = None

    int_kernel, shared_ptrs = gamma_aux_2e_cuda(int3c2e_opt, solve_j3c=False, ij_tasks=ij_tasks, 
                                                feri_mem_f8=feri_mem_f8)
    
    t1 = get_current_time()
    if irank_gpu == 0:
        occ_coeff_sorted = self.occ_coeff[int3c2e_opt.ao_idx]
    else:
        occ_coeff_sorted = None
    occ_coeff_gpu, occ_coeff_ptr = get_shared_cupy((nao, nocc), numpy_array=occ_coeff_sorted)
    accumulate_time(self.t_data, t1)

    shared_ptrs.append(occ_coeff_ptr)


    if shell_slice_gpu is not None:
        ao0, ao1 = ao_slice[igpu][0], ao_slice[igpu][1]
        
        #dm_gpu = cupy.asarray(dm_sorted)
        #occ_coeff_gpu = cupy.asarray(occ_coeff_sorted)
        

        nao_cpu = get_ncols_for_memory(max_memory*0.7, nocc*naoaux, nao)
        shell_seg, nao_cpu = get_seg_gpu(shell_slice_gpu, nao_cpu)

        buf_ialp_cpu = np.empty((nocc*nao_cpu*naoaux))
        buf_ialp_gpu = cupy.empty((max_nao1*nocc*naoaux))
        buf_occ_coeff_gpu = cupy.empty((max_nao1*nocc))
        bp_gpu = cupy.zeros(naoaux)

        c_li_counts = np.asarray(int3c2e_opt.cell0_ctr_li_counts)
        c_lj_counts = np.asarray(int3c2e_opt.cell0_ctr_lj_counts)
        uniq_li = np.asarray(int3c2e_opt.cell0_uniq_li)
        uniq_lj = np.asarray(int3c2e_opt.cell0_uniq_lj)
        nf_i = uniq_li * 2 + 1
        nf_j = uniq_lj * 2 + 1
        c_aoi_offsets = cum_offset(c_li_counts*nf_i)
        c_aoj_offsets = cum_offset(c_lj_counts*nf_j)

        ao_idx0 = 0
        save_idx0 = 0
        for iseg in shell_seg:
            AL0, AL1 = iseg[0][0][0], iseg[-1][0][1]
            nao_seg = AL1 - AL0
            ialp_cpu = buf_ialp_cpu[:nocc*nao_seg*naoaux].reshape(nocc, nao_seg, naoaux)
            ao_idx0 = 0
            for (al0, al1), cpidx_list in iseg:
                nao0 = al1 - al0
                ialp_gpu = buf_ialp_gpu[:nocc*nao0*naoaux].reshape(nocc, nao0, naoaux)
                ialp_gpu.fill(0.0)
                #ialp_gpu = cupy.zeros((nocc, nao0, naoaux))
                t2 = get_current_time()
                for cpi, cpj in cpidx_list:
                    be0, be1 = c_aoi_offsets[cpi:cpi+2]
                    nao1 = be1 - be0
                    t1 = get_current_time()
                    int3c_slice = int_kernel(cpi, cpj) # (be, al | P)
                    accumulate_time(self.t_feri, t1)

                    t1 = get_current_time()
                    #ialp_gpu += cupy.dot(occ_coeff_gpu[be0:be1].T, int3c_slice.reshape(nao1, -1)).reshape(nocc, nao0, self.naoaux)
                    dgemm_cupy(1, 0, occ_coeff_gpu[be0:be1], int3c_slice.reshape(nao1, -1), ialp_gpu.reshape(nocc, -1), 1.0, 1.0)
                    accumulate_time(self.t_k, t1)

                t1 = get_current_time()
                #bp_gpu += cupy.dot(occ_coeff_gpu[al0:al1].T.ravel(), ialp_gpu.reshape(-1, naoaux))
                occ_coeff_slice = buf_occ_coeff_gpu[:nao0*nocc].reshape(nocc, nao0)
                occ_coeff_slice[:] = occ_coeff_gpu[al0:al1].T
                #bp_gpu += cupy.einsum("ij,ijk->k", occ_coeff_slice, ialp_gpu, optimize=True)
                dgemv_cupy(0, ialp_gpu.reshape(-1, naoaux), occ_coeff_slice.ravel(), bp_gpu, 1.0, 1.0)
                accumulate_time(self.t_j, t1)

                ao_idx1 = ao_idx0 + nao0
                accumulate_time(self.t_gpu, t2)
                
                t1 = get_current_time()
                ialp_cpu[:, ao_idx0:ao_idx1] = cupy.asnumpy(ialp_gpu)
                accumulate_time(self.t_data, t1)
                ao_idx0 = ao_idx1
            
            t1 = get_current_time()
            ialp_data.write_direct(ialp_cpu, dest_sel=np.s_[:, AL0:AL1])
            accumulate_time(self.t_write, t1)
            
        t1 = get_current_time()
        bp = cupy.asnumpy(bp_gpu)
        accumulate_time(self.t_data, t1)
        
        #if irank_shm != 0:
        Accumulate_GA_shm(win_bp, bp_node, bp)

    file_ialp.close()
    #fence_and_free(win_low)
    comm_shm.Barrier()

    for ptr in shared_ptrs:
        close_ipc_handle(ptr)

    return win_bp, bp_node


def get_gamma_j_step2_cuda(self, win_vj, vj_node, bp_node):
    gpu_memory_f8 = ave_gpu_memory(self.gpu_memory) * 1e6 // 8
    nao = self.nao
    naoaux = self.auxmol.nao_nr()
    cell = self.mol
    auxcell = self.auxmol
    omega = self.with_df.df_builder.omega
    int3c2e_opt = self.intopt #SRInt3c2eOpt(cell, auxcell, -omega, fitting=True)

    '''feri_mem_f8 = gpu_memory_f8 - nao * nao
    min_nao, max_nao = get_pbc_nao_range(int3c2e_opt)
    max_nao1 = min(max_nao, max(min_nao, int(0.8*feri_mem_f8 / (2*max_nao*naoaux))))
    max_nao0 = max(min_nao, int(0.8*feri_mem_f8 / (2*max_nao1*naoaux)))
    if max_nao0 > max_nao:
        max_nao0 = max_nao
        max_nao1 = min(max_nao, int(0.8*feri_mem_f8 / (2*max_nao0*naoaux)))'''
    
    ncaux = auxcell.nao_nr(cart=True)
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    ngrid = np.prod(cell.symmetrize_mesh(cell.cutoff_to_mesh(ke_cutoff)))
    shared_buf_size = nao*nao + naoaux*ncaux + ngrid*(naoaux+ncaux)
    gpu_memory_f8 = ave_gpu_memory(max_mem=self.gpu_memory) * 1e6 // 8 - shared_buf_size
    gpu_memory_f8 /= nrank_per_gpu
    min_nao, max_nao = get_pbc_nao_range(int3c2e_opt)
    max_l = max(cell._bas[:, gto.ANG_OF])
    max_s2c = (2*max_l + 1) / ((max_l+1)*(max_l+2)//2) 

    max_nao1 = min(max_nao, max(min_nao, int(0.8*max_s2c*gpu_memory_f8 / (3*max_nao*naoaux))))
    max_nao0 = max(min_nao, int(0.8*max_s2c*gpu_memory_f8 / (3*max_nao1*naoaux)))
    max_nao1 = min(max_nao, int(0.8*max_s2c*gpu_memory_f8 / (3*max_nao0*naoaux)))
    feri_mem_f8 = int(0.8*gpu_memory_f8)

    int3c2e_opt.build(group_size_aoi=max_nao0, group_size_aoj=max_nao1)

    _, shell_slice_ranks = get_pbc_slice_ranks(int3c2e_opt, ncore=nrank)
    if igpu is None:
        shell_slice_gpu = None
    else:
        shell_slice_gpu = shell_slice_ranks[igpu]

    if shell_slice_gpu is not None:
        sorted_ao_indices = int3c2e_opt.ao_idx
        c_li_counts = np.asarray(int3c2e_opt.cell0_ctr_li_counts)
        c_lj_counts = np.asarray(int3c2e_opt.cell0_ctr_lj_counts)
        uniq_li = np.asarray(int3c2e_opt.cell0_uniq_li)
        uniq_lj = np.asarray(int3c2e_opt.cell0_uniq_lj)
        nf_i = uniq_li * 2 + 1
        nf_j = uniq_lj * 2 + 1
        c_aoi_offsets = cum_offset(c_li_counts*nf_i)
        c_aoj_offsets = cum_offset(c_lj_counts*nf_j)
        ij_tasks = []
        for _, ijs in shell_slice_gpu:
            ij_tasks.extend(ijs)
    else:
        ij_tasks = None
    
    int_kernel, shared_ptrs, feri_buf = gamma_aux_2e_cuda(int3c2e_opt, solve_j3c=False, ij_tasks=ij_tasks, 
                                                feri_mem_f8=feri_mem_f8, return_buf=True)

    vj_gpu, vj_ptr = get_shared_cupy((nao, nao), set_zeros=True)
    shared_ptrs.append(vj_ptr)

    if shell_slice_gpu is not None:
        t1 = get_current_time()
        bp_gpu = cupy.asarray(bp_node)
        accumulate_time(self.t_data, t1)
        

        t2 = get_current_time()
        for _, ij_ids in shell_slice_gpu:
            for cpi, cpj in ij_ids:
                al0, al1 = c_aoi_offsets[cpi], c_aoi_offsets[cpi+1]
                be0, be1 = c_aoj_offsets[cpj], c_aoj_offsets[cpj+1]
                #be0, be1 = self.intopt.ao_loc_sj[cpj], self.intopt.ao_loc_sj[cpj+1]
                nao0 = al1 - al0
                nao1 = be1 - be0

                t1 = get_current_time()
                int3c_slice = int_kernel(cpi, cpj)
                accumulate_time(self.t_feri, t1)

                t1 = get_current_time()
                #vj_gpu[al0:al1, be0:be1] += cupy.dot(int3c_slice.reshape(-1, self.naoaux), bp_gpu).reshape(nao0, nao1)
                '''sorted_al_indices = sorted_ao_indices[al0:al1]
                sorted_be_indices = sorted_ao_indices[be0:be1]

                vj_slice = cupy.dot(int3c_slice.reshape(-1, self.naoaux), bp_gpu)


                vj_gpu[sorted_al_indices[:,None], sorted_be_indices] = vj_slice.reshape(nao0, nao1)
                '''
            
                vj_gpu[al0:al1, be0:be1] = cupy.dot(int3c_slice.reshape(-1, self.naoaux), bp_gpu).reshape(nao0, nao1)


                accumulate_time(self.t_j, t1)
        #vj_gpu = self.intopt.unsort_orbitals(vj_gpu, axis=[0, 1])
        accumulate_time(self.t_gpu, t2)

    else:
        vj = None
    comm_shm.Barrier()

    
    if irank_gpu == 0:
        unsorted_ao_indices = cupy.empty(nao, dtype=cupy.int32)
        unsorted_ao_indices[int3c2e_opt.ao_idx] = cupy.arange(nao, dtype=cupy.int32)
        vj1_gpu = feri_buf[:nao*nao].reshape(nao, nao)
        cupy.take(vj_gpu, unsorted_ao_indices, axis=0, out=vj1_gpu)
        cupy.take(vj1_gpu, unsorted_ao_indices, axis=1, out=vj_gpu)

        t1 = get_current_time()
        vj = cupy.asnumpy(vj_gpu)
        accumulate_time(self.t_data, t1)

        Accumulate_GA_shm(win_vj, vj_node, vj)
    comm_shm.Barrier()

    for ptr in shared_ptrs:
        close_ipc_handle(ptr)

    return vj

from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
def fill_triu_bvk_conj(a, nao, bvk_kmesh):
    # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
    assert a.flags.c_contiguous
    conj_mapping = conj_images_in_bvk_cell(bvk_kmesh)
    conj_mapping = cupy.asarray(conj_mapping, dtype=np.int32)
    bvk_ncells = np.prod(bvk_kmesh)
    err = libpbc.aopair_fill_triu(
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(conj_mapping.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(bvk_ncells))
    if err != 0:
        raise RuntimeError('aopair_fill_triu failed')
    return a


def get_pp_loc_part1(cell, kpts=None, with_pseudo=True, verbose=None):
    log = logger.new_logger(cell, verbose)
    cell_exps, cs = extract_pgto_params(cell, 'diffused')
    omega = 0.2
    log.debug('omega guess in get_pp_loc_part1 = %g', omega)

    if kpts is None or is_zero(kpts):
        kpts = None
        bvk_kmesh = np.ones(3, dtype=int)
        bvk_ncells = 1
    else:
        bvk_kmesh = kpts_to_kmesh(cell, kpts)
        bvk_ncells = np.prod(bvk_kmesh)
    # TODO: compress

    t0 = get_current_time()
    fakenuc = aft_cpu._fake_nuc(cell, with_pseudo=with_pseudo)
    '''nuc = sr_aux_e2(cell, fakenuc, -omega, kpts, bvk_kmesh, j_only=True)
    charges = -cupy.asarray(cell.atom_charges())
    if kpts is None:
        nuc = contract('pqr,r->pq', nuc, charges)
    else:
        nuc = contract('kpqr,r->kpq', nuc, charges)'''
    
    nuc = sr_aux_e2_nuc(cell, fakenuc, -omega, kpts, bvk_kmesh, j_only=True)

    t0 = print_time(["sr_aux_e2", get_elapsed_time(t0)])

    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    Gv, (basex, basey, basez), kws = cell.get_Gv_weights(mesh)
    if with_pseudo:
        #TODO: call multigrid.eval_vpplocG after removing its part2 contribution
        ZG = ft_ao.ft_ao(fakenuc, Gv).conj()
        ZG = ZG.dot(charges)
        ZG *= asarray(_weighted_coulG_LR(cell, Gv, omega, kws))
        if ((cell.dimension == 3 or
            (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            exps = cupy.asarray(np.hstack(fakenuc.bas_exps()))
            ZG[0] -= charges.dot(np.pi/exps) / cell.vol
    else:
        basex = cupy.asarray(basex)
        basey = cupy.asarray(basey)
        basez = cupy.asarray(basez)
        b = cell.reciprocal_vectors()
        coords = cell.atom_coords()
        rb = cupy.asarray(coords.dot(b.T))
        SIx = cupy.exp(-1j*rb[:,0,None] * basex)
        SIy = cupy.exp(-1j*rb[:,1,None] * basey)
        SIz = cupy.exp(-1j*rb[:,2,None] * basez)
        SIx *= cupy.asarray(-cell.atom_charges())[:,None]
        ZG = cupy.einsum('qx,qy,qz->xyz', SIx, SIy, SIz).ravel().conj()
        ZG *= asarray(_weighted_coulG_LR(cell, Gv, omega, kws))
    ZG = cupy.ascontiguousarray(ZG)

    ft_opt = ft_ao.FTOpt(cell, bvk_kmesh=bvk_kmesh).build()
    sorted_cell = ft_opt.sorted_cell
    bvkcell = ft_opt.bvkcell
    uniq_l = ft_opt.uniq_l_ctr[:,0]
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    l_ctr_offsets = ft_opt.l_ctr_offsets

    img_idx_cache = ft_opt.make_img_idx_cache(True, log)

    # Determine the addresses of the non-vanished pairs and the diagonal indices
    # within these elements.
    nbas = sorted_cell.nbas
    ao_loc = sorted_cell.ao_loc
    nao = ao_loc[nbas]
    ao_loc = cupy.asarray(ao_loc)
    nf = (uniq_l + 1) * (uniq_l + 2) // 2
    cart_idx = [cupy.arange(n) for n in nf]
    aopair_idx = []
    p0 = p1 = 0
    for i, j in img_idx_cache:
        bas_ij = img_idx_cache[i, j][0]
        ish, J, jsh = cupy.unravel_index(bas_ij, (nbas, bvk_ncells, nbas))
        nfij = nf[i] * nf[j]
        p0, p1 = p1, p1 + nfij * len(bas_ij)
        # Note: corresponding to the storage order (nfj,nfi,npairs,nGv)
        iaddr = ao_loc[ish] + cart_idx[i][:,None]
        jaddr = ao_loc[jsh] + cart_idx[j][:,None]
        ijaddr = iaddr * nao + jaddr[:,None,:] + J * nao**2
        aopair_idx.append(ijaddr.ravel())
        iaddr = jaddr = ijaddr = None
    nao_pairs = p1
    aopair_idx = cupy.hstack(aopair_idx)

    t0 = print_time(["nuc1", get_elapsed_time(t0)])

    avail_mem = avail_gpu_mem(unit="B") * .8
    ngrids = len(Gv)
    Gblksize = max(16, int(avail_mem/(2*16*nao_pairs*bvk_ncells))//8*8)
    Gblksize = min(Gblksize, ngrids, 16384)
    log.debug2('ft_ao_iter ngrids = %d Gblksize = %d', ngrids, Gblksize)
    kern = libpbc.build_ft_aopair
    nuc_compressed = 0
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        # Padding zeros, allowing idle threads to access these data
        GvT = cupy.append(cupy.asarray(Gv[p0:p1]).T.ravel(), cupy.zeros(THREADS))
        nGv = p1 - p0
        pqG = cupy.empty((nao_pairs, nGv), dtype=np.complex128)
        pair0 = 0
        for i, j in img_idx_cache:
            bas_ij, img_offsets, img_idx = img_idx_cache[i, j]
            npairs = len(bas_ij)
            if npairs == 0:
                continue

            li = uniq_l[i]
            lj = uniq_l[j]
            ll_pattern = f'{l_symb[i]}{l_symb[j]}'
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            scheme = ft_ao.ft_ao_scheme(cell, li, lj, nGv)
            log.debug2('ft_ao_scheme for %s: %s', ll_pattern, scheme)
            err = kern(
                ctypes.cast(pqG[pair0:].data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), # Do not remove zero elements
                ctypes.byref(ft_opt.aft_envs), (ctypes.c_int*3)(*scheme),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.c_int(npairs), ctypes.c_int(nGv),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                bvkcell._atm.ctypes, ctypes.c_int(bvkcell.natm),
                bvkcell._bas.ctypes, ctypes.c_int(bvkcell.nbas),
                bvkcell._env.ctypes)
            if err != 0:
                raise RuntimeError(f'build_ft_aopair kernel for {ll_pattern} failed')
            pair0 += npairs * nf[i] * nf[j]
        #print(pqG.shape, ZG[p0:p1].shape)
        #nuc_compressed += contract('pG,G->p', pqG, ZG[p0:p1]).real
        nuc_compressed += cupy.dot(pqG, ZG[p0:p1]).real
        pqG = GvT = None

    nuc_raw = cupy.zeros((bvk_ncells * nao * nao))
    nuc_raw[aopair_idx] = nuc_compressed
    nuc_raw = nuc_raw.reshape(bvk_ncells, nao, nao)
    nuc_raw = fill_triu_bvk_conj(nuc_raw, nao, bvk_kmesh)
    nuc_raw = sandwich_dot(nuc_raw, ft_opt.coeff)

    if kpts is None:
        nuc += nuc_raw[0]
    else:
        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True)
        expLk = cupy.exp(1j*cupy.asarray(bvkmesh_Ls.dot(kpts.T)))
        nuc += contract('lk,lpq->kpq', expLk, nuc_raw)
    
    t0 = print_time(["nuc2", get_elapsed_time(t0)])

    return nuc

def get_nuc(cell, kpts=None):
    '''Get the periodic nuc-el AO matrix, with G=0 removed.
    '''
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    nuc = get_pp_loc_part1(cell, kpts, with_pseudo=False, verbose=log)
    log.timer('get_nuc', *t0)
    return nuc
