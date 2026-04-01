import sys
import os
import numpy as np
import scipy.linalg
from pyscf import lib
from osvmp2.__config__ import ngpu
from osvmp2.mpi_addons import *
from osvmp2.osvutil import *
from osvmp2.lib import localization
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank // nrank_shm
#ngpu = min(int(os.environ.get("ngpu", 0)), nrank)

if ngpu:
    #set up gpu parameters
    import cupy
    import cupyx
    import cupyx.scipy.linalg
    from osvmp2.gpu.cuda_utils import sliceJobsFor2DBlocks
    from osvmp2.lib import localizationCuda


    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    ranks_gpu_shm = np.arange(nrank_shm).reshape(ngpu_shm, -1)[igpu_shm]
    cupy.cuda.runtime.setDevice(igpu_shm)
    
    THREADS_PER_AXIS = 16
    THREADS_PER_BLOCK = 256


def sqrtm(s, out=None):
    e, v = np.linalg.eigh(s)
    return np.dot(v * np.sqrt(e), v.T.conj(), out=out)

def get_sc_cuda(smat, mocc):
    if smat.flags['F_CONTIGUOUS']:
        smat_col = smat
    elif smat.flags['C_CONTIGUOUS']:
        smat_col = smat.T
    else:
        smat_col = np.asfortranarray(smat)
    e, v = cupy.linalg.eigh(smat_col)
    cupy.dot(v * np.sqrt(e), v.T.conj(), out=smat_col.T)
    return cupy.dot(smat_col, mocc)

def get_cs_cuda(smat, mocc):
    if smat.flags['F_CONTIGUOUS']:
        smat_col = smat
    elif smat.flags['C_CONTIGUOUS']:
        smat_col = smat.T
    else:
        smat_col = cupy.asfortranarray(smat)

    e, v = cupy.linalg.eigh(smat_col)

    cupy.dot(v * np.sqrt(e), v.T.conj(), out=smat_col.T)
    cs = cupy.dot(mocc.T, smat_col)
    return cs

def lowdin(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v / np.sqrt(e), v.T.conj())


def sliceJobsFor2DBlockss(indices, dimListA, dimListB, threadsPerAxis):
    # Calculate total number of elements to preallocate memory
    blocks = 0
    for dimA, dimB in zip(dimListA, dimListB):
        nSegA = (dimA + threadsPerAxis - 1) // threadsPerAxis  # Ceiling division
        nSegB = (dimB + threadsPerAxis - 1) // threadsPerAxis
        blocks += nSegA * nSegB
    blocks = int(blocks)
    # Preallocate memory for the results
    indicesBlock = cupy.empty(blocks, dtype=cupy.int32)
    localIndicesBlock = cupy.empty(blocks, dtype=cupy.int32)
    dimSegBlockA = cupy.empty(blocks, dtype=cupy.int32)
    dimSegBlockB = cupy.empty(blocks, dtype=cupy.int32)

    indices = indices.reshape(-1, 1)
    offset = 0
    for localIdx, (idx, dimA, dimB) in enumerate(zip(indices, dimListA, dimListB)):
        # Generate segment indices for current dimensions
        dimSegA = cupy.arange(0, dimA, threadsPerAxis, dtype=cupy.int32)
        dimSegB = cupy.arange(0, dimB, threadsPerAxis, dtype=cupy.int32)
        nSegA, nSegB = len(dimSegA), len(dimSegB)
        sizeNow = nSegA * nSegB

        if sizeNow == 0:
            continue  # Skip if no indices generated

        # Compute block indices using vectorized operations
        indicesA = dimSegA.repeat(nSegB)
        indicesB = cupy.tile(dimSegB, nSegA)

        # Fill preallocated arrays
        dimSegBlockA[offset:offset + sizeNow] = indicesA
        dimSegBlockB[offset:offset + sizeNow] = indicesB
        indicesBlock[offset:offset + sizeNow] = cupy.repeat(idx, sizeNow)
        localIdx = cupy.asarray(localIdx, dtype=cupy.int32)
        localIndicesBlock[offset:offset + sizeNow] = cupy.repeat(localIdx, sizeNow)
        offset += sizeNow

    return indicesBlock, localIndicesBlock, dimSegBlockA, dimSegBlockB

def get_paij(mol, mo_coeff, smat, partition, iop, use_gpu=False, log=None):
    
    nao, nmo = mo_coeff.shape
    natom = len(partition)

    if use_gpu:
        
        if irank_shm == 0:
            
            if iop == 1:
                '''sc = sqrtm(smat).T.dot(mo_coeff)
                pija = np.empty((nmo, nmo, natom))
                for ia in np.arange(natom):
                    al0, al1 = partition[ia]
                    sc_ia = sc[al0:al1]
                    pija[:, :, ia] = np.dot(sc_ia.T, sc_ia)'''

                
                mocc_gpu = cupy.asarray(mo_coeff)
                smat_gpu = cupy.asarray(smat)
                ao_size_sum_gpu = cupy.asarray(partition[:, 1], dtype=cupy.int32)
                #print_time(["gpu data trans.", get_elapsed_time(t0)])

                #sc_gpu = get_sc_cuda(smat_gpu, mocc_gpu)
                cs_gpu = get_cs_cuda(smat_gpu, mocc_gpu)
                #cupy.cuda.runtime.deviceSynchronize()
                #print_time(["sc", get_elapsed_time(t0)])

                
                pija_gpu = cupy.empty((nmo, nmo, natom), dtype=cupy.float64)

                '''atoms_gpu = cupy.arange(natom, dtype=cupy.int32)
                nmo_atoms_gpu = cupy.repeat(cupy.asarray([nmo], dtype=cupy.int32), natom)

                atoms_block_gpu, _, mos_col_gpu, mos_row_gpu = sliceJobsFor2DBlocks(atoms_gpu, nmo_atoms_gpu, nmo_atoms_gpu, 
                                                                                    THREADS_PER_AXIS, THREADS_PER_AXIS)'''

                atoms = np.arange(natom, dtype=np.int32) 
                nmo_atoms = np.repeat(np.asarray([nmo], dtype=np.int32), natom)

                atoms_blocks, _, mos_col, mos_row = sliceJobsFor2DBlocks(atoms, nmo_atoms, nmo_atoms, 
                                                                          THREADS_PER_AXIS, THREADS_PER_AXIS)
                
                atoms_block_gpu = cupy.asarray(atoms_blocks, dtype=cupy.int32)
                mos_row_gpu = cupy.asarray(mos_row, dtype=cupy.int32)
                mos_col_gpu = cupy.asarray(mos_col, dtype=cupy.int32)

                localizationCuda.pijaLowdinCupy(cs_gpu, pija_gpu, atoms_block_gpu,
                                        ao_size_sum_gpu, mos_row_gpu, mos_col_gpu,
                                        nmo, natom, nao, len(atoms_block_gpu))
                pija = pija_gpu
            else:
                raise NotImplementedError()
        else:
            pija = None
        
        win_pija = None
                

    else:
        if iop == 2:
            na = 3
            cal_slice = get_slice(range(nrank), job_size=na)[irank]
        elif iop == 3:
            na = 1
        else:
            na = natom
            weights = partition[:, 1] - partition[:, 0]
            cal_slice = get_slice(range(nrank), job_size=natom, weight_list=weights)[irank]
        
        win_pija, pija = get_shared((nmo, nmo, na), dtype=np.float64, set_zeros=True)

        
        # Mulliken matrix
        if iop == 0:
            #cts = mo_coeff.T.dot(smat)
            #for ia, (al0, al1) in enumerate(partition):
            if cal_slice is not None:
                for ia in cal_slice:
                    al0, al1 = partition[ia]
                    #tmp = np.dot(cts[:, al0:al1], mo_coeff[al0:al1])
                    tmp = mo_coeff.T @ smat[:, al0:al1] @ mo_coeff[al0:al1]
                    pija[:, :, ia] = 0.5 * (tmp + tmp.T)
        # Lowdin
        elif iop == 1:
            t0 = get_current_time()
            #win_s12, s12 = get_shared((nao, nao))
            #win_ve, ve_node = get_shared((nao, nao))
            gopt = 1
            if gopt == 0:
                win_s12, s12 = get_shared((nao, nao))
                sqrtm(smat, out=s12)
                comm_shm.Barrier()

                if cal_slice is not None:
                    for ia in cal_slice:
                        al0, al1 = partition[ia]
                        #sc_ia = s12c[al0:al1]
                        sc_ia = s12.T[al0:al1] @ mo_coeff
                        pija[:, :, ia] = np.dot(sc_ia.T, sc_ia)
                comm_shm.Barrier()
                win_s12.Free()

            elif gopt == 1:
                win_vmo, vmo_node = get_shared((nao, nmo))
                win_v, v_node = get_shared((nao, nao))
                if irank_shm == 0:
                    #sqrtm(smat, s12)
                    e, v = np.linalg.eigh(smat)
                    np.copyto(v_node, v)
                    esqrt = np.sqrt(e, out=e)
                    np.dot((v * esqrt).T, mo_coeff, out=vmo_node)
                comm_shm.Barrier()
                #s12c = s12.T.dot(mo_coeff)
                #for ia, (al0, al1) in enumerate(partition):
                if cal_slice is not None:
                    for ia in cal_slice:
                        al0, al1 = partition[ia]
                        #sc_ia = s12c[al0:al1]
                        #sc_ia = s12.T[al0:al1] @ mo_coeff
                        sc_ia = v_node[al0:al1] @ vmo_node
                        pija[:, :, ia] = np.dot(sc_ia.T, sc_ia)
                comm_shm.Barrier()
                win_vmo.Free()
                win_v.Free()
        # Boys
        elif iop == 2:
            if cal_slice is not None:
                rmat = mol.intor_symmetric("cint1e_r_sph", 3)
                for icart in range(na):
                    pija[icart] = mo_coeff.T @ rmat[icart] @ mo_coeff
        elif iop == 3:
            pija[:] = np.dot(mo_coeff.T, mo_coeff).reshape(pija.shape)
        # P[i,j,a]
        #pija = np.ascontiguousarray(pija.transpose(1, 2, 0))
        comm.Barrier()
        Acc_and_get_GA(pija)
        comm.Barrier()
    return win_pija, pija

def get_ijdx(nmo, pija):
    # Compute all pairwise indices (i, j) where i < j
    ilist, jlist = np.triu_indices(nmo, k=1)

    # Vectorized computation for bij
    vij = pija[ilist, ilist] - pija[jlist, jlist]
    bij = np.abs(np.sum(pija[ilist, jlist] * vij, axis=-1))

    # Combine indices with bij values and sort by bij in descending order
    return np.column_stack((ilist, jlist))[np.argsort(-bij)]

def print_test(a, varname=None):
    a = a.ravel()
    msg = '%.8E  %.8E  %.8E'%(a.min(), a.max(), a.mean())
    if varname is not None:
        msg = '%s: '%varname + msg
    print(msg)

def loc_jacobi(mol, mo_coeff, tol=1e-6, max_cycle=1000, iop=0, 
               smat=None, log=None, use_cpp=True, use_gpu=False):
    if log is None:
        log = lib.logger.Logger(sys.stdout, mol.verbose)
    if smat is None:
        smat = mol.intor_symmetric("cint1e_ovlp_sph")
    partition = mol.offset_nr_by_atom()[:, 2:]
    
    t0 = get_current_time()
    win_pija, pija = get_paij(mol, mo_coeff, smat, partition, iop, use_gpu=use_gpu, log=log)
    
    print_time(["pija", get_elapsed_time(t0)], log=log)

    if irank_shm == 0:
        # Start
        nmo = mo_coeff.shape[1]
        
        t0 = get_current_time()
        if use_gpu:
            #pija in in gpu memory!
            ut_gpu = cupy.empty((nmo, nmo))
            localizationCuda.locJacobiCupy(pija, ut_gpu, max_cycle, tol)
            u = cupy.asnumpy(ut_gpu.T)

        elif use_cpp:
            u = np.identity(nmo)
            localization.locJacobiNumpy(pija, u, max_cycle, tol, irank)
        else:
            u = np.identity(nmo)
            pija_rav = pija.ravel()
            fun = np.dot(pija_rav, pija_rav)

            log.info(" initial funval = %.9f"%fun)
            for icycle in range(max_cycle):
                delta = 0.0
                # i>j
                pairs = get_ijdx(nmo, pija)
                for i, j in pairs:
                    # determine angle
                    pija_ij = pija[i, j]
                    vij = pija[i, i] - pija[j, j]
                    aij = np.dot(pija_ij, pija_ij) - 0.25 * np.dot(vij, vij)
                    bij = np.dot(pija_ij, vij)
                    
                    if abs(aij) < 1.0e-10 and abs(bij) < 1.0e-10:
                        continue
                    
                    theta = np.arctan2(bij, -aij)
                    a = theta / 4.0
                    cosa = np.cos(a)
                    sina = np.sin(a)

                    if np.abs(sina) < 1e-10:
                        continue

                    p1 = np.hypot(aij, bij)
                    cos4a = np.cos(theta)

                    # incremental value
                    delta += p1 * (1 - cos4a)

                    # Apply rotation to transformation matrix u
                    ui = u[:, i] * cosa + u[:, j] * sina
                    uj = -u[:, i] * sina + u[:, j] * cosa
                    u[:, i] = ui
                    u[:, j] = uj

                    # Bra-transformation of Integrals
                    tmp_ip = pija[i] * cosa + pija[j] * sina
                    tmp_jp = -pija[i] * sina + pija[j] * cosa
                    pija[i] = tmp_ip
                    pija[j] = tmp_jp

                    # Ket-transformation of Integrals
                    tmp_ip = pija[:, i] * cosa + pija[:, j] * sina
                    tmp_jp = -pija[:, i] * sina + pija[:, j] * cosa
                    pija[:, i] = tmp_ip
                    pija[:, j] = tmp_jp

                fun = fun + delta
                log.info("    Cycle %d, delta=%.4e, fun=%.8f"%(icycle, delta, fun))
                if delta < tol:
                    break

            # Check
            if delta < tol:
                log.info("    Localization converged!")
            else:
                log.info("    WARNING: Localization not converged")
        print_time(["jacobi sweeps", get_elapsed_time(t0)], log=log)
    else:
        u = None
    comm_shm.Barrier()
    if win_pija is not None:
        win_pija.Free()
    return u

def loc(mol, mo_coeff, tol=1e-6, max_cycle=1000, iop=0, 
        smat=None, opt=0, verbose=3, use_cpp=True, use_gpu=False):
    log = lib.logger.Logger(sys.stdout, verbose)

    # Jacobi sweeps
    if opt == 0:
        return loc_jacobi(mol, mo_coeff, tol, max_cycle, iop, smat, log, use_cpp, use_gpu)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    def read_xyz(xyz_file):
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            natom = int(lines[0])
            coord = ""
            for l in lines[2:2+natom]:
                coord += l
        return coord
    from pyscf import gto, scf
    import h5py
    
    molecule = "water8"; basis = "ccpvtz"
    #molecule = "water100"; basis = "def2-tzvp"
    #molecule = "water190"; basis = "ccpvtz"
    file_mo = f"{molecule}/mocc.chk"
    file_xyz = f"{molecule}/{molecule}.xyz"

    with h5py.File(file_mo, 'r') as fmo:
        mocc = np.asarray(fmo["mocc"])
    mol = gto.Mole()
    mol.atom = read_xyz(file_xyz)
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.build()

    use_cpp = use_gpu = False #True
    #nocc = mol.nelectron // 2
    ierr, uo = loc(mol, mocc, max_cycle=1000, iop=1, verbose=4, use_cpp=use_cpp, use_gpu=use_gpu)


