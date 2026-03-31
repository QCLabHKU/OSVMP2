import numpy as np
import h5py
from pyscf import gto, scf, df, lib
from pyscf.dft import gen_grid, numint, radi
from pyscf.dft.gen_grid import make_mask, BLKSIZE
from pyscf.solvent import ddcosmo#, ddpcm, ddcosmo_grad
from pyscf.symm import sph
from pyscf.grad import rks as rks_grad
import ctypes
from osvmp2.osvutil import *
from osvmp2.mpi_addons import *
from mpi4py import MPI


comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nnode = nrank/comm_shm.size

def get_eps(sol_name):
    eps_dic = {
    'water':                          78.3553,
    'acetonitrile':                   35.688,
    'methanol':                       32.613,
    'ethanol':                        24.852,
    'isoquinoline':                   11.00,
    'quinoline':                      9.16,
    'chloroform':                     4.7113,
    'diethylether':                   4.2400,
    'dichloromethane':                8.93,
    'dichloroethane':                 10.125,
    'carbontetrachloride':            2.2280,
    'benzene':                        2.2706,
    'toluene':                        2.3741,
    'chlorobenzene':                  5.6968,
    'nitromethane':                   36.562,
    'heptane':                        1.9113,
    'cyclohexane':                    2.0165,
    'aniline':                        6.8882,
    'acetone':                        20.493,
    'tetrahydrofuran':                7.4257,
    'dimethylsulfoxide':              46.826,
    'argon':                          1.430,
    'krypton':                        1.519,
    'xenon':                          1.706,
    'noctanol':                       9.8629,
    '111trichloroethane':             7.0826,
    '112trichloroethane':             7.1937,
    '124trimethylbenzene':            2.3653,
    '12dibromoethane':                4.9313,
    '12ethanediol':                   40.245,
    '14dioxane':                      2.2099,
    '1bromo2methylpropane':           7.7792,
    '1bromooctane':                   5.0244,
    '1bromopentane':                  6.269,
    '1bromopropane':                  8.0496,
    '1butanol':                       17.332,
    '1chlorohexane':                  5.9491,
    '1chloropentane':                 6.5022,
    '1chloropropane':                 8.3548,
    '1decanol':                       7.5305,
    '1fluorooctane':                  3.89,
    '1heptanol':                      11.321,
    '1hexanol':                       12.51,
    '1hexene':                        2.0717,
    '1hexyne':                        2.615,
    '1iodobutane':                    6.173,
    '1iodohexadecane':                3.5338,
    '1iodopentane':                   5.6973,
    '1iodopropane':                   6.9626,
    '1nitropropane':                  23.73,
    '1nonanol':                       8.5991,
    '1pentanol':                      15.13,
    '1pentene':                       1.9905,
    '1propanol':                      20.524,
    '222trifluoroethanol':            26.726,
    '224trimethylpentane':            1.9358,
    '24dimethylpentane':              1.8939,
    '24dimethylpyridine':             9.4176,
    '26dimethylpyridine':             7.1735,
    '2bromopropane':                  9.3610,
    '2butanol':                       15.944,
    '2chlorobutane':                  8.3930,
    '2heptanone':                     11.658,
    '2hexanone':                      14.136,
    '2methoxyethanol':                17.2,
    '2methyl1propanol':               16.777,
    '2methyl2propanol':               12.47,
    '2methylpentane':                 1.89,
    '2methylpyridine':                9.9533,
    '2nitropropane':                  25.654,
    '2octanone':                      9.4678,
    '2pentanone':                     15.200,
    '2propanol':                      19.264,
    '2propen1ol':                     19.011,
    '3methylpyridine':                11.645,
    '3pentanone':                     16.78,
    '4heptanone':                     12.257,
    '4methyl2pentanone':              12.887,
    '4methylpyridine':                11.957,
    '5nonanone':                      10.6,
    'aceticacid':                     6.2528,
    'acetophenone':                   17.44,
    'achlorotoluene':                 6.7175,
    'anisole':                        4.2247,
    'benzaldehyde':                   18.220,
    'benzonitrile':                   25.592,
    'benzylalcohol':                  12.457,
    'bromobenzene':                   5.3954,
    'bromoethane':                    9.01,
    'bromoform':                      4.2488,
    'butanal':                        13.45,
    'butanoicacid':                   2.9931,
    'butanone':                       18.246,
    'butanonitrile':                  24.291,
    'butylamine':                     4.6178,
    'butylethanoate':                 4.9941,
    'carbondisulfide':                2.6105,
    'cis12dimethylcyclohexane':       2.06,
    'cisdecalin':                     2.2139,
    'cyclohexanone':                  15.619,
    'cyclopentane':                   1.9608,
    'cyclopentanol':                  16.989,
    'cyclopentanone':                 13.58,
    'decalinmixture':                 2.196,
    'dibromomethane':                 7.2273,
    'dibutylether':                   3.0473,
    'diethylamine':                   3.5766,
    'diethylsulfide':                 5.723,
    'diiodomethane':                  5.32,
    'diisopropylether':               3.38,
    'dimethyldisulfide':              9.6,
    'diphenylether':                  3.73,
    'dipropylamine':                  2.9112,
    'e12dichloroethene':              2.14,
    'e2pentene':                      2.051,
    'ethanethiol':                    6.667,
    'ethylbenzene':                   2.4339,
    'ethylethanoate':                 5.9867,
    'ethylmethanoate':                8.3310,
    'ethylphenylether':               4.1797,
    'fluorobenzene':                  5.42,
    'formamide':                      108.94,
    'formicacid':                     51.1,
    'hexanoicacid':                   2.6,
    'iodobenzene':                    4.5470,
    'iodoethane':                     7.6177,
    'iodomethane':                    6.8650,
    'isopropylbenzene':               2.3712,
    'mcresol':                        12.44,
    'mesitylene':                     2.2650,
    'methylbenzoate':                 6.7367,
    'methylbutanoate':                5.5607,
    'methylcyclohexane':              2.024,
    'methylethanoate':                6.8615,
    'methylmethanoate':               8.8377,
    'methylpropanoate':               6.0777,
    'mxylene':                        2.3478,
    'nbutylbenzene':                  2.36,
    'ndecane':                        1.9846,
    'ndodecane':                      2.0060,
    'nhexadecane':                    2.0402,
    'nhexane':                        1.8819,
    'nitrobenzene':                   34.809,
    'nitroethane':                    28.29,
    'nmethylaniline':                 5.9600,
    'nmethylformamidemixture':        181.56,
    'nndimethylacetamide':            37.781,
    'nndimethylformamide':            37.219,
    'nnonane':                        1.9605,
    'noctane':                        1.9406,
    'npentadecane':                   2.0333,
    'npentane':                       1.8371,
    'nundecane':                      1.9910,
    'ochlorotoluene':                 4.6331,
    'ocresol':                        6.76,
    'odichlorobenzene':               9.9949,
    'onitrotoluene':                  25.669,
    'oxylene':                        2.5454,
    'pentanal':                       10.0,
    'pentanoicacid':                  2.6924,
    'pentylamine':                    4.2010,
    'pentylethanoate':                4.7297,
    'perfluorobenzene':               2.029,
    'pisopropyltoluene':              2.2322,
    'propanal':                       18.5,
    'propanoicacid':                  3.44,
    'propanonitrile':                 29.324,
    'propylamine':                    4.9912,
    'propylethanoate':                5.5205,
    'pxylene':                        2.2705,
    'pyridine':                       12.978,
    'secbutylbenzene':                2.3446,
    'tertbutylbenzene':               2.3447,
    'tetrachloroethene':              2.268,
    'tetrahydrothiophenessdioxide':   43.962,
    'tetralin':                       2.771,
    'thiophene':                      2.7270,
    'thiophenol':                     4.2728,
    'transdecalin':                   2.1781,
    'tributylphosphate':              8.1781,
    'trichloroethene':                3.422,
    'triethylamine':                  2.3832,
    'xylenemixture':                  2.3879,
    'z12dichloroethene':              9.2}
    if type(sol_name) == str:
        sol_name = sol_name.replace('-','').replace(',','').lower()
        return eps_dic[sol_name]
    else:
        return sol_name


def make_grids_one_sphere(lebedev_order):
    ngrid_1sph = gen_grid.LEBEDEV_ORDER[lebedev_order]
    leb_grid = np.empty((ngrid_1sph,4))
    gen_grid.libdft.MakeAngularGrid(leb_grid.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(ngrid_1sph))
    coords_1sph = leb_grid[:,:3]
    # Note the Lebedev angular grids are normalized to 1 in pyscf
    weights_1sph = 4*np.pi * leb_grid[:,3]
    return coords_1sph, weights_1sph
def _vstack_factor_fak_pol(fak_pol, lmax):
    fac_pol = []
    for l in range(lmax+1):
        fac = 4*np.pi/(l*2+1)
        fac_pol.append(fac * fak_pol[l])
    return np.vstack(fac_pol)

def get_sol_3c2e(pcmobj, cav_coords, blksize, cintopt):
    '''
    Pre-generation of 3c2e for solvent
    '''
    nao = pcmobj.mol.nao_nr()
    nao_tril = nao*(nao+1)//2
    if pcmobj.int_storage == 1:
        pcmobj.int3c2e = "sol_int3c2e_%d.tmp"%irank
        f_int = h5py.File(pcmobj.int3c2e, 'w')
        int_save = f_int.create_dataset("int3c2e", shape=(nao_tril, 
                                        cav_coords.shape[0]), dtype='f8')
    else:
        int_save = pcmobj.int3c2e = np.empty((nao_tril, cav_coords.shape[0]))

    for i0, i1 in lib.prange(0, cav_coords.shape[0], blksize):
        fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
        int_save[:, i0:i1] = df.incore.aux_e2(pcmobj.mol, fakemol, intor='int3c2e', 
                                                    aosym='s2ij', cintopt=cintopt)
    if pcmobj.int_storage == 1:
        f_int.close()

def get_grid_int(pcmobj, ni, grids, deriv, grid0, grid1, blksize, non0tab, buf):
    nao = pcmobj.mol.nao_nr()
    if pcmobj.int_storage == 1:
        pcmobj.grid_int = "grid_int_%d.tmp"%irank
        f_int = h5py.File(pcmobj.grid_int, 'w')
        int_save = f_int.create_dataset("grid_int", shape=((grid1-grid0), 
                                        nao), dtype='f8')
    else:
        int_save = pcmobj.grid_int = np.empty(((grid1-grid0), nao))

    save_idx1 = 0
    for ip0 in range(grid0, grid1, blksize):
        ip1 = min(grid1, ip0+blksize)
        save_idx0, save_idx1 = save_idx1, (save_idx1 + (ip1-ip0))
        coords = grids.coords[ip0:ip1]
        non0 = non0tab[ip0//BLKSIZE:]
        int_save[save_idx0:save_idx1] = ni.eval_ao(pcmobj.mol, coords, deriv=deriv, 
                                                   non0tab=non0, out=buf)
    if pcmobj.int_storage == 1:
        f_int.close()

def make_phi(pcmobj, dm, r_vdw, ui, ylm_1sph, with_nuc=True):
    '''
    Induced potential of ddCOSMO model
    Kwargs:
        with_nuc (bool): Mute the contribution of nuclear charges when
            computing the second order derivatives of energy
    The function has been parallelized
    '''
    mol = pcmobj.mol
    natm = mol.natm
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    dms = np.asarray(dm)
    is_single_dm = dms.ndim == 2

    nao = dms.shape[-1]
    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]
    diagidx = np.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm = lib.pack_tril(dms+dms.transpose(0,2,1))
    tril_dm[:,diagidx] *= .5

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    cav_coords = (atom_coords.reshape(natm,1,3)
                + np.einsum('r,gx->rgx', r_vdw, coords_1sph))

    win_v_phi, v_phi_node = get_shared((n_dm, natm, ngrid_1sph), set_zeros=True)
    atom_list = range(natm)
    #Disitribute the jobs accroding to atoms
    atom_slice = get_slice(job_list=atom_list, rank_list=range(nrank))[irank]
    if atom_slice is not None:
        if irank_shm == 0:
            v_phi = v_phi_node
        else:
            v_phi = np.zeros((n_dm, natm, ngrid_1sph))
        if with_nuc:
            for ia in atom_slice:
                # Note (-) sign is not applied to atom_charges, because (-) is explicitly
                # included in rhs and L matrix
                d_rs = atom_coords.reshape(-1,1,3) - cav_coords[ia]
                v_phi[:,ia] = np.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    else:
        v_phi = None
    extern_point_idx = ui > 0
    extern_point_idx_flat = (np.arange(ui.size).reshape(ui.shape))[extern_point_idx]
    cav_coords = cav_coords[extern_point_idx]
    job_slice = get_slice(job_size=cav_coords.shape[0], rank_list=range(nrank))[irank]
    if job_slice is not None:
        max_len = len(job_slice)
    else:
        max_len = None
    blksize = get_buff_len(mol, size_sub=nao**2, ratio=0.8, max_len=max_len, min_len=1)
    
    if job_slice is not None:
        if v_phi is None:
            v_phi = np.zeros((n_dm, natm, ngrid_1sph))
        job0, job1 = job_slice[0], job_slice[-1]+1
        cav_coords = cav_coords[job0:job1]
        v_phi_e = np.empty((n_dm, cav_coords.shape[0]))
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        if pcmobj.int_storage == 2:
            for i0, i1 in lib.prange(0, cav_coords.shape[0], blksize):
                fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
                v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s2ij', cintopt=cintopt)
                v_phi_e[:,i0:i1] = np.einsum('nx,xk->nk', tril_dm, v_nj)
        else:
            if pcmobj.int3c2e is None:
                get_sol_3c2e(pcmobj, cav_coords, blksize, cintopt)
            if pcmobj.int_storage == 1:
                f_int = h5py.File(pcmobj.int3c2e, 'r')
                int_save = f_int["int3c2e"]
            else:
                int_save = pcmobj.int3c2e
            for i0, i1 in lib.prange(0, cav_coords.shape[0], blksize):
                fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
                v_phi_e[:,i0:i1] = np.einsum('nx,xk->nk', tril_dm, int_save[:,i0:i1])
            if pcmobj.int_storage == 1:
                f_int.close()
        (v_phi.reshape(n_dm, -1))[:, extern_point_idx_flat[job0:job1]] -= v_phi_e
    if irank_shm != 0 and v_phi is not None:
        Accumulate_GA_shm(win_v_phi, v_phi_node, v_phi)
    comm.Barrier()
    Acc_and_get_GA(v_phi_node)
    comm_shm.Barrier()
    phi = -np.einsum('n,xn,jn,ijn->ijx', weights_1sph, ylm_1sph, ui, v_phi_node)
    comm_shm.Barrier()
    free_win(win_v_phi)
    if is_single_dm:
        phi = phi[0]
    return phi



def make_psi_vmat(pcmobj, dm, r_vdw, ui, ylm_1sph, cached_pol, Xvec, L,
                  with_nuc=True):
    '''
    The first order derivative of E_ddCOSMO wrt density matrix
    Kwargs:
        with_nuc (bool): Mute the contribution of nuclear charges when
            computing the second order derivatives of energy.
    '''
    def block_loop(self, mol, grids, grid0, grid1, nao=None, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = mol.nao
        ngrids = grids.coords.shape[0]
        ngrids_rank = grid1 - grid0
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nao*8*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids_rank, BLKSIZE*1200))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = np.ones(((ngrids_rank+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                 dtype=np.uint8)
        if buf is None:
            buf = np.empty((comp,blksize,nao))
        #for ip0 in range(0, ngrids, blksize):
        if pcmobj.int_storage != 2:
            if pcmobj.grid_int is None:
                get_grid_int(pcmobj, ni, grids, deriv, grid0, grid1, blksize, non0tab, buf)
        int_idx1 = 0
        for ip0 in range(grid0, grid1, blksize):
            ip1 = min(grid1, ip0+blksize)
            int_idx0, int_idx1 = int_idx1, (int_idx1 + (ip1-ip0))
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            if pcmobj.int_storage == 2:
                ao = self.eval_ao(mol, coords, deriv=deriv, non0tab=non0, out=buf)
            else:
                if pcmobj.int_storage == 1:
                    with h5py.File(pcmobj.grid_int, 'r') as f_int:
                        a0 = f_int["grid_int"][int_idx0:int_idx1]
                else:
                    ao = pcmobj.grid_int[int_idx0:int_idx1]
            yield ao, non0, weight, coords

            
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    dms = np.asarray(dm)
    is_single_dm = dms.ndim == 2
    ni = numint.NumInt()
    make_rho, n_dm, nao = ni._gen_rho_evaluator(mol, dms)
    grids = pcmobj.grids
    ngrids = grids.coords.shape[0]
    grid_idx = [None]*natm
    idx0 = 0
    for ia in range(natm):
        idx1 = idx0 + len(cached_pol[mol.atom_symbol(ia)][1])
        grid_idx[ia] = [idx0, idx1]
        idx0 = idx1

    atom_slice = get_slice(range(nrank), job_size=natm)[irank]
    win_weignts, scaled_weights = get_shared((n_dm, ngrids), set_zeros=True)
    if atom_slice is not None:
        t0 = create_timer()
        Xvec = Xvec.reshape(n_dm, natm, nlm)
        atm0 = atom_slice[0]

        i1 = grid_idx[atm0][0]
        for ia in atom_slice:
            fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
            fac_pol = _vstack_factor_fak_pol(fak_pol, lmax)
            i0, i1 = i1, i1 + fac_pol.shape[1]
            scaled_weights[:,i0:i1] = np.dot(Xvec[:,ia], fac_pol)#np.einsum('mn,im->in', fac_pol, Xvec[:,ia])
        #scaled_weights = np.empty((n_dm, grids.weights.size))
        #if irank==0: print_time([['step 1', np.asarray([logger.process_clock(), time.time()])-t0]])
    comm_shm.Barrier()
    if irank_shm == 0:
        scaled_weights *= grids.weights
    comm.Barrier()
    Acc_and_get_GA(scaled_weights)
    comm_shm.Barrier()
    grid0_list = np.arange(0, ngrids, BLKSIZE)
    grid0_slice = get_slice(range(nrank), job_list=grid0_list)[irank]
    max_memory = get_mem_spare(mol, ratio=0.7)
    win_den, dennode = get_shared((n_dm, ngrids), set_zeros=True)
    if grid0_slice is not None:
        dms = dms.reshape(n_dm,nao,nao)
        
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        vmat = np.zeros((n_dm, nao, nao))
        if irank_shm == 0:
            den = dennode
        else:
            den = np.zeros((n_dm, ngrids))
        aow = None
        grid0, grid1 = grid0_slice[0], min(grid0_slice[-1]+BLKSIZE, ngrids)
        p1 = grid0
        for ao, mask, weight, coords \
                in block_loop(ni, mol, grids, grid0, grid1, nao, 0, max_memory):
            p0, p1 = p1, p1 + weight.size
            
            for i in range(n_dm):
                den[i,p0:p1] = make_rho(i, ao, mask, 'LDA')
                aow = numint._scale_ao(ao, scaled_weights[i,p0:p1], out=aow)
                vmat[i] -= numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
        if irank_shm != 0:
            Accumulate_GA_shm(win_den, dennode, den)
        ao = aow = None
        
    else:
        vmat = None
    comm_shm.Barrier()
    free_win(win_weignts); scaled_weights = None
    if irank_shm == 0:
        dennode *= grids.weights
    comm.Barrier()
    Acc_and_get_GA(dennode)
    #Acc_and_get_GA(vmat_node)
    comm_shm.Barrier()

    #nelec_leak = 0
    #psi = np.zeros((n_dm, natm, nlm))
    win_psi, psi_node = get_shared((n_dm, natm, nlm), set_zeros=True)
    if atom_slice is not None:
        atm0, atm1 = atom_slice[0], atom_slice[-1]+1
        i1 = grid_idx[atm0][0]
        t0 = create_timer()
        for ia in atom_slice:
            fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
            fac_pol = _vstack_factor_fak_pol(fak_pol, lmax)
            i0, i1 = i1, i1 + fac_pol.shape[1]
            #nelec_leak += dennode[:,i0:i1][:,leak_idx].sum(axis=1)
            psi_node[:,ia] = -np.dot(dennode[:,i0:i1], fac_pol.T)#-np.einsum('in,mn->im', den[:,i0:i1], fac_pol)
        #logger.debug(pcmobj, 'electron leaks %s', nelec_leak)
    # Contribution of nuclear charges to the total density
    # The factor np.sqrt(4*np.pi) is due to the product of 4*pi * Y_0^0
            if with_nuc:
                #for ia in range(natm):
                psi_node[:,ia,0] += np.sqrt(4*np.pi)/r_vdw[ia] * mol.atom_charge(ia)
        #if irank==0: print_time([['step 3', np.asarray([logger.process_clock(), time.time()])-t0]])
    comm.Barrier()
    Acc_and_get_GA(psi_node)
    comm_shm.Barrier()
    psi = np.empty(psi_node.shape)
    psi[:] = psi_node
    comm_shm.Barrier()
    free_win(win_psi); psi_node = None
    
    t0 = create_timer()
    # <Psi, L^{-1}g> -> Psi = SL the adjoint equation to LX = g
    L_S = np.linalg.solve(L.reshape(natm*nlm,-1).T, psi.reshape(n_dm,-1).T)
    L_S = L_S.reshape(natm,nlm,n_dm).transpose(2,0,1)
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    # JCP, 141, 184108, Eq (39)
    xi_jn = np.einsum('n,jn,xn,ijx->ijn', weights_1sph, ui, ylm_1sph, L_S)
    extern_point_idx = ui > 0
    cav_coords = (mol.atom_coords().reshape(natm,1,3)
                  + np.einsum('r,gx->rgx', r_vdw, coords_1sph))
    cav_coords = cav_coords[extern_point_idx]
    xi_jn = xi_jn[:,extern_point_idx]
    #if irank==0: print_time([['step 4', np.asarray([logger.process_clock(), time.time()])-t0]])
    
    '''max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2, 400))'''
    t0 = create_timer()
    job_slice = get_slice(job_size=cav_coords.shape[0], rank_list=range(nrank))[irank]
    if job_slice is not None:
        max_len = len(job_slice)
    else:
        max_len = None
    blksize = get_buff_len(mol, size_sub=nao**2, ratio=0.8, max_len=max_len, min_len=1)
    if job_slice is not None:
        if vmat is None:
            vmat = np.zeros((n_dm, nao, nao))
        job0, job1 = job_slice[0], job_slice[-1]+1
        cav_coords = cav_coords[job0:job1]
        xi_jn = xi_jn[:, job0:job1]
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
        vmat_tril = 0
        if pcmobj.int_storage == 2:
            for i0, i1 in lib.prange(0, cav_coords.shape[0], blksize):
                fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
                v_nj = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s2ij',
                                        cintopt=cintopt)
                #vmat_tril += np.einsum('xn,in->ix', v_nj, xi_jn[:,i0:i1])
                vmat_tril += np.dot(xi_jn[:,i0:i1], v_nj.T)
        else:
            '''if pcmobj.int3c2e is None:
                get_sol_3c2e(pcmobj, cav_coords, blksize, cintopt)'''
            if pcmobj.int_storage == 1:
                f_int = h5py.File(pcmobj.int3c2e, 'r')
                int_save = f_int["int3c2e"]
            else:
                int_save = pcmobj.int3c2e
            for i0, i1 in lib.prange(0, cav_coords.shape[0], blksize):
                fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
                vmat_tril += np.dot(xi_jn[:,i0:i1], int_save[:,i0:i1].T)
            if pcmobj.int_storage == 1:
                f_int.close()
        vmat += lib.unpack_tril(vmat_tril)
    #if irank==0: print_time([['step 5', np.asarray([logger.process_clock(), time.time()])-t0]])
    win_vmat, vmat_node = get_shared((n_dm, nao, nao), set_zeros=True)
    if vmat is not None:
        Accumulate_GA_shm(win_vmat, vmat_node, vmat)
    comm.Barrier()
    Acc_and_get_GA(vmat_node)
    comm_shm.Barrier()
    vmat[:] = vmat_node
    comm_shm.Barrier()
    for win_i in [win_den, win_vmat]:
        free_win(win_i)
    if is_single_dm:
        psi = psi[0]
        L_S = L_S[0]
        vmat = vmat[0]
    return psi, vmat, L_S

def get_veff_sol(self, dm):
    #from pyscf.solvent.ddcosmo import make_phi, make_psi_vmat
    #from pyscf.solvent.ddcosmo import make_phi
    '''A single shot solvent effects for given density matrix.
    '''
    if not self._intermediates or self.grids.coords is None:
        self.build()
    if irank == 0:
        verbose=5
    else:
        verbose=0
    log = logger.Logger(self.stdout, verbose)
    tt = (logger.process_clock(), time.time())
    mol = self.mol
    r_vdw      = self._intermediates['r_vdw'     ]
    ylm_1sph   = self._intermediates['ylm_1sph'  ]
    ui         = self._intermediates['ui'        ]
    Lmat       = self._intermediates['Lmat'      ]
    cached_pol = self._intermediates['cached_pol']

    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        # spin-traced DM for UHF or ROHF
        dm = dm[0] + dm[1]
    t0 = create_timer()
    phi = make_phi(self, dm, r_vdw, ui, ylm_1sph)
    print_time(['make_phi', get_elapsed_time(t0)], log)
    Xvec = np.linalg.solve(Lmat, phi.ravel()).reshape(mol.natm,-1)
    t0 = create_timer()
    psi, vmat = make_psi_vmat(self, dm, r_vdw, ui, ylm_1sph,
                            cached_pol, Xvec, Lmat)[:2]
    print_time(['make_psi_vmat', get_elapsed_time(t0)], log)
    dielectric = self.eps
    if dielectric > 0:
        f_epsilon = (dielectric-1.)/dielectric
    else:
        f_epsilon = 1
    epcm = .5 * f_epsilon * np.einsum('jx,jx', psi, Xvec)
    vpcm = .5 * f_epsilon * vmat
    print_time(['Solvent energy and potential', get_elapsed_time(tt)], log)
    return epcm, vpcm

###############################################################################################################
#Gradient
def atoms_with_vdw_overlap(atm_id, atom_coords, r_vdw):
    atm_dist = atom_coords - atom_coords[atm_id]
    atm_dist = np.einsum('pi,pi->p', atm_dist, atm_dist)
    atm_dist[atm_id] = 1e200
    vdw_sum = r_vdw + r_vdw[atm_id]
    atoms_nearby = np.where(atm_dist < vdw_sum**2)[0]
    return atoms_nearby

def regularize_xt1(t, eta):
    xt = np.zeros_like(t)
    # no response if grids are inside the cavity
    # inner = t <= 1-eta
    # xt[inner] = 0
    on_shell = (1-eta < t) & (t < 1)
    ti = t[on_shell]
    xt[on_shell] = -30./eta**5 * (1-ti)**2 * (1-eta-ti)**2
    return xt

def multipoles1(r, lmax, reorder_dipole=True):
    ngrid = r.shape[0]
    xs = np.ones((lmax+1,ngrid))
    ys = np.ones((lmax+1,ngrid))
    zs = np.ones((lmax+1,ngrid))
    for i in range(1,lmax+1):
        xs[i] = xs[i-1] * r[:,0]
        ys[i] = ys[i-1] * r[:,1]
        zs[i] = zs[i-1] * r[:,2]
    ylms = []
    for l in range(lmax+1):
        nd = (l+1)*(l+2)//2
        c = np.empty((nd,3,ngrid))
        k = 0
        for lx in reversed(range(0, l+1)):
            for ly in reversed(range(0, l-lx+1)):
                lz = l - lx - ly
                c[k,0] = lx * xs[lx-1] * ys[ly] * zs[lz]
                c[k,1] = ly * xs[lx] * ys[ly-1] * zs[lz]
                c[k,2] = lz * xs[lx] * ys[ly] * zs[lz-1]
                k += 1
        ylm = gto.cart2sph(l, c.reshape(nd,3*ngrid).T)
        ylm = ylm.reshape(3,ngrid,l*2+1).transpose(0,2,1)
        ylms.append(ylm)

# when call libcint, p functions are ordered as px,py,pz
# reorder px,py,pz to p(-1),p(0),p(1)
    if (not reorder_dipole) and lmax >= 1:
        ylms[1] = ylms[1][:,[1,2,0]]
    return ylms

def make_fi(pcmobj, r_vdw):
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    mol = pcmobj.mol
    eta = pcmobj.eta
    natm = mol.natm
    atom_coords = mol.atom_coords()
    ngrid_1sph = coords_1sph.shape[0]
    win_fi, fi_node = get_shared((natm,ngrid_1sph), set_zeros=True)
    job_list = []
    for ia in range(natm):
        for ja in ddcosmo.atoms_with_vdw_overlap(ia, atom_coords, r_vdw):
            job_list.append([ia, ja])
    job_slice = get_slice(range(nrank), job_list=job_list)[irank]
    if job_slice is not None:
        fi = np.zeros(fi_node.shape)
        for ia, ja in job_slice:
            v = r_vdw[ia]*coords_1sph + atom_coords[ia] - atom_coords[ja]
            rv = lib.norm(v, axis=1)
            t = rv / r_vdw[ja]
            xt = pcmobj.regularize_xt(t, eta, r_vdw[ja])
            fi[ia] += xt
        Accumulate_GA_shm(win_fi, fi_node, fi)
    comm.Barrier()
    Acc_and_get_GA(fi_node)
    if irank_shm == 0:
        fi_node[fi_node < 1e-20] = 0
    comm_shm.Barrier()
    fi[:] = fi_node
    comm_shm.Barrier()
    free_win(win_fi)
    return fi

def make_fi1(pcmobj, r_vdw):
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    mol = pcmobj.mol
    eta = pcmobj.eta
    natm = mol.natm
    atom_coords = mol.atom_coords()
    ngrid_1sph = coords_1sph.shape[0]
    win_fi1, fi1_node = get_shared((natm,3,natm,ngrid_1sph), set_zeros=True)
    #fi1 = np.zeros((natm,3,natm,ngrid_1sph))
    atom_slice = get_slice(range(nrank), job_list=range(natm))[irank]
    if atom_slice is not None:
        for ia in atom_slice:
        #for ia, ja in job_slice:
            for ja in atoms_with_vdw_overlap(ia, atom_coords, r_vdw):
                v = r_vdw[ia]*coords_1sph + atom_coords[ia] - atom_coords[ja]
                rv = lib.norm(v, axis=1)
                t = rv / r_vdw[ja]
                xt1 = regularize_xt1(t, eta*r_vdw[ja])
                s_ij = v.T / rv
                xt1 = 1./r_vdw[ja] * xt1 * s_ij
                fi1_node[ia,:,ia] += xt1
                fi1_node[ja,:,ia] -= xt1
    fi = ddcosmo.make_fi(pcmobj, r_vdw)
    #fi1[:,:,fi<1e-20] = 0
    comm.Barrier()
    Acc_and_get_GA(fi1_node)
    if irank_shm == 0:
        fi1_node[:,:,fi<1e-20] = 0
    comm_shm.Barrier()
    return win_fi1, fi1_node

def make_L(pcmobj, r_vdw, ylm_1sph, fi):
    # See JCTC, 9, 3637, Eq (18)
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    eta = pcmobj.eta
    nlm = (lmax+1)**2

    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = weights_1sph.size
    atom_coords = mol.atom_coords()
    ylm_1sph = ylm_1sph.reshape(nlm,ngrid_1sph)

# JCP, 141, 184108 Eq (9), (12) is incorrect
# L_diag = <lm|(1/|s-s'|)|l'm'>
# Using Laplace expansion for electrostatic potential 1/r
# L_diag = 4pi/(2l+1)/|s| <lm|l'm'>
    win_lmat, lmat_node = get_shared((natm,nlm,natm,nlm))
    if irank == 0:
        L_diag = np.zeros((natm,nlm))
        p1 = 0
        for l in range(lmax+1):
            p0, p1 = p1, p1 + (l*2+1)
            L_diag[:,p0:p1] = 4*np.pi/(l*2+1)
        L_diag *= 1./r_vdw.reshape(-1,1)
        lmat_node[:] = np.diag(L_diag.ravel()).reshape(natm,nlm,natm,nlm)
    comm_shm.Barrier()

    job_list = []
    for ja in range(natm):
        for ka in atoms_with_vdw_overlap(ja, atom_coords, r_vdw):
            job_list.append([ja, ka])
    job_slice = get_slice(range(nrank), job_list=job_list)[irank]
    #for ja in range(natm):
    if job_slice is not None:
        for ja, ka in job_slice:
        # scale the weight, precontract d_nj and w_n
        # see JCTC 9, 3637, Eq (16) - (18)
        # Note all values are scaled by 1/r_vdw to make the formulas
        # consistent to Psi in JCP, 141, 184108
            part_weights = weights_1sph.copy()
            part_weights[fi[ja]>1] /= fi[ja,fi[ja]>1]
        #for ka in atoms_with_vdw_overlap(ja, atom_coords, r_vdw):
            vjk = r_vdw[ja] * coords_1sph + atom_coords[ja] - atom_coords[ka]
            tjk = lib.norm(vjk, axis=1) / r_vdw[ka]
            wjk = pcmobj.regularize_xt(tjk, eta, r_vdw[ka])
            wjk *= part_weights
            pol = sph.multipoles(vjk, lmax)
            p1 = 0
            for l in range(lmax+1):
                fac = 4*np.pi/(l*2+1) / r_vdw[ka]**(l+1)
                p0, p1 = p1, p1 + (l*2+1)
                a = np.einsum('xn,n,mn->xm', ylm_1sph, wjk, pol[l])
                lmat_node[ja,:,ka,p0:p1] += -fac * a
    comm.Barrier()
    Acc_and_get_GA(lmat_node)
    comm_shm.Barrier()
    return win_lmat, lmat_node

def make_L1(pcmobj, r_vdw, ylm_1sph, fi):
    # See JCTC, 9, 3637, Eq (18)
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    eta = pcmobj.eta
    nlm = (lmax+1)**2

    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = weights_1sph.size
    atom_coords = mol.atom_coords()
    ylm_1sph = ylm_1sph.reshape(nlm,ngrid_1sph)

    
    win_fi1, fi1 = make_fi1(pcmobj, pcmobj.get_atomic_radii())
    #fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
    #Lmat = np.zeros((natm,3,natm,nlm,natm,nlm))
    win_lmat, lmat_node = get_shared((natm,3,natm,nlm,natm,nlm), set_zeros=True)
    job_list = []
    for ja in range(natm):
        for ka in atoms_with_vdw_overlap(ja, atom_coords, r_vdw):
            job_list.append([ja, ka])
    job_slice = get_slice(range(nrank), job_list=job_list)[irank]
    #for ja in range(natm):
    if job_slice is not None:
        ja_check = [False]*natm
        for ja, ka in job_slice:
    #for ja in range(natm):
            if ja_check[ja] == False:
                part_weights = weights_1sph.copy()
                part_weights[fi[ja]>1] /= fi[ja,fi[ja]>1]

                part_weights1 = np.zeros((natm,3,ngrid_1sph))
                tmp = part_weights[fi[ja]>1] / fi[ja,fi[ja]>1]
                part_weights1[:,:,fi[ja]>1] = -tmp * fi1[:,:,ja,fi[ja]>1]
            else:
                ja_check[ja] = True
        #for ka in ddcosmo.atoms_with_vdw_overlap(ja, atom_coords, r_vdw):
            vjk = r_vdw[ja] * coords_1sph + atom_coords[ja] - atom_coords[ka]
            rv = lib.norm(vjk, axis=1)
            tjk = rv / r_vdw[ka]
            wjk0 = pcmobj.regularize_xt(tjk, eta, r_vdw[ka])
            wjk1 = regularize_xt1(tjk, eta*r_vdw[ka])
            sjk = vjk.T / rv
            wjk1 = 1./r_vdw[ka] * wjk1 * sjk

            wjk01 = wjk0 * part_weights1
            wjk0 *= part_weights
            wjk1 *= part_weights

            pol0 = sph.multipoles(vjk, lmax)
            pol1 = multipoles1(vjk, lmax)
            p1 = 0
            for l in range(lmax+1):
                fac = 4*np.pi/(l*2+1) / r_vdw[ka]**(l+1)
                p0, p1 = p1, p1 + (l*2+1)
                a = np.einsum('xn,zn,mn->zxm', ylm_1sph, wjk1, pol0[l])
                a+= np.einsum('xn,n,zmn->zxm', ylm_1sph, wjk0, pol1[l])
                lmat_node[ja,:,ja,:,ka,p0:p1] += -fac * a
                lmat_node[ka,:,ja,:,ka,p0:p1] -= -fac * a
                a = np.einsum('xn,azn,mn->azxm', ylm_1sph, wjk01, pol0[l])
                lmat_node[:,:,ja,:,ka,p0:p1] += -fac * a
    comm.Barrier()
    Acc_and_get_GA(lmat_node)
    comm_shm.Barrier()
    free_win(win_fi1)
    return win_lmat, lmat_node



def make_phi1(pcmobj, dm, r_vdw, ui, ylm_1sph):
    mol = pcmobj.mol
    natm = mol.natm

    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    tril_dm = lib.pack_tril(dm+dm.T)
    nao = dm.shape[0]
    diagidx = np.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm[diagidx] *= .5

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()

    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    #extern_point_idx = ui > 0

    win_fi1, fi1 = make_fi1(pcmobj, pcmobj.get_atomic_radii())
    if irank_shm == 0:
        fi1[:,:,ui==0] = 0
        fi1 *= -1
    comm_shm.Barrier()
    ui1 = fi1

    ngrid_1sph = weights_1sph.size
    win_vphi0, vphi0_node = get_shared((natm, ngrid_1sph), set_zeros=True)
    #v_phi0 = np.empty((natm,ngrid_1sph))
    atom_slice = get_slice(job_list=range(natm), rank_list=range(nrank))[irank]
    #for ia in range(natm):
    if atom_slice is not None:
        for ia in atom_slice:
            cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
            d_rs = atom_coords.reshape(-1,1,3) - cav_coords
            vphi0_node[ia] = np.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    comm.Barrier()
    Acc_and_get_GA(vphi0_node)
    comm_shm.Barrier()

    win_phi1, phi1_node = get_shared((natm, 3, natm, ylm_1sph.shape[0]))
    if irank_shm == 0:
        phi1_node[:] = -np.einsum('n,ln,azjn,jn->azjl', weights_1sph, ylm_1sph, ui1, vphi0_node)
    comm_shm.Barrier()
    #phi1 = -np.einsum('n,ln,azjn,jn->azjl', weights_1sph, ylm_1sph, ui1, vphi0_node)

    atom_slice = get_slice(range(nrank), job_list=range(natm))[irank]
    if atom_slice is not None:
        for ia in atom_slice:
            cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
            for ja in range(natm):
                rs = atom_coords[ja] - cav_coords
                d_rs = lib.norm(rs, axis=1)
                v_phi = atom_charges[ja] * np.einsum('px,p->px', rs, 1./d_rs**3)
                tmp = np.einsum('n,ln,n,nx->xl', weights_1sph, ylm_1sph, ui[ia], v_phi)
                phi1_node[ja,:,ia] += tmp  # response of the other atoms
                phi1_node[ia,:,ia] -= tmp  # response of cavity grids
    comm_shm.Barrier()
    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    aoslices = mol.aoslice_by_atom()
    #if job_slice is not None:
    
    #for ia in range(natm):
    if atom_slice is not None:
        for ia in atom_slice:
            cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
            #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
            fakemol = gto.fakemol_for_charges(cav_coords)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1')
            v_phi = np.einsum('ij,ijk->k', dm, v_nj)
            phi1_node[:,:,ia] += np.einsum('n,ln,azn,n->azl', weights_1sph, ylm_1sph, ui1[:,:,ia], v_phi)

            v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3, aosym='s1')
            phi1_e2_nj  = np.einsum('ij,xijr->xr', dm, v_e1_nj)
            phi1_e2_nj += np.einsum('ji,xijr->xr', dm, v_e1_nj)
            phi1_node[ia,:,ia] += np.einsum('n,ln,n,xn->xl', weights_1sph, ylm_1sph, ui[ia], phi1_e2_nj)

            for ja in range(natm):
                shl0, shl1, p0, p1 = aoslices[ja]
                phi1_nj  = np.einsum('ij,xijr->xr', dm[p0:p1  ], v_e1_nj[:,p0:p1])
                phi1_nj += np.einsum('ji,xijr->xr', dm[:,p0:p1], v_e1_nj[:,p0:p1])
                phi1_node[ja,:,ia] -= np.einsum('n,ln,n,xn->xl', weights_1sph, ylm_1sph, ui[ia], phi1_nj)
    comm.Barrier()
    Acc_and_get_GA(phi1_node)
    comm_shm.Barrier()
    free_win(win_fi1)
    return win_phi1, phi1_node



def make_e_psi1(pcmobj, dm, r_vdw, ui, ylm_1sph, cached_pol, Xvec, L):
    def grids_response_cc(atom_list, grids):
        def _radii_adjust(mol, atomic_radii):
            charges = mol.atom_charges()
            if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
                rad = np.sqrt(atomic_radii[charges]) + 1e-200
            elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
                rad = atomic_radii[charges] + 1e-200
            else:
                fadjust = lambda i, j, g: g
                gadjust = lambda *args: 1
                return fadjust, gadjust

            rr = rad.reshape(-1,1) * (1./rad)
            a = .25 * (rr.T - rr)
            a[a<-.5] = -.5
            a[a>0.5] = 0.5

            def fadjust(i, j, g):
                return g + a[i,j]*(1-g**2)

            #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
            def gadjust(i, j, g):
                return 1 - 2*a[i,j]*g
            return fadjust, gadjust
        def gen_grid_partition(coords, atom_id):
            ngrids = coords.shape[0]
            grid_dist = []
            grid_norm_vec = []
            for ia in range(mol.natm):
                v = (atm_coords[ia] - coords).T
                normv = np.linalg.norm(v,axis=0) + 1e-200
                v /= normv
                grid_dist.append(normv)
                grid_norm_vec.append(v)

            def get_du(ia, ib):  # JCP 98, 5612 (1993); (B10)
                uab = atm_coords[ia] - atm_coords[ib]
                duab = 1./atm_dist[ia,ib] * grid_norm_vec[ia]
                duab-= uab[:,None]/atm_dist[ia,ib]**3 * (grid_dist[ia]-grid_dist[ib])
                return duab

            pbecke = np.ones((mol.natm,ngrids))
            dpbecke = np.zeros((mol.natm,mol.natm,3,ngrids))
            for ia in range(mol.natm):
                for ib in range(ia):
                    g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                    p0 = fadjust(ia, ib, g)
                    p1 = (3 - p0**2) * p0 * .5
                    p2 = (3 - p1**2) * p1 * .5
                    p3 = (3 - p2**2) * p2 * .5
                    t_uab = 27./16 * (1-p2**2) * (1-p1**2) * (1-p0**2) * gadjust(ia, ib, g)

                    s_uab = .5 * (1 - p3 + 1e-200)
                    s_uba = .5 * (1 + p3 + 1e-200)
                    pbecke[ia] *= s_uab
                    pbecke[ib] *= s_uba
                    pt_uab =-t_uab / s_uab
                    pt_uba = t_uab / s_uba

    # * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
    #   How to remove this error?
                    duab = get_du(ia, ib)
                    duba = get_du(ib, ia)
                    if ia == atom_id:
                        dpbecke[ia,ia] += pt_uab * duba
                        dpbecke[ia,ib] += pt_uba * duba
                    else:
                        dpbecke[ia,ia] += pt_uab * duab
                        dpbecke[ia,ib] += pt_uba * duab

                    if ib == atom_id:
                        dpbecke[ib,ib] -= pt_uba * duab
                        dpbecke[ib,ia] -= pt_uab * duab
                    else:
                        dpbecke[ib,ib] -= pt_uba * duba
                        dpbecke[ib,ia] -= pt_uab * duba

    # * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                    if ia != atom_id and ib != atom_id:
                        ua_ub = grid_norm_vec[ia] - grid_norm_vec[ib]
                        ua_ub /= atm_dist[ia,ib]
                        dpbecke[atom_id,ia] -= pt_uab * ua_ub
                        dpbecke[atom_id,ib] -= pt_uba * ua_ub

            for ia in range(mol.natm):
                dpbecke[:,ia] *= pbecke[ia]

            return pbecke, dpbecke

        '''natm = mol.natm
        for ia in range(natm):'''
        mol = grids.mol
        atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                                grids.radi_method,
                                                grids.level, grids.prune)
        atm_coords = np.asarray(mol.atom_coords() , order='C')
        atm_dist = gto.inter_distance(mol, atm_coords)
        fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)
        for ia in atom_list:
            coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
            coords = coords + atm_coords[ia]
            pbecke, dpbecke = gen_grid_partition(coords, ia)
            z = 1./pbecke.sum(axis=0)
            w1 = dpbecke[:,ia] * z
            w1 -= pbecke[ia] * z**2 * dpbecke.sum(axis=1)
            w1 *= vol
            w0 = vol * pbecke[ia] * z
            yield ia, coords, w0, w1
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    grids = pcmobj.grids
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    ni = numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm)
    grid_idx = [None]*natm
    idx0 = 0
    for ia in range(natm):
        idx1 = idx0 + len(cached_pol[mol.atom_symbol(ia)][1])
        grid_idx[ia] = [idx0, idx1]
        idx0 = idx1

    win_vtmp, vtmp_node = get_shared((3, nao, nao), set_zeros=True)
    win_psi1, psi1_node = get_shared((natm,3), set_zeros=True)
    atom_slice = get_slice(range(nrank), job_size=natm)[irank]
    if atom_slice is not None:
        #den = np.empty((4,grids.weights.size))
        ao_loc = mol.ao_loc_nr()
        if irank_shm == 0:
            vmat = vtmp_node
        else:
            vmat = np.zeros((3,nao,nao))
        psi1 = np.zeros((natm,3))
        #i1 = 0
        for ia, coords, weight, weight1 in grids_response_cc(atom_slice, grids):
            #i0, i1 = i1, i1 + weight.size
            i0, i1 = grid_idx[ia]
            ao = ni.eval_ao(mol, coords, deriv=1)
            mask = gen_grid.make_mask(mol, coords)
            #den[:,i0:i1] = make_rho(0, ao, mask, 'GGA')
            den = make_rho(0, ao, mask, 'GGA')

            fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
            eta_nj = 0
            p1 = 0
            for l in range(lmax+1):
                fac = 4*np.pi/(l*2+1)
                p0, p1 = p1, p1 + (l*2+1)
                eta_nj += fac * np.einsum('mn,m->n', fak_pol[l], Xvec[ia,p0:p1])
            '''psi1 -= np.einsum('n,n,zxn->zx', den[0,i0:i1], eta_nj, weight1)
            psi1[ia] -= np.einsum('xn,n,n->x', den[1:4,i0:i1], eta_nj, weight)'''
            psi1 -= np.einsum('n,n,zxn->zx', den[0], eta_nj, weight1)
            psi1[ia] -= np.einsum('xn,n,n->x', den[1:4], eta_nj, weight)

            vtmp = np.zeros((3,nao,nao))
            aow = np.einsum('pi,p->pi', ao[0], weight*eta_nj)
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat += vtmp
        if irank_shm != 0:
            Accumulate_GA_shm(win_vtmp, vtmp_node, vmat)
    comm.Barrier()
    Acc_and_get_GA(vtmp_node)
    comm_shm.Barrier()
    if atom_slice is not None:
        aoslices = mol.aoslice_by_atom()
        #for ia in range(natm):
        for ia in atom_slice:
            shl0, shl1, p0, p1 = aoslices[ia]
            psi1[ia] += np.einsum('xij,ij->x', vtmp_node[:,p0:p1], dm[p0:p1]) * 2
        Accumulate_GA_shm(win_psi1, psi1_node, psi1)
    else:
        psi1 = np.empty((psi1_node.shape))
    Acc_and_get_GA(vtmp_node)
    comm_shm.Barrier()
    psi1[:] = psi1_node
    comm_shm.Barrier()
    for win_i in [win_vtmp, win_psi1]:
        free_win(win_i)
    return psi1

def get_grad_sol(pcmobj, dm, log):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    if pcmobj.grids.coords is None:
        pcmobj.grids.build(with_non0tab=True)

    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        # UHF density matrix
        dm = dm[0] + dm[1]
    t0 = create_timer()
    r_vdw = ddcosmo.get_atomic_radii(pcmobj)
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ylm_1sph = np.vstack(sph.real_sph_vec(coords_1sph, lmax, True))
    fi = ddcosmo.make_fi(pcmobj, r_vdw)
    ui = 1 - fi
    ui[ui<0] = 0
    cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)
    print_time(["prepare", get_elapsed_time(t0)], log)

    t0 = create_timer()
    nlm = (lmax+1)**2
    win_l0, L0 = make_L(pcmobj, r_vdw, ylm_1sph, fi)
    L0 = L0.reshape(natm*nlm,-1)
    win_l1, L1 = make_L1(pcmobj, r_vdw, ylm_1sph, fi)
    print_time(["L term", get_elapsed_time(t0)], log)

    t0 = create_timer()
    phi0 = make_phi(pcmobj, dm, r_vdw, ui, ylm_1sph) 
    print_time(["phi0 term", get_elapsed_time(t0)], log)
    t0 = create_timer()
    win_phi1, phi1 = make_phi1(pcmobj, dm, r_vdw, ui, ylm_1sph)
    print_time(["phi1 term", get_elapsed_time(t0)], log)

    t0 = create_timer()
    L0_X = np.linalg.solve(L0, phi0.ravel()).reshape(natm,-1)
    psi0, vmat, L0_S = make_psi_vmat(pcmobj, dm, r_vdw, ui, ylm_1sph,
                                  cached_pol, L0_X, L0)

    e_psi1 = make_e_psi1(pcmobj, dm, r_vdw, ui, ylm_1sph,
                         cached_pol, L0_X, L0)
    print_time(["psi term", get_elapsed_time(t0)], log)
    
    t0 = create_timer()
    dielectric = pcmobj.eps
    if dielectric > 0:
        f_epsilon = (dielectric-1.)/dielectric
    else:
        f_epsilon = 1
    comm.Barrier()
    de = .5 * f_epsilon * e_psi1
    de+= .5 * f_epsilon * np.einsum('jx,azjx->az', L0_S, phi1)
    de-= .5 * f_epsilon * np.einsum('aziljm,il,jm->az', L1, L0_S, L0_X)
    print_time(["cal grad", get_elapsed_time(t0)], log)
    comm_shm.Barrier()
    for win_i in [win_l0, win_l1, win_phi1]:
        free_win(win_i)
    return de

