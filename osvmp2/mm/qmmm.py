from copy import copy
from dataclasses import replace
import sys
import os
from sys import stdout
import itertools
from operator import itemgetter
import numpy as np
from pyscf import gto, df, lib, qmmm, scf
from simtk import openmm
import parmed as pmd
import osvmp2
from osvmp2.__config__ import inputs
from osvmp2.mpi_addons import *
from osvmp2.mm.xyz2pdb import xyz_to_pdb
from osvmp2.osvutil import *
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm= comm_shm.rank # rank index in sub-comm
nnode = nrank//comm_shm.size

def strl2list(strl, dtype='i'):
    if dtype == 'i':
        type_format = int
    else:
        type_format = float
    return [type_format(i) for i in strl.replace("[","").replace("]","").split(',')]

def region_qmmm(coords, qm_region=None, qm_atoms=None, qm_center=None, nwater_qm=20):
    natm = len(coords)
    #Get the C-O and O-O distances
    ci_oidx = 0
    ci_cidx = 2
    co_distlist = []
    solvent_region = np.delete(range(natm), qm_atoms)
    owater_list = [int(solvent_region[io_idx]) for io_idx in np.arange(len(solvent_region),step=3)]
    for oi in owater_list:
        co_distlist.append([np.linalg.norm(coords[ci_cidx][1]-coords[oi][1]), oi])
    co_distlist.sort()
    min_dist_co = co_distlist[0][0]
    o_close2ci = co_distlist[0][1]

    oh_distlist = []
    for oi in [oi for dco, oi in co_distlist[:nwater_qm*2]]:
        for ih_w in [oi+1, oi+2]:
            oh_distlist.append([np.linalg.norm(coords[ci_oidx][1]-coords[ih_w][1]), ih_w])
    oh_distlist.sort()
    min_dist_oh = oh_distlist[0][0]    
    h_close2ci = oh_distlist[0][1]
    
    if qm_region is None:
        '''
        Adaptive selection of QM atoms. It works for water solvent only
        '''
        reaction_end = False
        if min_dist_co < 1.6: #Step 1 is finished
            qm_region = list(copy.deepcopy(qm_atoms))
            o_close = co_distlist[0][1]
            dist_oh_close = []
            for hi in [o_close+1, o_close+2]:
                dist_oh_close.append([np.linalg.norm(coords[o_close][1]-coords[hi][1]), hi])
            dist_oh_close.sort()
            if dist_oh_close[1][0] < 1.3:
                qm_region.extend([o_close, o_close+1, o_close+2])
                nwater_qm -= 1

                #PT has not happened
                qm_center = o_close
                coord_center = (coords[qm_center][1] + coords[ci_oidx][1]) / 2
                oo_distlist = []
                for io in [oi for di, oi in co_distlist[1:nwater_qm*2]]:
                    oo_distlist.append([np.linalg.norm(coord_center-coords[io][1]), io])
                oo_distlist.sort()
                for idist, io in oo_distlist[:nwater_qm]:
                    qm_region.extend([io, io+1, io+2])
            else:
                olist_water = [oi for di, oi in co_distlist[:nwater_qm*3]]
                #hlist_water = [dist_oh_close[1][1]]
                hlist_water = []
                for oi in olist_water:
                    hlist_water.extend([oi+1, oi+2])
                
                oidx_list = [None] * natm
                hidx_list = [None] * natm
                dist_oh_array = np.zeros((len(olist_water), len(hlist_water)))
                dist_oh_o = [None]*natm
                dist_oh_h = [None]*natm
                for hi in hlist_water:
                    dist_oh_h[hi] = []
                for idx_o, oi in enumerate(olist_water):
                    oidx_list[oi] = idx_o
                    dlist_i = []
                    for idx_h, hi in enumerate(hlist_water):
                        hidx_list[hi] = idx_h
                        dist_i = np.linalg.norm(coords[oi][1]-coords[hi][1])
                        dist_oh_array[idx_o, idx_h] = dist_i
                        dlist_i.append([dist_i, hi])
                        dist_oh_h[hi].append([dist_i, oi])
                    dlist_i.sort()
                    dist_oh_o[oi] = dlist_i
                for hi in hlist_water:
                    dist_oh_h[hi].sort()

                #Check the localtion of the transferred proton
                dist_twoh = []
                dist_threeh = []
                for oi in olist_water:
                    dist_twoh.append(dist_oh_o[oi][1][0])
                    dist_threeh.append(dist_oh_o[oi][2][0])
                
                dist_o_oh = max(dist_twoh)
                dist_o_h3o = min(dist_threeh)

                #Check if the reaction is finished
                if min_dist_oh < 1.2:
                    if dist_o_h3o > 1.5 and min_dist_oh < 1.1:
                        #Reaction is finished
                        reaction_end = True

                #Trace the route of proton transfer
                o_pre = o_close2ci
                dist_oh = dist_oh_close
                while dist_oh[1][0] > 1.4:
                    h_left = dist_oh[1][1]
                    if h_left == h_close2ci:
                        qm_center = ci_oidx
                        break
                    else:
                        doh, o_recv = dist_oh_h[h_left][0]
                        if doh > 1.2:
                            break
                        else:
                            if o_recv == o_pre:
                                break
                            else:
                                dist_oh = [[dist_oh_array[oidx_list[o_recv], hidx_list[hi]], hi] for hi in [o_recv+1, o_recv+2]]
                                dist_oh.sort()
                                o_pre = qm_center = o_recv

                check_sel_o = [False]*natm
                #include all oxygen atoms involved in the PT
                for idx_o, oi in enumerate(olist_water):
                    hclose_list = [oi+1, oi+2]
                    doh_close = []
                    for hi in hclose_list:
                        doh_close.append(dist_oh_array[idx_o, hidx_list[hi]])
                    if max(doh_close) > 1.3:
                        qm_region.append(oi)
                        qm_region.extend(hclose_list)
                        check_sel_o[oi] = True
                        nwater_qm -= 1

                coord_center = (coords[qm_center][1] + coords[ci_oidx][1]) / 2
                oo_distlist = []
                for oi in olist_water:
                    if check_sel_o[oi] == False:
                        oo_distlist.append([np.linalg.norm(coord_center-coords[oi][1]), oi])
                oo_distlist.sort()
                for di, oi in oo_distlist[:nwater_qm]:
                    qm_region.append(oi)
                    qm_region.extend([oi+1, oi+2])
                
        else:
            if qm_center == ci_cidx:
                dist_list = co_distlist#[[co_distlist[oidx], iow] for oidx, iow in enumerate(owater_list)]
            else:
                dist_list = []
                for io_w in owater_list:
                    dist_i = np.linalg.norm(coords[qm_center][1]-coords[io_w][1])
                    dist_list.append([dist_i, io_w])
                dist_list.sort()
            qm_region = list(copy.deepcopy(qm_atoms))
            for dist_i, io in dist_list[:nwater_qm]:
                qm_region.extend([io, io+1, io+2])

        
        #Check if the reaction has been finished
        if os.path.isfile("ci_co_oh_dist.log"):
            data = np.genfromtxt("ci_co_oh_dist.log")
            if data.size > 100:
                dco_md = max(data[:,0][-100:])
                doh_md = max(data[:,1][-100:])
                if (dco_md < 1.6) and (doh_md < 1.1) and (reaction_end):
                    if irank == 0:
                        with open("CONGRATULATIONS.log", "w") as f:
                            f.write("Simulation ends at step %d\n"%(data.shape[0]))
                    sys.exit()
        comm.Barrier()
        if irank == 0:
            with open("ci_co_oh_dist.log", "a") as f:
                f.write("%.4f   %.4f\n"%(min_dist_co, min_dist_oh))
        

    elif qm_atoms == qm_region:
        dist_list = []
        for ia in qm_atoms:#np.delete(qm_atoms, [ci_cidx, ci_oidx]):
            dist_ia = []
            for io_w in owater_list:
                dist_ia.append(np.linalg.norm(coords[ia][1]-coords[io_w][1]))
            dist_list.append(min(dist_ia))
        min_dist = np.min(dist_list)
        max_dist = np.max(dist_list)
        #co_distlist.sort()
        min_dist_co = co_distlist[0][0]
        if irank == 0:
            with open("ci_water_dist.log", "a") as f:
                f.write("%.4f   %.4f   %.4f   %.4f\n"%(min_dist_co, min_dist_oh, min_dist, max_dist))
        if (max_dist < 3.5):
            if irank == 0:
                data = np.genfromtxt("ci_water_dist.log")
                with open("CONGRATULATIONS.log", "w") as f:
                    f.write("Simulation ends at step %d\n"%(data.shape[0]))
            #sys.exit()
    qm_region.sort()
    mm_region = np.delete(range(natm), qm_region)
    #Cutoff of QM-MM interations
    mm_region_full = mm_region
    '''if cutoff is not None:
        cutoff = 100
        mm_region_c = []
        for ia in mm_region:
            dist_min = min([np.linalg.norm(coords[ia][1]-coords[ja][1]) for ja in qm_region])
            if dist_min < cutoff:
                mm_region_c.append(ia)
        mm_region = mm_region_c'''
    qm_coords = [coords[ia] for ia in qm_region]
    mm_coords = [coords[ia] for ia in mm_region]
    if irank == 0:
        #with open("coord_qm.xyz", "w") as f:
        #molecule = os.environ.get('molecule')
        molecule = inputs["molecule"]
        print(os.getcwd(), molecule)
        with open(f"{molecule}_qm.xyz", "w") as f:
            coord_msg = "%d\n\n"%len(qm_coords)
            for ia, (x, y, z) in qm_coords:
                coord_msg += "%s %12.6f %12.6f %12.6f\n"%(ia, x, y, z)
            f.write(coord_msg)
        #sys.exit()
        with open("coord_now.xyz", "w") as f:
            coord_msg = "%d\n\n"%len(coords)
            for ia, (x, y, z) in coords:
                coord_msg += "%s %12.6f %12.6f %12.6f\n"%(ia, x, y, z)
            f.write(coord_msg)
    return qm_region, mm_region, qm_coords, mm_coords


def get_hcore_qmmm(mf, h1e):
    '''
    Return qmmm hcore 
    '''
    mol = mf.mol
    coords = mf.mol_mm.atom_coords()
    charges = mf.mol_mm.atom_charges()
    '''v = 0
    for i,q in enumerate(charges):
        mol.set_rinv_origin(coords[i])
        v += mol.intor('int1e_rinv') * -q
    v_node = v'''
    if mol.cart:
        intor = 'int3c2e_cart'
    else:
        intor = 'int3c2e_sph'
    nao = mol.nao_nr()
    nao_pair = nao*(nao+1)//2
    win_v, v_node = get_shared((nao_pair), set_zeros=True)
    atom_slice = get_slice(range(nrank), job_size=charges.size)[irank]
    if atom_slice is not None:
        size_sub = nao**2
        max_len = len(atom_slice)
    else:
        size_sub = None
        max_len = None
    
    if atom_slice is not None:
        blksize = get_buff_len(mol, size_sub, 0.5, max_len=len(atom_slice))
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                            mol._env, intor)
        if irank_shm == 0:
            v = v_node
        else:
            v = np.zeros((nao_pair))
        ia0, ia1 = atom_slice[0], atom_slice[-1]
        for i0, i1 in lib.prange(ia0, ia1+1, blksize):
            fakemol = gto.fakemol_for_charges(coords[i0:i1])
            j3c = df.incore.aux_e2(mol, fakemol, intor=intor,
                                    aosym='s2ij', cintopt=cintopt)
            v += np.einsum('xk,k->x', j3c, -charges[i0:i1])
        if irank_shm != 0:
            Accumulate_GA_shm(win_v, v_node, v)
    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(v_node)
    
    if irank_shm == 0:
        h_qmmm = h1e + lib.unpack_tril(v_node)
    else:
        h_qmmm = None
    comm_shm.Barrier()
    #free_win(win_v)
    return h_qmmm

def energy_nuc_qmmm(self, nuc):
    # interactions between QM nuclei and MM particles
    #nuc = self.mol.energy_nuc()
    coords = self.mol_mm.atom_coords()
    charges = self.mol_mm.atom_charges()
    for j in range(self.mol.natm):
        q2, r2 = self.mol.atom_charge(j), self.mol.atom_coord(j)
        r = lib.norm(r2-coords, axis=1)
        nuc += q2*(charges/r).sum()
    return nuc

def grad_nuc_ele_qmmm(self, dm_tmp=None, mol=None):
    ''' (QM 1e grad) + <-d/dX i|q_mm/r_mm|j>'''
    if mol is None: mol = self.mol
    coords = self.mol_mm.atom_coords()
    charges = self.mol_mm.atom_charges()
    nao = self.mol.nao_nr()
    if mol.cart:
        intor = 'int3c2e_ip1_cart'
    else:
        intor = 'int3c2e_ip1_sph'

    atom_slice = get_slice(range(nrank), job_size=charges.size)[irank]
    if atom_slice is not None:
        size_sub = 3*nao**2
        max_len = len(atom_slice)
    else:
        size_sub = None
        max_len = None
    
    win_gmm, gmm_node = get_shared((self.mol_mm.natm, 3), set_zeros=True)
    if atom_slice is not None:
        blksize = get_buff_len(mol, size_sub, 0.5, max_len=len(atom_slice))
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                            mol._env, intor)
        v = np.zeros((3, nao, nao))
        ia0, ia1 = atom_slice[0], atom_slice[-1]
        for i0, i1 in lib.prange(ia0, ia1+1, blksize):
            fakemol = gto.fakemol_for_charges(coords[i0:i1])
            j3c = df.incore.aux_e2(mol, fakemol, intor, aosym='s1',
                                comp=3, cintopt=cintopt)
            v += np.einsum('ipqk,k->ipq', j3c, -charges[i0:i1])
            for idx, ia in enumerate(range(i0, i1)):
                #gmm_node -= charges[ia]*j3c[:,:,:,idx].reshape(-3, -1), dm_tmp.ravel()
                d_rinv = j3c[:,:,:,idx]
                d_rinv = d_rinv + d_rinv.transpose(0,2,1)
                gmm_node[ia] -= charges[ia]*np.einsum('ipq,pq->i', d_rinv, dm_tmp)
    else:
        v = None
    comm.Barrier()
    if nnode > 1:
        Acc_and_get_GA(var=gmm_node)
    comm_shm.Barrier()
    g_mm = np.copy(gmm_node)
    comm_shm.Barrier()
    free_win(win_gmm)
    return v, g_mm

def grad_nuc_nuc_qmmm(self, mol=None, atmlst=None):
    if mol is None: mol = self.mol
    mm_coords = self.mol_mm.atom_coords()
    mm_charges = self.mol_mm.atom_charges()
    qm_coords = mol.atom_coords()
    qm_charges = mol.atom_charges()

    # nuclei lattice interaction
    g_qm = np.zeros((mol.natm,3))
    for i in range(mol.natm):
        q1 = mol.atom_charge(i)
        r1 = mol.atom_coord(i)
        r = lib.norm(r1-mm_coords, axis=1)
        g_qm[i] = -q1 * np.einsum('i,ix,i->x', mm_charges, r1-mm_coords, 1/r**3)
    g_mm = np.zeros((self.mol_mm.natm,3))
    for i in range(self.mol_mm.natm):
        q1 = self.mol_mm.atom_charge(i)
        r1 = self.mol_mm.atom_coord(i)
        r = lib.norm(r1-qm_coords, axis=1)
        g_mm[i] = -q1 * np.einsum('i,ix,i->x', qm_charges, r1-qm_coords, 1/r**3)
    if atmlst is not None:
        g_qm = g_qm[atmlst]
    return g_qm, g_mm

def xyz_str2tuple(coords_str):
    coords_str = coords_str.split("\n")
    coords_tup = []
    for coi in coords_str:
        coi_split = coi.split()
        if len(coi_split) > 0:
            ia_sym = coi_split[0]
            ico = [float(i) for i in coi_split[1:]]
            coords_tup.append((ia_sym, ico))
    return coords_tup

def create_new_residue_template(forcefield, topology):
    """
    Create a new OpeMM residue template when there is no matching residue 
    and registers it into self.forcefield forcefield object.

    Note
    ----
    currently, if there is unmatched name, currently only checks original 
    unmodified residue, N-terminus form, and C-terminus form. 
    This may not be robust.

    Parameters
    ----------
    topology : OpenMM topology object

    Examples
    --------
    >>> create_new_residue_template(topology)
    """
    template, unmatched_res = forcefield.generateTemplatesForUnmatchedResidues(topology)
    #print(template, unmatched_res)
    # Loop through list of unmatched residues
    #print('Loop through list of unmatched residues')
    for i, res in enumerate(unmatched_res):
        res_name = res.name                             # get the name of the original unmodifed residue
        n_res_name = 'N' + res.name                     # get the name of the N-terminus form of original residue
        c_res_name = 'C' + res.name                     # get the name of the C-terminus form of original residue
        name = 'Modified_' + res_name                   # assign new name
        template[i].name = name

        # loop through all atoms in modified template and all atoms in orignal template to assign atom type
        #print('loop through all atoms in modified template and all atoms in orignal template to assign atom type')
        for atom in template[i].atoms:
            for atom2 in forcefield._templates[res_name].atoms:
                if atom.name == atom2.name:
                    atom.type = atom2.type
            # the following is for when there is a unmatched name, check the N and C terminus residues
            if atom.type == None:
                #print('check n')
                for atom3 in forcefield._templates[n_res_name].atoms:
                    if atom.name == atom3.name:
                        atom.type = atom3.type
            if atom.type == None:
                #print('check c')
                for atom4 in forcefield._templates[c_res_name].atoms:
                    if atom.name == atom4.name:
                        atom.type = atom4.type

        # override existing modified residues with same name
        if name in forcefield._templates:
            #print('override existing modified residues with name {}'.format(name))
            template[i].overrideLevel = forcefield._templates[name].overrideLevel + 1

        # register the new template to the forcefield object
        #print('register the new template to the forcefield object')
        forcefield.registerResidueTemplate(template[i])
        
class mm_gradient():
    def __init__(self, coords, qm_region=None, nonwater_region=None, 
                 forcefield_all='mm/criegee.xml', forcefield_water='tip3p.xml', 
                 cg_residue="CG1", charges=None, box_pbc=None, log=None):
        """
        Only support MM for water clusters for now
        Make sure the coordinates of a water molecules are next to each others
        """
        if log is None:
            self.log = lib.logger.Logger(sys.stdout, verbose=4)
        else:
            self.log = log
        self.forcefield_water = forcefield_water
        #Assume MM molecules are all water molecules
        is_qm = [False]*len(coords)
        if qm_region is not None:
            for ia in qm_region:
                is_qm[ia] = True
        co = []
        if nonwater_region is None:
            water_coords = coords
            nonwater_coords = None
        else:
            water_region = set(range(len(coords))).difference(set(nonwater_region))
            nonwater_coords = [coords[i] for i in nonwater_region]
            water_coords = [coords[i] for i in water_region]
        xyz_to_pdb(water_coords, coords_nonwater=nonwater_coords, pdb_name="input.pdb", cg_residue=cg_residue, box_pbc=box_pbc)
        self.pdb = openmm.app.PDBFile('input.pdb')
        #forcefield = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        forcefield_all = "%s/mm/data/criegee.xml"%(os.path.dirname(osvmp2.__file__))
        forcefield = openmm.app.ForceField(forcefield_all)#, forcefield_water)
        unmatched = forcefield.getUnmatchedResidues(self.pdb.topology)
        if unmatched:
            create_new_residue_template(forcefield, self.pdb.topology)
        self.system = forcefield.createSystem(self.pdb.topology, nonbondedMethod=openmm.app.NoCutoff, 
                                              constraints=None, rigidWater=False)
        for force in self.system.getForces():
            if type(force) is openmm.NonbondedForce:
                if charges is not None:
                    for ia in range(len(coords)):
                        if not is_qm[ia]:
                            ipara = force.getParticleParameters(ia)
                            force.setParticleParameters(ia, charge=charges[ia], sigma=ipara[1], epsilon=ipara[2])
                if qm_region is not None:
                    for idx_i, ia in enumerate(qm_region):
                        ipara = force.getParticleParameters(ia)
                        force.setParticleParameters(ia, charge=0.0, sigma=ipara[1], epsilon=ipara[2])
                        for ja in qm_region[idx_i+1:]:
                            #sigma must not be zero
                            force.addException(ia, ja, chargeProd=0.0, sigma=1e-10, epsilon=0.0, replace=True)       
            elif type(force) is openmm.HarmonicBondForce:
                for ibond in range(force.getNumBonds()):
                    ipara = force.getBondParameters(ibond)
                    ia, ja = ipara[:2]
                    if is_qm[ia] and is_qm[ja]:
                        force.setBondParameters(ibond, ia, ja, ipara[2], k=0.0)
            elif type(force) is openmm.HarmonicAngleForce:
                for iang in range(force.getNumAngles()):
                    ipara = force.getAngleParameters(iang)
                    ia, ja, ka = ipara[:3]
                    if is_qm[ia] and is_qm[ja] and is_qm[ka]:
                        force.setAngleParameters(iang, ia, ja, ka, ipara[3], k=0.0)
        integrator = openmm.VerletIntegrator(1.0*openmm.unit.femtoseconds)	
        #integrator = openmm.LangevinIntegrator(300.0*openmm.unit.kelvin, 1, 1.0*openmm.unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        self.simulation = openmm.app.Simulation(self.pdb.topology, self.system, integrator, platform)
        self.atom_list = [ia_sym for ia_sym, ico in coords]
        
    def energy_minimization(self, tolerance=None, maxIterations=0):
        if tolerance is None:
            tolerance = 10*openmm.unit.kilojoules_per_mole/openmm.unit.nanometer
        self.simulation.context.setPositions(self.pdb.positions)
        self.simulation.minimizeEnergy(tolerance=tolerance, maxIterations=maxIterations)
        state = self.simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True) * 10 / openmm.unit.nanometer
        msg = "%d\n\n"%(pos.shape[0])
        for idx, i in enumerate(pos):
            x, y, z = i
            msg += "%s %10.8f %10.8f %10.8f\n"%(self.atom_list[idx], x, y, z)
        with open("output.xyz", "w") as f:
            f.write(msg)
    def md_simulation(self, ensemble="nvt", temperature=300, time_step=1, nstep=20000, nstripe=100):
        if ensemble.lower() == "nvt":
            integrator = openmm.LangevinMiddleIntegrator(temperature*openmm.unit.kelvin, 
                                                        1/openmm.unit.picosecond, 
                                                        time_step*openmm.unit.femtoseconds)
        elif ensemble.lower() == "nve":
            integrator = openmm.VerletIntegrator(time_step*openmm.unit.femtoseconds)
        else:
            raise NotImplementedError("Ensemble %s is not implemented"%ensemble)
        simulation = openmm.app.Simulation(self.pdb.topology, self.system, integrator)
        simulation.context.setPositions(self.pdb.positions)
        #simulation.minimizeEnergy()
        simulation.reporters.append(openmm.app.PDBReporter('output.pdb', nstripe))
        simulation.reporters.append(openmm.app.StateDataReporter(stdout, nstripe, step=True, 
                                    potentialEnergy=True, temperature=True))
        simulation.step(nstep)
    def kernel(self):
        self.log.info("----------------------------MM energy and gradient--------------------------")
        t0 = get_current_time()
        eqcgmx = 2625.5002
        fqcgmx = 49621.9
        #pos = [Vec3(self.M.xyzs[0][i,0]/10, self.M.xyzs[0][i,1]/10, self.M.xyzs[0][i,2]/10) for i in range(self.M.na)]*u.nanometer
        self.simulation.context.setPositions(self.pdb.positions)
        state = self.simulation.context.getState(getEnergy=True, getForces=True)
        energy_mm = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole) / eqcgmx
        force_mm = state.getForces(asNumpy=True).flatten() / fqcgmx
        parm = pmd.openmm.load_topology(self.pdb.topology, system=self.system, xyz=self.pdb.positions)
        kcalmol2kjmol = 4.184
        hartree2kcalmol = 627.5
        ev2hartree = 0.036749308136649
        ev2kcalmol = 23
        charge_mm = np.asarray([ia.charge for ia in parm if ia.charge != 0.0])
        self.log.info("MM energy (Eh):")
        energy_list = []
        etot = 0.0
        for iforce, ie in pmd.openmm.energy_decomposition_system(parm, self.system):
            e_hartree = ie/hartree2kcalmol
            energy_list.append([iforce, "%.10f"%e_hartree])
            etot += e_hartree
        energy_list.extend([['', '-'*len("%.10f"%etot)],  ["Total MM energy", "%.10f"%etot]])
        print_align(energy_list, align='lr', indent=0, log=self.log)
        print_time(["MM energy and gradients", get_elapsed_time(t0)], self.log)
        return energy_mm, force_mm, charge_mm

if __name__ == '__main__':
    class grad_scanner():
        def __init__ (self, mol=None):
            from pyscf import gto
            self.mol = gto.M(basis=basis,verbose=3)
            self.mol.atom = test_xyz
            self.mol.build()
            self.converged = True
            self.verbose=4
            self.base = None
            self.stdout = sys.stdout
            with open("opt_traj.xyz", 'w') as f:
                pass
        def osvgrad(self, mol=None):
            if mol is not None:
                self.mol = mol
            with open("opt_traj.xyz", 'a') as f:
                f.write(get_coords_from_mol(self.mol))
            from pyscf import mp
            qm_region=None
            qm_atoms = range(5)
            qm_center = 0
            nwater_qm = 20
            #qm_region = range(5+3*4)
            nonwater_region = range(5)
            if qm_atoms == range(self.mol.natm):
                mf = scf.RHF(self.mol).density_fit()
                energy_qm = mf.kernel()
                GRAD_QM = mf.nuc_grad_method()
                grad_qm = GRAD_QM.kernel().reshape(-1,3)
                return energy_qm, grad_qm
            else:
                coord_ang = []
                for ia, (ia_sym, co_i) in enumerate(self.mol._atom):
                    coord_ang.append((ia_sym, np.asarray(co_i)*lib.param.BOHR))
                if qm_atoms is None:
                    GRAD_MM = mm_gradient(coord_ang, qm_region=qm_region)
                    energy_mm, force_mm, charge_mm = GRAD_MM.kernel()
                    grad_mm = -force_mm.reshape(-1,3)
                    return energy_mm, grad_mm
                else:
                    qm_region, mm_region, qm_coords, mm_coords = region_qmmm(coord_ang, qm_region=qm_region, 
                                                                             qm_atoms=qm_atoms, qm_center=qm_center, 
                                                                             nwater_qm=nwater_qm)
                    #print(qm_coords)
                    #print(mm_coords)
                    GRAD_MM = mm_gradient(coord_ang, qm_region=qm_region, nonwater_region=nonwater_region)
                    energy_mm, force_mm, charge_mm = GRAD_MM.kernel()
                    grad_mm = -force_mm.reshape(-1,3)
                    print(energy_mm)
                    print(grad_mm)
                    print(charge_mm)
                    #qm mol
                    mol_qm = gto.M(atom=qm_coords, basis=basis, verbose=3)
                    mol_qm.build()

                    #mm mol
                    mol_mm = qmmm.mm_mole.create_mm_mol(mm_coords, charges=charge_mm)
                    mf = qmmm.mm_charge(scf.RHF(mol_qm), mm_coords, charge_mm)
                    mf.verbose=4
                    energy_qm = mf.kernel()
                    '''mymp = mp.MP2(mf)
                    energy_qm += mymp.kernel()[0]
                    GRAD_QM = mymp.nuc_grad_method()'''
                    GRAD_QM = mf.nuc_grad_method()
                    grad_qm = GRAD_QM.kernel().reshape(-1,3)
                    mf.mol_mm = mol_mm
                    v, gmm_ne = grad_nuc_ele_qmmm(mf, dm_tmp=mf.make_rdm1())
                    gqm_nn, gmm_nn = grad_nuc_nuc_qmmm(mf)
                    grad_mm[mm_region] += gmm_ne.reshape(-1,3) + gmm_nn.reshape(-1,3)
                    grad_mm[qm_region] += grad_qm
                    grad_total = grad_mm
                    return energy_qm+energy_mm, grad_total
    def kernel():
        conv_params = {  # They are default settings
        'convergence_energy':  5e-6,  # Eh
        'convergence_grms': 1e-4,    # Eh/Bohr
        'convergence_gmax': 3e-4,  # Eh/Bohr
        'convergence_drms': 1.2e-3,  # Angstrom
        'convergence_dmax': 1.8e-3,  # Angstrom]
        }
        scanner = grad_scanner()
        sol = geometric_solver.GeometryOptimizer(method=scanner)
        #sol = berny_solver.GeometryOptimizer(method=scanner)
        sol.params = conv_params
        sol.max_cycle = 600
        sol.verbose = scanner.verbose
        sol.kernel()
    from osvmp2.grad_addons import read_xyz
    test_mol = "criegee_ch2o2_water100.xyz"
    test_xyz = read_xyz(test_mol)[1]
    coords = xyz_str2tuple(test_xyz)
    basis = "321G"
    GRAD_OSV = grad_scanner()
    #print(GRAD_OSV.osvgrad())
    from osvmp2.geometric import geometric_solver
    #from osvmp2.berny import berny_solver
    
    kernel()
    
