import os
import sys
import numpy as np
from simtk import openmm
import parmed as pmd
from pyscf import lib
import osvmp2
from osvmp2.mm.xyz2pdb import xyz_to_pdb

def print_align(msg_list, align='l', align_1=None, indent=0, log=None, printout=True):
    if align_1 is None:
        align_1 = align
    align_list = []
    for align_i in [align_1, align]:
        align_format = []
        for i in list(align_i):
            if i == 'l':
                align_format.append('<')
            elif i == 'c':
                align_format.append('^')
            elif i == 'r':
                align_format.append('>')
        align_list.append(align_format)
    len_col = []
    for col_i in zip(*msg_list):
        len_col.append(max([len(str(i)) for i in col_i])+2)
    msg = ''
    for idx, msg_i in enumerate(msg_list):
        if idx == 0:
            align_i = align_list[0]
        else:
            align_i = align_list[1]
        msg += ' ' * indent
        msg += ''.join([('{:%s%d} '%(ali, li)).format(mi) for ali, li, mi in zip(align_i, len_col, msg_i)])
        if idx != len(msg_list)-1:
            msg += '\n'
    if printout:
        if log is None:
            print(msg)
        else:
            log.info(msg)
    return msg

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
                 cg_residue="CG1", charges=None, log=None):
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
        xyz_to_pdb(water_coords, coords_nonwater=nonwater_coords, pdb_name="input.pdb", cg_residue=cg_residue)
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
        simulation.reporters.append(openmm.app.StateDataReporter(sys.stdout, nstripe, step=True, 
                                    potentialEnergy=True, temperature=True))
        simulation.step(nstep)
    def kernel(self):
        self.log.info("----------------------------MM energy and gradient--------------------------")
        #t0 = get_current_time()
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
        #print_time(["MM energy and gradients", get_elapsed_time(t0)], self.log)
        return energy_mm, force_mm, charge_mm