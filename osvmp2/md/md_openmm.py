import os
import sys
import time
import pickle
import numpy as np
from simtk import openmm
import simtk.openmm.app
import osvmp2
from osvmp2.__config__ import inputs
from osvmp2.mm.xyz2pdb import xyz_to_pdb
from osvmp2.md.driver import GaussDriver, OSVMP2Driver, OrcaDriver, OpenmmDriver
from osvmp2.osvutil import *
from osvmp2.ga_addons import *
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()    # Size of communicator
irank = comm.Get_rank()    # Ranks in communicator
inode = MPI.Get_processor_name()     # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//nrank_shm

j2kjmol = 6.02214076e20 
j2hartree = 2.294e+17
m2nm = 1e9
ang2m = 1e-10
ang2nm = 0.1

def strl2list(strl, dtype='i'):
    if strl is None:
        return None
    else:
        if dtype in ['i', int]:
            type_format = int
        else:
            type_format = float
        ilist = [type_format(i) for i in strl.replace("[","").replace("]","").split(',')]
        return np.asarray(ilist)

def read_chk(self, chk_read):
    with open(chk_read, 'rb') as f:
        dict_para = pickle.load(f)
    self.stride = dict_para["stride"]
    self.total_steps = dict_para["total_steps"]
    self.seed = dict_para["seed"]
    self.temperature = dict_para["temperature"]
    self.pressure = dict_para["pressure"]
    #self.dyn_mode = dict_para["dyn_mode"]
    self.fric_coeff = dict_para["fric_coeff"]
    self.time_step = dict_para["time_step"]
    self.atom_list = dict_para["atom_list"]
    self.coord_init = dict_para["co_new"]
    self.vel_init = dict_para["vel_new"]
    self.pbc_box = dict_para["pbc_box"]
    print(f"Read check file: {chk_read}")
    
    return self

def save_chk(self, co_new, vel_new, box_now, chk_save):
    dict_para = {"stride": self.stride,
                 "total_steps": self.total_steps,
                 "seed": self.seed,
                 "temperature": self.temperature,
                 "pressure": self.pressure,
                 #"dyn_mode": self.dyn_mode,
                 "fric_coeff": self.fric_coeff,
                 "time_step": self.time_step,
                 "atom_list": self.atom_list,
                 "co_new": co_new,
                 "vel_new": vel_new,
                 "pbc_box": box_now
    }

    with open(chk_save, 'wb') as f:
        pickle.dump(dict_para, f)

def read_xyz(xyz_file):
    atom_list = []
    coord_list = []
    with open(xyz_file, "r") as f:
        lines = f.readlines()
        natm = int(lines[0])
        for l in lines[2:2+natm]:
            lsplit = l.split()
            atom_list.append(lsplit[0])
            coord_list.append([float(i) for i in lsplit[1:]])
    return atom_list, np.asarray(coord_list)

def save_info(self, istep, state, coord_now, box_now, t_for, t_sol):
    if istep == 0:
        file_mode = "w"
    else:
        file_mode = "a"

    vel_new = state.getVelocities(asNumpy=True) / (openmm.unit.nanometer/openmm.unit.picoseconds)
    save_chk(self, coord_now, vel_new, box_now, self.chk_save)

    msg_traj = f"{coord_now.shape[0]}\n"
    if self.pbc_box is not None:
        bx, by, bz = box_now
        msg_traj += "%.4f %.4f %.4f"%(bx, by, bz)
    msg_traj += "\n"
    msg_vel = f"{coord_now.shape[0]}\n\n"
    for idx, (ico, ive) in enumerate(zip(coord_now, vel_new)):
        x, y, z = ico
        msg_traj += "%s %10.4f %10.4f %10.4f\n"%(self.atom_list[idx], x, y, z)

        x, y, z = ive
        msg_vel += "%s %12.4E %12.4E %12.4E\n"%(self.atom_list[idx], x, y, z)
    
    with open(f"{self.output_dir}/traj_md.xyz", file_mode) as f:
        f.write(msg_traj)
    
    with open(f"{self.output_dir}/vel_md.xyz", file_mode) as f:
        f.write(msg_vel)
    
    #save md info
    temp, vol = self.state_reporter._constructReportValues(self.simulation, state)
    ekin = state.getKineticEnergy() / openmm.unit.kilocalories_per_mole
    epot = state.getPotentialEnergy() / openmm.unit.kilocalories_per_mole
    etot = (ekin+epot) 
    msg = f"%d %.2f %.2f %2f %2f %2f %.3f %.3f"%(istep, temp, vol, ekin, epot, etot, t_for/(istep+1), t_sol/(istep+1))
    print(msg)
    if istep == 0:
        msg = "step temperature volume T_force T_solver\n" + msg 
    msg += "\n"
    with open(f"{self.output_dir}/sim_info.out", file_mode) as f:
        f.write(msg)

class openmm_solver():
    def __init__(self, traj_name="md_traj.xyz", chk_save="md_restart.chk"):
        '''xyz_file = sys.argv[1]
        self.main_dir = os.environ.get("main_dir", '.')
        self.output_dir = os.environ.get("output_dir", ".")
        chk_read = os.environ.get("chkfile_md", None)
        self.platform = os.environ.get("platform", "CPU")
        self.dyn_mode = os.environ.get("dyn_mode", 'nvt').lower()
        self.pbc_box = strl2list(os.environ.get("pbc_box", None), dtype=float)
        self.stride = int(os.environ.get("stride", '10'))
        self.total_steps = int(os.environ.get("total_steps", 1000))
        self.seed = int(os.environ.get("seed", 3348))
        self.temperature = float(os.environ.get("temperature", 298.15)) 
        self.pressure = float(os.environ.get("pressure", 1)) #atmosphere
        
        self.fric_coeff = float(os.environ.get("fric_coeff", 1)) #1 / ps
        self.time_step = int(os.environ.get("time_step", 1))  #fs 
        self.basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()
        md_driver = os.environ.get("md_driver", 'osvmp2')'''

        xyz_file          = inputs["xyz_file"]
        self.main_dir     = inputs["main_dir"]
        self.output_dir   = inputs["output_dir"]
        chk_read          = inputs["chkfile_md"]
        self.platform     = inputs["platform"]
        self.dyn_mode     = inputs["dyn_mode"]
        self.pbc_box      = inputs["pbc_box"]
        self.stride       = inputs["stride"]
        self.total_steps  = inputs["total_steps"]
        self.seed         = inputs["seed"]
        self.temperature  = inputs["temperature"]
        self.pressure     = inputs["pressure"]

        self.fric_coeff   = inputs["fric_coeff"]
        self.time_step    = inputs["time_step"]
        self.basis        = inputs["basis"]
        md_driver         = inputs["md_driver"]


        self.traj_name = f"{self.output_dir}/{traj_name}"
        self.chk_save = f"{self.output_dir}/{chk_save}"
        
        if chk_read is None:
            self.atom_list, self.coord_init = read_xyz(xyz_file)
            #MD parameters
        else:
            self = read_chk(self, chk_read)

        
        if md_driver == "gaussian":
            self.base_driver = GaussDriver
        elif md_driver == "orca":
            self.base_driver = OrcaDriver
        elif md_driver == "osvmp2":
            self.base_driver = OSVMP2Driver
        elif md_driver == "openmm":
            self.base_driver = OpenmmDriver
        if irank == 0:
            #openmm
            water_coords = [(ia_sym, ico) for ia_sym, ico in zip(self.atom_list, self.coord_init)]
            label_list = res_list = [[sym.upper()] for sym in self.atom_list]
            file_pdb = f"{self.output_dir}/initial.pdb"
            xyz_to_pdb(water_coords, pdb_name=file_pdb, res_list=res_list, label_list=label_list, box_pbc=self.pbc_box)
            self.pdb = openmm.app.PDBFile(file_pdb)
            self.ff = "blank_water"#tip3p, blank_water
            #forcefield = openmm.app.ForceField(f"{self.main_dir}/{self.ff}.xml")
            path_ff = "%s/mm/data/blank.xml"%(os.path.dirname(osvmp2.__file__))
            forcefield = openmm.app.ForceField(path_ff)
            
            if self.ff == "tip3p":
                cutoff = 0.8
            else:
                cutoff = 0.2
            if self.pbc_box is None:
                self.system = forcefield.createSystem(self.pdb.topology, constraints=None, rigidWater=False)
            else:
                self.system = forcefield.createSystem(self.pdb.topology, constraints=None, nonbondedMethod=openmm.app.PME, 
                                                    nonbondedCutoff=cutoff*openmm.unit.nanometer,rigidWater=False)
            if self.ff == "blank_water": #or True:
                for force in self.system.getForces():
                    if type(force) is openmm.NonbondedForce:
                        #force.setPMEParameters(2.5002898720871846, 5, 5, 5)
                        #print(force.getPMEParameters())
                        #force.setPMEParameters(1, 3, 3, 3)
                        force.setEwaldErrorTolerance(0.1)
                        for ia in range(len(self.coord_init)):
                            ipara = np.asarray(force.getParticleParameters(ia))
                            ipara[:] = 0
                            force.setParticleParameters(ia, charge=ipara[0], sigma=ipara[1], epsilon=ipara[2])
                    elif type(force) is openmm.HarmonicBondForce:
                        for ibond in range(force.getNumBonds()):
                            ipara = force.getBondParameters(ibond)
                            ia, ja = ipara[:2]
                            force.setBondParameters(ibond, ia, ja, ipara[2], k=0.0)
                    elif type(force) is openmm.HarmonicAngleForce:
                        for iang in range(force.getNumAngles()):
                            ipara = force.getAngleParameters(iang)
                            ia, ja, ka = ipara[:3]
                            force.setAngleParameters(iang, ia, ja, ka, ipara[3], k=0.0)
            time_step = self.time_step * openmm.unit.femtoseconds
            fric_coeff = self.fric_coeff / openmm.unit.picoseconds
            temperature = self.temperature * openmm.unit.kelvin
            pressure = self.pressure * openmm.unit.atmosphere
            if self.dyn_mode == "nve":
                self.integrator = openmm.VerletIntegrator(time_step)
            else:
                self.integrator = openmm.LangevinIntegrator(temperature, fric_coeff, time_step)
                if self.dyn_mode == "npt":
                    self.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
                
            #Initialize force
            self.force = openmm.CustomExternalForce("-x*fx-y*fy-z*fz+de")    # define a custom force for adding gradients
            self.force.addPerParticleParameter('fx')
            self.force.addPerParticleParameter('fy')
            self.force.addPerParticleParameter('fz')
            self.force.addGlobalParameter('de', 0.0)
            self.natm = self.system.getNumParticles()
            for i in range(self.natm):
                self.force.addParticle(i, np.array([0.0, 0.0, 0.0]))
            self.system.addForce(self.force)
            platform = openmm.Platform.getPlatformByName(self.platform)
            self.simulation = openmm.app.Simulation(self.pdb.topology, self.system, self.integrator, platform=platform)
            self.simulation.context.setPositions(self.pdb.positions)
            if chk_read is None:
                self.simulation.context.setVelocitiesToTemperature(temperature)
            else:
                self.simulation.context.setVelocities(self.vel_init)
            '''self.simulation.reporters.append(simtk.openmm.app.StateDataReporter(sys.stdout, self.stride, step=True,
                                            potentialEnergy=True, temperature=True, density=True))'''
            self.state_reporter = simtk.openmm.app.StateDataReporter(sys.stdout, self.stride, temperature=True, volume=True)
            self.state_reporter._initializeConstants(self.simulation)
    def kernel(self):
        coord_now = self.coord_init
        box_now = self.pbc_box
        t_for = 0.0
        t_sol = 0.0
        get_eg = self.base_driver(atoms=self.atom_list, basis=self.basis, stride=self.stride)
        
        for istep in range(self.total_steps):
            t0 = time.time()
            win_coord, coord_node = get_shared((len(coord_now), 3))
            if irank == 0:
                state = self.simulation.context.getState(getPositions=True)
                coord_node[:] = state.getPositions(asNumpy=True) * 10 / openmm.unit.nanometer
            comm.Barrier()
            coord_now = np.copy(coord_node)
            comm.Barrier()
            win_coord.Free()
            get_eg.md_step = istep
            if self.pbc_box is None:
                box_nm = None
            else:
                box_nm = box_now*ang2nm
            ene_now, grad_now = get_eg.grad(coord_now*ang2m)#, pbc_box=box_nm) # jole, jole/m
            t_for += time.time() - t0
            if irank == 0:
                t0 = time.time()
                force_now = - grad_now * (j2kjmol / m2nm) #kj/mol * nm
                np.einsum("ab,ab->", coord_now, force_now)
                delta_e = (ene_now*j2kjmol + np.einsum("ab,ab->", coord_now, force_now) * ang2nm) / self.natm
                #print("HERE HERE")
                for ia, iforce in enumerate(force_now):
                    self.force.setParticleParameters(ia, ia, iforce)
                self.simulation.context.setParameter("de", delta_e) #To update energy
                self.force.updateParametersInContext(self.simulation.context)
                
                if istep%self.stride==0:
                    getVelocities = True
                    getEnergy = True
                else:
                    getVelocities = False
                    getEnergy = False
                state = self.simulation.context.getState(getEnergy=getEnergy, getVelocities=getVelocities)
                '''state = self.simulation.context.getState(getEnergy=True, getForces=True, getVelocities=getVelocities)
                print(state.getPotentialEnergy())
                print(state.getForces(asNumpy=True))'''
                if self.pbc_box is not None:
                    box_now = np.diag(state.getPeriodicBoxVectors(asNumpy=True) * 10 / openmm.unit.nanometer)
                
                if istep%self.stride==0:
                    save_info(self, istep, state, coord_now, box_now, t_for, t_sol)
                self.simulation.step(1)
            
            t_sol += time.time() - t0
if __name__ == '__main__':
    RUNMD = openmm_solver()
    RUNMD.kernel()


    