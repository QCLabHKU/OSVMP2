import numpy as np
import pickle 
from mpi4py import MPI
from osvmp2.__config__ import inputs
from osvmp2.md.driver import GaussDriver, OSVMP2Driver, OrcaDriver, OpenmmDriver
from osvmp2.osvutil import *
from osvmp2.ga_addons import *

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()    # Size of communicator
irank = comm.Get_rank()    # Ranks in communicator
inode = MPI.Get_processor_name()     # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//nrank_shm

########### AMBER #############
Na = 6.02214076e23
epsilon = 0.995792#0.0656888
sigma = 0.34#0.34#0.106907846177
#epsilon = sigma = 1
radius = 1.88 #ang
rchg = 0.01 #ang
cutoff_rep = 50
cutoff_att = 30
coulombscale = 1#0.8333
ljscale = 1#0.5
ke = 138.9421367512152 #nm^2 * kJ/mol
kboltz = 1.380649e-23 #j/k
kboltz = kboltz * Na * 0.001 #nm^2 * dal/(ps^2 * K)
diele = 1
fqcgmx = 49621.9
IADD = 453806245
IMUL = 314159269
MASK = 2147483647
SCALE = 0.4656612873e-9
randSeedP = 17

dalton2kg = 1.66054e-27
j2jmol = 6.02214076e23
j2kjmol = 6.02214076e20 
m2nm = 1e9
ang2nm = 0.1
ang2m = 1e-10
ang2bohr = 1.88973
bohr2m = 5.29177e-11
bohr2nm = 0.0529177
hartree2kcalmol = 627.5
hartree2kjmol = 2625.5
hartree2j = 4.35974e-18
fs2s = 1e-15
fs2ns = 1e-6
fs2ps = 1e-3
#################################################################

mass_dict = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
            'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
            'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
            'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
            'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
            'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
            'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
            'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
            'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
            'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
            'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
            'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
            'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
            'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
            'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
            'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
            'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
            'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
            'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
            'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
            'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
            'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
            'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
            'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
            'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
            'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
            'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
            'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
            'OG' : 294}

def velocity_verlet(co_now, vel_now, force_now, dt, mass_list, get_eg):
    """
    Velocity Verlet.

    Arguments:
    co_now  -- atom positions in reduced coordinates (meter)
    vel_now  -- velocity
    force_now  -- forces 
    dt -- time step
    mass_list  -- atomic masses
    
    Returns:
    co_new -- new atom positions
    vel_new -- new velocities
    force_new -- new forces
    epot_new -- new potential energy
    """
    coord_now = co_now * m2nm #meter to nanometer 
    mass_list = mass_list[:, np.newaxis]  # enable broadcasting in numpy

    # update positions
    coord_new = coord_now + vel_now * dt + (0.5 * force_now * (dt ** 2)) / mass_list 
    co_new = coord_new / m2nm

    epot_new, grad_new = get_eg.grad(co_new) #grad unit (Jole/meter)
    force_new = -grad_new * j2kjmol/m2nm #(kJ/mol/nm)

    vel_new = vel_now + 0.5 * (force_now + force_new) * dt / mass_list #nm/ps
    return co_new, vel_new, force_new, epot_new

def langevin_baoab(co_now, vel_now, force_now, dt, temp, mass_list, get_eg, gamma=100, ndim=3):
    """
    Velocity Verlet.

    Arguments:
    co_now  -- atom positions in reduced coordinates (meter)
    vel_now  -- velocity
    force_now  -- forces 
    gamma -- friction constant
    dt -- time step
    mass_list  -- atomic masses
    
    Returns:
    co_new -- new atom positions
    vel_new -- new velocities
    force_new -- new forces
    epot_new -- new potential energy
    """
    def position_update(co_now, vel, dt):
        coord_now = co_now * m2nm #meter to nanometer

        coord_new = coord_now + vel*dt/2.
        co_new = coord_new / m2nm
        return co_new

    #this is step B
    def velocity_update(v, F, dt):
        v_new = v + F*dt/2.
        return v_new

    def random_velocity_update(v, gamma, kBT, dt):
        #R = np.random.normal()
        #Generate gaussian noise
        if ndim == 2:
            vrand = np.random.normal(loc=0, size=(mass_list.size, 2))
            vrand = np.asarray([[x, y, 0] for x, y in vrand])
        elif ndim == 3:
            vrand = np.random.normal(loc=0, size=(mass_list.size, 3))

        c1 = np.exp(-gamma*dt)
        c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT/mass_list[:, np.newaxis])
        v_new = c1*v + vrand*c2
        return v_new


    kBT = kboltz * temp

    # B
    vel_new = velocity_update(vel_now, force_now, dt)
    
    #A
    co_new = position_update(co_now, vel_new, dt)

    #O
    win_newv, newv_node = get_shared((mass_list.size, 3))
    if irank == 0:
        newv_node[:] = random_velocity_update(vel_new, gamma, kBT, dt)
    comm.Barrier()
    vel_new = np.copy(newv_node)
    comm.Barrier()
    win_newv.Free()
    
    #A
    co_new = position_update(co_new, vel_new, dt)
    
    # B
    epot_new, grad_new = get_eg.grad(co_new) #grad unit (Jole/meter)
    force_new = -grad_new * j2kjmol/m2nm #(kJ/mol/nm)
    vel_new = velocity_update(vel_new, force_new, dt)
 
    return co_new, vel_new, force_new, epot_new


def langevin_gromacs(co_now, vel_now, force_now, dt, temp, mass_list, 
                     get_eg, gamma=10, ndim=3, frozen_atoms=None):
    """
    Velocity Verlet.

    Arguments:
    co_now  -- atom positions in reduced coordinates (meter)
    vel_now  -- velocity
    force_now  -- forces 
    gamma -- friction constant
    dt -- time step
    mass_list  -- atomic masses
    
    Returns:
    co_new -- new atom positions
    vel_new -- new velocities
    force_new -- new forces
    epot_new -- new potential energy
    """
    def position_update(co_now, vel_prime, delta_vel, dt):
        coord_now = co_now * m2nm #meter to nanometer

        coord_new = coord_now + (vel_prime + 0.5 * delta_vel) * dt
        co_new = coord_new / m2nm
        return co_new


    def get_delta_vel(vel_prime, alpha, kBT):
        #R = np.random.normal()
        #Generate gaussian noise
        vrand = np.random.normal(size=(mass_list.size, ndim))
        if ndim == 2:
            vrand = np.asarray([[x, y, 0] for x, y in vrand])
        term2 = np.sqrt(kBT * alpha * (2 - alpha) /mass_list[:, np.newaxis]) * vrand
        delta_vel = -alpha * vel_prime + term2
        if frozen_atoms is not None:
            for ia in frozen_atoms:
                delta_vel[ia] = 0.0
        return delta_vel


    kBT = kboltz * temp
    alpha = 1 - np.exp(-gamma * dt)

    vel_prime = vel_now + force_now * dt / mass_list[:, np.newaxis]
    if frozen_atoms is not None:
        for ia in frozen_atoms:
            vel_prime[ia] = 0.0

    win_dv, dv_node = get_shared((mass_list.size, 3))
    if irank == 0:
        dv_node[:] = get_delta_vel(vel_prime, alpha, kBT)
    comm.Barrier()

    co_new = position_update(co_now, vel_prime, dv_node, dt)

    vel_new = vel_prime + dv_node

    comm.Barrier()
    win_dv.Free()

    epot_new, grad_new = get_eg.grad(co_new) #grad unit (Jole/meter)
    force_new = -grad_new * j2kjmol/m2nm #(kJ/mol/nm)

    return co_new, vel_new, force_new, epot_new

def record_traj(atom_list, coord_list, step, traj_name):
    msg = "%d\nStep %d\n"%(len(coord_list), step)
    coord_list = coord_list / ang2m
    for ia_sym, (x, y, z) in zip(atom_list, coord_list):
        msg += "%s %10.6f %10.6f %10.6f\n"%(ia_sym, x, y, z)
    with open(traj_name, "a") as f:
        f.write(msg)


def velocity_init(temp, mass_list, natm, ndim=3, frozen_atoms=None):
    '''velocities = []
    for mass in mass_list:
        factor = math.sqrt(kboltz * temp / mass)
        ivel = np.random.normal(loc=0, scale=factor, size=ndim)
        if ndim == 2:
            velocities.append(ivel + [0])
        else:
            velocities.append(ivel)'''
    vrand = np.random.normal(size=(mass_list.size, ndim))
    if ndim == 2:
        vrand = np.asarray([[x, y, 0] for x, y in vrand])

    velocities = np.sqrt(kboltz * temp / mass_list[:, np.newaxis]) * vrand
    if frozen_atoms is not None:
        for ia in frozen_atoms:
            velocities[ia] = 0

    #scale the velocities to match the temperature
    mvsq_target = ndim * mass_list.size * kboltz * temp
    mvsq_now = np.sum(velocities**2 * mass_list[:, np.newaxis])
    velocities *= np.sqrt(mvsq_target / mvsq_now)
    return velocities

def print_info(step, epot, vel, mass_list, output, ndim):
    epot = epot * j2kjmol
    ekin = 0.5 * np.sum(vel**2 * mass_list[:, np.newaxis]) #/ hartree2kjmol
    itemp = ekin*2/(ndim*mass_list.size*kboltz)#*hartree2kjmol
    msg = "Iter. %d Etot: %10.6f, Ekin: %10.6E, Epot: %10.6f, temp %5.2f" % (step, epot+ekin, ekin, epot, itemp)
    print(msg)
    with open(output, "a") as f:
        f.write(msg + "\n")

def save_chk(self, co_new, vel_new, chk_save):
    dict_para = {"stride": self.stride,
                 "total_steps": self.total_steps,
                 "seed": self.seed,
                 "temperature": self.temperature,
                 "pressure": self.pressure,
                 "dyn_mode": self.dyn_mode,
                 "tau": self.tau,
                 "time_step": self.time_step,
                 "atom_list": self.atom_list,
                 "ndim": self.ndim,
                 "co_new": co_new,
                 "vel_new": vel_new,
    }

    with open(chk_save, 'wb') as f:
        pickle.dump(dict_para, f)

def read_chk(self, chk_read):
    with open(chk_read, 'rb') as f:
        dict_para = pickle.load(f)
    self.stride = dict_para["stride"]
    self.total_steps = dict_para["total_steps"]
    self.seed = dict_para["seed"]
    self.temperature = dict_para["temperature"]
    self.pressure = dict_para["pressure"]
    self.dyn_mode = dict_para["dyn_mode"]
    self.tau = dict_para["tau"]
    self.time_step = dict_para["time_step"]
    self.atom_list = dict_para["atom_list"]
    self.ndim = dict_para["ndim"]
    self.coord_init = dict_para["co_new"]
    if irank == 0:
        self.vel_init = dict_para["vel_new"]
        print(f"Read check file: {chk_read}")
    return self

def read_xyz(xyz_file):
    atom_list = []
    coord_list = []
    with open(xyz_file, "r") as f:
        lines = f.readlines()
        natm = int(lines[0])
        for l in lines[2:2+natm]:
            lsplit = l.split()
            atom_list.append(lsplit[0])
            coord_list.append([float(i)*ang2m for i in lsplit[1:]])
    return atom_list, np.asarray(coord_list)

def get_mass(atom_list):
    mass_list = []
    for ia in atom_list:
        mass_list.append(mass_dict[ia.upper()])
    return np.asarray(mass_list)

def str2list(lstr, dtype=int):
    if lstr is None:
        return None
    str_list = [float(i) for i in lstr.replace("[","").replace("]","").split(",")]
    return np.asarray(str_list, dtype=dtype)

class run_md():
    def __init__(self, traj_name="md_traj.xyz", output="md_info.log", chk_save="md_restart.chk"):
        with open(traj_name, "w") as f:
            pass
        with open(output, "w") as f:
            pass
        self.traj_name = traj_name
        self.output = output
        self.chk_save = chk_save
        '''xyz_file = sys.argv[1]
        chk_read = os.environ.get("chkfile_md", None)
        self.stride = int(os.environ.get("stride", '10'))
        self.total_steps = int(os.environ.get("total_steps", 1000))
        self.seed = int(os.environ.get("seed", 3348))
        self.temperature = float(os.environ.get("temperature", 25))
        self.pressure = float(os.environ.get("pressure", 10))
        self.dyn_mode = os.environ.get("dyn_mode", 'nvt')
        if self.dyn_mode.lower() not in ['nvt', 'npt']:
            self.temperature = None
        self.tau = int(os.environ.get("tau", 100))
        self.time_step = int(os.environ.get("time_step", 1)) * fs2ps #fs to s
        self.ndim = int(os.environ.get("ndim", 3))
        self.basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()
        md_driver = os.environ.get("md_driver", 'osvmp2')
        self.frozen_atoms = str2list(os.environ.get("frozen_atoms", None))'''

        xyz_file          = inputs["xyz_file"]
        chk_read          = inputs["chkfile_md"]
        self.stride       = inputs["stride"]
        self.total_steps  = inputs["total_steps"]
        self.seed         = inputs["seed"]
        self.temperature  = inputs["temperature"]
        self.pressure     = inputs["pressure"]
        self.dyn_mode     = inputs["dyn_mode"]
        self.tau          = inputs["tau"]
        self.time_step    = inputs["time_step"] * fs2ps  # fs to s
        self.ndim         = inputs["ndim"]
        self.basis        = inputs["basis"]
        md_driver         = inputs["md_driver"]
        self.frozen_atoms = str2list(inputs["frozen_atoms"])


        if self.dyn_mode.lower() not in ["nvt", "npt"]:
            self.temperature = None

        if chk_read is None:
            self.atom_list, self.coord_init = read_xyz(xyz_file)
            self.vel_init = None
            #MD parameters
            
        else:
            self = read_chk(self, chk_read)
        self.mass_list = get_mass(self.atom_list)
        
        if md_driver == "gaussian":
            self.base_driver = GaussDriver
        elif md_driver == "orca":
            self.base_driver = OrcaDriver
        elif md_driver == "osvmp2":
            self.base_driver = OSVMP2Driver
        elif md_driver == "openmm":
            self.base_driver = OpenmmDriver
        
        

    def kernel(self):
        #unit of dt is fs
        natm = len(self.coord_init)

        get_eg = self.base_driver(atoms=self.atom_list, basis=self.basis, stride=self.stride)
        ene_now, grad_now = get_eg.grad(self.coord_init) # jole, jole/m
        
        force_now = -grad_now * j2kjmol/m2nm #(kJ/mol/nm)
        win_vel, vel_node = get_shared((natm, 3))
        if irank == 0:
            record_traj(self.atom_list, self.coord_init, 0, self.traj_name)
            if self.vel_init is None:
                vel_node[:] = velocity_init(self.temperature, self.mass_list, natm, self.ndim, self.frozen_atoms)
            else:
                vel_node[:] = self.vel_init
            print_info(0, ene_now, vel_node, self.mass_list, self.output, self.ndim)
            save_chk(self, self.coord_init, vel_node, self.chk_save)
        comm.Barrier()
        vel_now = vel_node
        co_now = self.coord_init
        

        # Velocity Verlet time-stepping
        for i in range(self.total_steps):
            if self.temperature is None:
                co_new, vel_new, force_new, epot_new = velocity_verlet(co_now, vel_now, force_now, self.time_step, get_eg, self.mass_list)
            #co_new, vel_new, force_new, epot_new = langevin_baoab(co_now, vel_now, force_now, self.time_step, temp, mass_list, get_eg, gamma=10)
            else:
                co_new, vel_new, force_new, epot_new = langevin_gromacs(co_now, vel_now, force_now, self.time_step, 
                                                                        self.temperature, self.mass_list, get_eg, 
                                                                        gamma=self.tau, ndim=self.ndim, frozen_atoms=self.frozen_atoms)
            if irank == 0:
                if ((i+1) % self.stride == 0):
                    record_traj(self.atom_list, co_new, i+1, self.traj_name)
                    save_chk(self, co_new, vel_new, self.chk_save)
                print_info(i+1, epot_new, vel_new, self.mass_list, self.output, self.ndim)
            
            co_now = co_new
            vel_now = vel_new
            force_now = force_new
        comm.Barrier()
        free_win(win_vel)

if __name__ == '__main__':
    RUNMD = run_md()
    RUNMD.kernel()