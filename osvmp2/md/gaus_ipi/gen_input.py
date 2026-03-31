import sys
import os
import itertools
import numpy as np
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nproc = comm.Get_size()   # Size of communicator
iproc = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm

def get_SideLength(xyz_name):
    with open(xyz_name, 'r') as xyz:
        lines = xyz.readlines()
        coord_list = []
        for idx, l in enumerate(lines):
            if idx > 1:
                coord_i = []
                for i in l.split()[1:]:
                    coord_i.append(float(i))
                coord_list.append(coord_i)
        idx_list = range(len(coord_list))
        atom_dist = []
        for idx0, co0 in enumerate(coord_list):
            idx1_list = idx_list[idx0:]
            for idx1 in idx1_list:
                co1 = coord_list[idx1]
                atom_dist.append(np.linalg.norm(np.asarray(co1)-np.asarray(co0)))
    return max(atom_dist)

def get_cen_atom(xyz_name):
    with open(xyz_name, 'r') as xyz:
        lines = xyz.readlines()
        coord_list = []
        for idx, l in enumerate(lines):
            if idx > 1:
                coord_i = []
                for i in l.split()[1:]:
                    coord_i.append(float(i))
                coord_list.append(coord_i)
        idx_list = range(len(coord_list))

        atom_dist = []
        for co0 in coord_list:
            dist_i = 0.0
            for co1 in coord_list:
                dist_i += np.linalg.norm(np.asarray(co1)-np.asarray(co0))
            atom_dist.append(dist_i)
        return atom_dist.index(min(atom_dist))

class md_parameters():
    def __init__(self):
        self.xyz_name = sys.argv[1]
        self.len_side = get_SideLength(self.xyz_name)*10
        self.cen_atom = get_cen_atom(self.xyz_name)
        self.chkfile = os.environ.get("chkfile", None)
        self.output_opt = os.environ.get("output_opt", 'prop')
        #Set up MD units
        self.time_unit = os.environ.get("time_unit", 'femtosecond')
        self.temp_unit = os.environ.get("temp_unit", 'kelvin')
        self.potential_unit = os.environ.get("potential_unit", 'electronvolt')
        self.press_unit = os.environ.get("press_unit", 'bar')
        self.cell_units = os.environ.get("cell_units", 'angstrom')
        self.force_unit = os.environ.get("force_unit", 'piconewton')
        #Set up MD parameters
        self.verbose = int(os.environ.get("verbose", 5))
        if self.verbose == 5:
            self.verbose = 'high'
        else:
            self.verbose = 'medium'
        self.stride = os.environ.get("stride", '20')
        self.nbeads = os.environ.get("nbeads", '1')
        self.port = int(os.environ.get("port", 31415))
        self.total_steps = os.environ.get("total_steps", 1000)
        self.seed = os.environ.get("seed", 3348)
        self.temperature = os.environ.get("temperature", 25)
        self.pressure = os.environ.get("pressure", 10)
        self.if_fixatoms = bool(int(os.environ.get("if_fixatoms", 0)))
        self.fixatoms = os.environ.get("fixatoms", None)
        if self.if_fixatoms and self.fixatoms is None:
            self.fixatoms = [self.cen_atom]

        self.fixcom = bool(int(os.environ.get("fixcom", 1)))
        self.dyn_mode = os.environ.get("dyn_mode", 'nvt')
        self.temp_ensem = bool(int(os.environ.get("temp_ensem", 0)))
        if self.dyn_mode.lower() in ['nvt', 'npt']:
            self.temp_ensem = True
        self.therm_mode = os.environ.get("therm_mode", 'pile_g')
        self.baro_mode = os.environ.get("baro_mode", 'isotropic')
        self.tau = os.environ.get("tau", 100)
        self.time_step = os.environ.get("time_step", 0.25)

    def gen_input(self):
        text = ""
        text += "<simulation verbosity='%s'>\n"%self.verbose
        #Set up output
        text += "\t<output prefix='sim_%s'>\n"%self.dyn_mode
        self.output_opt = self.output_opt.split()
        if 'prop' in self.output_opt:
            text += "\t\t<properties stride='%s' filename='out'>  [step, time{%s}, conserved{%s}, temperature{%s}, kinetic_md{%s}, potential{%s}] </properties>\n"%(self.stride, self.time_unit, self.temp_unit, self.temp_unit, self.potential_unit, self.potential_unit)
        if 'pos' in self.output_opt:
            text += "\t\t<trajectory filename='pos' stride='%s' format='xyz' cell_units='%s'> positions{%s} </trajectory>\n"%(self.stride, self.cell_units, self.cell_units)
        if 'force' in self.output_opt:
            text += "\t\t<trajectory filename='force' stride='%s' format='xyz' cell_units='%s'> forces{%s} </trajectory>\n\t</output>\n"%(self.stride, self.cell_units, self.force_unit)
        if 'vel' in self.output_opt:
            text += "\t\t<trajectory filename='vel' stride='%s' format='xyz' cell_units='%s'> velocities </trajectory>\n"%(self.stride, self.cell_units)
        if 'chk' in self.output_opt:
            text += "\t\t<checkpoint filename='chk' stride='100' overwrite='True'/>\n\t</output>\n"#%self.stride
        #Set up steps
        text += "\t<total_steps> %s </total_steps>\n"%self.total_steps
        #Set up prng
        text += "\t<prng>\n\t\t<seed> %s </seed>\n\t</prng>\n"%self.seed
        #Set up socket
        text += "\t<ffsocket mode='inet' name='driver' pbc='False'>\n\t\t<address>localhost</address>\n\t\t<port> %s </port>\n\t</ffsocket>\n"%self.port
        #Set up system
        text += "\t<system>\n"
        #Set up initialize
        text += "\t\t<initialize nbeads='%s'>\n"%self.nbeads
        if self.chkfile is not None:
            text += "\t\t\t<file mode='chk'> %s </file>\n\t\t</initialize>\n"%(self.chkfile)
        else:
            text += "\t\t\t<file mode='xyz' units='%s'> %s </file>\n"%(self.cell_units, self.xyz_name)
            text += "\t\t\t<cell mode='abc' units='%s'> [ %.1f, %.1f, %.1f ] </cell>\n"%(self.cell_units, self.len_side, self.len_side, self.len_side)
            #if self.dyn_mode == 'nvt' or self.dyn_mode == 'npt':
            text += "\t\t\t<velocities mode='thermal' units='%s'> %s </velocities>\n"%(self.temp_unit, self.temperature)
            text += "\t\t</initialize>\n"
        #Set up forces
        text += "\t\t<forces>\n\t\t\t<force forcefield='driver'/>\n\t\t</forces>\n"
        #Set up ensemble
        if self.temp_ensem:
            text += "\t\t<ensemble>\n\t\t\t<temperature units='%s'> %s </temperature>\n"%(self.temp_unit, self.temperature)
            if self.dyn_mode == 'npt':
                text += "\t\t\t<pressure units='%s'> %s </pressure>\n"%(self.press_unit, self.pressure)
            text += "\t\t</ensemble>\n"
        #Set up motion
        text += "\t\t<motion mode='dynamics'>\n"

        if self.fixatoms is not None:
            text += "\t\t\t<fixatoms> %s </fixatoms>\n"%(self.fixatoms)
            text += "\t\t\t<fixcom> False </fixcom>\n"
        else:
            text += "\t\t\t<fixcom> %s </fixcom>\n"%(self.fixcom)
        text += "\t\t\t<dynamics mode='%s'>\n"%self.dyn_mode
        if self.dyn_mode == 'npt':
            text += "\t\t\t\t<barostat mode='%s'>\n\t\t\t\t\t<tau units='%s'> %s </tau>\n"%(self.baro_mode, self.time_unit, self.tau)
            text += "\t\t\t\t\t<thermostat mode='%s'>\n\t\t\t\t\t\t<tau units='%s'> %s </tau>\n\t\t\t\t\t</thermostat>\n\t\t\t\t</barostat>\n"%(self.therm_mode, self.time_unit, self.tau)
        if self.dyn_mode == 'nvt' or self.dyn_mode == 'npt':
            text += "\t\t\t\t<thermostat mode='%s'>\n\t\t\t\t\t<tau units='%s'> %s </tau>\n\t\t\t\t</thermostat>\n"%(self.therm_mode, self.time_unit, self.tau)
        text += "\t\t\t\t<timestep units='%s'> %s </timestep>\n"%(self.time_unit, self.time_step)
        text += "\t\t\t</dynamics>\n\t\t</motion>\n\t</system>\n</simulation>"
        with open('input.xml', 'w') as f:
            f.write(text)
#Generate input
if iproc == 0:
    md_para = md_parameters()
    md_para.gen_input()
comm.Barrier()

