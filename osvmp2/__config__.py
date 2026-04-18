import os
import sys
from mpi4py import MPI

def num_from_environ(val, dtype):
    if val is None:
        return None
    else:
        if dtype == "i":
            formater = int
        elif dtype == "f":
            formater = float
        else:
            raise NotImplementedError
        return formater(val)

def strl2list(strl, dtype='i'):
    if strl is None:
        return None
    else:
        if dtype == "i":
            formater = int
        elif dtype == "f":
            formater = float
        else:
            raise NotImplementedError
        return [formater(i) for i in strl.replace("[","").replace("]","").split(',')]
    
def str_to_bool(val):
    val = val.lower()
    if "true" in val:
        return True
    elif "false" in val:
        return False
    else:
        return bool(int(val))

def read_inputs():
    inputs = {
        "ecp": None,
        "use_ecp": False,
        "xyz_file": None,
        "basis": "def2svp",
        "auxbasis_hf": None,
        "auxbasis_mp2": None,
        "basis_molpro": False,
        "charge": 0,
        "spin": 0,
        "verbose": 4,
        "max_memory": -1,
        "lat_vec": None,
        "k_points": None,
        
        # Check files
        "chkfile_init": None,
        "chkfile_hf": None,
        "chkfile_loc": None,
        "chkfile_ialp_hf": None,
        "chkfile_ialp_mp2": None,
        "chkfile_fitratio_hf": None,
        "chkfile_fitratio_mp2": None,
        "chkfile_ti": None,
        "chkfile_qcp": None,
        "chkfile_qmat": None,
        "chkfile_qao": None,
        "chkfile_imup": None,
        "chkfile_save": None,
        "chkfile_md": None,

        # QM/MM
        "qm_atoms": None,
        "qm_region": None,
        "nonwater_region": None,
        "qm_center": 2,
        "cg_residue": "CG1",
        "nwater_qm": 20,
        "gauss_template": "../template.gjf",

        # MD
        "func": "b3lyp",
        "stride": 20,
        "forcefield": "amber",
        "path_output": ".",
        "total_steps": 1000,
        "temperature": 298.15,
        "pressure": 1, 
        "fixatoms": None,
        "fixcom": None,
        "thermostat": "langevin",
        "barostat": None,
        "dyn_mode": "nvt",
        "time_step": 1,
        "md_driver": "osvmp2",
        "pbc_box": None,
        "ndim": 3,

        # openMM
        "fric_coeff": 1.0, # 1/ps
        "platform": "CPU",
        "main_dir": ".",
        "output_dir": ".",

        # i-pi
        "output_opt": "prop",
        "time_unit": "femtosecond",
        "temp_unit": "kelvin",
        "potential_unit": "electronvolt",
        "press_unit": "bar",
        "cell_units": "angstrom",
        "force_unit": "piconewton",
        "nbead": "1",
        "port": "31415",
        "seed": "3348",
        "tau": "100",

        # GPU
        "ngpu": 0,
        "gpu_memory": 1100,

        "omp_threads": "1",
        "opt_solver": "geometric",
        "method": "mbeosvmp2",
        "svd_method": 1,
        "cal_mode": "energy",
        "save_pene": False,
        "frozen_atoms": None,
        
        "loc_fit": True,
        "double_buffer": False,
        "int_storage": 0,
        "max_cycle": 30,
        "local_type": 1,
        "pop_method": "lowdin",
        "use_frozen": True,
        "use_sl": True,
        "pop_hf": False,
        "pop_uremp2": False,
        "pop_remp2": True,
        "charge_method": "meta_lowdin",
        "charge_method_mp2": "meta_lowdin",
        "use_cposv": True,
        "use_cpl": False,
        "nosv_ml": None,
        "nosv_id": None,

        # Thresholds
        "shell_tol": 1e-10,
        "fit_tol": 1e-6,
        "bfit_tol": 1e-2,
        "cposv_tol": 1e-10,
        "osv_tol": 1e-4,
        "threeb_tol": 0.2,
        "remo_tol": 1e-2,
        "disc_tol": 1e-7,
        "loc_tol": 1e-6,
        
        "solvent": None,
        "use_df_hf": True,
        "fully_direct": False,
    }

    def load_variable(inputs, var_name, value):
        if var_name not in inputs:
            print(var_name, inputs, inputs[var_name])
            raise ValueError(f"Variable {var_name} is not supported")
        elif var_name in ["qm_atoms", "qm_region", "nonwater_region", "frozen_atoms"]:
            inputs[var_name] = strl2list(value, dtype="i")
        elif var_name in ["pbc_box", "lat_vec", "k_points"]:
            inputs[var_name] = strl2list(value, dtype="f")
        elif var_name in ["ngpu", "verbose", "max_memory", "gpu_memory", "qm_center", "nwater_qm", 
                        "charge", "spin", "max_cycle", "local_type", "svd_method", 
                        "fully_direct", "stride", "int_storage"]:
            inputs[var_name] = int(value)
        elif var_name in ["shell_tol", "fit_tol", "bfit_tol", "cposv_tol", "osv_tol", 
                        "threeb_tol", "remo_tol", "disc_tol", "loc_tol", "temperature", 
                        "pressure", "time_step"]:
            inputs[var_name] = float(value)
        elif var_name in ["nosv_ml", "nosv_id"]:
            inputs[var_name] = num_from_environ(value, dtype="i")
        elif var_name in ["basis", "auxbasis_hf", "auxbasis_mp2"]:
            inputs[var_name] = value.replace("-", "").lower()
        elif var_name in ["use_ecp", "save_pene", "loc_fit",  "double_buffer", 
                        "pop_hf", "pop_uremp2", "pop_remp2", "use_cposv", 
                        "use_cpl", "use_frozen", "basis_molpro", "use_df_hf"]:
            inputs[var_name] = str_to_bool(value)
        else:
            inputs[var_name] = value

    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
        for l in lines:
            if "#" in l:
                l = l.split("#")[0]
            for x in l.split(";"):
                lsplit = x.replace("\n", "").replace(" ", "").split("=")
                if len(lsplit) != 2: continue
                    
                var_name, value = lsplit
                
                
                load_variable(inputs, var_name, value)
    inputs["molecule"] = inputs["xyz_file"].split("/")[-1].replace(".xyz", "")
    return inputs

inputs = read_inputs()

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()
irank = comm.Get_rank()

comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED)
irank_shm = comm_shm.rank
nrank_shm = comm_shm.size

nnode = nrank // nrank_shm
inode = irank // nrank_shm
ngpu = inputs["ngpu"]
if ngpu:
    if ngpu != nrank:
        raise ValueError(f"The number of GPUs {ngpu} is not identical to number of CPU cores {nrank}")
    import cupy
    ngpu_shm = ngpu // nnode
    nrank_per_gpu = nrank_shm // ngpu_shm
    igpu = irank // nrank_per_gpu
    igpu_shm = irank_shm // nrank_per_gpu
    irank_gpu = irank % nrank_per_gpu
    cupy.cuda.runtime.setDevice(igpu_shm)
    comm_gpu = comm_shm.Split(color=igpu_shm, key=irank_shm)
