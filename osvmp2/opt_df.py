import os
import sys
from osvmp2.__config__ import inputs
os.environ["OMP_NUM_THREADS"] = inputs["omp_threads"]
os.environ["VECLIB_MAXIMUM_THREADS"] = inputs["omp_threads"]
os.environ["OPENBLAS_NUM_THREADS"] = inputs["omp_threads"]
os.environ["MKL_NUM_THREADS"] = inputs["omp_threads"]
import numpy as np
from pyscf import gto, lib
from pyscf.pbc import gto as pbc_gto
from osvmp2.scanner import *
from mpi4py import MPI

# Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nnode = nrank // comm_shm.size # number of nodes


def set_params(cal_mode):
    if cal_mode == 'tightopt':
        gradientmax = 3e-4; gradientrms = 1e-4; maxsteps = 25
    elif cal_mode == 'supertightopt':
        gradientmax = 1e-4; gradientrms = 3e-5; maxsteps = 50
    elif cal_mode == 'ultratightopt':
        gradientmax = 1e-5; gradientrms = 3e-6; maxsteps = 50
    elif cal_mode == "grad":
        gradientmax = 4.5e-4; gradientrms = 3e-4; maxsteps = 1
    elif cal_mode == 'longopt':
        gradientmax = 4.5e-4; gradientrms = 3e-4; maxsteps = 500
    else:
        gradientmax = 4.5e-4; gradientrms = 3e-4; maxsteps = 25
    return gradientmax, gradientrms, maxsteps


def read_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        natom = int(lines[0])
        coord = ""
        for l in lines[2:2 + natom]:
            if ":" in l:
                lsplit = l.split()
                if lsplit[0][-1] == ":":
                    ia_sym_old = lsplit[0]
                    ia_atom = ia_sym_old.replace(":")
                    ia_sym_new = f"X:{ia_atom}"
                    l = l.replace(ia_sym_old, ia_sym_new)
            coord += l
    return natom, coord


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
        return [formater(i) for i in strl.replace("[", "").replace("]", "").split(',')]


def get_mole(mol):
    if mol is None:
        if irank == 0:
            verbose = 3
        else:
            verbose = 0

        '''
        xyz_file = sys.argv[1]
        basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()
        use_ecp = bool(int(os.environ.get("use_ecp", 0)))
        ecp = os.environ.get("ecp", None)
        basis_molpro = bool(int(os.environ.get("basis_molpro", 0)))
        lat_vec = strl2list(os.environ.get("lat_vec", None), dtype='f')  # Angstroms
        k_points = strl2list(os.environ.get("k_points", None), dtype='f')
        charge = int(os.environ.get("charge", 0))
        spin = int(os.environ.get("spin", 0))'''

        xyz_file     = inputs["xyz_file"]
        basis        = inputs["basis"].replace("-", "").lower()
        use_ecp      = inputs["use_ecp"]
        ecp          = inputs["ecp"]
        basis_molpro = inputs["basis_molpro"]
        lat_vec      = inputs["lat_vec"]  # Angstroms
        k_points     = inputs["k_points"]
        charge       = inputs["charge"]
        spin         = inputs["spin"]


        natom, coord = read_xyz(xyz_file)

        if lat_vec is not None:  # PBC
            mole = pbc_gto.Cell()
            lat_vec = np.asarray(lat_vec)
            if len(lat_vec) == 3:
                mole.a = np.diag(lat_vec)
            else:
                mole.a = lat_vec

            if k_points is None:
                k_points = np.zeros(3)
            k_points = np.asarray(k_points)

            if np.all(k_points == 0):  # Gamma points
                mole.gamma_only = True
            else:
                mole.gamma_only = False

            mole.kpts = k_points.reshape(-1, 3)
            mole.atom = coord
            mole.basis = basis
            mole.charge = charge
            mole.spin = spin
            mole.build(verbose=verbose)
            mole.pbc = True
        else:
            if basis_molpro:
                from osvmp2.get_mol_special import get_M_molpro
                mole = get_M_molpro(coord, basis, verbose)
            elif (use_ecp or "Be" in coord) and ecp is None:
                from osvmp2.get_mol_special import get_M
                mole = get_M(coord, verbose=verbose)
            else:
                mole = gto.M()
                mole.atom = coord
                mole.basis = basis
                mole.charge = charge
                mole.spin = spin
                mole.ecp = ecp
                mole.build(verbose=verbose)
            mole.pbc = False
        mole.opt_cycle = 0
    else:
        mole = mol
        mole.opt_cycle += 1
    return mole


class grad_scanner:
    def __init__(self, mol=None):
        self.mol = get_mole(mol)
        self.converged = True
        if irank == 0:
            self.verbose = inputs["verbose"]
        else:
            self.verbose = 0
        self.base = None
        self.stdout = sys.stdout

    def osvgrad(self, mol=None):
        if mol is not None:
            self.mol = mol
            self.mol.opt_cycle += 1
        e, h = gradient(self.mol)
        return e, h.reshape(-1, 3)


if __name__ == "__main__":
    #opt_solver = os.environ.get("opt_solver", "geometric")
    #cal_mode = str_letter(os.environ.get('cal_mode', 'energy'))
    opt_solver = inputs["opt_solver"]
    cal_mode = inputs["cal_mode"]
    if cal_mode in ["nve" or "nvt"]:
        cal_mode = "grad"
    if (irank == 0) and ("opt" in cal_mode):
        try:
            os.remove('hf_mat.chk')
        except OSError:
            pass

    if "opt" not in cal_mode:
        mp2 = grad_scanner()
        e, g = mp2.osvgrad()
        hartree2ev = 27.211386245988
        bohr2ang = 0.529177
        with open('energy.txt', 'w') as f:
            f.write('%.9f\n' % (e * hartree2ev))
        with open('gradient.txt', 'w') as f:
            g = g * hartree2ev / bohr2ang
            for (x, y, z) in g:
                f.write('%.9f\t%.9f\t%.9f\n' % (x, y, z))
    else:
        if irank == 0:
            # Initialize output files
            for output in ["opt_traj.xyz", "opt_traj_qm.xyz", "opt_eg.xyz"]:
                with open(output, 'w') as f:
                    pass
        if opt_solver == "berny":
            from berny import berny_solver, optimize
            gradientmax, gradientrms, max_cycle = set_params(cal_mode)
            scanner = grad_scanner()
            sol = berny_solver.GeometryOptimizer(method=scanner)
            sol.max_cycle = max_cycle
            sol.params = {
                'gradientmax': gradientmax,
                'gradientrms': gradientrms,
                'stepmax': 1.8e-3,
                'steprms': 1.2e-3,
                'trust': 0.3,
                'dihedral': True,
                'superweakdih': False,
            }
            if irank == 0:
                sol.verbose = scanner.verbose
            else:
                sol.verbose = 0

            sol.kernel()
            log = lib.logger.Logger(sys.stdout, sol.verbose)
            log.info("\nThe final coordinates:")
            log.info(get_coords_from_mol(sol.mol, coord_only=True))
        else:
            from geometric import geometric_solver, optimize
            from geometric.geometric_solver import gen_coords

            def kernel():
                gradientmax, gradientrms, max_cycle = set_params(cal_mode)
                conv_params = {
                    'convergence_energy': 5e-6,  # Eh
                    'convergence_grms': gradientrms,  # Eh/Bohr
                    'convergence_gmax': gradientmax,  # Eh/Bohr
                    'convergence_drms': 1.2e-3,  # Angstrom
                    'convergence_dmax': 1.8e-3,  # Angstrom
                }
                scanner = grad_scanner()
                sol = geometric_solver.GeometryOptimizer(method=scanner)
                sol.params = conv_params
                sol.max_cycle = max_cycle
                sol.verbose = scanner.verbose
                sol.kernel()
                log = lib.logger.Logger(sys.stdout, sol.verbose)
                if irank == 0:
                    log.info("\nThe final coordinate:")
