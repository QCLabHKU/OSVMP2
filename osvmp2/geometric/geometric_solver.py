#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Interface to geomeTRIC library https://github.com/leeping/geomeTRIC
'''
import sys
import tempfile
import numpy
#from geometric import molecule
from pyscf import lib
#from addons import (as_pyscf_method, dump_mol_geometry,
                                  #symmetrize)
from pyscf.geomopt.addons import dump_mol_geometry, symmetrize
from osvmp2 import geometric
#import osvmp2.geometric.molecule
from osvmp2.geometric import __config__

try:
    from osvmp2.geometric import internal, optimize, nifty, engine, molecule
except ImportError:
    msg = ('Geometry optimizer geomeTRIC not found.\ngeomeTRIC library '
           'can be found on github https://github.com/leeping/geomeTRIC.\n'
           'You can install geomeTRIC with "pip install geometric"')
    raise ImportError(msg)
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()   # Size of communicator
iproc = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs

# Overwrite units defined in geomeTRIC
internal.ang2bohr = optimize.ang2bohr = nifty.ang2bohr = 1./lib.param.BOHR
engine.bohr2ang = internal.bohr2ang = molecule.bohr2ang = nifty.bohr2ang = \
        optimize.bohr2ang = lib.param.BOHR
del(internal, optimize, nifty, engine, molecule)


INCLUDE_GHOST = getattr(__config__, 'geomopt_berny_solver_optimize_include_ghost', True)
ASSERT_CONV = getattr(__config__, 'geomopt_berny_solver_optimize_assert_convergence', True)

def set_geom_(mol, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        PTR_COORD  = 1
        save_unit = unit
        '''Update geometry
        '''
        import copy
        if inplace:
            mol = mol
        else:
            mol = copy.copy(mol)
            mol._env = mol._env.copy()
        if unit is None:
            unit = mol.unit
        else:
            mol.unit = unit
        if symmetry is None:
            symmetry = mol.symmetry

        if isinstance(atoms_or_coords, numpy.ndarray):
            mol.atom = list(zip([x[0] for x in mol._atom],
                                atoms_or_coords.tolist()))
        else:
            mol.atom = atoms_or_coords

        if isinstance(atoms_or_coords, numpy.ndarray) and not symmetry:
            mol._atom = mol.atom
            #if sys.version_info[0] >= 3:
            unicode = str
            if isinstance(unit, (str, unicode)):
                if unit.upper().startswith(('B', 'AU')):
                    unit = 1.
                else: #unit[:3].upper() == 'ANG':
                    unit = 1./param.BOHR
            else:
                unit = 1./unit
            ptr = mol._atm[:,PTR_COORD]
            mol._env[ptr+0] = unit * atoms_or_coords[:,0]
            mol._env[ptr+1] = unit * atoms_or_coords[:,1]
            mol._env[ptr+2] = unit * atoms_or_coords[:,2]
        else:
            mol.symmetry = symmetry
            mol.build(False, False)
        if iproc == 0:
            if mol.verbose >= lib.logger.INFO:
                if save_unit == "Bohr":
                    lib.logger.info(mol, '\nNew geometry (Bohrs)')#%s', unit)
                elif save_unit == "Angstrom":
                    lib.logger.info(mol, '\nNew geometry (Angstrom)')#%s', unit)
                else:
                    lib.logger.info(mol, '\nNew geometry %s', unit)

                
                coords = mol.atom_coords()
                for ia in range(mol.natm):
                    lib.logger.note(mol, ' %3d %-4s %16.12f %16.12f %16.12f',
                                ia+1, mol.atom_symbol(ia), *coords[ia])
        return mol

def gen_coords(mol, log):
    atm_list = []
    for atm in range(mol.natm):
        atm_list.append(mol.atom_pure_symbol(atm))
    xyz_list = mol.atom_coords()*lib.param.BOHR
    a = 0
    xyz = []
    for i in xyz_list:
       xyz_for = []
       for j in i.tolist():
           j = format(j, '.9f')
           xyz_for.append(j)
       xyz.append(xyz_for)
       a += 1
    for i in range(len(atm_list)):
        xyz[i].insert(0,atm_list[i])

    lens = []
    for column in zip(*xyz):
        lens.append(max([len(x) for x in column]))
    xyz_for = "  ".join(["{:<" + str(x) + "}" for x in lens])
    for row_i in xyz:
        log.info(xyz_for.format(*row_i))
    return xyz

class OSVMP2Engine(geometric.engine.Engine):
    def __init__(self, scanner):
        molecule = geometric.molecule.Molecule()
        self.mol = mol = scanner.mol
        molecule.elem = [mol.atom_symbol(i) for i in range(mol.natm)]
        # Molecule is the geometry parser for a bunch of formats which use
        # Angstrom for Cartesian coordinates by default.
        molecule.xyzs = [mol.atom_coords()*lib.param.BOHR]  # In Angstrom
        super(OSVMP2Engine, self).__init__(molecule)
        self.verbose = scanner.verbose
        self.scanner = scanner
        self.cycle = 0
        self.callback = None
        self.maxsteps = 100
        self.assert_convergence = False

    def calc_new(self, coords, dirname):
        if self.cycle >= self.maxsteps:
            raise NotConvergedError('Geometry optimization is not converged in '
                                    '%d iterations' % self.maxsteps)

        g_scanner = self.scanner
        if g_scanner.verbose is None:
            g_scanner.verbose = 4
        mol = self.mol
        log = lib.logger.Logger(sys.stdout, self.verbose)
        if iproc ==0:
            msg = '\n\n\n     *************************************************************'
            msg += '\n     *                Geometry optimization step %d               *'%self.cycle
            msg += '\n     *************************************************************\n'
            log.info(msg)
        else:
            coords = None
        coords = comm.bcast(coords,root=0)
        self.cycle += 1

        # geomeTRIC requires coords and gradients in atomic unit
        coords = coords.reshape(-1,3)
        if g_scanner.verbose >= lib.logger.NOTE:
            dump_mol_geometry(mol, coords*lib.param.BOHR)

        if mol.symmetry:
            coords = symmetrize(mol, coords)

        mol = set_geom_(mol,coords, unit='Bohr')
        energy, gradients = g_scanner.osvgrad(mol)

        if callable(self.callback):
            self.callback(locals())

        if self.assert_convergence and not g_scanner.converged:
            raise RuntimeError('Nuclear gradients of %s not converged' % g_scanner.base)
        if iproc ==0:
            log.info("\nCoordinate at step %d:" %(self.cycle-1))
            gen_coords(mol, log)
        return {"energy": energy, "gradient": gradients.ravel()}

def kernel(method, assert_convergence=ASSERT_CONV,
           include_ghost=INCLUDE_GHOST, constraints=None, callback=None,
           maxsteps=100, **kwargs):
    '''Optimize geometry with geomeTRIC library for the given method.
    
    To adjust the convergence threshold, parameters can be set in kwargs as
    below:

    .. code-block:: python
        conv_params = {  # They are default settings
            'convergence_energy': 1e-6,  # Eh
            'convergence_grms': 3e-4,    # Eh/Bohr
            'convergence_gmax': 4.5e-4,  # Eh/Bohr
            'convergence_drms': 1.2e-3,  # Angstrom
            'convergence_dmax': 1.8e-3,  # Angstrom
        }
        from pyscf import geometric_solver
        opt = geometric_solver.GeometryOptimizer(method)
        opt.params = conv_params
        opt.kernel()
    '''
    g_scanner = method
    tmpf = tempfile.mktemp(dir=lib.param.TMPDIR)
    engine = OSVMP2Engine(g_scanner)
    engine.callback = callback
    engine.maxsteps = maxsteps
    # To avoid overwritting method.mol
    engine.mol = g_scanner.mol.copy()

    # When symmetry is enabled, the molecule may be shifted or rotated to make
    # the z-axis be the main axis. The transformation can cause inconsistency
    # between the optimization steps. The transformation is muted by setting
    # an explict point group to the keyword mol.symmetry (see symmetry
    # detection code in Mole.build function).
    if engine.mol.symmetry:
        engine.mol.symmetry = engine.mol.topgroup

    engine.assert_convergence = assert_convergence
    try:
        m = geometric.optimize.run_optimizer(customengine=engine, input=tmpf,
                                             constraints=constraints, **kwargs)
        conv = True
        # method.mol.set_geom_(m.xyzs[-1], unit='Angstrom')
    except NotConvergedError as e:
        lib.logger.note(method, str(e))
        conv = False
    return conv, engine.mol

def optimize(method, assert_convergence=ASSERT_CONV,
             include_ghost=INCLUDE_GHOST, constraints=None, callback=None,
             maxsteps=100, **kwargs):
    '''Optimize geometry with geomeTRIC library for the given method.
    
    To adjust the convergence threshold, parameters can be set in kwargs as
    below:

    .. code-block:: python
        conv_params = {  # They are default settings
            'convergence_energy': 1e-6,  # Eh
            'convergence_grms': 3e-4,    # Eh/Bohr
            'convergence_gmax': 4.5e-4,  # Eh/Bohr
            'convergence_drms': 1.2e-3,  # Angstrom
            'convergence_dmax': 1.8e-3,  # Angstrom
        }
        from pyscf import geometric_solver
        newmol = geometric_solver.optimize(method, **conv_params)
    '''
    return kernel(method, assert_convergence, include_ghost, callback,
                  maxsteps, **kwargs)[1]

class GeometryOptimizer():
    '''Optimize the molecular geometry for the input method.

    Note the method.mol will be changed after calling .kernel() method.
    '''
    def __init__(self, method):
        self.method = method
        self.callback = None
        self.params = {}
        self.converged = False
        self.max_cycle = 100

    @property
    def mol(self):
        return self.method.mol
    @mol.setter
    def mol(self, x):
        self.method.mol = x

    def kernel(self):
        self.converged, self.mol = \
                kernel(self.method, callback=self.callback,
                       maxsteps=self.max_cycle, **self.params)
        return self.mol
    optimize = kernel

class NotConvergedError(RuntimeError):
    pass

del(INCLUDE_GHOST, ASSERT_CONV)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf, dft, cc, mp
    mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
                ''',
                basis='3-21g')

    mf = scf.RHF(mol)
    conv_params = {
        'convergence_energy': 1e-4,  # Eh
        'convergence_grms': 3e-3,    # Eh/Bohr
        'convergence_gmax': 4.5e-3,  # Eh/Bohr
        'convergence_drms': 1.2e-2,  # Angstrom
        'convergence_dmax': 1.8e-2,  # Angstrom
    }
    opt = GeometryOptimizer(mf).set(params=conv_params)#.run()
    opt.max_cycle=1
    opt.run()
    mol1 = opt.mol
    log.info(mf.kernel() - -153.219208484874)
    log.info(scf.RHF(mol1).kernel() - -153.222680852335)

    mf = dft.RKS(mol)
    mf.xc = 'pbe,'
    mf.conv_tol = 1e-7
    mol1 = optimize(mf)

    mymp2 = mp.MP2(scf.RHF(mol))
    mol1 = optimize(mymp2)

    mycc = cc.CCSD(scf.RHF(mol))
    mol1 = optimize(mycc)
