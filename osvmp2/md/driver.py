import os
import sys
import socket
import struct
import time
import shutil
import numpy as np
import errno
import psutil
from pyscf import gto, lib
from osvmp2.__config__ import inputs
from osvmp2.grad_addons import gradient
from osvmp2.mm import qmmm
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nproc = comm.Get_size()   # Size of communicator
iproc = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm


# CONSTANTS
BOHR2M = 5.291772108e-11  # BOHR2M -> m
ANG2M = 1e-10  # ANG2M -> m
AMU = 1.660539040e-27  # amu -> kg
FEMTO = 1e-15
PICO = 1e-12
EH2J = 4.35974417e-18  # Hartrees -> J
EV = 1.6021766209e-19  # eV -> J
H = 6.626069934e-34  # Planck const
KB = 1.38064852e-23  # Boltzmann const
MOLE = 6.02214129e23
KJ = 1000.0
KCAL = 4184.0

# HEADERS
if sys.version[0] == '2':
    STATUS = "STATUS      "
    NEEDINIT = "NEEDINIT    "
    READY = "READY       "
    HAVEDATA = "HAVEDATA    "
    FORCEREADY = "FORCEREADY  "
else:
    STATUS = b"STATUS      "
    NEEDINIT = b"NEEDINIT    "
    READY = b"READY       "
    HAVEDATA = b"HAVEDATA    "
    FORCEREADY = b"FORCEREADY  "
# BYTES
INT = 4
FLOAT = 8

def soc_send(soc, data):
    step = 0
    while True:
        step += 1
        if step > 100:
            break
        try:
            soc.send(data)
            break
        except socket.error as e:
            if isinstance(e.args, tuple):
                print("Errno is %s"%e.args[1])
                if e.args[0] == errno.EPIPE:
                    # remote peer disconnected
                    print("Detected remote disconnect")
                else:
                    # determine and handle different error
                    pass
            else:
                print("socket error ", e)
                break
        except IOError as e:
            # Hmmm, Can IOError actually be raised by the socket module?
            print("Got IOError: ", e)
            break

class ExitSignal(BaseException):
    pass


class TimeOutSignal(BaseException):
    pass

class BaseDriver(object):
    """
    Base class of Socket driver.
    """

    def __init__(self, port, addr="127.0.0.1"):
        if iproc == 0:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            '''if sys.version[0]=='2':
                try:
                    self.socket.connect((addr, port))
                    self.socket.settimeout(None)
                except (socket.timeout, ConnectionRefusedError), e:
                    raise TimeOutSignal("Time out, quit.")
            else:'''
            try:
                self.socket.connect((addr, port))
                self.socket.settimeout(None)
            except ConnectionRefusedError:
                #raise ExitSignal("Lost connection")
                sys.exit()
            except socket.timeout:
                raise TimeOutSignal("Time out, quit.")
        self.job_now = 0
        self.job_next = 0
        self.ifInit = False
        self.ifForce = False
        self.cell = None
        self.inverse = None
        self.crd = None
        self.energy = None
        self.force = None
        if sys.version[0]=='2':
            self.extra = ""
        else:
            self.extra = b""
        self.nbead = -1
        self.natom = -1

    def grad(self, crd):
        """
        Calculate gradient.
        Need to be rewritten in inheritance.
        """
        return None, None

    def update(self, text):
        """
        Update system message from INIT motion.
        Need to be rewritten in inheritance.
        Mostly we don't need it.
        """
        pass

    def init(self):
        """
        Deal with message from INIT motion.
        """
        if iproc == 0:
            self.nbead = np.frombuffer(
                self.socket.recv(INT * 1), dtype=np.int32)[0]
            offset = np.frombuffer(self.socket.recv(INT * 1), dtype=np.int32)[0]
            self.update(self.socket.recv(offset))
            self.ifInit = True

    def status(self):
        """
        Reply STATUS.
        """
        if iproc == 0:
            if self.ifInit and not self.ifForce:
                soc_send(self.socket, READY)
            elif self.ifForce:
                soc_send(self.socket, HAVEDATA)
            else:
                soc_send(self.socket, NEEDINIT)

    def posdata(self):
        """
        Read position data.
        """
        if iproc == 0:
            self.cell = np.frombuffer(self.socket.recv(
                FLOAT * 9), dtype=np.float64) * BOHR2M
            self.inverse = np.frombuffer(self.socket.recv(
                FLOAT * 9), dtype=np.float64) / BOHR2M
            self.natom = np.frombuffer(
                self.socket.recv(INT * 1), dtype=np.int32)[0]
            message = bytearray()
            size = FLOAT * 3 * self.natom
            ite = 1
            t0 = time.time()
            len_list = []
            while len(message) < size:
                buffer = self.socket.recv(size-len(message))
                '''if not buffer:
                    raise EOFError("Could not receive all messeage!!")'''
                len_list.append(len(buffer))
                message.extend(buffer)
                ite += 1
                if ite == 1000:
                    os.system("kill %s"%os.getpgid())
            with open("recvtime.log", "a") as f:
                f.write("%.5f   %d   %s\n"%((time.time()-t0), size, len_list))
            crd = np.frombuffer(bytes(message), dtype=np.float64)
        else:
            crd = None
        crd = comm.bcast(crd, root=0)
        self.crd = crd.reshape((self.natom, 3)) * BOHR2M
        energy, force = self.grad(self.crd)
        self.energy = energy
        self.force = -force
        self.ifForce = True

    def getforce(self):
        """
        Reply GETFORCE.
        """
        if iproc == 0:
            soc_send(self.socket, FORCEREADY)
            soc_send(self.socket, struct.pack("d", self.energy / EH2J))
            soc_send(self.socket, struct.pack("i", self.natom))
            for f in self.force.ravel():
                soc_send(self.socket, struct.pack("d", f / (EH2J / BOHR2M))
                                )  # Force unit: xx
            virial = np.diag((self.force * self.crd).sum(axis=0)).ravel() / EH2J
            for v in virial:
                soc_send(self.socket, struct.pack("d", v))
            if len(self.extra) > 0:
                extra = self.extra  
            elif sys.version[0] == '2':
                extra = " "
            else:
                extra = b" "
            lextra = len(extra)
            soc_send(self.socket, struct.pack("i", lextra))
            soc_send(self.socket, extra)
            self.ifForce = False

    def exit(self):
        """
        Exit.
        """
        if iproc == 0:
            self.socket.close()
        raise ExitSignal()

    def parse(self):
        """
        Reply the request from server.
        """
        if iproc == 0:
            try:
                self.socket.settimeout(10)
                header = self.socket.recv(12).strip()
                if sys.version[0] == '3':
                    header = header.decode()
                self.socket.settimeout(None)
            except socket.timeout as e:
                raise TimeOutSignal("Time out, quit.")
            if len(header) < 2:
                raise TimeOutSignal()
        else:
            header = None
        header = comm.bcast(header, root=0)
        if header == "STATUS":
            self.status()
        elif header == "INIT":
            self.init()
        elif header == "POSDATA":
            self.posdata()
        elif header == "GETFORCE":
            self.getforce()
        elif header == "EXIT":
            self.exit()

def data2file(fname, data):
    try:
        with open(fname, 'a') as f:
            f.write(data)
    except IOError:
        with open(fname, 'w') as f:
            f.write(data)

def strl2list(strl, dtype='i'):
    if strl is None:
        return None
    else:
        if dtype == 'i':
            type_format = int
        else:
            type_format = float
        return [type_format(i) for i in strl.replace("[","").replace("]","").split(',')]

class GaussDriver(BaseDriver):
    """
    Driver for QM calculation with Gaussian.
    """

    def __init__(self, port=None, addr=None, atoms=None, basis=None, stride=None):
        if port is not None:
            BaseDriver.__init__(self, port, addr)
        '''self.memory = int(os.environ.get("max_memory", 250000))*1e-3
        self.ncore = int(os.environ.get("OMP_NUM_THREADS", 1))
        self.func = os.environ.get("func", 'b3lyp')
        self.basis = os.environ.get("basis", "631g")
        self.charge = int(os.environ.get("charge", 0))
        self.stride = int(os.environ.get("stride", '20'))
        spin = int(os.environ.get("spin", 0))
        self.method = (os.environ.get("method", 'dft')).lower()
        self.forcefield = os.environ.get("forcefield", 'amber')'''

        self.memory     = inputs["max_memory"]
        self.ncore      = int(os.environ.get("OMP_NUM_THREADS", 1))
        self.func       = inputs["func"]
        self.basis      = inputs["basis"]
        self.charge     = inputs["charge"]
        self.stride     = inputs["stride"]
        spin            = inputs["spin"]
        self.method     = inputs["method"].lower()
        self.forcefield = inputs["forcefield"]


        self.md_step = 0
        self.t_md = 0.0
        
        self.multiplicity = 2 * spin + 1
        
        with open("dip_moment.out", 'w') as f:
            pass
        if self.method == 'dft':
            self.atoms = atoms
        elif self.method == 'ff':
            #template = os.environ.get("template", '../template.gjf')
            template = inputs["gauss_template"]
            with open(template, 'r') as f:
                lines = f.readlines()
                for idx, l in enumerate(lines):
                    if l.split() == ['%d'%self.charge, '%d'%self.multiplicity]:
                        idx1 = idx + 1
                        self.atoms = []
                        while lines[idx1] != '\n':
                            self.atoms.append(lines[idx1].split()[0])
                            idx1 += 1

                        '''idx1 = idx1 + 1
                        self.para_ff = []
                        while lines[idx1] != '\n':
                            self.para_ff.append(lines[idx1])
                            idx1 += 1'''

    def gen_inp(self, coord):
        '''
        Generate input (.com) file for Gaussian
        '''
        coord_str = ""
        for idx, (x, y, z) in enumerate(coord):
            coord_str += "%s\t%.9f\t%.9f\t%.9f\n"%(self.atoms[idx], x, y, z)
        msg = "%%NProcshared=%d\n%%mem=%dgb\n%%Chk=gauss.chk"%(self.ncore, self.memory)
        if self.method == "dft":
            msg += "\n#n %s/%s force"%(self.func, self.basis)
        elif self.method == "ff":
            msg += "\n#n %s force"%self.forcefield
            #coord_str += "\n%s".join(self.para_ff)
        msg += "\n\nTitle\n\n%d %d\n%s\n"%(self.charge, self.multiplicity, coord_str)
        
        with open('test.com', 'w') as f:
            f.write(msg)

    def readlog(self):
        """
        Get energy and force from .log file.
        """
        with open("test.log", "r") as f:
            lines = f.readlines()
            g = []
            for idx, l in enumerate(lines):
                if 'SCF Done' in l and "E" in l:
                    e = float(l.split()[4])
                elif ("Energy=" in l) and ("Dipole moment" in lines[idx+1]):
                    e = float(l.split()[1])
                elif "Forces (Hartrees/Bohr)" in l:
                    idx0 = idx + 3
                    while "-------" not in lines[idx0]:
                        g.append([float(i) for i in lines[idx0].split()[2:]])
                        idx0 += 1
                elif ("\\D" in l) or ("Dipole=" in l):
                    l_com = l + lines[idx+1]
                    l_com = l_com = l_com.replace('\n', '').replace(' ', '').split('\\')
                    for text in l_com:
                        if "Dipole=" in text:
                            text = text.replace(" ", "").replace("Dipole=", "")
                            dip_mom = [float(i) for i in text.split(',')]
        msg = "%s\n\n"%e
        for x, y, z in g:
            msg += "%.9f    %.9f    %.9f\n"%(x, y, z)
        with open("eg.log", 'w') as f:
            f.write("%s"%msg)
        return e, - np.array(g), dip_mom

    def grad(self, crd, cell=None):
        t0 = time.time()
        self.gen_inp(crd / ANG2M)
        print("Time for input %.4f"%(time.time()-t0)); t0 = time.time()
        os.system("g16 test.com")
        print("Time for g16 %.4f"%(time.time()-t0)); t0 = time.time()

        energy, grad, dip_mom = self.readlog()
        energy = energy * EH2J
        grad = grad * (EH2J / BOHR2M)
        with open("dip_moment.out", 'a') as f:
            dx, dy, dz = dip_mom
            f.write("%.9f\t%.9f\t%.9f\n"%(dx, dy, dz))
        self.t_md += time.time() - t0
        self.md_step += 1
        if self.md_step%self.stride==0:
            msg = "MD step %d: esapsed time: %.2f, average time: %.2f\n"%(self.md_step, self.t_md, self.t_md/self.md_step)
            data2file("simulation_speed.out", msg)
        print("Time for output %.4f"%(time.time()-t0)); t0 = time.time()
        return energy, grad


class OrcaDriver(BaseDriver):
    """
    Driver for MD calculation with DLPNO-MP2 on ORCA.
    """

    def __init__(self, port=None, addr=None, atoms=None, basis=None, stride=None):
        if port is not None:
            BaseDriver.__init__(self, port, addr)
        '''self.path_orca = os.environ.get("path_orca", "orca")
        self.memory = int(os.environ.get("max_memory", psutil.virtual_memory()[3]*1e-6))
        self.ncore = int(os.environ.get("ncore_orca", 1))
        self.func = os.environ.get("func", 'b3lyp')
        self.basis = os.environ.get("basis", '6-31g**').lower()
        self.charge = int(os.environ.get("charge", 0))
        self.stride = int(os.environ.get("stride", '20'))
        spin = int(os.environ.get("spin", 0))
        self.method = (os.environ.get("method", 'dft')).lower()
        self.verbose = int(os.environ.get("verbose", 4))
        self.qm_region = strl2list(os.environ.get("qm_region", None))
        self.nonwater_region = strl2list(os.environ.get("nonwater_region", self.qm_atoms))
        self.qm_center = int(os.environ.get("qm_center", 0))
        self.cg_residue = os.environ.get("cg_residue", "CG1")
        self.nwater_qm = int(os.environ.get("nwater_qm", 20))
        self.qm_atoms = strl2list(os.environ.get("qm_atoms", None))'''

        self.path_orca      = inputs["path_orca"]
        self.memory         = inputs["max_memory"]
        self.ncore          = inputs["ncore_orca"]
        self.func           = inputs["func"]
        self.basis          = inputs["basis"].lower()
        self.charge         = inputs["charge"]
        self.stride         = inputs["stride"]
        self.method         = inputs["method"].lower()
        self.verbose        = inputs["verbose"]
        self.qm_region      = inputs["qm_region"]
        self.nonwater_region= inputs["nonwater_region"]
        self.qm_center      = inputs["qm_center"]
        self.cg_residue     = inputs["cg_residue"]
        self.nwater_qm      = inputs["nwater_qm"]
        self.qm_atoms       = inputs["qm_atoms"]
        spin                = inputs["spin"]


        if self.basis == "ccpvdz":
            self.basis = "cc-pvdz"
        
        self.md_step = 0
        self.t_md = 0.0
        
        self.multiplicity = 2 * spin + 1
        
        #self.point_charge = os.environ.get("basis", '6-31g**')
        self.multiplicity = 2 * spin + 1
        
        with open("traj_qm.xyz", 'w') as f:
            pass
        self.atoms = atoms
        
        '''if self.qm_atoms is not None: #Non-qmmm case
            #QM-MM
            
            #self.qm_atoms = strl2list(os.environ.get("qm_atoms", None))'''
            


    def gen_inp(self, coord):
        '''
        Generate input (.inp) file for ORCA
        '''
        log = lib.logger.Logger(sys.stdout, self.verbose)
        self.coords_all = []
        for idx, (x, y, z) in enumerate(coord):
            self.coords_all.append((self.atoms[idx], np.asarray([x, y, z])))
        if self.qm_atoms is not None:
            self.qm_region, self.mm_region, self.qm_coords, self.mm_coords = \
                             qmmm.region_qmmm(self.coords_all, qm_region=self.qm_region,
                                              qm_atoms=self.qm_atoms, qm_center=self.qm_center, 
                                              nwater_qm=self.nwater_qm)
            GRAD_MM = qmmm.mm_gradient(self.coords_all, qm_region=self.qm_region,
                                           nonwater_region=self.nonwater_region, 
                                           cg_residue=self.cg_residue, log=log)
                #print_time(["Initialization", get_elapsed_time(tt)], log)
            self.energy_mm, self.force_mm, self.charge_mm = GRAD_MM.kernel()
            self.grad_mm = -self.force_mm.reshape(-1, 3)
            #Create charge file
            msg_chg = "%d\n"%(len(self.mm_coords))
            for ichg, (iatm, (x, y, z)) in zip(self.charge_mm, self.mm_coords):
                msg_chg += "%.2f %10.6f %10.6f %10.6f\n"%(ichg, x, y, z)
            with open("pointcharges.pc", "w") as f:
                f.write(msg_chg)
            
            #Record qm trajetory
            coord_str = "%d\n\n"%(len(self.qm_coords))
            for iatm, (x, y, z) in self.qm_coords:
                coord_str += "%s %10.6f %10.6f %10.6f\n"%(iatm, x, y, z)
            with open("traj_qm.xyz", "a") as f:
                f.write(coord_str)
        else:
            self.qm_coords = self.coords_all
        coord_str = "%d\n\n"%(len(self.qm_coords))
        for iatm, (x, y, z) in self.qm_coords:
            coord_str += "%s %10.6f %10.6f %10.6f\n"%(iatm, x, y, z)
        file_co = "mol_coords.xyz"
        with open(file_co, "w") as f:
            f.write(coord_str)

        msg = "!%s D3BJ %s opt\n\n"%(self.func, self.basis)
        if self.qm_atoms is not None:
            msg += '%pointcharges "pointcharges.pc"\n'
        msg += "%" + "geom MaxIter 1 END\n\n"
        msg += "%%PAL NPROCS %d END\n%%maxcore %d\n\n"%(self.ncore, self.memory//self.ncore)
        msg += "* xyzfile %d %d %s\n"%(self.charge, self.multiplicity, file_co)
        with open('test.inp', 'w') as f:
            f.write(msg)

    def readlog(self):
        """
        Get energy and gradient from .engrad file.
        """

        def get_gtot(qm_region, mm_region, gqm, gmm, gmm_qm):
            gtot = np.copy(gmm).reshape(-1, 3)
            gtot[qm_region] += gqm.reshape(-1,3)
            gtot[mm_region] += gmm_qm.reshape(-1,3)
            return gtot
        
        with open("test.engrad", "r") as f:
            lines = f.readlines()
            g = []
            for idx, l in enumerate(lines):
                if 'The current total energy' in l:
                    e = float(lines[idx+2])
                elif "The current gradient" in l:
                    idx0 = idx + 2
                    while "#" not in lines[idx0]:
                        g.append(float(lines[idx0]))
                        idx0 += 1
        g = np.asarray(g).reshape(-1, 3)
        if self.qm_atoms is not None:
            e_qm, g_qm = e, g
            g_mm_qm = []
            with open("test.pcgrad", "r") as f:
                lines = f.readlines()
                for l in lines[1:]:
                    g_mm_qm.append([float(i) for i in l.split()])
            g_mm_qm = np.asarray(g_mm_qm)
            e = e_qm + self.energy_mm 
            g = get_gtot(self.qm_region, self.mm_region, g_qm, self.grad_mm, g_mm_qm)
        return e, g

    def grad(self, crd, cell=None):
        self.gen_inp(crd / ANG2M)
        os.system(f"{self.path_orca} test.inp")
        energy, grad = self.readlog()
        energy = energy * EH2J
        grad = grad * (EH2J / BOHR2M)
        return energy, grad
    
class OSVMP2Driver(BaseDriver):
    """
    Driver for MD calculation with OSV-MP2.
    """
    def __init__(self, port=None, addr=None, atoms=None, basis=None, stride=None):
        if port is not None:
            BaseDriver.__init__(self, port, addr)
        self.atoms = atoms
        self.basis = basis
        '''self.path_output = os.environ.get("path_output", '.')
        self.stride = int(os.environ.get("stride", stride))'''
        self.path_output = inputs["path_output"]
        self.stride = inputs["stride"]

        self.md_step = 0
        self.t_md = 0.0
    def grad(self, crd):
        t0 = time.time()
        crd = crd/ANG2M
        atom = ""
        for i in range(len(self.atoms)):
            atom += self.atoms[i]+'\t'+str(crd[i, 0])+'\t'+str(crd[i, 1])+'\t'+str(crd[i, 2])+'\n'
        #charge = int(os.environ.get("charge", 0))
        charge = inputs["charge"]
        mol = gto.M(atom=atom, basis=self.basis, charge=charge, verbose=0)
        mol.md_step = self.md_step
        mol.stride = self.stride
        mol.path_output = self.path_output
        e, h = gradient(mol)
        self.t_md += time.time() - t0
        if iproc == 0:
            if self.md_step%self.stride==0:
                msg = "MD step %d: esapsed time: %.2f, average time: %.2f\n"%(self.md_step, self.t_md, self.t_md/(self.md_step+1))
                data2file("%s/simulation_speed.out"%mol.path_output, msg)
            elif self.md_step%(2*self.stride)==0:
                os.system("mv sim_nvt.chk sim_nvt_%d.chk"%self.md_step)
            elif self.md_step%self.stride==1 and mol.path_output != '.':
                file_list = os.listdir(os.getcwd())
                for f in file_list:
                    if ('sim' in f) or ('xml' in f):
                        shutil.copy(f, mol.path_output)
        self.md_step += 1
        return e*EH2J, h.reshape(-1,3)* (EH2J/BOHR2M)

class OpenmmDriver():
    """
    Driver for MD calculation with OpenMM.
    """
    def __init__(self, port=None, addr=None, atoms=None, basis=None, stride=None):
        if port is not None:
            BaseDriver.__init__(self, port, addr)
        self.atoms = atoms
        self.basis = basis
        '''self.path_output = os.environ.get("path_output", '.')
        self.stride = int(os.environ.get("stride", stride))
        self.nonwater_region = strl2list(os.environ.get("nonwater_region", None))
        #self.qm_atoms = strl2list(os.environ.get("qm_atoms", None))
        self.cg_residue = os.environ.get("cg_residue", "CG1")
        self.nwater_qm = int(os.environ.get("nwater_qm", 20))'''
        self.path_output = inputs["path_output"]
        self.stride = inputs["stride"]
        self.nonwater_region = inputs["nonwater_region"]
        self.qm_atoms = inputs["qm_atoms"]
        self.cg_residue = inputs["cg_residue"]
        self.nwater_qm = inputs["nwater_qm"]
        self.md_step = 0
        self.t_md = 0.0
        self.qm_region = None
        

    def grad(self, crd):
        log = lib.logger.Logger(sys.stdout, 5)
        t0 = time.time()
        crd = crd/ANG2M
        self.coords_all = []
        for idx, (x, y, z) in enumerate(crd):
            self.coords_all.append((self.atoms[idx], np.asarray([x, y, z])))
        GRAD_MM = qmmm.mm_gradient(self.coords_all, qm_region=self.qm_region,
                                        nonwater_region=self.nonwater_region, 
                                        cg_residue=self.cg_residue, log=log)
            #print_time(["Initialization", get_elapsed_time(tt)], log)
        energy_mm, force_mm, charge_mm = GRAD_MM.kernel()
        grad_mm = -force_mm

        self.t_md += time.time() - t0
        if iproc == 0:
            if self.md_step%self.stride==0:
                msg = "MD step %d: esapsed time: %.2f, average time: %.2f\n"%(self.md_step, self.t_md, self.t_md/(self.md_step+1))
                data2file("%s/simulation_speed.out"%self.path_output, msg)
            elif self.md_step%(2*self.stride)==0:
                os.system("mv sim_nvt.chk sim_nvt_%d.chk"%self.md_step)
            elif self.md_step%self.stride==1 and self.path_output != '.':
                file_list = os.listdir(os.getcwd())
                for f in file_list:
                    if ('sim' in f) or ('xml' in f):
                        shutil.copy(f, self.path_output)
        self.md_step += 1
        return energy_mm*EH2J, grad_mm.reshape(-1,3)* (EH2J/BOHR2M)


if __name__ == '__main__':
    driver = HarmonicDriver(31415, "127.0.0.1", 100.0)
    while True:
        driver.parse()
