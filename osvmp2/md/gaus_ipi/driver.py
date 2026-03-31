"""
---------------------------------------------------------------------
|I-PI socket client.
|
|Version: 0.1
|Program Language: Python 3.6
|Developer: Xinyan Wang
|Homepage:https://github.com/WangXinyan940/i-pi-driver
|
|Receive coordinate and send force back to i-PI server using socket.
|Read http://ipi-code.org/assets/pdf/manual.pdf for details.
---------------------------------------------------------------------
"""
import os
import socket
import struct
from tempfile import template
import numpy as np
import sys
import numpy as np
from pkg_resources import resource_filename
import time
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nproc = comm.Get_size()   # Size of communicator
iproc = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm


# CONSTANTS
BOHR = 5.291772108e-11  # Bohr -> m
ANGSTROM = 1e-10  # angstrom -> m
AMU = 1.660539040e-27  # amu -> kg
FEMTO = 1e-15
PICO = 1e-12
EH = 4.35974417e-18  # Hartrees -> J
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
                self.socket.send(READY)
            elif self.ifForce:
                self.socket.send(HAVEDATA)
            else:
                self.socket.send(NEEDINIT)

    def posdata(self):
        """
        Read position data.
        """
        if iproc == 0:
            self.cell = np.frombuffer(self.socket.recv(
                FLOAT * 9), dtype=np.float64) * BOHR
            self.inverse = np.frombuffer(self.socket.recv(
                FLOAT * 9), dtype=np.float64) / BOHR
            self.natom = np.frombuffer(
                self.socket.recv(INT * 1), dtype=np.int32)[0]
            crd = np.frombuffer(self.socket.recv(
                FLOAT * 3 * self.natom), dtype=np.float64)
        else:
            crd = None
        crd = comm.bcast(crd, root=0)
        self.crd = crd.reshape((self.natom, 3)) * BOHR
        energy, force = self.grad(self.crd)
        self.energy = energy
        self.force = -force
        self.ifForce = True

    def getforce(self):
        """
        Reply GETFORCE.
        """
        if iproc == 0:
            self.socket.send(FORCEREADY)
            self.socket.send(struct.pack("d", self.energy / EH))
            self.socket.send(struct.pack("i", self.natom))
            for f in self.force.ravel():
                self.socket.send(struct.pack("d", f / (EH / BOHR))
                                )  # Force unit: xx
            virial = np.diag((self.force * self.crd).sum(axis=0)).ravel() / EH
            for v in virial:
                self.socket.send(struct.pack("d", v))
            if len(self.extra) > 0:
                extra = self.extra  
            elif sys.version[0] == '2':
                extra = " "
            else:
                extra = b" "
            lextra = len(extra)
            self.socket.send(struct.pack("i", lextra))
            self.socket.send(extra)
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

class GaussDriver(BaseDriver):
    """
    Driver for QM calculation with Gaussian.
    """

    def __init__(self, port, addr, atoms, template=None):
        BaseDriver.__init__(self, port, addr)
        self.memory = int(os.environ.get("max_memory", 250000))*1e-3
        self.ncore = int(os.environ.get("OMP_NUM_THREADS", 1))
        self.func = os.environ.get("func", 'b3lyp')
        self.basis = os.environ.get("basis", '6-31g**')
        self.charge = int(os.environ.get("charge", 0))
        self.stride = int(os.environ.get("stride", '20'))
        spin = int(os.environ.get("spin", 0))
        self.multiplicity = 2 * spin + 1
        self.template = os.environ.get("template", None)
        if self.template is not None:
            with open(self.template) as f:
                lines = f.readlines()
                for lidx, l in enumerate(lines):
                    if "coord" in l:
                        break
                self.template = "".join(lines[:lidx])
        self.md_step = 0
        self.t_md = 0.0
        self.atoms = atoms
        with open("dip_moment.out", 'w') as f:
            pass

    def gen_inp(self, coord):
        '''
        Generate input (.com) file for Gaussian
        '''
        coord_str = ""
        for idx, (x, y, z) in enumerate(coord):
            coord_str += "%s\t%.9f\t%.9f\t%.9f\n"%(self.atoms[idx], x, y, z)
        if self.template is None:
            msg = "%%NProcshared=%d\n%%mem=%dgb\n%%Chk=gauss.chk"%(self.ncore, self.memory)
            msg += "\n#n %s/%s force"%(self.func, self.basis)
            msg += "\n\nTitle\n\n%d %d\n%s\n"%(self.charge, self.multiplicity, coord_str)
        else:
            msg = "%s%s\n"%(self.template, coord_str)
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
        self.gen_inp(crd / ANGSTROM)
        os.system("g16 test.com")
        energy, grad, dip_mom = self.readlog()
        energy = energy * EH
        grad = grad * (EH / BOHR)
        with open("dip_moment.out", 'a') as f:
            dx, dy, dz = dip_mom
            f.write("%10.9f %10.9f %10.9f\n"%(dx, dy, dz))
        self.t_md += time.time() - t0
        self.md_step += 1
        if self.md_step%self.stride==0:
            msg = "MD step %d: esapsed time: %.2f, average time: %.2f\n"%(self.md_step, self.t_md, self.t_md/self.md_step)
            data2file("simulation_speed.out", msg)
        return energy, grad

