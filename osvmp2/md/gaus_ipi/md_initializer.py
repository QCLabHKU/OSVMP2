import sys
from driver import GaussDriver, ExitSignal, TimeOutSignal
import os

def get_atomlist(xyz_name):
    atom_list = []
    with open(xyz_name, 'r') as xyz:
        lines = xyz.readlines()
        for idx, l in enumerate(lines):
            if idx > 1:
                atom_list.append(l.split()[0])
    return atom_list
xyz_name = sys.argv[1]
atom_list = get_atomlist(xyz_name)
basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()
port = int(os.environ.get("port", 31415))
driver = GaussDriver(port, "127.0.0.1", atom_list, basis)

while True:
    try:
        driver.parse()
    except ExitSignal as e:
        driver = GaussDriver(port, "127.0.0.1", atom_list, basis)
    except TimeOutSignal as e:
        print("Time out. Check whether the server is closed.")
        exit()
