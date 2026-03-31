import os
import sys
import numpy as np

def split_traj(f_traj, interval, dir_out):
    with open(f_traj, "r") as f:
        lines = f.readlines()
        natm = int(lines[0])
        idx0_list = np.arange(len(lines), step=interval*(natm+2))
        ndig = len(str(len(idx0_list)))
        for idx_out, idx0 in enumerate(idx0_list):
            idx1 = natm+2
            with open("%s/%s.xyz"%(dir_out, "{:0%d}"%ndig.format(idx_out)), "w") as fout:
                fout.write("".join(lines[idx0:idx0+natm+2]))

