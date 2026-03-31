import os, sys
import numpy as np
import matplotlib.pyplot as plt

def get_dat(temp_list):
    #Get drift
    x = np.arange(len(temp_list))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, temp_list, rcond=None)[0]
    t0 = m*x[0]+c
    t1 = m*x[-1]+c
    t_drift = (t1-t0)
    #Get RMSD
    t_d = [(abs(temp_list[idx]) - abs(m*x[idx]+c)) for idx in range(len(temp_list))]
    #print(t_d)
    t_rmsd = np.sqrt(np.mean((np.asarray(t_d))**2))
    return t_rmsd, t_drift

output = sys.argv[1]
temp_list = []
with open(output, 'r') as f:
    lines = f.readlines()
    for l in lines[6:]:
        temp_list.append(float(l.split()[3]))

t_rmsd, t_drift = get_dat(temp_list)
print(np.mean(temp_list), t_rmsd, t_drift)

plt.plot(np.arange(len(temp_list)), temp_list)
plt.plot(np.arange(len(temp_list)), [290]*len(temp_list), color='black')
plt.show()