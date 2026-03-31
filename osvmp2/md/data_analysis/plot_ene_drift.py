import sys, os
import numpy as np
import matplotlib.pyplot as plt


def get_temp(e_k, natom):
    #T = e_k*2/(N_dof*kb)
    #N_dof = 3natom - Nc - Ncomm: degree of freedom of the system
    #Boltzmann constant in eV/K: 8.617333262e-5 eV/K
    '''if fix_atom:
        N_dof = 3*(natom-1)
    elif fix_com:
        N_dof = 3*natom - 3
    else:
        N_dof = 3*natom'''
    N_dof = 3*(natom-1)
    kb = 8.617333262e-5
    return e_k*2/(N_dof*kb)
def get_dat(ene_list):
    
    #Get drift
    x = np.arange(len(ene_list))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, ene_list, rcond=None)[0]
    e0 = m*x[0]+c
    e1 = m*x[-1]+c
    e_drift = (e1-e0)
    #Get RMSD
    e_d = [(abs(ene_list[idx]) - abs(m*x[idx]+c)) for idx in range(len(ene_list))]
    #print(e_d)
    e_rmsd = np.sqrt(np.mean((np.asarray(e_d))**2))
    return e_rmsd, e_drift

#output = sys.argv[1]
mol = 'eigen'; natom = 13
#mol = 'zundel'; natom = 19
dir_list = os.listdir(os.getcwd())
mode_list = [dir_i for dir_i in dir_list if 'mbe' in dir_i]
for mol in ['eigen', 'zundel']:
    rmsd_list = []
    drift_list = []
    for dir_i in mode_list:
            output = '%s/%s_1e-2_0.2/sim_nve.out'%(dir_i, mol)
            fix_com = True
            temp_list = []
            ene_kin = []
            ene_list = []
            with open(output, 'r') as f:
                lines = f.readlines()
                for l in lines[6:]:
                    #temp_list.append(float(l.split()[3]))
                    ene_kin.append(float(l.split()[-2]))
                    ene_list.append(96.487 * (float(l.split()[-1]) + float(l.split()[-2])))
            ene_list = np.asarray(ene_list) 
            e_rmsd, e_drift = get_dat(ene_list)
            rmsd_list.append(e_rmsd)
            drift_list.append(e_drift)
            '''for e_k in ene_kin:
                temp_i = get_temp(e_k, natom)
                temp_list.append(temp_i)'''
            '''print(dir_i)
            print("Average temperature: %.2f K"%np.mean(temp_list))
            e_rmsd, e_drift = get_dat(ene_list)
            print("Energy RMSD: %.4f kj/mol"%e_rmsd)
            print("Energy drift: %.4f kj/mol"%e_drift)'''
            '''plt.plot(np.arange(len(ene_list)), ene_list-min(ene_list),  label='Original data')
            #plt.ylim(0,5)
            plt.legend()
            plt.show()'''
    print(mol)
    for data in zip(mode_list, rmsd_list, drift_list):
        print(*data)
