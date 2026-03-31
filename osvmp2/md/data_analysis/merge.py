import numpy as np
import os, sys

#dir_list = ['part1', 'part2']#, 'part3']
dir_list = ['por_1e-4_nvt_1','por_1e-4_nvt_2', 'por_1e-4_nvt_3']
slice_list = [6300, 6600, 10000]
opt = 1
if opt == 0:
    traj_list = []
    for dir_i in dir_list:
        with open('%s/sim_nvt.pos_0.xyz'%dir_i, 'r') as f:
            traj_list.extend(f.readlines())
    with open('sim_nvt.pos_0.xyz', 'w') as f:
        f.write(''.join(traj_list))
else:
    dir_col = 'por_nvt_mp2'
    try:
        os.mkdir(dir_col)
    except FileExistsError: pass
    #fil = 'dip_mom_porphycene_relaxed_MP2.dat'
    fil = 'dip_mom_porphycene_unrelaxed_MP2.dat'
    file_list = []
    for f in os.listdir('por_1e-4_nvt_1'):
        if '_0' in f or 'dip' in f or 'sim_nvt' in f:
            file_list.append(f)
    print(file_list)
    idx0 = 0
    for fi in file_list:
        if '_0' in fi: 
            delta_l = 40
        else: 
            delta_l = 1
        out_list = []
        for idx, dir_i in enumerate(dir_list):
            print('%s/%s'%(dir_i, fi))
            '''with open('%s/%s'%(dir_i, fi), 'r') as f:
                out_list.extend(f.readlines()[idx0:idx0+slice_list[idx]*delta_l])'''
            with open('%s/%s'%(dir_i, fi), 'r') as f:
                out_list.extend(f.readlines())
        with open('%s/%s'%(dir_col, fi), 'w') as f:
            f.write(''.join(out_list))
