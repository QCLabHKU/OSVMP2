import sys, os
import numpy as np
import matplotlib.pyplot as plt

calc_por = False
if calc_por:
    for ensem in ['nve']:#, 'nvt']:
        fname = 'por/gofr_NN_%s.dat'%ensem
        #fname = "por/nothermo/gofr_NN.dat"
        mol = 'por'
        bond = 'NN'
        dat_i = np.genfromtxt(fname)
        plt.figure()
        ax = plt.subplot(111)
        plt.plot(dat_i[:,0], dat_i[:,1], linewidth=1, color='b')
        plt.axvline(x=2.63, color='r', linewidth=1.5,linestyle='--', label='$cis$-isomer')
        plt.axvline(x=2.68, color='r', linewidth=1.5, label='$trans$-isomer')
        plt.xlim(2, 4.5)
        plt.xticks(np.arange(2, 4.6, step=0.5))
        plt.ylim(0, max(dat_i[:,1]))
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #fig.subplots_adjust(hspace=0)
        plt.legend()
        # Remove horizontal space between axes

        plt.xlabel("%s distance [Angstrom]"%(bond[0]+'-'+bond[1]),fontsize=11)
        #plt.show()
        plt.savefig('/Users/calvin/Dropbox/code/MPI/work/results/figs/rdf_%s_%s_%s.eps'%(mol, bond, ensem))
else:
    '''data_dic = {'Normal OSV-MP2':0,
                'MBE(3)-OSV-MP2':0,
                'c-MBE(3)-OSV-MP2':0,
                'g-MBE(3)-OSV-MP2':0}
    dir_list = os.listdir(os.getcwd())'''
    data_dic = {'Normal OSV-MP2':0,
                 'MBE(3)-OSV-MP2':0}
    dir_list = ['mbe_conv', 'mbe_opt']
    for mol in ['eigen', 'zundel']:
        for bond in ['OH', 'OO']:
            #data_list = []
            max_d = []
            for dir_i in dir_list:
                if 'mbe' in dir_i:
                    fname = '%s/%s_1e-2_0.2/gofr_%s.dat'%(dir_i, mol, bond)
                    dat_i = np.genfromtxt(fname)
                    #data_list.append([dir_i, dat_i])
                    if 'conv' in dir_i:
                        data_dic['Normal OSV-MP2'] = dat_i
                    else:
                        data_dic['MBE(3)-OSV-MP2'] = dat_i
                    '''elif 'no' in dir_i:
                        data_dic['MBE(3)-OSV-MP2'] = dat_i
                    elif 'csg' in dir_i:
                        data_dic['c-MBE(3)-OSV-MP2'] = dat_i
                    else:
                        data_dic['g-MBE(3)-OSV-MP2'] = dat_i'''
                    max_d.append(max(dat_i[:,1]))
            max_d = max(max_d)
            plot_opt=1
            #color_list = ['b', 'g', 'r', 'c']#'tab:orange']
            color_list = ['b', 'r']
            if plot_opt == 0:
                fig, axs = plt.subplots(len(data_list), 1, sharex=True)
                #for idx, (mode, dat_i) in enumerate(data_list):
                idx = 0
                for mode in data_dic.keys():
                    dat_i = data_dic[mode]
                    axs[idx].plot(dat_i[:,0], dat_i[:,1], linewidth=1, color=color_list[idx], label=mode)
                    axs[idx].set_ylim(0, max_d)
                    axs[idx].set_yticks([])
                    if bond == 'OO':
                        axs[idx].set_xlim(2, 5)
                        axs[idx].set_xticks(np.arange(2, 5.1, step=0.5))
                    elif bond == 'OH':
                        axs[idx].set_xlim(0, 4)
                        axs[idx].set_xticks(np.arange(4.1, step=0.5))
                    axs[idx].spines['right'].set_visible(False)
                    axs[idx].spines['top'].set_visible(False)
                    axs[idx].spines['left'].set_visible(False)
                    axs[idx].legend()
                    idx += 1
                fig.subplots_adjust(hspace=0)
            else:
                plt.figure()
                ax = plt.subplot(111)
                #or idx, (mode, dat_i) in enumerate(data_list):
                idx = 0
                for mode in list(data_dic.keys()):
                    dat_i = data_dic[mode]
                    plt.plot(dat_i[:,0], dat_i[:,1], linewidth=1, color=color_list[idx], label=mode)
                    if bond == 'OH':
                        if mol == 'eigen':
                            idx_weak = tuple([dat_i[:,0]>1.3])
                            times = 3
                        else:
                            idx_weak = tuple([dat_i[:,0]>1.05])
                            times = 5
                        plt.plot(dat_i[:,0][idx_weak], 3+times*dat_i[:,1][idx_weak], linewidth=1, color='black')
                    idx += 1
                if bond == 'OO':
                    plt.xlim(2, 5)
                    plt.xticks(np.arange(2, 5.1, step=0.5))
                elif bond == 'OH':
                    plt.xlim(0, 4)
                    plt.xticks(np.arange(4.1, step=0.5))
                plt.ylim(0, max_d)
                ax.set_yticks([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                #fig.subplots_adjust(hspace=0)
                plt.legend()
            # Remove horizontal space between axes

            plt.xlabel("%s distance [Angstrom]"%(bond[0]+'-'+bond[1]),fontsize=11)
            #plt.show()
            plt.savefig('/Users/calvin/Dropbox/code/MPI/work/results/figs/rdf_%s_%s.eps'%(mol, bond))
