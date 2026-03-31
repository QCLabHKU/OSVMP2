import numpy as np
#from osvmp2.osvutil import *

def xyz_str2tuple(coords_str):
    coords_str = coords_str.split("\n")
    coords_tup = []
    for coi in coords_str:
        if len(coi) > 10:
            coi_split = coi.split()
            ia_sym = coi_split[0]
            ico = [float(i) for i in coi_split[1:]]
            coords_tup.append((ia_sym, ico))
    return coords_tup

def get_cell_lenth(xyz, times=10):
    '''atom_dist = []
    for ia, (ia_sym, ico) in enumerate(xyz):
        for (ja_sym, jco) in xyz[ia:]:
            atom_dist.append(np.linalg.norm(np.asarray(jco)-np.asarray(ico)))
    return times*max(atom_dist)'''
    coord_list = np.asarray([ico for ia_sym, ico in xyz])
    max_xyz = np.max(coord_list, axis=0)
    min_xyz = np.min(coord_list, axis=0)
    return times*np.linalg.norm(max_xyz-min_xyz)

def msg_align(msg_list, align='l', align_1=None, indent=0):
    if align_1 is None:
        align_1 = align
    align_list = []
    for align_i in [align_1, align]:
        align_format = []
        for i in list(align_i):
            if i == 'l':
                align_format.append('<')
            elif i == 'c':
                align_format.append('^')
            elif i == 'r':
                align_format.append('>')
        align_list.append(align_format)
    len_col = []
    for col_i in zip(*msg_list):
        len_col.append(max([len(str(i)) for i in col_i])+2)
    msg = ''
    for idx, msg_i in enumerate(msg_list):
        if idx == 0:
            align_i = align_list[0]
        else:
            align_i = align_list[1]
        msg += ' ' * indent
        msg += ''.join([('{:%s%d} '%(ali, li)).format(mi) for ali, li, mi in zip(align_i, len_col, msg_i)])
        if idx != len(msg_list)-1:
            msg += '\n'
    return msg

def xyz_to_pdb(coords_water, res_list=None, label_list=None, 
               coords_nonwater=None, cg_residue="CG1", 
               pdb_name="input.pdb", box_pbc=None):
    '''
    The format of xyz must be [(ia_sym, [x, y, z])]
    '''
    def label_cg(coords_cg, cg_residue):
        natm = len(coords_cg)
        label_list = [None]*natm
        index_list = list(range(natm))
        dist_array = np.zeros((natm, natm))
        c_list = []
        o_list = []
        h_list = []
        for ia, (ia_sym, ico) in enumerate(coords_cg):
            if ia_sym == "C":
                c_list.append(ia)
            elif ia_sym == "O":
                o_list.append(ia)
            elif ia_sym == "H":
                h_list.append(ia)
            for ja in index_list[ia+1:]:
                jco = coords_cg[ja][1]
                dist_array[ia, ja] = dist_array[ja, ia] = np.linalg.norm(ico-jco)
        #Label O
        oc_dist = []
        for io in o_list:
            oc_min = min([dist_array[io, ic] for ic in c_list])
            oc_dist.append([oc_min, io])
        oc_dist.sort() #0: O2, 1: O1
        label_list[oc_dist[0][1]] = "O2"
        label_list[oc_dist[1][1]] = "O1"
        
        #Label C
        o2 = oc_dist[0][1]
        oc_dist = [[dist_array[o2, ic], ic] for ic in c_list]
        oc_dist.sort()
        for idx, (idist, ic) in enumerate(oc_dist):
            label_list[ic] = "C%d"%(idx+1)

        #Label H
        if cg_residue == "CG1": #CH2OO
            for idx, ih in enumerate(h_list):
                label_list[ih] = "H%d"%(idx+1)
        else:
            c1 = oc_dist[0][1]
            ch_dist = [[dist_array[c1, ih], ih] for ih in h_list]
            ch_dist.sort()
            h1 = ch_dist[0][1]
            label_list[h1] = "H1"
            h_list.remove(h1)
            if cg_residue == "CG2": #CH3CHOO
                for idx, ih in enumerate(h_list):
                    label_list[ih] = "H%d"%(idx+2)
            elif cg_residue == "CG3": #MACRO
                for idist, ic in oc_dist[-2:]:
                    ch_dist = [[dist_array[ic, ih], ih] for ih in h_list]
                    ch_dist.sort()
                    print(np.std([idist for idist, ih in ch_dist[:3]]))
                    if np.std([idist for idist, ih in ch_dist[:3]]) < 0.1:
                        label_list[ic] = "C3"
                        hidx = 2
                        for idist, ih in ch_dist[:3]:
                            label_list[ih] = "H%d"%hidx
                            hidx += 1
                    else:
                        label_list[ic] = "C4"
                        hidx = 5
                        for idist, ih in ch_dist[:2]:
                            label_list[ih] = "H%d"%hidx
                            hidx += 1

        return label_list
    if coords_nonwater is not None:
        coords_all = list(coords_nonwater) + list(coords_water)
    else:
        coords_all = coords_water
    if box_pbc is None:
        cell_len = get_cell_lenth(coords_all)
        bx = by = bz = cell_len
    else:
        bx, by, bz = box_pbc
    msg_list = ["CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n"%(bx, by, bz)]
    msg_list.append("MODEL        0\n")
    if res_list is not None:
        ia = 1
        for ires, res_seg in enumerate(res_list):
            ia_res = 1
            for ridx, residue in enumerate(res_seg):
                ia_sym, (x, y, z) = coords_all[ia-1]
                if label_list is not None:
                    ilab = label_list[ires][ridx]
                else:
                    ilab = f"{ia_sym}{ia_res}"
                msg_list.append("ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n"%(ia, ilab, residue, ires+1, x, y, z))
                ia_res += 1
                ia += 1
            msg_list.append("TER\n")
    else:
        
        ia = 1
        ires = 1
        if coords_nonwater is not None:
            if "CG" in cg_residue:       
                label_list = label_cg(coords_nonwater, cg_residue)
            print(label_list)
            for idx, (ia_sym, (x, y, z)) in enumerate(coords_nonwater):
                msg_list.append("ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n"%(ia, label_list[idx], cg_residue, ires, x, y, z))
                ia += 1
            ires += 1
            msg_list.append("TER\n")
        #for iw, ia0 in enumerate(np.arange(natm, step=3)):
        for iw, ia0 in enumerate(np.arange(len(coords_water), step=3)):
            hcount = 1
            for ia_sym, (x, y, z) in coords_water[ia0: ia0+3]:
                if ia_sym == "H":
                    ia_sym = "H%d"%hcount
                    hcount += 1
                
                msg_list.append("ATOM  %5d %4s HOH A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n"%(ia, ia_sym, (ires+iw), x, y, z))
                ia += 1
            msg_list.append("TER\n")
    msg_list.append("END\n")
    with open(pdb_name, "w") as f:
        f.write("".join(msg_list))
    


if __name__ == '__main__':
    test_xyz = '''
O      0.000000    0.000000    0.000000
H      0.000000   -0.751841    0.568201
H     -0.000000    0.751841    0.568201
O     -2.511474    0.000000   -1.450000
H     -1.651960    0.000000   -1.063537
H     -2.374291    0.000000   -2.382362'''
    test_xyz2 = '''
C       -4.0582987569      0.3293231553     -0.1723586110                 
O       -2.8490391066     -0.1903737284     -0.6588439357                 
O       -1.9531804572      0.1608534205      0.1972543050       
H       -4.9249008272      0.1575419975     -0.8443095155                 
H       -3.9339659341      1.3804075741      0.1165361401           
O       -2.1959492499      2.8067518911     -0.6520177457                 
H       -1.4634849979      2.2150106616     -0.9454589779                 
H       -2.0559342836      2.9683775542      0.2823481764                 
O        0.6980268089      0.7800535028     -0.9061942604                 
H        1.5853481095      0.3640187515     -0.7734948940                 
H        0.1911198992      0.1902991960     -1.5060556033                 
O        0.4946134473     -1.9582752653     -0.1228953666                 
H        1.4397817034     -1.8379246049      0.1233808939                 
H        0.0227559448     -2.4015959351      0.6202917219                 
O       -2.9072700046     -2.4545751068      1.2868084918                 
H       -2.1974106899     -2.2999192851      0.6219145870                 
H       -3.0693769118     -1.5540339269      1.6608695355                                 
O       -5.7420927329      2.1449682024      1.5430301096                 
H       -4.8200274944      2.4684689911      1.6608268714                 
H       -6.0605083562      2.0868702985      2.4733417798                 
O       -6.0251833404     -1.8213966201      0.7701174145                 
H       -5.0506872929     -1.7451209736      0.6777328200                 
H       -6.1263102497     -2.7046425278      1.1983839656                 
O       -1.0758572476      1.8364175393     -3.5412511307                 
H       -0.0882759259      1.6983232715     -3.5869847895                 
H       -1.3594427614      1.8571916596     -4.4825846146                 
O       -0.7004109456     -0.5360653093     -3.6947402530                 
H        0.3014208494     -0.5465173160     -3.6800259658                 
H       -0.9358510659     -0.3360383100     -4.6116715137                 
O        0.4443180046      0.7432001124      1.6530563876                 
H        1.4320646824      0.8018832444      1.7112158155                 
H        0.1044960897      1.3345523461      2.3674182959                 
O        0.7108101323     -1.6119387051      2.6374812117                 
H        1.6365202408     -1.3523173334      2.3893566481                 
H        0.3531059238     -0.8847583126      3.2036406547                 
O       -0.5821860031     -2.7155780922     -2.1786513295                 
H        0.3399331642     -2.4411633173     -2.4701753421                 
H       -1.1816144178     -2.5434448404     -2.9245980516                 
O        0.4621423552      3.3092987653      0.8851489104                 
H        1.3632532615      2.9575011538      1.0734580715                 
H        0.0643077006      2.5764529328      0.3732093303                 
O       -0.4795766806      4.1269341973     -2.8718241572                 
H        0.3084843711      3.5674982185     -2.6421401177                 
H       -1.0378726472      4.0475274745     -2.0769782957                 
O       -2.5168103962      2.8843735626      2.5002300208                 
H       -1.5398808902      2.9363866747      2.4014079080                 
H       -2.7164015156      2.0775273080      1.9663614949                 
O       -4.0246925018      0.9025459760      3.3373730420                 
H       -3.1100679036      0.5182143650      3.3533027348                 
H       -4.3887448147      0.7955829335      4.2454240981                 
O       -4.7873585684     -1.5790769908      3.7031939183                 
H       -3.8017130449     -1.4880530091      3.7503874430                 
H       -5.0402974536     -1.1486852515      2.8490683907                 
O       -4.7740317961      3.0581354825     -1.8128403699                 
H       -3.8388867863      2.8869670823     -2.0507526744                 
H       -5.0965625022      3.4553945325     -2.6560710568                 
O       -5.2333204392      1.4392383390     -3.7188129576                 
H       -4.2374456178      1.4550776975     -3.7382837940                 
H       -5.5658658619      0.6291496259     -4.1714675669                 
O       -3.8606087970     -0.7717290723     -3.9750252567                 
H       -2.8831351863     -0.6726971098     -3.9344761555                 
H       -4.0375183274     -1.6854225564     -4.2490922818  
'''
    test_xyz = '''
He      -4.9544500000      2.8910700000      0.0000000000                 
He      -2.9506100000      2.9280200000      0.0000000000                 
He      -3.9433400000      0.6558000000      0.0000000000                 
He      -6.5354600000      0.6927500000      0.0000000000                 
He      -1.3879800000      0.7666400000      0.0000000000                 
He      -3.9617200000      4.5906100000      0.0000000000                 
He      -7.1237500000      5.3849600000      0.0000000000                 
He      -0.8180800000      5.3664900000      0.0000000000                 
He      -2.6387112815      7.0059473933      0.0000000000 
    '''
    res_list = None
    coords = xyz_str2tuple(test_xyz)
    natm = len(coords)
    res_list = [["HEH", "HEH", "HEH"]] * (natm//3)
    '''nonwater_region = range(5)
    water_region = set(range(len(coords))).difference(set(nonwater_region))
    nonwater_coords = [coords[i] for i in nonwater_region]
    water_coords = [coords[i] for i in water_region]
    xyz_to_pdb(water_coords, nonwater_coords)'''
    xyz_to_pdb(coords, res_list=res_list)