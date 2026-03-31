import os

var_list = "verbose, xyz_name,  chkfile, output_opt, stride, len_side, total_steps, port, nbeads, seed, temperature, pressure, fixatoms, fixcom, dyn_mode, temp_ensem, therm_mode, baro_mode, tau), time_step".replace(",","").split()

with open("gen_input.py", 'r') as f:
    lines = f.readlines()
    l_sel = []
    for idx, l in enumerate(lines):
        if "gen_input(verbose" in l:
            while "class md_parameters()" not in lines[idx]:
                l_sel.append(lines[idx])
                idx += 1
            break
    text = "".join(l_sel)
    for i in var_list:
        text = text.replace(i, "self.%s"%i)
    print(text)