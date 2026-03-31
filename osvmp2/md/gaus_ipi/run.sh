#For DFT on Gaussian
export func="b3lyp"

#Set up MD parameters
#Units
export time_unit=femtosecond
export temp_unit=kelvin
export potential_unit=electronvolt
export press_unit=bar
#megapascal
export cell_units=angstrom
export force_unit=piconewton

####################################################################################
#Parameters
export verbose=0
#export charge=1
export dyn_mode='nvt'
export stride='1'
export nbeads='1'
export port=5`date +"%M%S"`
export seed=32225
export total_steps=20000
export time_step=0.5
export temperature=350


#For NVT and NPT
export tau=100
export therm_mode='langevin'
#For NPT
export pressure=10
export baro_mode='isotropic'

#Option of output: prop(properties), pos(trajectory), force, chk(checkpoint)
export output_opt="prop pos vel chk"
#####################################################################################
export molecule=o2h.xyz; export basis=6-31G
export template=template.gjf
#!!!!!!If a template is used, functional, basis set, memory and number of cores 
#!!!!!!should be specified in the template
export ncore=10  
md_test () {
    python3 gen_input.py $molecule
    nohup i-pi input.xml &
    export use_mbe=1; export use_ga=1; nohup mpirun -n $ncore python3 md_initializer.py $molecule &
    #export use_mbe=0; export use_ga=0; python3 md_initializer.py $mol
}
bach_test () {
    mol=$(echo $molecule | cut -d. -f1)
    rm -r "$mol"_*
    export OMP_NUM_THREADS=$ncore
    export ncore=1
    dir_i="$mol"_"$dyn_mode"
    mkdir $dir_i; cp $template driver.py gen_input.py md_initializer.py $molecule $dir_i
    cd $dir_i;md_test;cd ..
}

bach_test
