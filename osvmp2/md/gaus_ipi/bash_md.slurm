#!/bin/bash
# parallel job using 128 cores. and runs for 4 hours (max)
#SBATCH -N 1 # node count
#SBATCH -t 48:00:00
# sends mail when process begins, and 
# when it ends. Make sure you define your email 
# address.
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
##SBATCH --mail-user=juny@hku.hk

#make sure calculations are carried out in /scratch on each node
export PYTHONPATH=/home/qjliang/OSV-BOMD-TEST:$PYTHONPATH
source /home/qjliang/software/i-pi-master/env.sh
export OMP_NUM_THREADS=1
export molecule=$1
export threshold=$thres
export date_time=`date +"%m-%d-%Y_%H:%M:%S"`_"$basis"

if [[ $frag == 1 ]];then
   export name="$basis"_"$molecule"_thres"$threshold"_scrp"$scr"_3b"$threeb_thres"
else
   export name="$basis"_"$molecule"_thres"$threshold"_scrp"$scr"
fi

mkdir -p /home/qjliang/test/results/$date_time/
srun mkdir -p /scratch/qjliang/
srun rm -rf /scratch/qjliang/*

cd /scratch/qjliang
srun mkdir -p "$name"_"$date_time"
cd "$name"_"$date_time"

#copy the input and executable files to /scratch
srun cp -r $SLURM_SUBMIT_DIR/$molecule .
srun cp $SLURM_SUBMIT_DIR/{gen_input.py,driver.py,osvdriver.py} ./$molecule
ulimit -l unlimited #don't remove this line, or parallel calculation will fail.

#run the calculation
cd ./$molecule
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
pwd
for molecule_conformer in *.xyz; do
   molname=${molecule_conformer:0:$((${#molecule_conformer}-4))} 
   export path_output=/home/qjliang/test/results/"$date_time"_scrp"$scr"_3b"$threeb_thres"/$molname
   echo "Output directory:" $path_output
   mkdir -p $path_output
   python gen_input.py "$molecule_conformer"
   i-pi input.xml & 
   srun -n 16 --mpi=pmi2 python osvdriver.py "$molecule_conformer"
   cp *simulation* $path_output
   rm *simulation*
done

srun rm -rf /scratch/qjliang/*
