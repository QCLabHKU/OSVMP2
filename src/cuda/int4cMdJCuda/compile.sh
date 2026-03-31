#rys_contract_jk.cu #unrolled_os.cu unrolled_rys.cu \
  #nr_sr_estimator.c cart2xyz.c \
  #count_tasks.cu  -I".." \ 
lib_dir="/mnt/c/Users/lando/OneDrive-HKU/OneDrive/code/cuda/lib"
nvcc -arch=sm_75 -Xcompiler -fPIC -Xcompiler -fopenmp -Xptxas=-v -shared -I$lib_dir \
md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu \
  md_j_driver.cu md_pairdata.c  -o ../../gvhf_md.so


