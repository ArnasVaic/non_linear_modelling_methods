#!/bin/bash
#SBATCH -p main
#SBATCH -n 1

# go to the directory where your files are
cd $SLURM_SUBMIT_DIR

# optional: list files to confirm
ls -l

module load openmpi
module load openblas

OPENBLAS_DIR=/usr/share/spack/root/opt/spack/linux-ubuntu22.04-cascadelake/gcc-11.4.0/openblas-0.3.26-6i5mj2eoxl4u7riotfxo2plwl7to4dzg

mpic++ \
    -I${OPENBLAS_DIR}/include \
    -L${OPENBLAS_DIR}/lib \
    -o build/main src/main.cpp \
    -lopenblas \
    -std=c++23 \
    -fopenmp \
    -I extern/spdlog/include \
    -Wl,-rpath,${OPENBLAS_DIR}/lib

cd build

srun ./main initial_condition_my_function.bin initial_condition_linear.bin