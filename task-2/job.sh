#!/bin/bash
#SBATCH -p main
#SBATCH -n 1

module load openmpi
module load openblas

OPENBLAS_DIR=/usr/share/spack/root/opt/spack/linux-ubuntu22.04-cascadelake/gcc-11.4.0/openblas-0.3.26-6i5mj2eoxl4u7riotfxo2plwl7to4dzg

mpic++ -I${OPENBLAS_DIR}/include -L${OPENBLAS_DIR}/lib \
    -o main main.cpp -lopenblas -std=c++23

mpirun ./main
