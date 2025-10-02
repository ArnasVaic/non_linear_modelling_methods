#!/bin/bash
#SBATCH -p main
#SBATCH -n 1

module load openmpi
module load openblas/0.3.26-gcc-11.4.0-rvj5pay
module load cmake/3.22.1-oneapi-2024.1.0-ihalehd

# Build
export CXX=mpic++
mkdir -p build
cd build
cmake ..
make -j 4

# Run
mpirun ./main
