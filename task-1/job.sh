#!/bin/bash
#SBATCH -p main
#SBATCH -n1

module load openmpi
mpic++ -o main main.cpp
mpirun main
