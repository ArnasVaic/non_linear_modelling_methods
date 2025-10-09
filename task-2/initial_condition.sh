#!/bin/bash
#SBATCH -p main
#SBATCH -n 1

g++ -o build/generate_ic src/init_cond.cpp -std=c++23

./build/generate_ic