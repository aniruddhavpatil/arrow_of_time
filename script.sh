#!/bin/bash
#SBATCH -n 32
#SBATCH --mem-per-cpu=2048
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL

python3 arrow_of_time.py
