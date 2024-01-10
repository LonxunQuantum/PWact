#!/bin/sh
#SBATCH --partition=cpu
#SBATCH --job-name=avg_bond
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1



#module load intel/2020
#source ~/.bashrc 
conda activate mlff

#conda pack -j 8 -n pwkit_env -o pwkit_env.tar.gz
python3 /data/home/liuhanyu/hyliu/code/matersdk/demo/feature/avg/avgbond4xhm.py
