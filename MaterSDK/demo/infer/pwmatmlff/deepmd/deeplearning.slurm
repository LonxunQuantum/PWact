#!/bin/sh
#SBATCH --partition=3080ti
#SBATCH --job-name=deeplearning
#SBATCH --nodes=1		# Number of nodes
#SBATCH --ntasks-per-node=2	# Number of processes per node
#SBATCH --gpus-per-task=1       # Number of GPUs per process
#SBATCH --gres=gpu:2		# Number of total GPUs

#module load intel/2020

#source /share/app/anaconda3/etc/profile.d/conda.sh
#module load cuda/11.3
#conda activate mlff

mpirun -np $SLURM_NPROCS   python3 /data/home/liuhanyu/hyliu/code/matersdk/demo/infer/pwmatmlff/deepmd/dp_inference.py  > output 
