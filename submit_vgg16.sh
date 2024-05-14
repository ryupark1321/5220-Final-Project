#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH -C gpu
#SBATCH --account=m4341
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J vgg16_train
#SBATCH -o module_job_log_%j.out

# load libs
module load cudnn
module load python
module load tensorflow

python vgg16_mirrored_strategy_script.py
