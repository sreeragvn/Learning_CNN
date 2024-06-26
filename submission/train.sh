#!/bin/bash
#SBATCH --job-name=check
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=gpu04,gpu06
#SBATCH --mem=10000
#SBATCH --mail-type=end
#SBATCH --mail-user=sreerag.naveenachandran@tu-braunschweig.de

# Load modules
module load cuda/10.0
module load lib/cudnn/7.6.1.34_cuda_10.0
module load anaconda/3-5.0.1

source activate my_env

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

# Execute programs
srun python -u train.py
