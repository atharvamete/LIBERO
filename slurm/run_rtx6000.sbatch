#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH --account=gts-agarg35                   # charge account
#SBATCH -N1 --gres=gpu:V100:1               # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=32G                       # Memory per core
#SBATCH -t8:00:00                               # Duration of the job (8 hours)
#SBATCH -q embers                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=END,FAIL                    # Mail preferences
#SBATCH --mail-user=amete7@gatech.edu           # E-mail address for notifications
#SBATCH --output=slurm_out_reviews/fewshot/Report-%A.out
cd $HOME/p-agarg35-0/diff-skill/LIBERO                      # Change to working directory

module load anaconda3/2022.05.0.1
conda activate libero

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

echo "Running the following command:"
echo $@

srun $@