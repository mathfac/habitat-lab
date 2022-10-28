#!/bin/bash
## SLURM scripts have a specific format. 

#SBATCH --job-name=nav_pick_t7
#SBATCH --output=../rearrange_policies/nav_pick_t7/slurm.out
#SBATCH --error=../rearrange_policies/nav_pick_t7/slurm.err
## SBATCH --partition=learnfair
#SBATCH --partition=scavenge
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=18
#SBATCH --open-mode=append
#SBATCH --time=6:00:00
#SBATCH --signal=USR1@60

# setup conda and shell environments
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate habitat-lab

# Setup slurm multinode
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
set -x

echo 1
# Run training
#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/ddppo_pick.yaml --run-type train
#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/ddppo_nav_to_obj.yaml --run-type train
#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/ddppo_place.yaml --run-type train

#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/hab/ddppo_tidy_house.yaml --run-type train


srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/hab/ddppo_nav_pick.yaml --run-type train