#!/bin/bash
## SLURM scripts have a specific format. 

#SBATCH --job-name=nav_pick_t
#SBATCH --output=../rearrange_policies/nav_pick_t/slurm_err.out
#SBATCH --error=../rearrange_policies/nav_pick_t/slurm_err.err
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=10
#SBATCH --open-mode=append
#SBATCH --time=28:00:00
#SBATCH --signal=USR1@60
## SBATCH --begin=now+2hour

# setup conda and shell environments
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate new_habitat2

# Setup slurm multinode
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
set -x

echo 1
# Run training
#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/ddppo_pick.yaml --run-type eval
#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/ddppo_place.yaml --run-type eval
#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/ddppo_nav_to_obj.yaml --run-type eval

#srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/hab/ddppo_tidy_house.yaml --run-type eval

srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/rearrange/hab/ddppo_nav_pick.yaml --run-type eval