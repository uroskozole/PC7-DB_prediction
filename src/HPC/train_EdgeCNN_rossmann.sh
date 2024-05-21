#!/bin/bash
#SBATCH --job-name="EdgeCNN"
#SBATCH --output=sling_logs/EdgeCNN-%J.out
#SBATCH --error=sling_logs/EdgeCNN-%J.err
#SBATCH --time=08:00:00 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1 # number of nodes
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH --ntasks=1 # number of tasks
#SBATCH --mem-per-gpu=32G # memory allocation
#SBATCH --partition=gpu # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12 # number of allocated cores

source /d/hpc/projects/FRI/mj5835/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate pc7_project # activate the previously created environment
srun python /d/hpc/projects/FRI/mj5835/PC7-DB_prediction/src/HPC/train.py --dataset rossmann --target_table historical --target Customers --task regression --model_name EdgeCNN

# --dataset rossmann --target_table historical --target Customers --task regression --model_name EdgeCNN 
# --dataset Biodegradability_v1 --target_table molecule --target activity --task regression --model_name EdgeCNN