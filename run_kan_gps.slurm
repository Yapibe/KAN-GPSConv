#!/bin/tcsh
#SBATCH --job-name=kan_gps_training
#SBATCH --output=logs/kan_gps_training_%j.out
#SBATCH --error=logs/kan_gps_training_%j.err
#SBATCH --partition=gpu-ai
#SBATCH --account=gpu-research
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

# Navigate to your project directory
cd /home/bnet/yairp1/KAN-GPSConv

# Run your Python script
python experiments/train_model.py --dataset Cora --task node --optimizer sgd --epochs 100
