#!/bin/bash
#SBATCH --job-name="NLP-finetune"
#SBATCH --account=3189081
#SBATCH --partition=ai
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=output/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3189081@studbocconi.it

cd ~/NLP-Latent-Learning

module load modules/miniconda3
conda activate /home/3189081/.conda/envs/uv

uv run efficient_b_train.py