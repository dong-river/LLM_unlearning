#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=Unlearning
#SBATCH --output=%x.%j.log
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=200:00:00
#SBATCH --nodelist=nlpgpu05

source /home1/r/riverd/miniconda3/etc/profile.d/conda.sh
conda activate watermark
cd /home1/r/riverd/LLM_unlearning

python main.py --method contrasive --num_epochs 10 --lr 0.0001 --contrastive_coef 0.25 --strat relu2 --cd_num_token 1000 --model_name_or_path EleutherAI/gpt-neo-1.3B --train_batch_size 16 --eval_batch_size 32 --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method contrasive --num_epochs 10 --lr 2e-05 --contrastive_coef 0.25 --strat relu2 --cd_num_token 1000 --model_name_or_path EleutherAI/gpt-neo-1.3B --train_batch_size 16 --eval_batch_size 32 --eval_num 5000 --output_folder outputs_1.3B_baseline 
