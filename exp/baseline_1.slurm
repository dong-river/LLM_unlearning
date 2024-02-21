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

python main.py --method DP --DP_coef 0.2 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method DP --DP_coef 0.4 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method DP --DP_coef 0.6 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method DP --DP_coef 0.8 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
