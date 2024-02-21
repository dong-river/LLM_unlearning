#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=Unlearning
#SBATCH --output=%x.%j.log
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=200:00:00
#SBATCH --nodelist=nlpgpu08

source /home1/r/riverd/miniconda3/etc/profile.d/conda.sh
conda activate watermark
cd /home1/r/riverd/LLM_unlearning

python main.py --method unlikelihood --lr 5e-05 --num_epochs 1 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method unlikelihood --lr 5e-05 --num_epochs 5 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method weight_subtraction --weight_subtraction_coef 0.01 --lr 0.0001 --num_epochs 10 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method weight_subtraction --weight_subtraction_coef 0.01 --lr 2e-05 --num_epochs 10 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method weight_subtraction --weight_subtraction_coef 0.05 --lr 0.0001 --num_epochs 10 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
python main.py --method weight_subtraction --weight_subtraction_coef 0.05 --lr 2e-05 --num_epochs 10 --train_batch_size 16 --eval_batch_size 64 --model_name_or_path EleutherAI/gpt-neo-1.3B --eval_num 5000 --output_folder outputs_1.3B_baseline 
