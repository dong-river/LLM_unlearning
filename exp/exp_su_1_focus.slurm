#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=Unlearning
#SBATCH --output=%x.%j.log
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=100:00:00
#SBATCH --nodelist=nlpgpu07

source /home1/r/riverd/miniconda3/etc/profile.d/conda.sh
conda activate watermark
cd /home1/r/riverd/LLM_unlearning

python main.py --method soft_unlikelihood --model_name_or_path EleutherAI/gpt-neo-1.3B --train_batch_size 16 --eval_batch_size 64 --eval_num 5000 --lr_su 1e-08 --su_strength 10 --num_epochs_su 100 --gradient_accu 4 --output_folder outputs_early_stop --early_stop True --early_stop_criteria 1.03 --warmup_steps 100 
python main.py --method soft_unlikelihood --model_name_or_path EleutherAI/gpt-neo-1.3B --train_batch_size 32 --eval_batch_size 64 --eval_num 5000 --lr_su 1e-05 --su_strength 3 --num_epochs_su 100 --gradient_accu 2 --output_folder outputs_early_stop --early_stop True --early_stop_criteria 1.03 --peft lora --rank 8 --lora_alpha 16 --warmup_steps 100 
python main.py --method soft_unlikelihood --model_name_or_path EleutherAI/gpt-neo-1.3B --train_batch_size 32 --eval_batch_size 64 --eval_num 5000 --lr_su 1e-05 --su_strength 10 --num_epochs_su 100 --gradient_accu 2 --output_folder outputs_early_stop --early_stop True --early_stop_criteria 1.03 --peft lora --rank 8 --lora_alpha 16 --warmup_steps 100 
