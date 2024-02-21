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

python main.py --method raw_gpt --eval_batch_size 32 --eval_num 5000 --model_name_or_path EleutherAI/gpt-neo-2.7B --output_folder outputs_early_stop 
