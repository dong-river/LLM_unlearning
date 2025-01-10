# LLM_unlearning_DI
This repo contains the official code implementation for our paper **UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models**

(The old version of the paper is called **Unmemorization in Large Language Models via Self-Distillation and Deliberate Imagination**)

## Install environment
```bash
conda create --name unlearning python=3.9.16
conda activate unlearning
pip install -r requirements.txt
```
## Run one experiment (125M FT)
```
python main.py --method DI --cd_num_token 1000 --model_name_or_path EleutherAI/gpt-neo-125m --train_batch_size 64 --eval_batch_size 256 --eval_num 5000 --num_epochs_di 10 --lr_di 1e-06 --di_strength 3 --output_folder outputs_new
```

## Run one experiment (1.3B LoRA)
```
python main.py --method DI --model_name_or_path EleutherAI/gpt-neo-1.3B --train_batch_size 32 --eval_batch_size 64 --eval_num 5000 --lr_di 5e-06 --di_strength 3 --num_epochs_di 100 --gradient_accu 2 --early_stop True --early_stop_criteria 1.03 --peft lora --rank 8 --lora_alpha 16 --warmup_steps 100 --output_folder outputs_new
```

## Run full experiments
Python files under ./exp would create thorough experiments for different purposes.

## Logs and Visualization
Running main.py will produce a result file and a generation example file. You then use parse_log.py to convert that to CSV file. We have our old results in the ./output folder and you can use visualization.ipynb to visualize it.
