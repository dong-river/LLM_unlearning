model = 'EleutherAI/gpt-neo-1.3B'
eval_num = 5000
train_batch_size = 16
eval_batch_size = 64
output_folder = "outputs_1.3B_baseline"

ul_dic = {
    'method': 'unlikelihood',
    'lr': [5e-5],
    'num_epochs': [1, 5],
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size,
    'model_name_or_path': model,
    'eval_num': eval_num,
    'output_folder': output_folder
}

## They use lr = 1e-5, bs = 32, num_epochs = 5, weight_subtraction_coef={0, 0.1, ... , 1}
ws_dic = {
    'method': 'weight_subtraction',
    'weight_subtraction_coef': [0.01, 0.05, 0.1],
    'lr': [1e-4, 2e-5],
    'num_epochs': 10,
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size,
    'model_name_or_path': model,
    'eval_num': eval_num,
    'output_folder': output_folder
}

dp_dic = {
    'method': 'DP',
    'DP_coef': [0.2, 0.4, 0.6, 0.8],
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size,
    'model_name_or_path': model,
    'eval_num': eval_num,
    'output_folder': output_folder
}

cd_dic = {
    'method': 'contrasive',
    'num_epochs': 10,
    'lr': [1e-4, 2e-5],
    'contrastive_coef': [0.25, 0.5, 0.75],
    'strat': 'relu2',
    'cd_num_token': 1000,
    'model_name_or_path': model,
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size // 2,
    'eval_num': eval_num,
    'output_folder': output_folder
}

# dic_list = {"ul": ul_dic, "ws": ws_dic, "dp": dp_dic, "cd": cd_dic}
dic_list = {"ul": ul_dic, "ws": ws_dic, "dp": dp_dic, "cd": cd_dic}
num_commands_per_file = 6

slurm_header = """#!/bin/bash
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
"""

import os
import itertools
all_commands = []
def create_experiment_commands(dic):
    commands = []
    keys, values = zip(*dic.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for experiment in experiments:
        command = 'python main.py '
        for key, value in experiment.items():
            command += f'--{key} {value} '
        commands.append(command)
    return commands

for dic in dic_list.values():
    for key, value in dic.items():
        if type(value) == str:
            dic[key] = [value]
        if type(value) == int or type(value) == float or type(value) == bool:
            dic[key] = [str(value)]
    
    commands = create_experiment_commands(dic)
    print('num commands', len(commands), 'for dic', dic)
    all_commands += commands

print('total commands', len(all_commands))
# for name, dic in dic_list.items():
#     output_path = f'exp_{name}.slurm'
#     commands = create_experiment_commands(dic)
    
#     with open(output_path, 'a') as f:
#         f.write(slurm_header + '\n')
#     for command in commands:
#         with open(output_path, 'a') as f:
#             f.write(command + '\n')

for idx in range(1 + len(all_commands) // num_commands_per_file):
    file_commands = all_commands[idx * num_commands_per_file : (idx+1) * num_commands_per_file]
    output_path = f'baseline_{idx}.slurm'
    print('writing to', output_path)
    with open(output_path, 'a') as f:
        f.write(slurm_header + '\n')
        for command in file_commands:
            f.write(command + '\n')