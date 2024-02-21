## soft_unlikelihood
raw_gpt = {
    'method': 'raw_gpt',
    'eval_batch_size': 256,
    'eval_num': 5000,
    'model_name_or_path': "EleutherAI/gpt-neo-125m",
}

su_dic = {
    'method': 'soft_unlikelihood',
    'cd_num_token': 1000,
    'model_name_or_path': "EleutherAI/gpt-neo-125m",
    'train_batch_size': 64,
    'eval_batch_size': 256,
    'eval_num': 5000,
    'num_epochs_su': [3, 10, 30, 100],
    'lr_su': [1e-4, 1e-5, 1e-6, 1e-7],
    'su_strength': [3, 10, 30],
}

su_dic_add = {
    'method': 'soft_unlikelihood',
    'cd_num_token': 1000,
    'model_name_or_path': "EleutherAI/gpt-neo-125m",
    'train_batch_size': 64,
    'eval_batch_size': 256,
    'eval_num': 5000,
    'num_epochs_su': [3, 10, 30],
    'lr_su': [1e-5, 5e-5, 1e-6],
    'su_strength': [3, 10],
    'kl': True,
    'output_folder': 'outputs_kl',
}

dic_list = {"su": su_dic_add}
# file_prefix = "exp_soft_unlikelihood"
# dic_list = [cd_ul_dic]
num_commands_per_file = 4

slurm_header = """#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=Unlearning
#SBATCH --output=%x.%j.log
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=100:00:00
#SBATCH --nodelist=nlpgpu05

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
    output_path = f'kl_{idx}.slurm'
    print('writing to', output_path)
    with open(output_path, 'a') as f:
        f.write(slurm_header + '\n')
        for command in file_commands:
            f.write(command + '\n')