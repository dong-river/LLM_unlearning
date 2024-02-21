model = 'EleutherAI/gpt-neo-1.3B'
eval_num = 5000
train_batch_size = 16
eval_batch_size = 64

su_dic = {
    'method': 'soft_unlikelihood',
    'model_name_or_path': model,
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size,
    'eval_num': eval_num,
    'lr_su': [1e-6, 1e-7],
    'su_strength': [3, 10],
    'num_epochs_su': [1, 2, 3, 10, 30],
    'gradient_accu': 4,
    'output_folder': 'outputs_not_important'
}

su_dic_2 = {
    'method': 'soft_unlikelihood',
    'model_name_or_path': 'EleutherAI/gpt-neo-2.7B',
    'train_batch_size': 8,
    'eval_batch_size': 32,
    'eval_num': eval_num,
    'lr_su': [1e-7, 1e-8],
    'su_strength': [3, 10],
    'num_epochs_su': [1, 2, 3, 10, 30],
    'gradient_accu': 8,
    'output_folder': 'outputs_new'
}


# dic_list = {"raw_gpt": raw_gpt, "su": su_dic, "su_2": su_dic_2}
dic_list = {"su": su_dic_2}
# file_prefix = "exp_soft_unlikelihood"
# dic_list = [cd_ul_dic]
num_commands_per_file = 3

slurm_header = """#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=Unlearning
#SBATCH --output=%x.%j.log
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=200:00:00
#SBATCH --nodelist=nlpgpu06

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
    output_path = f'exp_su_1.3B_{idx}_add.slurm'
    print('writing to', output_path)
    with open(output_path, 'a') as f:
        f.write(slurm_header + '\n')
        for command in file_commands:
            f.write(command + '\n')