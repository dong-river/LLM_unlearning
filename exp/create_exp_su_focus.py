## soft_unlikelihood

## Raw GPT on focus dataset
raw_gpt = {
    'method': 'raw_gpt',
    'eval_batch_size': 256,
    'eval_num': 5000,
    'model_name_or_path': "EleutherAI/gpt-neo-125m",
    'focus_dataset': 'True',
    'output_folder': 'outputs_new'
}

## No focus
su_dic = {
    'method': 'soft_unlikelihood',
    'cd_num_token': 1000,
    'model_name_or_path': "EleutherAI/gpt-neo-125m",
    'train_batch_size': 64,
    'eval_batch_size': 256,
    'eval_num': 5000,
    'num_epochs_su': [3, 10, 30, 100],
    'lr_su': [1e-6, 1e-7],
    'su_strength': [3, 10, 30],
    'focus_dataset': 'True',
    'output_folder': 'outputs_new'
}

## Focus Unlearning (Hard)
su_dic_2 = {
    'method': 'soft_unlikelihood',
    'cd_num_token': 1000,
    'model_name_or_path': "EleutherAI/gpt-neo-125m",
    'train_batch_size': 64,
    'eval_batch_size': 256,
    'eval_num': 5000,
    'num_epochs_su': [3, 10, 30, 100],
    'lr_su': [1e-6, 1e-7],
    'su_strength': [3, 10, 30],
    'focus': 'True',
    'focus_hard': 'True',
    'focus_type': ['entity', 'none'],
    'output_folder': 'outputs_new'
}

## Focus Unlearning (Hard 1.3B)
## This experiment is trying to check if focus unlearning can delay overfits
su_dic_3 = {
    'method': 'soft_unlikelihood',
    'cd_num_token': 1000,
    'model_name_or_path': 'EleutherAI/gpt-neo-1.3B',
    'train_batch_size': 16,
    'eval_batch_size': 64,
    'eval_num': 5000,
    'num_epochs_su': [3, 10, 30],
    'gradient_accu': 4,
    'lr_su': 1e-7,
    'su_strength': [3, 10],
    'focus': 'True',
    'focus_hard': 'True',
    'focus_type': 'entity',
    'output_folder': 'outputs_replicate_old'
}

## This experiment is trying to check if batch_size (and thus gradient accumulation) helps with training
su_dic_3_2 = {
    'method': 'soft_unlikelihood',
    'cd_num_token': 1000,
    'model_name_or_path': 'EleutherAI/gpt-neo-1.3B',
    'train_batch_size': 16,
    'eval_batch_size': 64,
    'eval_num': 5000,
    'num_epochs_su': [3, 10, 30],
    'gradient_accu': 1,
    'lr_su': 1e-7,
    'su_strength': [3, 10],
    'focus': 'True',
    'focus_hard': 'True',
    'focus_type': 'entity',
    'output_folder': 'outputs_replicate_old'
}

## This experiment is simply trying to replicate the old results
su_dic_4 = {
    'method': 'soft_unlikelihood',
    'cd_num_token': 1000,
    'model_name_or_path': 'EleutherAI/gpt-neo-1.3B',
    'train_batch_size': 16,
    'eval_batch_size': 64,
    'eval_num': 5000,
    'num_epochs_su': [3, 10, 30],
    'gradient_accu': 4,
    'lr_su': 1e-7,
    'su_strength': [3, 10],
    'output_folder': 'outputs_replicate_old'
}

## This experiment already checked that with small learning rate like 1e-7, the model still overfits
# su_dic_4 = {
#     'method': 'soft_unlikelihood',
#     'model_name_or_path': 'EleutherAI/gpt-neo-1.3B',
#     'train_batch_size': 16,
#     'eval_batch_size': 64,
#     'eval_num': 5000,
#     'lr_su': 1e-7,
#     'su_strength': [3, 10],
#     'num_epochs_su': 100,
#     'gradient_accu': 4,
#     'output_folder': 'outputs_early_stop'
# }

## Try to use early stop to prevent overfitting
su_dic_5 = {
    'method': 'soft_unlikelihood',
    'model_name_or_path': 'EleutherAI/gpt-neo-1.3B',
    'train_batch_size': 16,
    'eval_batch_size': 64,
    'eval_num': 5000,
    'lr_su': [1e-7, 1e-8],
    'su_strength': [3, 10],
    'num_epochs_su': 100,
    'gradient_accu': 4,
    'output_folder': 'outputs_early_stop',
    'early_stop': True,
    'early_stop_criteria': 1.1
}

# su_dic_6 = {
#     'method': 'soft_unlikelihood',
#     'model_name_or_path': 'EleutherAI/gpt-neo-2.7B',
#     'train_batch_size': 8,
#     'eval_batch_size': 32,
#     'eval_num': 5000,
#     'lr_su': [1e-6, 1e-7, 1e-8],
#     'su_strength': [3, 10],
#     'num_epochs_su': 100,
#     'gradient_accu': 8,
#     'output_folder': 'outputs_early_stop',
#     'early_stop': True
# }

dic_list = {"su_3": su_dic_3, "su_3_2": su_dic_3_2, "su_4": su_dic_4, "su_5": su_dic_5}
# file_prefix = "exp_soft_unlikelihood"
# dic_list = [cd_ul_dic]
num_commands_per_file = 2


slurm_header = """#!/bin/bash
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
    output_path = f'exp_su_{idx}_focus.slurm'
    print('writing to', output_path)
    with open(output_path, 'a') as f:
        f.write(slurm_header + '\n')
        for command in file_commands:
            f.write(command + '\n')