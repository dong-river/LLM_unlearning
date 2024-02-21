import json
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.other_utils import get_gen_args
from utils.data_utils import Custom_Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer

## All the code and data used for computing NLU benchmarks are adopted from https://github.com/joeljang/knowledge-unlearning.git. We are grateful their code

def get_hparams(config_path = "data/example.json"):
    with open(config_path) as config_file:
        config = json.load(config_file)
        hparams = argparse.Namespace(**config)
    return hparams

def get_rid_of_pad(tokens, tokenizer):
    while tokens[-1] == -100 or tokens[-1] == tokenizer.pad_token_id:
        tokens.pop()
    return tokens

def _model_call(args, model, inps, kwargs):
    """
    inps: a torch tensor of shape [batch, sequence]
    the size of sequence may vary from call to call
    returns: a torch tensor of shape [batch, sequence, vocab] with the
    logits returned from the model
    """
    with torch.no_grad():
        res = model(inps)[0]

        if kwargs['teacher_student']:
            model_kwargs_student = kwargs['model_kwargs_student']
            res_student = kwargs['student_lm'](inps)['logits']
            relu = torch.nn.ReLU()
            if model_kwargs_student['strat'] == 'relu':
                res = res - kwargs['st_coef'] * relu(res_student)
            elif model_kwargs_student['strat'] == 'relu2':
                res = res - kwargs['st_coef'] * relu(res - res_student)
            elif model_kwargs_student['strat'] == 'relu_offset':
                relu_student = relu(res_student)
                sum_pos = torch.sum(relu_student, dim=-1)
                count_pos = torch.sum(relu_student > 0, dim=-1)
                offset = sum_pos / (count_pos + 1)
                offset_matrix = (relu_student == 0) * offset.unsqueeze(-1)
                res = res - kwargs['st_coef'] * relu_student - offset_matrix
        
        if kwargs['DP']:
            uniform_dist = torch.ones(res.shape[0], res.shape[1], res.shape[2]).to(res.device)
            mean = torch.mean(res, dim=-1).unsqueeze(-1)
            uniform_dist = uniform_dist * mean
            res = kwargs['DP_coef'] * res + (1-kwargs['DP_coef']) * uniform_dist
        return res[:, :, :]

def classification_verbalizer(args, hparams, model, memo_model, tokenizer, padding_length, task, batch, choices, answer_index):
    source_ids = batch["source_ids"].tolist()
    target_ids = batch["target_ids"]
    batch_size = len(source_ids)
    answer_idx = [-1] * batch_size
    for i in range(batch_size):
        answer_idx[i] = answer_index[i]

    batch_acc = 0
    device = args.device

    inps = []
    cont_toks_list = []
    inplens = []

    answers = torch.zeros(batch_size, len(choices), device=device)

    for c_idx in range(len(choices)):
        choice_ids = tokenizer.batch_encode_plus(
            list(choices[c_idx]),
            max_length=hparams.input_length,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors="pt")["input_ids"].tolist()
        for i in range(batch_size):
            context_enc = get_rid_of_pad(source_ids[i], tokenizer)
            continuation_enc = get_rid_of_pad(choice_ids[i], tokenizer)

            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):][:-1],
                dtype=torch.long
            ).to(device)
            inplen, = inp.shape
            cont = continuation_enc

            # pad length from seq to padding_length
            inp = torch.cat([
                inp,  # [seq]
                # [padding_length - seq]
                torch.zeros(padding_length - inplen,
                            dtype=torch.long).to(inp.device) + tokenizer.pad_token_id
            ], dim=0)
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        kwargs = get_gen_args(args, memo_model)
        multi_logits = F.log_softmax(_model_call(args, model, batched_inps, kwargs), dim=-1)  # [batch, padding_length, vocab]
        cnt = 0
        for logits, inp, inplen, cont_toks \
                in zip(multi_logits, inps, inplens, cont_toks_list):

            # Slice to original seq length
            contlen = len(cont_toks)
            original_logits = logits

            # [1, seq, vocab]
            logits = logits[inplen - contlen:inplen].unsqueeze(0)
            # Check if per-token argmax is exactly equal to continuation
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq]
            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
            # Answer: (log prob, is-exact-match)
            loss = -float(logits.sum())
            answers[cnt][c_idx] = loss
            cnt += 1
        inps = []
        cont_toks_list = []
        inplens = []

    answer_idx = torch.Tensor(answer_idx).to(device)
    answers = torch.argmin(answers, dim=1)

    batch_acc = int(torch.where(answers == answer_idx, 1, 0).sum())

    batch_acc_avg = batch_acc / batch_size

    return batch_acc_avg

def get_dataset(args, hparams, dataset_name, tokenizer,
                valid_subset_path, type_path, length=None):
    input_length = length if length else hparams.input_length
    output_length = length if length else hparams.output_length
    hparams.cache_dir = args.cache_dir
    dataset = Custom_Dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        valid_subset_path=valid_subset_path,
        type_path=type_path,
        input_length=input_length,
        output_length=output_length,
        args=hparams)
    return dataset

def compute_benchmarks(args, tokenizer, model, memo_model=None, device='cuda'):
    metrics = {}
    hparams = get_hparams()
    print(model)
    
    tokenizer = GPT2Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
    if 'gpt' in hparams.tokenizer_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        
    for i in range(len(hparams.valid_sets)):
        dataset_name = hparams.valid_sets[i]
        valid_subset_path = hparams.valid_subset_path[i]
        type_path = hparams.valid_type_path[i]
        print(dataset_name)

        dataset = get_dataset(
            args = args,
            hparams = hparams,
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            valid_subset_path=valid_subset_path,
            type_path=type_path
            )
        print("len: ", dataset.__len__())
        if dataset[0]['task_type'] != 'classification':
            continue
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size // 2, shuffle=False)
        res_list = []
        for batch in tqdm(dataloader):
            task = batch["task"][0]
            task_type = batch["task_type"][0]
            
            print(task_type)
            
            if task_type == 'classification':
                res = classification_verbalizer(
                    args = args,
                    hparams = hparams,
                    model = model,
                    memo_model = memo_model,
                    tokenizer = tokenizer, 
                    padding_length=hparams.input_length,
                    task=task,
                    batch=batch,
                    choices=batch["choices"],
                    answer_index=batch["answer_index"])
                res_list.append(res)
                print('acc: ', res)
        
        final_acc = sum(res_list) / len(res_list)
        print("final_acc: ", final_acc, 'for dataset: ', dataset_name)
        metrics[dataset_name + "_" + valid_subset_path] = final_acc
        # print("*" * 50)
        # print("final_acc: ", final_acc, 'for dataset: ', dataset_name)
    print(metrics)
    return metrics