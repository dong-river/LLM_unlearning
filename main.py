import argparse
import torch
import sys
import os
import time
import math
from tqdm import tqdm
from utils.data_utils import ExtractChallengeDataset, GenEvalDataset
from utils.eval_utils import compute_unlearning_metrics, compute_gen_metrics
from utils.other_utils import log_results, torch_save, torch_load, get_gen_args
from utils.ft_utils import get_train_args, GradientAscentTrainer, DITrainer, MyDataCollator, load_peft_model, merge_model, get_train_args_di, EarlyStoppingCallback
from utils.benchmark import compute_benchmarks
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
os.environ["WANDB_DISABLED"] = "true"

def get_args_parser():
    parser = argparse.ArgumentParser('llm_unlearning', add_help=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    
    ## Path
    parser.add_argument('--output_folder', default='outputs', type=str)
    parser.add_argument('--model_folder', default='./model', type=str)
    parser.add_argument("--extract_challenge_data_path", default = "./data/extract_challenge/train_dataset.npy", type=str)
    parser.add_argument("--filtered_extract_challenge_data_path", default = "./data/extract_challenge/filtered.npy", type=str)
    parser.add_argument('-m', "--model_name_or_path", default = "EleutherAI/gpt-neo-1.3B", type=str) #EleutherAI/gpt-neox-20b, EleutherAI/gpt-j-6b, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-neo-125m
    parser.add_argument('--teacher_model', default = "EleutherAI/gpt-neo-125m", type=str) 
    parser.add_argument("--cache_dir", default='./data', type=str)

    ## Methods and their params
    parser.add_argument('--method', default='raw_gpt', choices=['GPT', 'UL', 'TA', 'CD', 'DI'], type=str) ## UL for unlikelihood training, TA for task arithemic, CD for Contrastive Decoding, DI (ours) for deliberate imagination
    ## CD
    parser.add_argument('--contrastive_coef', type=float, default=0.1)
    parser.add_argument('--strat', choices=['relu', 'relu2', 'relu3', 'relu_offset'], default='relu2') ## Varients of CD
    parser.add_argument('--relu_threshold', type=float, default=0)
    ## TA
    parser.add_argument("--weight_subtraction_coef", default=0.25, type=float)
    ## DP
    parser.add_argument("--DP", default=False, type=bool)
    parser.add_argument("--DP_coef", default=0, type=float)
    ## DI (ours) params
    parser.add_argument("--num_epochs_di", default=2, type=int)
    parser.add_argument("--lr_di", default=6e-4, type=float)
    parser.add_argument("--di_strength", default=5, type=float)
    ## Focus params
    parser.add_argument("--focus", default=False, type=bool)
    parser.add_argument("--focus_dataset", default=False, type=bool)
    parser.add_argument("--focus_coeff", default=10, type=float)
    parser.add_argument("--focus_type", default='entity', type=str)
    parser.add_argument("--focus_hard", default=False, type=bool)

    ## Train params
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--peft", choices = ['ft', 'lora', 'adapter'], default='ft')
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--train_num", default=15000, type=int)
    parser.add_argument("--gradient_accu", default=1, type=int)
    parser.add_argument("--lr_sc", default="", choices=['cosine', 'polynomial'], type=str)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--early_stop", default=False, type=bool)
    parser.add_argument("--early_stop_criteria", default=1.03, type=float) ## A number > 1 that indicates the percentage of ppl change
    
    ## Eval params
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--eval_window_size", default=40, type=int) # The window calculating the extraction likehood during evaluation
    parser.add_argument("--eval_num", default=500, type=int)
    parser.add_argument("--oracle_model", default='EleutherAI/gpt-j-6b', type=str)
    
    ## Generation params
    parser.add_argument("--do_sample", default=True, type=bool)
    parser.add_argument("--top_p", default=0.95, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--relu3_topk", default=500, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--max_length", default=200, type=int)
    parser.add_argument("--cd_num_token", default=1000, type=int)
    
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    ## Set paths
    log_path, log_sample_path = f'./{args.output_folder}/log.txt', f'./{args.output_folder}/log_sample.txt'
    print("Log path: ", log_path, log_sample_path)
    model_save_folder= f"{args.method}_{args.peft}_{args.rank}_{args.model_name_or_path.replace('/', '-')}_{args.num_epochs}_{args.lr}_{args.train_batch_size}"
    if args.method == 'DI':
        model_save_folder += f"_{args.num_epochs_di}_{args.lr_di}_{args.di_strength}_{args.weight_decay}_{args.warmup_steps}_{args.gradient_accu}_{args.early_stop}_{args.early_stop_criteria}"
    if args.focus:
        model_save_folder += f"_focus_{args.focus_coeff}_type_{args.focus_type}_hard_{args.focus_hard}"
    model_save_path = f"{args.model_folder}/{model_save_folder}"
    if not os.path.exists(args.output_folder): 
        os.mkdir(args.output_folder)
    if not os.path.exists(args.model_folder): 
        os.mkdir(args.model_folder)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    peft_flag = True if (args.peft == 'lora' or args.peft == 'adapter') else False
    
    print("Loading data...")
    train_dataset = ExtractChallengeDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='train')
    test_dataset = ExtractChallengeDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='test')
    test_ma_dataset = ExtractChallengeDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_ma_dataloader = torch.utils.data.DataLoader(test_ma_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    wiki_eval_dataset = GenEvalDataset(args, tokenizer, gen_data_type='wikitext')
    validation_set = GenEvalDataset(args, tokenizer, gen_data_type='wikitext_valid')
    news_eval_dataset = GenEvalDataset(args, tokenizer, gen_data_type='news')
    wiki_eval_dataloader = torch.utils.data.DataLoader(wiki_eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
    news_eval_dataloader = torch.utils.data.DataLoader(news_eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
    print("Data loaded.")
    
    ## Unlearning
    print("Unlearning/Memorizing")
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    DI_data_collator = MyDataCollator(tokenizer)
    train_args = get_train_args(args)
    
    # If the model already exists, we skip training
    if not os.path.exists(model_save_path) and args.method != 'raw_gpt': 
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
        if args.peft == 'lora':
            model = load_peft_model(args, model, args.peft)
        
        if args.method == 'UL':
            trainer = GradientAscentTrainer(model=model, args=train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
            trainer.train()
            torch_save(model, model_save_path)
        
        elif args.method == 'TA':
            trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
            trainer.train()
            torch_save(model, model_save_path)
        
        elif args.method == 'CD':
            trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
            trainer.train()
            torch_save(model, model_save_path)
        
        elif args.method == "DI":
            print("Training DI...")
            di_kwargs = {'di_strength': args.di_strength, "focus": args.focus, "focus_coeff": args.focus_coeff, 'focus_hard': args.focus_hard, 'teacher_model': args.teacher_model}
            di_train_args = get_train_args_di(args)
            if args.early_stop:
                trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, tokenizer=tokenizer, data_collator=DI_data_collator)
                eval_results = trainer.evaluate(validation_set)
                initial_loss = eval_results.get("eval_loss")
                initial_perplexity = math.exp(initial_loss)
                
                trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, callbacks=[EarlyStoppingCallback(initial_perplexity = initial_perplexity, ppl_change=args.early_stop_criteria)], tokenizer=tokenizer, data_collator=DI_data_collator)
                trainer.train()
                for callback in trainer.callback_handler.callbacks:
                    if isinstance(callback, EarlyStoppingCallback):
                        early_stopping_callback = callback
                        break
                # early_stopping_callback = trainer.callback_handler.callbacks[0]
                early_stop_epoch = early_stopping_callback.early_stop_epoch
                if early_stop_epoch is None:
                    early_stop_epoch = args.num_epochs_di
                args.early_stop_epoch = early_stop_epoch
            else:
                trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, tokenizer=tokenizer, data_collator=DI_data_collator)
                trainer.train()
            torch_save(model, model_save_path, peft=peft_flag)
        del model
        torch.cuda.empty_cache()
    print("Unlearning Done")
    
    if args.method == 'GPT' or args.method == "DP":  
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
        memo_model = None
    elif args.method == 'TA':
        ori_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
        memo_model = torch_load(model_save_path, device=args.device)
        model = merge_model(ori_model, memo_model, weight_subtraction_coef=args.weight_subtraction_coef)
        memo_model = None
    elif args.method == 'UL': 
        model = torch_load(model_save_path, device=args.device)
        memo_model = None
    elif args.method == 'CD':
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
        memo_model = torch_load(model_save_path, device=args.device)
    elif args.method == "DI":
        model = torch_load(model_save_path, device=args.device, peft=peft_flag)
        memo_model = None     
    
    print("Evaluating...")
    metrics = {}
    gen_sample = {}
    
    # Evaluating the generation performance (wikitext)
    prompts, gens, targets = [], [], []
    for batch in tqdm(wiki_eval_dataloader, desc="Generating for evaluating model generation performance"):
        kwargs = get_gen_args(args, memo_model)
        output_ids = model.generate(input_ids=batch['input_ids'].to(device), attention_mask = batch['attention_mask'].to(device), **kwargs)
        output = tokenizer.batch_decode(output_ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        prompt = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        target = tokenizer.batch_decode(batch['gold'], skip_special_tokens=True)
        prompts += prompt; gens += output; targets += target
    print("computing generation metrics...")
    metrics_gen = compute_gen_metrics(args, prompts, gens, targets, prefix = 'wikitext')
    metrics.update(metrics_gen)
    print("*" * 50 + str(metrics))
    for i in range(5):
        gen_sample[i] = str(prompts[i]) + "\n--------Generation Below-------\n" + str(gens[i]) + "\n" + "*" * 50 + "\n"
    
    # Evaluating the generation performance (news)
    prompts, gens, targets = [], [], []
    for batch in tqdm(news_eval_dataloader, desc="Generating for evaluating model generation performance"):
        kwargs = get_gen_args(args, memo_model)
        output_ids = model.generate(input_ids=batch['input_ids'].to(device), attention_mask = batch['attention_mask'].to(device), **kwargs)
        output = tokenizer.batch_decode(output_ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        prompt = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        target = tokenizer.batch_decode(batch['gold'], skip_special_tokens=True)
        prompts += prompt; gens += output; targets += target
    print("computing generation metrics...")
    metrics_gen = compute_gen_metrics(args, prompts, gens, targets, prefix = 'news')
    metrics.update(metrics_gen)
    print("*" * 50 + str(metrics))
    
    # Evaluating the unlearning/memorization
    gen_seq, target_seq, prompt_seq = [], [], []
    for dp in tqdm(test_dataloader, desc="Generating for evaluating unlearning/memorization"):
        kwargs = get_gen_args(args, memo_model)
        output_ids = model.generate(input_ids=dp['input_ids'].to(device), attention_mask = dp['attention_mask'].to(device), **kwargs)
        gen_seq += tokenizer.batch_decode(output_ids[:, dp['input_ids'].shape[1]:], skip_special_tokens=True)
        target_seq += tokenizer.batch_decode(dp['targets'].int(), skip_special_tokens=True)
        prompt_seq += tokenizer.batch_decode(dp['input_ids'].int(), skip_special_tokens=True)
        break
    
    prompt_ids, target_ids, gen_ids = tokenizer(prompt_seq)['input_ids'], tokenizer(target_seq)['input_ids'], tokenizer(gen_seq)['input_ids']
    print("computing unlearning metrics...")
    metrics_unlearning = compute_unlearning_metrics(args, prompt_ids, gen_ids, target_ids, tokenizer)
    metrics.update(metrics_unlearning)
    for i in range(30):
        gen_sample[i+6] = str(prompt_seq[i]) + "\n--------Generation Below-------\n" + str(gen_seq[i]) + "\n" + "*" * 50 + "\n"

    # Evaluating MA in unlearning
    ma = []
    for dp in tqdm(test_ma_dataloader, desc="Generating for evaluating memorization acc"):
        kwargs = get_gen_args(args, memo_model)
        kwargs.update({"max_new_tokens": 1, "num_beams": 1, "do_sample": False})
        output_ids = model.generate(input_ids=dp['input_ids'].to(device), attention_mask = dp['attention_mask'].to(device), **kwargs)
        pred = output_ids[:,-1].to("cpu")
        label = dp['targets'][:,0]
        ma += (pred == label).detach().cpu().tolist()
    ma = sum(ma) / len(ma)
    metrics['ma'] = ma
    
    ## Evaluating the generation benchmarks
    print("computing benchmark metrics...\n")
    metrics_benchmark = compute_benchmarks(args, tokenizer, model, memo_model)
    metrics.update(metrics_benchmark)
    print(metrics)
    
    log_results(args, metrics, log_path)
    log_results(args, gen_sample, log_sample_path)
    print(str(metrics))
