import os
import torch
from functools import partial
from utils.ft_utils import merge_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import time
from peft import PeftModel, PeftConfig

# openai.api_key = os.environ['OPENAI_API_KEY']

def get_openai_response(prompt, model = 'gpt-3.5-turbo', max_tokens = 1000, temperature = 0, top_p = 0.9, frequency_penalty = 0, presence_penalty = 0, stop = ['###', '==='], recursion_depth = 0, max_depth = 3) -> str:
    if model not in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'gpt-4', 'text-davinci-003', 'text-davinci-002', 'text-curie-001']:
        print("model name is wrong, check you model name")
    if recursion_depth > max_depth:
        return ''
    try:
        if model in ['text-davinci-003', 'text-davinci-002', 'text-curie-001']:
            res = openai.Completion.create(
                model = model,
                prompt = prompt,
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
                frequency_penalty = frequency_penalty,
                presence_penalty = presence_penalty,
                stop = stop
            )
            return res['choices'][0]['text']

        elif model == 'gpt-4' or model == 'gpt-3.5-turbo' or model == 'gpt-3.5-turbo-16k':
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature = temperature,
                top_p = top_p,
                max_tokens=max_tokens,
                stop=stop
            )
            return completion['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        print('wait for 60 seconds')
        time.sleep(60)
        return get_openai_response(prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop, recursion_depth=recursion_depth+1, max_depth=max_depth)


def log_text(text, log_file_path):
    with open(log_file_path, 'a') as f:
        f.write(text)

def log_results(args, metrics, output_path): 
    txt = ""
    for arg in vars(args):
        if 'eval' not in arg and arg not in ['output_folder', 'model_folder', 'data_type', 'extract_challenge_data_path', 'device', 'oracle_model', 'relu3_topk', 'temperature', 'max_length', 'num_beams', 'top_p', 'repetition_penalty'] or isinstance(arg, int):
            txt += f"{arg}: {getattr(args, arg)}\n"
    txt += "-" * 50 + "\n"
    for key, value in metrics.items():
        txt += f"{key}: {value}\n"
    txt += "\n" + "*" * 50 + "\n"
    log_text(txt, output_path)

def torch_save(model, save_path, peft=False):
    if peft:
        model.save_pretrained(save_path)
    if not peft:
        if os.path.dirname(save_path) != '':
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model, save_path)

def torch_load(save_path, device=None, peft=False):
    if not peft:
        model = torch.load(save_path)
    if peft:
        config = PeftConfig.from_pretrained(save_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, save_path)
        model = model.merge_and_unload()
        model = model.to(device) 
    if device is not None:
        model = model.to(device)
    return model

def load_unlearned_model(model_save_path, args):
    if args.method == 'contrasive':
        memo_model = torch_load(model_save_path, device=args.device)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
        model_generate = partial(model.generate, teacher_student = True, top_p=0.95, do_sample=True, model_kwargs_student={}, contrastive_coef=args.contrastive_coef, student_lm = memo_model)
        
    elif args.method == 'weight_subtraction':
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
        memo_model = torch_load(model_save_path, device=args.device)
        merged_model = merge_model(model, memo_model, weight_subtraction_coef=args.weight_subtraction_coef)
        model_generate = partial(merged_model.generate, teacher_student = False, top_p=0.95, do_sample=True) 
    else:
        model = torch_load(model_save_path, device=args.device)
        model_generate = partial(model.generate, teacher_student = False, top_p=0.95, do_sample=True)
    return model_generate

def get_model_kwargs(args):
    if args.do_sample:
        model_kwargs = {'top_p': args.top_p, 'top_k': args.top_k, 'temperature': args.temperature, 'do_sample': True, 'repetition_penalty': args.repetition_penalty}
    else:
        model_kwargs = {'num_beams': args.num_beams, 'temperature': args.temperature, 'do_sample': False, 'repetition_penalty': args.repetition_penalty}
    
    model_kwargs.update({'max_new_tokens': 200})
    model_kwargs.update({'epsilon_cutoff': args.epsilon_cutoff, 'relative_top': args.relative_top, 'min_tokens_to_keep': args.min_tokens_to_keep})
    
    if args.method == 'contrasive':
        model_kwargs.update({'teacher_student': True, 'model_kwargs_student': {}, 'contrastive_coef': args.contrastive_coef})
        model_kwargs['model_kwargs_student']['strat'] = args.strat
    else:
        model_kwargs.update({'teacher_student': False})
    return model_kwargs

def get_gen_args(args, memo_model):
    ## Set up the mode and parameters for testing generation
    kwargs = {"max_new_tokens": args.max_length, 'temperature': args.temperature, 'repetition_penalty': args.repetition_penalty}
    if args.do_sample:
        # kwargs.update({"top_p": args.top_p, "top_k": args.top_k, "do_sample": True})
        kwargs.update({"top_p": args.top_p, "do_sample": True})
    elif args.do_sample == False:
        kwargs.update({"num_beams": args.num_beams, "do_sample": False})

    if args.method == 'contrasive' or args.method == 'soft_unlikelihood_with_CD':
        kwargs.update({"teacher_student": True, "model_kwargs_student": {"strat": args.strat, "relu_threshold": args.relu_threshold, "relu3_topk": args.relu3_topk}, "st_coef": args.contrastive_coef, "student_lm": memo_model})
    else:
        kwargs.update({"teacher_student": False})
    if args.method == 'DP':
        kwargs.update({"DP": True, "DP_coef": args.DP_coef})
    else:
        kwargs.update({"DP": False})    
    return kwargs

def get_gpu_usage():
    import GPUtil
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]

        # Get GPU details
        gpu_load = f"{gpu.load*100}%"
        gpu_free_memory = f"{gpu.memoryFree}MB"
        gpu_used_memory = f"{gpu.memoryUsed}MB"
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        gpu_temperature = f"{gpu.temperature} °C"

        print(f"GPU Load: {gpu_load}")
        print(f"GPU Free Memory: {gpu_free_memory}")
        print(f"GPU Used Memory: {gpu_used_memory}")
        print(f"GPU Total Memory: {gpu_total_memory}")
        print(f"GPU Temperature: {gpu_temperature}")
    else:
        print("No GPU found")