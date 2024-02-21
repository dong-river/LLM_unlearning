import torch
import numpy as np
from collections import defaultdict, Counter
from nltk import ngrams as ngrams_nltk
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _seq_rep_n(text_list, n = 4):
    print("Evaluating repetition score")
    rep_n_list = []
    for text in text_list:
        token_list = text.split(' ')
        stats = defaultdict(float)
        ngs = [ng for ng in ngrams_nltk(token_list, n)]
        counter = Counter([ng for ng in ngrams_nltk(token_list, n)])
        if len(ngs) == 0:
            continue
        rep_n = 1.0 - len(counter)/len(ngs)
        assert rep_n >= 0 and rep_n <= 1
        rep_n_list.append(rep_n)
    if len(rep_n_list) > 0: 
        rep_n = sum(rep_n_list) / len(rep_n_list)
    else: 
        rep_n = 0
    return rep_n

def _diversity_n(text_list, n = 2):
    print("Evaluating diversity score")
    div_n_list = []
    for text in text_list:
        token_list = text.split(' ')
        stats = defaultdict(float)
        ngs = [ng for ng in ngrams_nltk(token_list, n)]
        counter = Counter([ng for ng in ngrams_nltk(token_list, n)])
        if len(ngs) == 0:
            continue
        num_words = len(text.split(" "))
        div_n = len(counter)/num_words
        assert div_n >= 0 and div_n <= 1
        div_n_list.append(div_n)
    if len(div_n_list) > 0: 
        div_n = sum(div_n_list) / len(div_n_list)
    else: 
        div_n = 0
    return div_n

def _get_overlap(gen, target, n):
    n_grams_gen = set(ngrams_nltk(gen, n))
    n_grams_target = set(ngrams_nltk(target, n))
    overlap = n_grams_gen.intersection(n_grams_target)
    if len(n_grams_gen) == 0 or len(target) == 0:
        print("Warning: empty n_grams_gen in _get_overlap (eval_utils.py line 49)")
        return 0
    overlap = len(overlap) / len(n_grams_gen)
    return overlap

def get_extraction_likelihood(gens, targets, n):
    el = 0
    for gen, target in zip(gens, targets):
        el += _get_overlap(gen, target, n)
    el = el / len(gens)
    return el

def get_mem_acc(gens, targets):
    mem_acc_list = []
    for gen, target in zip(gens, targets):
        min_length = min(len(gen), len(target))
        gen, target = np.array(gen)[:min_length], np.array(target)[:min_length]
        count = np.sum(gen == target)
        mem_acc_list.append(count / len(gen))
    mem_acc = sum(mem_acc_list) / len(mem_acc_list)
    return mem_acc
    
def get_cosine_similarity(args, gens, targets):
    s_bert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device) #'sentence-transformers/all-roberta-large-v1' is better
    score = []
    for idx in range(0, len(gens), args.eval_batch_size): 
        batch_gen, batch_target = gens[idx : idx+args.eval_batch_size], targets[idx : idx+args.eval_batch_size]
        # batch_gen, batch_target = tokenizer.batch_decode(batch_gen), tokenizer.batch_decode(batch_target)
        embeddings1 = s_bert_model.encode(batch_gen, convert_to_tensor=True).to(device)
        embeddings2 = s_bert_model.encode(batch_target, convert_to_tensor=True).to(device)
        score += torch.sum(embeddings1 * embeddings2, dim=-1).detach().cpu().tolist()
    score = sum(score) / len(score)
    # gens, targets = tokenizer.batch_decode(gens), tokenizer.batch_decode(targets)
    # embeddings1 = s_bert_model.encode(gens, convert_to_tensor=True).to(device)
    # embeddings2 = s_bert_model.encode(targets, convert_to_tensor=True).to(device)
    # score = torch.sum(embeddings1 * embeddings2, dim=-1)
    return score

def k_extractable(prompts, gens, targets, k):     ## prompt the LLM with 50 tokens as prompt, and then if the model emits the next 50 tokens, then it is k-extractable
    count = 0
    for prompt, gen, target in zip(prompts, gens, targets):
        if gens[:k] == targets[:k]:
            count += 1
    ratio = count / len(gens)
    return ratio

def get_ppl(prompts, gens, args):
    tokenizer = AutoTokenizer.from_pretrained(args.oracle_model)
    tokenizer.pad_token = tokenizer.eos_token
    oracle_model = AutoModelForCausalLM.from_pretrained(args.oracle_model).half().to(device)
    score_lst = []
    ## must decode first and then encode because the tokenizer may not be the same one
    oracle_batch_size = 16
    for i in range(len(gens) // oracle_batch_size):
        gen_i = gens[i * oracle_batch_size : (i+1) * oracle_batch_size]
        prompt_i = prompts[i * oracle_batch_size : (i+1) * oracle_batch_size]
        text_list_i = [prompt + gen for prompt, gen in zip(prompt_i, gen_i)]
        inputs = tokenizer(text_list_i, return_tensors='pt', truncation=True, padding=True, max_length=200)
        with torch.no_grad():
            labels = inputs['input_ids'].cuda() 
            labels[labels== tokenizer.pad_token] = -100 
            out = oracle_model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), labels=labels)
            score_lst.append(out.loss.item())
            
    # score_lst = np.array(score_lst.detach().cpu().tolist())
    score_lst = np.array(score_lst)
    ppl = np.e ** score_lst.mean()
    print("Perplexity Computed")
    return ppl.item()

def get_mauve(args, prompts, gens, targets):
    from evaluate import load
    mauve = load('mauve')
    assert len(prompts) == len(gens) == len(targets)
    gens = [prompts[i] + gens[i] for i in range(len(prompts))]
    targets = [prompts[i] + targets[i] for i in range(len(prompts))]
    mauve_results = mauve.compute(predictions=gens, references=targets, device_id=0).mauve
    return mauve_results

def get_coherence(prompts, gens):
    from simcse import SimCSE
    sim_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    similarities = sim_model.similarity(prompts, gens)
    similarities = np.array(similarities)
    coherence_score = similarities.trace() / len(similarities) 
    return coherence_score

def compute_rep_and_div(gens):
    rep_2, rep_3, rep_4 = _seq_rep_n(gens, 2), _seq_rep_n(gens, 3), _seq_rep_n(gens, 4)
    div_2, div_3 = _diversity_n(gens, 2), _diversity_n(gens, 3)
    return rep_2, rep_3, rep_4, div_2, div_3
    
def compute_unlearning_metrics(args, prompt_ids, gen_ids, target_ids, tokenizer): ## list of token ids
    assert len(prompt_ids) == len(gen_ids) == len(target_ids)
    metrics = {}
    metrics['el_3'] = get_extraction_likelihood(gen_ids, target_ids, 3)
    metrics['el_5'] = get_extraction_likelihood(gen_ids, target_ids, 5)
    metrics['el_10'] = get_extraction_likelihood(gen_ids, target_ids, 10)

    gens = tokenizer.batch_decode(gen_ids)
    targets = tokenizer.batch_decode(target_ids)
    metrics['similarity_ul'] = get_cosine_similarity(args, gens, targets)
    print(metrics)
    return metrics

def compute_gen_metrics(args, prompts, gens, targets, prefix = ""): ## list of text
    metrics = {}
    metrics[f'{prefix}_perplexity'] = get_ppl(prompts, gens, args)
    metrics[f'{prefix}_rep_2'], metrics[f'{prefix}_rep_3'], metrics[f'{prefix}_rep_4'], metrics[f'{prefix}_div_2'], metrics[f'{prefix}_div_3'] = compute_rep_and_div(gens)
    metrics[f'{prefix}_mauve'] = get_mauve(args, prompts, gens, targets)
    metrics[f'{prefix}_coherence'] = get_coherence(prompts, gens)
    metrics[f'{prefix}_similarity_gen'] = get_cosine_similarity(args, gens, targets)
    return metrics
