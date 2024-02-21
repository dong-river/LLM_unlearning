import copy
import transformers
import torch
import math
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, TrainerCallback
from peft import prepare_model_for_kbit_training, LoraConfig, PrefixTuningConfig, TaskType, PeftType, get_peft_model, get_peft_config, PeftModel, PeftConfig

def merge_model(model_1, model_2, weight_subtraction_coef, operation = 'subtraction'):
    tmp = copy.deepcopy(model_1)
    for param1, param2 in zip(model_1.parameters(), model_2.parameters()):
        if operation == 'subtraction':
            param1.data -= weight_subtraction_coef * param2.data
        else:
            raise NotImplementedError("operation not implemented")
    return model_1

def get_train_args(args):
    train_args = transformers.TrainingArguments(
            per_device_train_batch_size = args.train_batch_size, 
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=1,
            save_strategy="no",
            output_dir='outputs',
        )
    if "DI" in args.method:
        train_args.remove_unused_columns = False
    if args.gradient_accu > 1:
        train_args.gradient_accumulation_steps = args.gradient_accu
    if args.warmup_steps > 0:
        train_args.warmup_steps = args.warmup_steps
    if args.weight_decay > 0:
        train_args.weight_decay = args.weight_decay
    # if args.lr_sc:
    #     train_args.learning_rate = args.lr_sc
    return train_args

def get_train_args_di(args):    
    train_args = transformers.TrainingArguments(
            per_device_train_batch_size = args.train_batch_size // 2, 
            num_train_epochs=args.num_epochs_di,
            learning_rate=args.lr_di,
            fp16=True,
            save_strategy="no",
            output_dir='outputs',
            evaluation_strategy="steps",      # Evaluation is done at the end of each epoch
            eval_steps=100,     
            logging_dir="./logs",             # Directory for storing logs
            logging_steps=1,  
        )
    if "DI" in args.method:
        train_args.remove_unused_columns = False
    if args.gradient_accu > 1:
        train_args.gradient_accumulation_steps = args.gradient_accu
    if args.warmup_steps > 0:
        train_args.warmup_steps = args.warmup_steps
    if args.weight_decay > 0:
        train_args.weight_decay = args.weight_decay
    return train_args

def get_lora_model(model, rank=8, lora_alpha=16):
    if 'gpt_neo' in str(type(model)):
        target_modules = ["q_proj", "v_proj"]
    elif 'llama' in str(type(model)):
        target_modules = ["q_proj", "v_proj"]
    else:
        raise NotImplementedError("specify lora layer in ft_utils.py")
    
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def load_peft_model(args, model, method):
    if method == 'lora':
        return get_lora_model(model, rank=args.rank, lora_alpha=args.lora_alpha)
    elif method == 'adapter':
        raise NotImplementedError

class GradientAscentTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        return -1 * super(GradientAscentTrainer,self).compute_loss(model, inputs, return_outputs=False)
    
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, initial_perplexity, ppl_change):
        self.best_perplexity = initial_perplexity
        self.ppl_change = ppl_change
        self.early_stop_epoch = None
        print("Initial Perplexity: {}".format(self.best_perplexity))

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        current_perplexity = math.exp(kwargs['metrics']['eval_loss'])
        if current_perplexity > self.ppl_change * self.best_perplexity and state.epoch >= 1:
            print("Perplexity increased, stopping training at epoch {}".format(state.epoch))
            control.should_training_stop = True
            self.early_stop_epoch = state.epoch  # Record the epoch number
        print("Current Perplexity: {}, Best Perplexity: {} at epoch {}".format(current_perplexity, self.best_perplexity, state.epoch))


class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        if "focus_idx" in batch[0]:
            focus_idx = torch.stack([item["focus_idx"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "focus_idx": focus_idx}    
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

class DITrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        di_kwargs = kwargs.pop('di_kwargs')
        super().__init__(*args, **kwargs)
        self.di_kwargs = di_kwargs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unlearn_teacher_model = AutoModelForCausalLM.from_pretrained(di_kwargs['teacher_model']).to(self.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids = inputs['input_ids'].to(self.device), attention_mask = inputs['attention_mask'].to(self.device))
        logits = outputs.logits
        input_ids = inputs['input_ids'].to(self.device)
        shift_labels = input_ids[..., 1:].contiguous().to(self.device)
        shift_logits = logits[..., :-1, :].contiguous().to(self.device)       

        with torch.no_grad():
            teacher_logits = self.unlearn_teacher_model(input_ids = input_ids, attention_mask = inputs['attention_mask']).logits
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        
        mask = torch.zeros_like(shift_logits).to(self.device)
        mask[torch.arange(mask.shape[0]).view(-1, 1, 1), torch.arange(mask.shape[1]).view(1, -1, 1), shift_labels.unsqueeze(-1)] = 1
        pre_softmax = shift_teacher_logits - mask * self.di_kwargs['di_strength']
        soft_label = F.softmax(pre_softmax, dim=-1)
        
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), soft_label.view(-1, soft_label.size(-1)))
        
        if self.di_kwargs['focus']:
            print("Focused Unlikelihood Training")
            # focus_idx = inputs['focus_idx']
            shift_focus_idx = inputs['focus_idx'] - 1   
            shift_focus_idx[shift_focus_idx < 0] = 0  
            shift_focus_idx[shift_focus_idx >= 199] = 0 
            shift_focus_idx = shift_focus_idx[:,: -1]
            
            reweight_vector = torch.zeros(shift_focus_idx.size(0), shift_focus_idx.size(1))            
            rows, cols = torch.meshgrid(torch.arange(shift_focus_idx.size(0)), torch.arange(shift_focus_idx.size(1)))
            rows = rows.to(self.device)
            row_indices = rows[shift_focus_idx!=0].flatten()
            col_indices = shift_focus_idx.flatten()
            col_indices = col_indices[col_indices!=0]
            
            ## check col_indices out of bound
            if not (col_indices < shift_teacher_logits.size(1)).all():
                print("col_indices out of bound")
                import pdb; pdb.set_trace()
            
            if self.di_kwargs['focus_hard']:
                reweight_vector[row_indices.long(), col_indices.long()] = 1
                reweight_vector = reweight_vector.flatten().to(self.device)
            else:
                reweight_vector[row_indices.long(), col_indices.long()] = self.di_kwargs['focus_coeff'] - 1
                reweight_vector = reweight_vector.flatten().to(self.device) + 1
            loss = loss * reweight_vector
        return loss.mean()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Setting the model to evaluation mode
        self.model.eval()

        total_loss = 0
        total_examples = 0

        for batch in self.get_eval_dataloader(eval_dataset):
            # Move batch to device
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            batch['labels'] = batch['input_ids']

            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_examples += batch['input_ids'].size(0)

        # Calculate average loss
        avg_loss = total_loss / total_examples

        # Calculate perplexity
        perplexity = math.exp(avg_loss)
        print(f"Perplexity: {perplexity}")
        logs = {"eval_loss": avg_loss, "perplexity": perplexity}
        if self.callback_handler is not None:
            self.callback_handler.on_evaluate(self.args, self.state, self.control, logs)

        return {"eval_loss": avg_loss, "perplexity": perplexity}
