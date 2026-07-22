from tqdm import tqdm 
from datasets import load_dataset

import math
import os
import glob
from itertools import chain
import torch 
import datasets 

from transformers import get_inverse_sqrt_schedule, AdamW
from transformers import (
    OlmoForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig,
    default_data_collator,
)
import argparse 
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 

# adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py 
def find_latest_checkpoint(output_dir):
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))

def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]
    
# tie feedforwards in olmo
def tie_weights(model, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie: 
        if layer != reference:
            model.model.layers[layer].mlp.gate_proj.weight = model.model.layers[reference].mlp.gate_proj.weight
            model.model.layers[layer].mlp.up_proj.weight = model.model.layers[reference].mlp.up_proj.weight
            model.model.layers[layer].mlp.down_proj.weight = model.model.layers[reference].mlp.down_proj.weight

def load_dolma(tokenizer):
    dolma_dataset = datasets.load_dataset('allenai/dolma', 'v1_6-sample', split='train', trust_remote_code=True, streaming=True)
    def tokenize_function(examples):
        return tokenizer(examples['text'])
    dolma_dataset = dolma_dataset.map(tokenize_function, batched=True, remove_columns=["text", "id", "added", "created", "source"])
    # dolma_dataset['labels'] = dolma_dataset['input_ids'].copy()
    return dolma_dataset

def load_wikitext(tokenizer, split='train'):
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    def tokenize_function(examples):
        return tokenizer(examples['text'])
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return dataset

def wrap_peft(model, args):
    if ',' in args.lora_modules:
        target_modules = args.lora_modules.split(',')
    else:
        target_modules = args.lora_modules
    peft_config = LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_a, bias="none", lora_dropout=0.1, target_modules=target_modules)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def remove_layers(model, remove_layers):
    n_layers = len(model.model.layers) - remove_layers
    print(f'model has {n_layers} layers')
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in range(n_layers)])
    return model

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def main(args):
    # load model 
    model_name = 'allenai/OLMo-1B-0724-hf'
    if args.model and args.drop_layers == False:
        model = OlmoForCausalLM.from_pretrained(args.model)
    else:
        model = OlmoForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
   
    orig_params = model.num_parameters()
    print(f'Original params {orig_params}')

    # tie feed-forward layers and print new param count
    if args.tie_ff:
        layers_to_tie = get_layer_list(args.tie_ff)
        tie_weights(model, layers_to_tie)
    if args.drop_layers:
        model = remove_layers(model, args.drop_layers)
    compressed_params = model.num_parameters()
    print(f'Compressed params {compressed_params}')
    print(f'ratio {compressed_params / orig_params}')

    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    # get peft model 

    # if this is 4096 it is too big 
    block_size = 2048
 
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return {}
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        accuracy = float((preds == labels).mean())
        return {"accuracy": accuracy}

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result    
    
    if args.peft:
        #model = prepare_model_for_kbit_training(model)
        model.half()
        model = wrap_peft(model, args)
    elif args.qlora:
        model = prepare_model_for_kbit_training(model)

        model = wrap_peft(model, args)


    #train_dataset = load_dolma(tokenizer)
    train_dataset = load_wikitext(tokenizer, split='train')
    val_dataset = load_wikitext(tokenizer, split='validation')

    grouped_train = train_dataset.map(group_texts, batched=True)     
    grouped_val = val_dataset.map(group_texts, batched=True)

    #model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        do_train=True,
        do_eval=True,        
        fp16=True,
        logging_dir='./logs',
        eval_strategy='steps',
        eval_steps=1000,
        save_steps=1000,
        max_steps=args.num_updates,
        logging_steps=100,
        save_safetensors=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
    )
    # if training_args.gradient_checkpointing:
    #     training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped_train,
        eval_dataset=grouped_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,       
        optimizers=(optimizer, scheduler)
    )

    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer.save_model()
    # trainer.model.print_trainable_parameters()
    # if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    #     from peft.utils.other import fsdp_auto_wrap_policy

    #     fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    #     fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # unfreeze only teh ff layers:
    if args.ff_tune:
        for i, param in enumerate(model.named_parameters()):
            if 'mlp' not in param[0]:
                param[1].requires_grad = False
    elif args.tune_all:
        for i, param in enumerate(model.named_parameters()):
            param[1].requires_grad = True

    latest_checkpoint = find_latest_checkpoint(args.outdir)
    if latest_checkpoint:
        print('resuming from checkpoint')
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print('from scratch')
        train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    metrics = trainer.evaluate()

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
    parser.add_argument('--num-updates', type=int, default=20000, help='Number of updates')
    parser.add_argument('--tie-ff', help='Comma sep list of which ff to tie')
    parser.add_argument('--outdir', type=str, help='Output directory',)
    parser.add_argument('--drop-layers', type=int)
    parser.add_argument('--peft', action='store_true', help='Use peft')
    parser.add_argument('--ff-tune', action='store_true', help='tune only ff layers')
    parser.add_argument('--tune-all', action='store_true', help='tune all layers')
    parser.add_argument('--qlora', action='store_true', help='quantize model')
    parser.add_argument(
        '--lora-modules',
        type=str,
        default='all-linear'
    )
    parser.add_argument(
        '--lora-a',
        type=int,
        default=None
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=None
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)

