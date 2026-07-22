import torch
import argparse 
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from tqdm import tqdm 
from datasets import load_dataset

import math
import os
import glob
from itertools import chain

import evaluate
from transformers import get_inverse_sqrt_schedule, AdamW
from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


# adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py 
def find_latest_checkpoint(output_dir):
    # List all directories that match the checkpoint pattern
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None

    # Sort directories by creation time and return the latest one
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint

def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]

def load_wikitext(tokenizer, split='train'):
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    return dataset

# tie feedforwards in gpt2
def tie_weights(model, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie: 
        if layer != reference:
            model.transformer.h[layer].mlp.c_fc.weight = model.transformer.h[reference].mlp.c_fc.weight
            model.transformer.h[layer].mlp.c_fc.bias = model.transformer.h[reference].mlp.c_fc.bias
            model.transformer.h[layer].mlp.c_proj.bias = model.transformer.h[reference].mlp.c_proj.bias
            model.transformer.h[layer].mlp.c_proj.weight = model.transformer.h[reference].mlp.c_proj.weight

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

def main(args):
    if args.large:
        model_name = 'gpt2-large'
    else:
        model_name = 'gpt2'

    # load model 
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    if args.baseline is False:
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict)

    # print original params 
    orig_params = model.num_parameters()
    print(f'Original params {orig_params}')

    # tie feed-forward layers and print new param count
    if args.tie_ff:
        layers_to_tie = get_layer_list(args.tie_ff)
        tie_weights(model, layers_to_tie)
    if args.drop_layers:
        layer_list = get_layer_list(args.drop_layers)
        model.transformer.h = torch.nn.ModuleList([model.transformer.h[i] for i in range(len(model.transformer.h)) if i not in layer_list])
    compressed_params = model.num_parameters()
    print(f'Compressed params {compressed_params}')
    print(f'ratio {compressed_params / orig_params}')

    # data 
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    
    block_size = tokenizer.model_max_length
    if block_size > max_pos_embeddings:
        if max_pos_embeddings > 0:
            block_size = min(1024, max_pos_embeddings)
        else:
            block_size = 1024

    def tokenize_function(examples):
        output = tokenizer(examples['text'])
         # Filter out examples with empty 'input_ids'
        output = {k: [v for v in output[k] if len(v) > 0] for k in output}
        return output
    
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

    train_dataset = load_wikitext(tokenizer, 'train')
    val_dataset = load_wikitext(tokenizer, 'validation')
    column_names = train_dataset.column_names

    grouped_train = train_dataset.map(tokenize_function, batched=True, remove_columns=column_names).map(group_texts, batched=True)
    grouped_val = val_dataset.map(tokenize_function, batched=True, remove_columns=column_names).map(group_texts, batched=True)

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
        save_steps=3000,
        max_steps=args.num_updates,
        logging_steps=100,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=2,
        save_safetensors=False,
    )
    class CustomTrainer(Trainer):

        def get_eval_dataloader(self, eval_dataset=None):
            # Reinitialize the streaming dataset
            eval_dataset = load_wikitext(self.tokenizer, split='validation')
            eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=column_names).map(group_texts, batched=True)
            return super().get_eval_dataloader(eval_dataset)


    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0)
    trainer = CustomTrainer(
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
    
    latest_checkpoint = find_latest_checkpoint(args.outdir)
    if latest_checkpoint:
        print('resuming from checkpoint')
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print('from scratch')
        train_results = trainer.train()
    train_results = trainer.train(resume_from_checkpoint=True)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
    parser.add_argument('--large', action='store_true', help='Use large model')
    parser.add_argument('--num-updates', type=int, default=100000, help='Number of updates')
    parser.add_argument('--tie-ff', help='Comma sep list of which ff to tie')
    parser.add_argument('--outdir', type=str, help='Output directory',)
    parser.add_argument('--drop-layers', type=str)

    args = parser.parse_args()



    main(args)

