import argparse 
import math
import torch
import glob
import os 

import evaluate
from datasets import load_dataset
import datasets
from datasets import Dataset

from transformers import (
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    get_inverse_sqrt_schedule,
    AdamW,
)
import numpy as np
from datasets import load_metric

from torch.optim.lr_scheduler import LambdaLR

# Load BLEU metric
bleu_metric = load_metric("sacrebleu")

# adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py 

def find_latest_checkpoint(output_dir):
    # List all directories that match the checkpoint pattern
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None

    # Sort directories by creation time and return the latest one
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint
'''
Tie feedforwards in one side of MT model. side like encoder or decoder is 
passed in
'''
def tie_weights(model_side, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie: 
        if layer != reference:
            model_side.layers[layer].fc1.weight = model_side.layers[reference].fc1.weight
            model_side.layers[layer].fc1.bias = model_side.layers[reference].fc1.bias
            model_side.layers[layer].fc2.weight = model_side.layers[reference].fc2.weight
            model_side.layers[layer].fc2.bias = model_side.layers[reference].fc2.bias

def load_opus( split, dataset=None):
    if split == 'train' and dataset == None:
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh", split=split)
    elif split == 'train' and dataset == 'un_pc':
        dataset = load_dataset("Helsinki-NLP/un_pc", "en-zh", split=split)
    elif split == 'validation':
        dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-zho", split=split)
    # elif split=='validation':
    #     dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh", split=split)
    return dataset

def load_opus_from_file(split, dataset=None, data_dir='data'):
    filepath = os.path.join(data_dir, 'release/v2020-07-28/eng-zho')

    with open(filepath + '/train.trg', 'r') as f:
        train_zh = f.readlines()
    with open(filepath + '/train.src', 'r') as f:
        train_en = f.readlines()
    assert len(train_zh) == len(train_en)

    data_dict = {'translation': [{'zh': zh.strip(), 'en': en.strip()} for zh, en in zip(train_zh, train_en)]}
    return Dataset.from_dict(data_dict)



# based from https://huggingface.co/docs/transformers/v4.17.0/en/tasks/translation 
def preprocess_function(examples, tokenizer, split='train'):
    if split == 'train':
        inputs= [example['zh'] for example in examples['translation']]
        targets = [example['en'] for example in examples['translation']]
    else:
        inputs = [example for example in examples['targetString']]
        targets = [example for example in examples['sourceString']]
    model_inputs = tokenizer(text=inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]  # BLEU expects a list of references (list of lists)
    return preds, labels


def remove_layers(model, layers):
    model.model.encoder.layers = torch.nn.ModuleList([model.model.encoder.layers[i] for i in range(len(model.model.encoder.layers)) if i not in layers])
    model.model.decoder.layers = torch.nn.ModuleList([model.model.decoder.layers[i] for i in range(len(model.model.decoder.layers)) if i not in layers])
    return model



def main(args):

    model_name = 'Helsinki-NLP/opus-mt-zh-en'

    config = AutoConfig.from_pretrained(model_name)
    if args.dropout:
        config.dropout =args.dropout # Set the dropout rate (e.g., 0.1)
        config.attention_dropout =args.dropout # Set attention dropout rate (if applicable)
    # load model 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.baseline is False:
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict)

    # print original params 
    orig_params = model.num_parameters()
    print(f'Original params {orig_params}')

    # tie feed-forward layers and print new param count
    if args.enc_tie_ff:
        layers_to_tie = get_layer_list(args.enc_tie_ff)
        tie_weights(model.model.encoder, layers_to_tie)
    if args.dec_tie_ff:
        layers_to_tie = get_layer_list(args.dec_tie_ff)
        tie_weights(model.model.decoder, layers_to_tie)

    if args.drop_layers:
        layer_list = get_layer_list(args.drop_layers)
        model = remove_layers(model, layer_list)
    compressed_params = model.num_parameters()
    print(f'Compressed params {compressed_params}')
    print(f'ratio {compressed_params / orig_params}')

    if args.dataset == 'full':
        # first time
        # train_dataset = load_opus_from_file('train')
        # processed_train = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer, split='train'), batched=True)
        # processed_train.save_to_disk(os.path.join(args.data_dir, 'opus-zh-en-full.hf'))
        processed_train = datasets.load_from_disk(os.path.join(args.data_dir, 'opus-zh-en-full.hf'))
        val_dataset = load_opus('validation')
        processed_val = val_dataset.map(lambda examples: preprocess_function(examples, tokenizer, split='validation'), batched=True)
    else:
        train_dataset = load_opus('train')
        val_dataset = load_opus('validation')
        processed_val = val_dataset.map(lambda examples: preprocess_function(examples, tokenizer, split='validation'), batched=True)


    # # comment this when done 
    # print('loaded dataset')
    # processed_train = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer, split='train'), batched=True)
    # print('processed dataset')


    # processed_train.save_to_disk(os.path.join(args.data_dir, 'opus-zh-en-full.hf'))
    # # end comment here 

    # # get a subsample of the validation set for evaluation because otherwise it is too large 
    validation_subset = processed_val.train_test_split(test_size=2000, seed=42)['test']

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        fp16=True,
        do_train=True,
        do_eval=True,
        logging_dir='./logs',
        eval_strategy='steps',
        eval_steps=3000,
        save_steps=3000,
        max_steps=args.num_updates,
        logging_steps=100,
        save_total_limit=1,
        save_safetensors=False,
        load_best_model_at_end=True,
        max_grad_norm=1.0, # adding gradient clipping 9.16
        predict_with_generate=True,
        metric_for_best_model='eval_bleu',
        greater_is_better=True,
    )
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0)


    # Define a function to compute BLEU
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        with tokenizer.as_target_tokenizer():
            # Replace -100s used for padding as we can't decode them
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
        
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=validation_subset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics  # Pass the compute_metrics function
    )
    print('set up trainer')

    latest_checkpoint = find_latest_checkpoint(args.outdir)
    if latest_checkpoint:
        print('resuming from checkpoint')
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print('from scratch')
        train_results = trainer.train()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    metrics = trainer.evaluate(processed_val)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue pretraining gpt2')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
    parser.add_argument('--large', action='store_true', help='Use large model')
    parser.add_argument('--num-updates', type=int, default=20000, help='Number of updates')
    parser.add_argument('--enc-tie-ff', help='Comma sep list of which enc ff to tie')
    parser.add_argument('--dec-tie-ff', help='Comma sep list of which dec ff to tie')
    parser.add_argument('--outdir', type=str, help='Output directory',)
    parser.add_argument('--dataset', type=str, help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the prepared OPUS/Tatoeba data')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--drop-layers', type=str, help='Layer indices to drop')


    args = parser.parse_args()



    main(args)

