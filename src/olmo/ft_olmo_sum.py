
import math
import os
import glob
import torch 
import evaluate
from transformers import (
    OlmoForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse 
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 
from nltk.tokenize import sent_tokenize


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
    
# tie feedforwards in olmo
def tie_weights(model, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie: 
        if layer != reference:
            model.model.layers[layer].mlp.gate_proj.weight = model.model.layers[reference].mlp.gate_proj.weight
            model.model.layers[layer].mlp.up_proj.weight = model.model.layers[reference].mlp.up_proj.weight
            model.model.layers[layer].mlp.down_proj.weight = model.model.layers[reference].mlp.down_proj.weight


'''
dropped layer models get reloaded as 0-n layers, + enough to fit original model. The rest are
randomly initialized. This function removes the extra layers, given # layers to remove
'''
def remove_layers(model, remove_layers):
    n_layers = len(model.model.layers) - remove_layers
    print(f'model has {n_layers} layers')
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in range(n_layers)])
    return model


def load_samsum(split='train'):
    dataset = load_dataset('samsum', trust_remote_code=True, split=split)
    return dataset

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def generate_prompt(dialogue, summary=None, eos_token='<|endoftext|>'):
  instruction = "Summarize the following:\n"
  input = f"{dialogue}\n"
  summary = f"Summary: {summary + ' ' + eos_token if summary else ''}"
  prompt = ("").join([instruction, input, summary])
  return prompt 

def wrap_peft(model, args):
    if ',' in args.lora_modules:
        target_modules = args.lora_modules.split(',')
    else:
        target_modules = args.lora_modules
    peft_config = LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_a, bias="none", lora_dropout=0.2, target_modules=target_modules)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

def main(args):

    # load model 
    model_name = 'allenai/OLMo-7B-0724-hf'
    if args.qlora:
        # load model with quantization config for 4 bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.float16
        )
        model = OlmoForCausalLM.from_pretrained(args.model, 
                                                quantization_config=bnb_config,
                                                torch_dtype=torch.float16)
    else:
        model = OlmoForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    orig_params = model.num_parameters()
    print(f'Original params {orig_params}')

    # model is saved with same layers, just make sure they are tied. Same with drop, cannot load dropped model, need to 
    # redrop the final layers (they load uninitialized)
    if args.tie_ff:
        layers_to_tie = get_layer_list(args.tie_ff)
        tie_weights(model, layers_to_tie)
    if args.drop_layers:
        model = remove_layers(model, args.drop_layers)
  
    # print compression ratio
    compressed_params = model.num_parameters()
    print(f'Compressed params {compressed_params}')
    print(f'ratio {compressed_params / orig_params}')

    # this is a necessary step for qlora fine-tuning
    peft_config = None
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
        model, peft_config = wrap_peft(model, args)
    else:
        model.half()
   
    # load model and eval
    train_dataset = load_samsum('train')
    val_dataset = load_samsum('validation')
    metric = evaluate.load("rouge")

    # this ends up not getting used but leaving for demonstrative purposes 
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        return result

    #model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=4,
        fp16=True,
        logging_dir='./logs',
        eval_strategy='steps',
        eval_steps=50,
        save_steps=50,
        save_strategy='steps',
        logging_steps=1,
        max_steps=args.num_updates,
        save_total_limit=2,
        save_safetensors=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        learning_rate=5e-4,
        warmup_ratio=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=0.01,
    )

    def formatting_function(prompt):
        output = []
        for d, s in zip(prompt["dialogue"], prompt["summary"]):
            full_prompt = generate_prompt(d, s)
            output.append(full_prompt)
        return output
    
    response_template = 'Summary:'

    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(range(100)),
        processing_class=tokenizer,
        peft_config= peft_config,
        formatting_func=formatting_function,
        data_collator=collator,
        #compute_metrics=compute_metrics,
    )

    latest_checkpoint = find_latest_checkpoint(args.outdir)
    if latest_checkpoint:
        print('resuming from checkpoint')
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print('from scratch')
        train_results = trainer.train()

    trainer.save_model(args.outdir)
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
    parser.add_argument('--qlora', action='store_true', help='quantize model')
    parser.add_argument('--lora-modules', type=str, default='all-linear')
    parser.add_argument('--lora-a', type=int, default=None)
    parser.add_argument('--lora-r', type=int, default=None)
    parser.add_argument('--lr-scheduler', type=str, default='constant')
    args = parser.parse_args()

    main(args)

