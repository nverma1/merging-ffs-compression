import math
import os
import glob
import argparse
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

MODEL_NAME = "allenai/OLMo-3-1025-7B"


class DataCollatorForCompletionOnlyLM:
    def __init__(self, tokenizer, response_template):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors='pt')
        labels = batch['input_ids'].clone()
        tlen = len(self.response_template_ids)
        for i, seq in enumerate(labels):
            found = False
            for j in range(len(seq) - tlen + 1):
                if seq[j:j+tlen].tolist() == self.response_template_ids:
                    labels[i, :j+tlen] = -100
                    found = True
                    break
            if not found:
                labels[i] = torch.full_like(labels[i], -100)
        labels[batch['attention_mask'] == 0] = -100
        batch['labels'] = labels
        return batch


# ---------------------------------------------------------------------------
# Per-task dataset loading and prompt formatting.
#
# Each task provides a loader returning (train, validation) splits and a
# formatting function mapping an example to a prompt whose completion begins at
# `response_template` (everything before it is masked out during training).
# ---------------------------------------------------------------------------


def _completion(label, text, eos_token):
    """Render the completion segment, appending the eos token when non-empty."""
    return f"{label}: {text + ' ' + eos_token if text else ''}"


def load_samsum():
    return (load_dataset('samsum', trust_remote_code=True, split='train'),
            load_dataset('samsum', trust_remote_code=True, split='validation'))


def format_samsum(example, eos_token):
    return f"Summarize the following:\n{example['dialogue']}\n" + _completion("Summary", example["summary"], eos_token)


def load_narrativeqa():
    return (load_dataset('narrativeqa', split='train', trust_remote_code=True),
            load_dataset('narrativeqa', split='validation', trust_remote_code=True))


def format_narrativeqa(example, eos_token):
    doc = example["document"]
    q = example["question"]
    answers = example["answers"]
    summary = doc["summary"]["text"] if isinstance(doc["summary"], dict) else doc["summary"]
    question_text = q["text"] if isinstance(q, dict) else q
    answer_text = answers[0]["text"] if answers else ""
    return f"Story summary:\n{summary}\n\nQuestion: {question_text}\n" + _completion("Answer", answer_text, eos_token)


def load_hotpotqa():
    return (load_dataset('hotpot_qa', 'distractor', split='train', trust_remote_code=True),
            load_dataset('hotpot_qa', 'distractor', split='validation', trust_remote_code=True))


def _format_hotpot_context(context):
    parts = []
    for title, sentences in zip(context['title'], context['sentences']):
        para = ' '.join(sentences)
        parts.append(f"{title}: {para}")
    return '\n'.join(parts)


def format_hotpotqa(example, eos_token):
    background = _format_hotpot_context(example["context"])
    return f"Background:\n{background}\n\nQuestion: {example['question']}\n" + _completion("Answer", example["answer"], eos_token)


@dataclass
class TaskSpec:
    load: Callable[[], tuple]         # () -> (train_dataset, val_dataset)
    format_example: Callable          # (example, eos_token) -> str
    response_template: str
    batch_size: int
    grad_accum: int
    max_length: int


TASKS = {
    "samsum": TaskSpec(load_samsum, format_samsum, "Summary:", batch_size=4, grad_accum=2, max_length=2048),
    "narrativeqa": TaskSpec(load_narrativeqa, format_narrativeqa, "Answer:", batch_size=2, grad_accum=4, max_length=2048),
    "hotpotqa": TaskSpec(load_hotpotqa, format_hotpotqa, "Answer:", batch_size=2, grad_accum=4, max_length=2048),
}


# ---------------------------------------------------------------------------
# Compression + training (shared across all tasks)
# ---------------------------------------------------------------------------


def find_latest_checkpoint(output_dir):
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=os.path.getmtime)


def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]


def tie_weights(model, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie:
        if layer != reference:
            model.model.layers[layer].mlp.gate_proj.weight = model.model.layers[reference].mlp.gate_proj.weight
            model.model.layers[layer].mlp.up_proj.weight = model.model.layers[reference].mlp.up_proj.weight
            model.model.layers[layer].mlp.down_proj.weight = model.model.layers[reference].mlp.down_proj.weight


def drop_specific_layers(model, layer_indices):
    for i in sorted(layer_indices, reverse=True):
        del model.model.layers[i]
    model.config.num_hidden_layers -= len(layer_indices)
    return model


def make_peft_config(args):
    if ',' in args.lora_modules:
        target_modules = args.lora_modules.split(',')
    else:
        target_modules = args.lora_modules
    return LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_a, bias="none", lora_dropout=0.2, target_modules=target_modules)


def main(args):
    task = TASKS[args.task]

    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                     quantization_config=bnb_config,
                                                     torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    orig_params = model.num_parameters()
    print(f'Original params {orig_params}')

    if args.tie_ff:
        tie_weights(model, get_layer_list(args.tie_ff))
    if args.drop_layers:
        model = drop_specific_layers(model, get_layer_list(args.drop_layers))

    compressed_params = model.num_parameters()
    print(f'Compressed params {compressed_params}')
    print(f'ratio {compressed_params / orig_params}')

    peft_config = None
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
        peft_config = make_peft_config(args)
    else:
        model.half()

    train_dataset, val_dataset = task.load()

    training_args = SFTConfig(
        output_dir=args.outdir,
        per_device_train_batch_size=task.batch_size,
        per_device_eval_batch_size=task.batch_size,
        gradient_accumulation_steps=task.grad_accum,
        eval_accumulation_steps=4,
        bf16=True,
        max_length=task.max_length,
        gradient_checkpointing=True,
        logging_dir='./logs',
        eval_strategy='steps',
        eval_steps=50,
        save_steps=50,
        save_strategy='steps',
        logging_steps=1,
        max_steps=args.num_updates,
        save_total_limit=2,
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

    eos_token = tokenizer.eos_token

    def formatting_function(example):
        return task.format_example(example, eos_token)

    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=task.response_template)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(range(100)),
        processing_class=tokenizer,
        formatting_func=formatting_function,
        data_collator=collator,
        peft_config=peft_config,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=sorted(TASKS.keys()),
                        help='Downstream task / dataset to fine-tune on')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--num-updates', type=int, default=3000)
    parser.add_argument('--tie-ff', help='Comma sep list or range of layers to tie')
    parser.add_argument('--outdir', type=str, help='Output directory')
    parser.add_argument('--drop-layers', type=str)
    parser.add_argument('--qlora', action='store_true')
    parser.add_argument('--lora-modules', type=str, default='all-linear')
    parser.add_argument('--lora-a', type=int, default=None)
    parser.add_argument('--lora-r', type=int, default=None)
    parser.add_argument('--lr-scheduler', type=str, default='constant')
    args = parser.parse_args()

    main(args)
