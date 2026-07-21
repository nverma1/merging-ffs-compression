import argparse
import torch
from rouge_score import rouge_scorer
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODEL_NAME = "allenai/OLMo-3-1025-7B"


def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]


def remove_layers(model, layers):
    model.model.layers = torch.nn.ModuleList(
        [model.model.layers[i] for i in range(len(model.model.layers)) if i not in layers]
    )
    return model


def tie_weights(model, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie:
        if layer != reference:
            model.model.layers[layer].mlp.gate_proj.weight = model.model.layers[reference].mlp.gate_proj.weight
            model.model.layers[layer].mlp.up_proj.weight = model.model.layers[reference].mlp.up_proj.weight
            model.model.layers[layer].mlp.down_proj.weight = model.model.layers[reference].mlp.down_proj.weight
    return model


def load_model(path, base_model_name=None, drop_layers=None, tie_ff=None):
    base_model_name = base_model_name or MODEL_NAME
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tie_ff:
        base_model = tie_weights(base_model, get_layer_list(tie_ff))
    if drop_layers:
        base_model = remove_layers(base_model, get_layer_list(drop_layers))

    return PeftModel.from_pretrained(base_model, path).eval(), tokenizer


def generate_prompt(dialogue):
    return f"Summarize the following:\n{dialogue}\nSummary:"


def preprocess_function(sample, tokenizer):
    prompt = generate_prompt(sample['dialogue'])
    model_inputs = tokenizer(prompt, max_length=1024, truncation=True)
    model_inputs['labels'] = sample['summary']
    return model_inputs


def evaluate_model(model, tokenizer, dataset):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    totals = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}

    for sample in tqdm(dataset):
        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        reference = sample['labels']
        scores = scorer.score(reference, prediction)
        for k in totals:
            totals[k] += scores[k].fmeasure
        print(f"Pred: {prediction}  |  Ref: {reference}")

    n = len(dataset)
    return {k: round(v / n * 100, 4) for k, v in totals.items()}


def main(args):
    model, tokenizer = load_model(args.model, base_model_name=args.base_model,
                                  drop_layers=args.drop_layers, tie_ff=args.tie_ff)

    dataset = load_dataset('samsum', trust_remote_code=True, split=args.split)
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer),
                          remove_columns=['id', 'dialogue', 'summary'])

    scores = evaluate_model(model, tokenizer, dataset)
    print(f"SamSum ROUGE scores: {scores}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--base-model', type=str, default=None)
    parser.add_argument('--drop-layers', default=None, type=str)
    parser.add_argument('--tie-ff', default=None, type=str)
    args = parser.parse_args()
    main(args)
