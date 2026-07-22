import argparse
import re
import string
import torch
from collections import Counter
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


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def tie_weights(model, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie:
        if layer != reference:
            model.model.layers[layer].mlp.gate_proj.weight = model.model.layers[reference].mlp.gate_proj.weight
            model.model.layers[layer].mlp.up_proj.weight = model.model.layers[reference].mlp.up_proj.weight
            model.model.layers[layer].mlp.down_proj.weight = model.model.layers[reference].mlp.down_proj.weight
    return model


def load_model(path, base_model_name=None, drop_layers=None, tie_ff=None, quantization=None):
    base_model_name = base_model_name or MODEL_NAME
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tie_ff:
        base_model = tie_weights(base_model, get_layer_list(tie_ff))

    if drop_layers:
        layer_list = get_layer_list(drop_layers)
        base_model = remove_layers(base_model, layer_list)

    return PeftModel.from_pretrained(base_model, path).eval(), tokenizer


def format_context(context):
    parts = []
    for title, sentences in zip(context['title'], context['sentences']):
        para = ' '.join(sentences)
        parts.append(f"{title}: {para}")
    return '\n'.join(parts)


def preprocess_function(sample, tokenizer):
    background = format_context(sample['context'])
    prompt = f"Background:\n{background}\n\nQuestion: {sample['question']}\nAnswer:"
    model_inputs = tokenizer(prompt, max_length=2048, truncation=True)
    model_inputs['labels'] = sample['answer']
    return model_inputs


def evaluate_model(model, tokenizer, dataset):
    em_scores, f1_scores = [], []
    for sample in tqdm(dataset):
        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        reference = sample['labels']
        em_scores.append(float(exact_match(prediction, reference)))
        f1_scores.append(f1_score(prediction, reference))
        print(f"Pred: {prediction}  |  Ref: {reference}")

    return {
        "exact_match": round(100 * sum(em_scores) / len(em_scores), 4),
        "f1": round(100 * sum(f1_scores) / len(f1_scores), 4),
    }


def main(args):
    model, tokenizer = load_model(args.model, base_model_name=args.base_model, drop_layers=args.drop_layers, tie_ff=args.tie_ff, quantization=args.quantize)

    dataset = load_dataset("hotpot_qa", "distractor", split=args.split, trust_remote_code=True)
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), remove_columns=["id", "question", "answer", "type", "level", "supporting_facts", "context"])

    scores = evaluate_model(model, tokenizer, dataset)
    print(f"HotPotQA scores: {scores}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to PEFT adapter')
    parser.add_argument('--quantize', type=str, default=None, help='Quantization mode (e.g. 4bit)')
    parser.add_argument('--split', default='validation', type=str)
    parser.add_argument('--base-model', type=str, default=None)
    parser.add_argument('--drop-layers', default=None, type=str)
    parser.add_argument('--tie-ff', default=None, type=str)
    args = parser.parse_args()
    main(args)
