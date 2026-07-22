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


def preprocess_function(sample, tokenizer):
    doc = sample['document']
    summary = doc["summary"]["text"] if isinstance(doc["summary"], dict) else doc["summary"]
    question = sample['question']['text'] if isinstance(sample['question'], dict) else sample['question']
    prompt = f"Story summary:\n{summary}\n\nQuestion: {question}\nAnswer:"
    model_inputs = tokenizer(prompt, max_length=2048, truncation=True)
    model_inputs['labels'] = [a['text'] for a in sample['answers']]
    return model_inputs


def evaluate_model(model, tokenizer, dataset):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    predictions, references = [], []
    for sample in tqdm(dataset):
        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        predictions.append(prediction)
        references.append(sample['labels'])
        print(f"Pred: {prediction}  |  Refs: {sample['labels']}")

    totals = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
    for pred, refs in zip(predictions, references):
        best_scores = None
        for ref in refs:
            s = scorer.score(ref, pred)
            if best_scores is None or s['rougeL'].fmeasure > best_scores['rougeL'].fmeasure:
                best_scores = s
        for k in totals:
            totals[k] += best_scores[k].fmeasure
    n = len(predictions)
    return {k: round(v / n * 100, 4) for k, v in totals.items()}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(
        args.model,
        base_model_name=args.base_model,
        drop_layers=args.drop_layers,
        tie_ff=args.tie_ff,
        quantization=args.quantize,
    )
    model.to(device)

    dataset = load_dataset("narrativeqa", split=args.split, trust_remote_code=True)
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), remove_columns=["document", "question", "answers"])

    scores = evaluate_model(model, tokenizer, dataset)
    print(f"NarrativeQA ROUGE scores: {scores}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to PEFT adapter')
    parser.add_argument('--base-model', type=str, default=None)
    parser.add_argument('--quantize', type=str, default=None, help='Quantization mode (e.g. 4bit)')
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--drop-layers', default=None, type=str)
    parser.add_argument('--tie-ff', default=None, type=str)
    args = parser.parse_args()
    main(args)
