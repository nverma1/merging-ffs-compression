"""Wikitext-103 perplexity eval for OLMo3-7B merged/dropped models. Used for pre-finetune model selection."""
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-3-1025-7B"


def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]


def tie_layers(model, layer_indices):
    reference = layer_indices[0]
    for layer in layer_indices:
        if layer != reference:
            model.model.layers[layer].mlp.gate_proj.weight = model.model.layers[reference].mlp.gate_proj.weight
            model.model.layers[layer].mlp.up_proj.weight   = model.model.layers[reference].mlp.up_proj.weight
            model.model.layers[layer].mlp.down_proj.weight = model.model.layers[reference].mlp.down_proj.weight
    return model


def drop_layers(model, layer_indices):
    for i in sorted(layer_indices, reverse=True):
        del model.model.layers[i]
    model.config.num_hidden_layers -= len(layer_indices)
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.baseline:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)

    model.half()

    if args.tie_layers:
        layer_indices = get_layer_list(args.tie_layers)
        model = tie_layers(model, layer_indices)

    if args.drop_layers:
        layer_indices = get_layer_list(args.drop_layers)
        model = drop_layers(model, layer_indices)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test = load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = min(model.config.max_position_embeddings, 4096)
    stride = 512
    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}, max_length: {max_length}, stride: {stride}")
    print(f"memory footprint: {model.get_memory_footprint()}")

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"perplexity: {ppl.item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to merged model directory')
    parser.add_argument('--baseline', action='store_true', help='Eval the unmodified base model')
    parser.add_argument('--tie-layers', type=str, help='Comma-sep or range of layers to tie at eval time')
    parser.add_argument('--drop-layers', type=str, help='Comma-sep or range of layers to drop at eval time')
    parser.add_argument('--split', default='validation', type=str)
    args = parser.parse_args()
    main(args)
