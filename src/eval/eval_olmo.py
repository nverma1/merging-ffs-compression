import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    OlmoForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
)
import os 
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


# adapted from  https://huggingface.co/docs/transformers/perplexity

def load_quantized(args):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    if args.baseline:
        model_name = 'allenai/OLMo-1B-0724-hf'
    else:
        model_name = args.model
    model = OlmoForCausalLM.from_pretrained(model_name, quantization_config=quantization_config,
    device_map="auto")
    return model

def remove_layers(model, layers):
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in range(len(model.model.layers)) if i not in layers])
    return model

def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]
    

def wrap_peft(model):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["down_proj", "up_proj", "gate_proj"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def rename_lora_weights(state_dict):
    # Rename keys to match PEFT's expected format
    new_lora_weights = {}
    for key in state_dict.keys():
        if 'lora' in key:
            new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
            new_lora_weights[new_key] = state_dict[key]
        else:
            new_lora_weights[key] = state_dict[key]
    state_dict = None
    return new_lora_weights

def rename_base_weights(state_dict):
    new_base_weights = {}
    for key in state_dict.keys():
        new_key = key.replace("model.", "base_model.model.model.")
        new_base_weights[new_key] = state_dict[key]
    else:
        new_base_weights[key] = state_dict[key]
    state_dict = None
    return new_base_weights

def tie_layers(model, layer_indices, reference=None):
    if reference is None:
        reference = layer_indices[0]
    for layer in layer_indices:
        if layer != reference:
            model.model.layers[layer].mlp.gate_proj.weight = model.model.layers[reference].mlp.gate_proj.weight
            model.model.layers[layer].mlp.up_proj.weight = model.model.layers[reference].mlp.up_proj.weight
            model.model.layers[layer].mlp.down_proj.weight = model.model.layers[reference].mlp.down_proj.weight
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.quantize:
        model = load_quantized(args)
        if args.tie_layers:
            layer_indices = get_layer_list(args.tie_layers)
            model = tie_layers(model, layer_indices)
        breakpoint()
    else:
        model = OlmoForCausalLM.from_pretrained('allenai/OLMo-1B-0724-hf')
        model.half()
    if not args.baseline and not args.quantize and not args.drop_layers:
        if os.path.isdir(args.model):
            model = OlmoForCausalLM.from_pretrained(args.model)
        else:
            state_dict = torch.load(args.model)
            model.load_state_dict(state_dict)
        if args.tie_layers:
            layer_indices = get_layer_list(args.tie_layers)
            model = tie_layers(model, layer_indices)
        if args.peft:
            model = PeftModel.from_pretrained(model, args.peft)
        model.half()
    

    if args.drop_layers: 
        breakpoint()
        layer_list = get_layer_list(args.drop_layers)
        model = remove_layers(model, layer_list)
        if not args.baseline:
            # check if args.modle is file or directory 
            if os.path.isdir(args.model):
                state_dict = torch.load(args.model + '/pytorch_model.bin')
            else:
                state_dict = torch.load(args.model)

            model.load_state_dict(state_dict)
        
    if not args.quantize:
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-1B-0724-hf')

    test = load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)
    print('seq_len ', seq_len)
    
    model.eval()
    print(model.get_memory_footprint())

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ppl of GPT2 model')
    parser.add_argument('--model', type=str, required=False, help='Path to model')
    parser.add_argument('--baseline', action='store_true', help='Whether to use the baseline model')
    parser.add_argument('--better-baseline', action='store_true', help='Whether to use the better baseline model')
    parser.add_argument('--quantize', action='store_true', help='Whether to quantize the model')
    parser.add_argument('--split', default='validation', type=str, help='Dataset split to evaluate on')
    parser.add_argument('--drop-layers', default=None, type=str, help='Layer indices to drop')
    parser.add_argument('--peft', type=str, required=False, help='Path to peft model')
    parser.add_argument('--tie-layers', type=str, help='Comma separated list of layers to tie')

    
    args = parser.parse_args()
    main(args)
