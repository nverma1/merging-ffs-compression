import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel, 
    GPT2TokenizerFast, 
    BitsAndBytesConfig
)

# adapted from  https://huggingface.co/docs/transformers/perplexity


def load_quantized(args):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    if args.baseline:
        model_name = 'gpt2-large'
    else:
        model_name = args.model
    model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=quantization_config,
    device_map="auto")
    return model

def remove_layers(model, layers):
    model.transformer.h = torch.nn.ModuleList([model.transformer.h[i] for i in range(len(model.transformer.h)) if i not in layers])
    return model 

def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.quantize:
        model = load_quantized(args)
    elif args.baseline:
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    elif args.drop_layers:
        model = GPT2LMHeadModel.from_pretrained(args.model)
        layer_list = get_layer_list(args.drop_layers)
        model = remove_layers(model, layer_list)
        state_dict = torch.load(args.model)
        model.load_state_dict(state_dict)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model)
        
    if not args.quantize:
        model.to(device)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')

    # load dataset and tokenize
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    # get ppl 
    max_length = model.config.n_positions
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
    parser.add_argument('--quantize', action='store_true', help='Whether to quantize the model')
    parser.add_argument('--split', default='validation', type=str, help='Dataset split to evaluate on')
    parser.add_argument('--drop-layers', default=None, type=str, help='Layer indices to drop')

    
    args = parser.parse_args()
    main(args)
