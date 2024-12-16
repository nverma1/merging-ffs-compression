import os
import json
import argparse
import torch

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import GPT2Model, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sim_analysis.similarity import CKA

from utils import group_texts

# compute rbf or linear cka between two matrices
def compute_cka(matrix1, matrix2, max_toks=10000, type='linear'):
    sim = CKA()
    if max_toks:
        matrix1 = matrix1[:max_toks]
        matrix2 = matrix2[:max_toks]
    if type == 'rbf':
        x = sim.rbf_cka(matrix1, matrix2)
    elif type == 'linear':
        x = sim.linear_cka(matrix1, matrix2)
    # convert to numpy float 64
    x = np.float64(x)
    return x

# loop through attn and ff, and get cka for each layer
def cka_loop(info_dict, args):
    n_layers = len(info_dict['layers'])
    last_layer = n_layers
    sublayers = ['attn', 'ff']
    sims = {}
    for i in tqdm(range(n_layers)):
        sims[i] = {}
        for j in range(i, last_layer):
            sims[i][j] = {component: 0 for component in sublayers}
            for sublayer in sublayers:
                if i != j:
                    mat1 = info_dict['layers'][i][sublayer][args.get_residual_info]
                    mat2 = info_dict['layers'][j][sublayer][args.get_residual_info]
                    sims[i][j][sublayer] = compute_cka(mat1.numpy(), mat2.numpy(),
                                                       type=args.cka,
                                                       max_toks=args.max_toks)
                elif i == j:
                    sims[i][j][sublayer] = 1.0
    return sims

# 
def add_hooks_gpt2(model,tmp_residual_info):
    def activation_hook(info_tuple):
        def hook(model, input, output):
            tmp_residual_info['layers'][int(info_tuple[1])][info_tuple[0]] = output.detach().cpu()
        return hook
    for i, block in enumerate(model.h):
        block.attn.c_proj.register_forward_hook(activation_hook(('attn', i)))
        block.mlp.c_proj.register_forward_hook(activation_hook(('ff', i)))


def add_hooks_vit(model,tmp_residual_info):
    def activation_hook(info_tuple):
        def hook(model, input, output):
            tmp_residual_info['layers'][int(info_tuple[1])][info_tuple[0]] = output.detach().cpu()
        return hook
    for i, block in enumerate(model.vit.encoder.layer):
        block.attention.output.dense.register_forward_hook(activation_hook(('attn', i)))
        block.output.dense.register_forward_hook(activation_hook(('ff', i)))

def add_hooks_opusmt(model, tmp_residual_info):
    def activation_hook(info_tuple):
        def hook(model, input, output):
            tmp_residual_info['layers'][int(info_tuple[1])][info_tuple[0]] = output.detach().cpu()
        return hook
    for i, block in enumerate(model.model.encoder.layers):
        block.self_attn.out_proj.register_forward_hook(activation_hook(('attn', i)))
        block.fc2.register_forward_hook(activation_hook(('ff', i)))
    for i, block in enumerate(model.model.decoder.layers):
        n_enc = len(model.model.encoder.layers)
        block.self_attn.out_proj.register_forward_hook(activation_hook(('attn', i+n_enc)))
        block.fc2.register_forward_hook(activation_hook(('ff', i+n_enc)))
        

def load_wikitext(tokenizer, batch_size):
    wikitext = load_dataset('wikitext','wikitext-103-raw-v1', split='validation')
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True)
    tokenized_wikitext = wikitext.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
    )
    
    grouped_wikitext = tokenized_wikitext.map(group_texts, batched=True, num_proc=4)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        grouped_wikitext,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    return dataloader 

def load_tatoeba(tokenizer, batch_size, model):
    dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-zho", split='validation')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def load_vit(token):
    # load vit model and processor
    model_name = 'google/vit-base-patch16-224'
    model = ViTForImageClassification.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    token_string = open(token).read().strip()
    dataset = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True, token=token_string)

    def collate_fn(batch):
        # Extract images and labels
        images = [item['image'].convert('RGB')  for item in batch]
        labels = [item['label'] for item in batch]
        
        # Preprocess images
        inputs = image_processor(images, return_tensors='pt')
        
        # Return images and labels as tensors
        return inputs['pixel_values'], torch.tensor(labels)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    model_layers_pointer = model.vit.encoder.layer
    n_layers = len(model_layers_pointer)
    return model, dataloader, n_layers
    

def load_opusmt():
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader = load_tatoeba(tokenizer, args.batch_size, model)
    n_layers = len(model.model.encoder.layers) + len(model.model.decoder.layers)
    return model, dataloader, n_layers

def load_gpt2(model_name):
    if model_name == 'gpt2-large':
        model = GPT2Model.from_pretrained('gpt2-large')
    elif model_name == 'gpt2':
        model = GPT2Model.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load WikiText-103 dataset
    dataloader = load_wikitext(tokenizer, args.batch_size)
    model_layers_pointer = model.h
    n_layers = len(model_layers_pointer)
    return model, dataloader, n_layers



def main(args):

    os.environ['HF_HOME'] = args.hf_cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    res_type = args.get_residual_info
    residual_info = {}
    residual_info['type'] = res_type
    residual_info['layers'] = {}

    print('loading model and data')
    # get model and processor/tokenizer, dataloader, and model_layers_pointer
    if args.model_type == 'vit':
        model, dataloader, n_layers = load_vit(args.hf_token)

    # setup gpt2 instead
    elif args.model_type == 'gpt2' or args.model_type == 'gpt2-large':
        model, dataloader, n_layers = load_gpt2(args.model_type)

    elif args.model_type == 'opusmt':
        model, dataloader, n_layers = load_opusmt()
        tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')

    # initialize info_dict
    for i in range(n_layers):
        residual_info['layers'][i] = {}
        for component in ['ff', 'attn']:
            residual_info['layers'][i][component] = {res_type: torch.Tensor()}

    tmp_residual_info = deepcopy(residual_info)    

    # add pytorch hooks
    if args.model_type == 'vit':
        add_hooks_vit(model, tmp_residual_info)
    elif args.model_type == 'gpt2' or args.model_type == 'gpt2-large':
        add_hooks_gpt2(model, tmp_residual_info) 
    elif args.model_type == 'opusmt':
        add_hooks_opusmt(model, tmp_residual_info)
    
    
    model.eval()
    total_toks = 0         

    print('running forwards')
    model.to(device)
    total_toks = 0

    # cka_gram_mats = create_gram_dict(n_layers)
    with torch.no_grad():
        for batch in tqdm(dataloader):
             # Move inputs to GPU
            if args.model_type == 'vit':
                images, labels = batch
                _ = model(images.to(device))
            elif args.model_type == 'gpt2' or args.model_type == 'gpt2-large':
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                _ = model(input_ids = input_ids, attention_mask=attention_mask)
            elif args.model_type == 'opusmt':
                
                # keys = ['input_ids', 'attention_mask', 'labels']
                inputs = tokenizer(text=batch["targetString"], return_tensors="pt", padding=False, truncation=True, max_length=128)
                labels = tokenizer(text_target=batch["sourceString"], return_tensors="pt", padding=False, truncation=True, max_length=128)
                batch = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels['input_ids']}
                inputs = {k: v.to(device) for k, v in batch.items()}
                _ = model(**inputs)

            for i in range(n_layers):
                for component in ['attn', 'ff']:
                    tmp = tmp_residual_info['layers'][i][component]
                    tmp = tmp.reshape(-1, tmp.shape[-1])
                    residual_info['layers'][i][component][res_type] = torch.cat((residual_info['layers'][i][component][res_type], tmp), dim=0)


            total_toks += tmp_residual_info['layers'][0]['attn'].shape[0] * tmp_residual_info['layers'][0]['attn'].shape[1]
            print(total_toks, args.max_toks)
            if total_toks > args.max_toks:
                break
    print('computing ckas')
    sims = cka_loop(residual_info, args)
    with open(os.path.join(args.outdir, f'sims_{args.max_toks}_{args.cka}_updated.json'), 'w+') as f:
        json.dump(sims, f)
    print('done')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--get-residual-info",
        default='sublayer',
        required=False,
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--outdir",
    )
    parser.add_argument(
        "--max-toks",
        default=10000,
        type=int,
        help='max tokens to compute cka over'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='gpt2',
        required=False,
        help='Type of model'
    )
    parser.add_argument(
        '--cka',
        type=str,
        help='rbf or linear'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
    )
    parser.add_argument(
        '--hf-cache',
        type=str,
    )
    args = parser.parse_args()
    main(args)
