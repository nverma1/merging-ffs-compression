import torch
import numpy as np
import datasets 
from datasets import load_dataset

from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OlmoForCausalLM
from transformers import MarianMTModel, AutoTokenizer
from torch.utils.data import DataLoader

from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForSeq2Seq
from functools import partial


# this can be updated to accommodate more model types
model_param_names = {
    'vit':{
        'encoder_prefix': 'vit.encoder.layer',
        'decoder_prefix': None,
        'fc1': 'intermediate.dense',
        'fc2': 'output.dense',
        'has_bias': True
    },
    'opusmt':{
        'encoder_prefix': 'model.encoder.layers',
        'decoder_prefix': 'model.decoder.layers',
        'fc1': 'fc1',
        'fc2': 'fc2',
        'has_bias': True
    },
    'gpt2-large':{
        'encoder_prefix': None,
        'decoder_prefix': 'transformer.h',
        'fc1': 'mlp.c_fc',
        'fc2': 'mlp.c_proj',
        'has_bias': True
    },
    'olmo':{
        'encoder_prefix': None,
        'decoder_prefix': 'model.layers',
        'fc1': 'mlp.up_proj',
        'fc1_gate': 'mlp.gate_proj',
        'fc2': 'mlp.down_proj',
        'has_bias': False
    },
    'olmo1b':{
        'encoder_prefix': None,
        'decoder_prefix': 'model.layers',
        'fc1': 'mlp.up_proj',
        'fc1_gate': 'mlp.gate_proj',
        'fc2': 'mlp.down_proj',
        'has_bias': False
    }
}


# this must also be updated for new model types
dim_map = {
    'vit': 3072,
    'gpt2-large': 5120,
    'opusmt':2048,
    'olmo': 11008,
    'olmo1b': 8192,
}


'''
layer manipulation helpers
'''
import operator
from functools import reduce

def get_submodule(model, path):
    """Fetch nested submodule or parameter via dot-separated path."""
    return reduce(getattr, path.split('.'), model)

def set_submodule(model, path, value):
    """Set nested submodule or parameter via dot-separated path."""
    parts = path.split('.')
    target = reduce(getattr, parts[:-1], model)
    setattr(target, parts[-1], value)

def tie_weights(model, layers_to_tie, model_type, is_decoder=True):
    """
    Tie FFN weights across specified layers using model_param_names.
    
    Args:
        model: The model object.
        layers_to_tie: List of layer indices to tie.
        model_type: String key from model_param_names.
        is_decoder: Whether to use decoder_prefix (True) or encoder_prefix (False).
    """

    config = model_param_names[model_type]
    prefix_keys = []
    if config['decoder_prefix'] is not None:
        prefix_keys.append(config['decoder_prefix'])
    if config['encoder_prefix'] is not None:
        prefix_keys.append(config['encoder_prefix'])

    for prefix in prefix_keys:
        def build_path(layer_idx, component):
            if prefix:
                return f"{prefix}.{layer_idx}.{component}"
            else:
                return f"{layer_idx}.{component}"

        reference = layers_to_tie[0]
        for layer in layers_to_tie:
            if layer == reference:
                continue
            
            for key in ['fc1', 'fc2']:
                if key in config:
                    src_path = build_path(reference, config[key])
                    tgt_path = build_path(layer, config[key])
                    set_submodule(model, tgt_path + '.weight', get_submodule(model, src_path + '.weight'))
                    if config.get('has_bias', False):
                        set_submodule(model, tgt_path + '.bias', get_submodule(model, src_path + '.bias'))

            # Special case: models like Olmo have fc1 split into gate + up projections
            if 'fc1_gate' in config:
                gate_path = config['fc1_gate']
                src_path = build_path(reference, gate_path)
                tgt_path = build_path(layer, gate_path)
                set_submodule(model, tgt_path + '.weight', get_submodule(model, src_path + '.weight'))

def get_all_layers(model, model_name):
    enc_layers, dec_layers = None, None
    if model_param_names[model_name]['encoder_prefix'] is not None:
        enc_layers = get_submodule(model, model_param_names[model_name]['encoder_prefix'] )
    if model_param_names[model_name]['decoder_prefix'] is not None:
        dec_layers = get_submodule(model, model_param_names[model_name]['decoder_prefix'] )
 
    layers = [enc_layers, dec_layers]

    return layers

# Retrieve a layer using its dotted path
def get_layers_by_path(model, layer_name):
    for part in layer_name.split('.'):
        model = getattr(model, part)
    return model

# Set a layer using its dotted path
def set_layers_by_path(model, layer_name, new_layer):
    parts = layer_name.split('.')
    for part in parts[:-1]:
        model = getattr(model, part)
    setattr(model, parts[-1], new_layer)

# remove layers from a model
def drop_layers(model, enc_indices, dec_indices, model_type):
    if enc_indices is not None:
        enc_prefix = model_param_names[model_type]['encoder_prefix']
        encoder_layers = get_layers_by_path(model, enc_prefix)
        filtered_layers = [layer for i, layer in enumerate(encoder_layers) if i not in enc_indices]
        set_layers_by_path(model, enc_prefix, torch.nn.ModuleList(filtered_layers))
    if dec_indices is not None:
        dec_prefix = model_param_names[model_type]['decoder_prefix']
        decoder_layers = get_layers_by_path(model, dec_prefix)
        filtered_layers = [layer for i, layer in enumerate(decoder_layers) if i not in dec_indices]
        set_layers_by_path(model, dec_prefix, torch.nn.ModuleList(filtered_layers))
    return model

# obtains a list of layer indices from a string input
def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]

# get a tuple with number of layers for encoder and decoder, handles models with no encoder or decoder
def get_num_layers(model, model_name):
    enc_layers = 0
    dec_layers = 0
    if model_param_names[model_name]['encoder_prefix'] is not None:
        enc_layers = len(get_layers_by_path(model, model_param_names[model_name]['encoder_prefix']))
    if model_param_names[model_name]['decoder_prefix'] is not None:
        dec_layers = len(get_layers_by_path(model, model_param_names[model_name]['decoder_prefix']))
    return (enc_layers, dec_layers)

# checks if a model has an encoder or decoder
def has_encoder(model_name):
    return model_param_names[model_name]['encoder_prefix'] is not None

def has_decoder(model_name):
    return model_param_names[model_name]['decoder_prefix'] is not None

'''
data helpers
'''

def get_dataloader(model_type, tokenizer, batch_size, token_string='', seq_len=None):
    if model_type == 'vit':
        return load_imagenet(tokenizer, token_string=token_string, batch_size=batch_size)
    elif model_type == 'gpt2-large':
        return load_wikitext(tokenizer, batch_size=batch_size, seq_len=seq_len)
    elif model_type == 'opusmt':
        return prepare_tatoeba_dataloader(tokenizer, batch_size=batch_size, seq_len=seq_len)
    
# preprocessing for wikitext
# original source: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
def group_texts(examples, block_size=512):
    
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result
        
# load wikitext dataset
def load_wikitext(tokenizer, batch_size, seq_len=512):
    wikitext = datasets.load_dataset('wikitext','wikitext-103-raw-v1', split='validation')
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True)
    tokenized_wikitext = wikitext.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
    )
    grouped_wikitext = tokenized_wikitext.map(
        partial(group_texts, block_size=seq_len), 
        batched=True, 
        num_proc=4)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        grouped_wikitext,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    return dataloader 

# load imagenet, needs hf token for gated dataset
def load_imagenet(image_processor, token_string='', batch_size=8):
    imagenet_valid = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True, token=token_string, trust_remote_code=True)
    def collate_fn(batch):
        # Extract images and labels
        images = [item['image'].convert('RGB')  for item in batch]
        labels = [item['label'] for item in batch]
        
        # Preprocess images
        inputs = image_processor(images, return_tensors='pt')
        
        # Return images and labels as tensors
        return inputs['pixel_values'], torch.tensor(labels)
    dataloader = DataLoader(imagenet_valid, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader

def load_olmomix(tokenizer, batch_size):
    dataset = load_dataset("allenai/olmo-mix-1124", split)

# load tatoeba data, loads zh-en direction
def prepare_tatoeba_dataloader(tokenizer, batch_size, seq_len=None):
    dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-zho", split="validation")
    # we need to flip it around because we are doing zh-en
    def preprocess_function(examples):
        # Tokenize source texts
        model_inputs = tokenizer(text=examples['targetString'], max_length=512, truncation=True)
        model_inputs["src_length"] = [len(ids) for ids in model_inputs["input_ids"]]
        # Tokenize target texts with labels
        labels = tokenizer(text_target=examples['sourceString'], max_length=512, truncation=True)
        model_inputs["tgt_length"] = [len(ids) for ids in labels["input_ids"]]

        # Add the labels to the model inputs
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    if seq_len is not None:
        tokenized_dataset = tokenized_dataset.filter(
            lambda ex:  len(ex["input_ids"]) <= seq_len and len(ex["labels"]) <= seq_len
        )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'src_length', 'tgt_length']) #

    if seq_len is not None:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, max_length=seq_len, padding='max_length')
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader


'''
activation helpers
'''

# converts a covariance matrix to a correlation matrix
def cov_to_corr(cov, std1, std2):
    outer_std = np.outer(std1, std2)
    # Avoid division by zero and handle nan values
    outer_std = np.clip(np.nan_to_num(outer_std), a_min=1e-7, a_max=None)
    corr = cov / outer_std
    return corr

# loads a model given a name for it
# can update this to include more models
def load_model(model_type, model_path=None):
    if model_type == 'vit':
        model_name = 'google/vit-base-patch16-224'
        if model_path:
            return ViTForImageClassification.from_pretrained(model_path)
        return ViTForImageClassification.from_pretrained(model_name)
    elif model_type == 'gpt2-large':
        model_name = 'gpt2-large'
        if model_path:
            return GPT2LMHeadModel.from_pretrained(model_path)
        return GPT2LMHeadModel.from_pretrained(model_name)
    elif model_type == 'opusmt':
        model_name = "Helsinki-NLP/opus-mt-zh-en"
        if model_path:
            return MarianMTModel.from_pretrained(model_path)
        return MarianMTModel.from_pretrained(model_name)
    elif model_type == 'olmo':
        model_name = 'allenai/OLMo-7B-0724-hf'
        model = OlmoForCausalLM.from_pretrained(model_name)
    elif model_type == 'olmo1b':
        model_name = 'allenai/OLMo-2-0425-1B'
        model = OlmoForCausalLM.from_pretrained(model_name)
    return model

# loads a tokenizer given a model type
# can update this to include more models
def load_tokenizer(model_type):
    if model_type == 'vit':
        model_name = 'google/vit-base-patch16-224'
        tokenizer = ViTImageProcessor.from_pretrained(model_name)
    elif model_type == 'gpt2-large':
        model_name = 'gpt2-large'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif model_type == 'opusmt':
        model_name = "Helsinki-NLP/opus-mt-zh-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == 'olmo':
        model_name = 'allenai/OLMo-7B-0724-hf'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == 'olmo1b':
        model_name = 'allenai/OLMo-2-0425-1B'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer