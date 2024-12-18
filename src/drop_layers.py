import argparse 
import torch
from transformers import ViTForImageClassification
from transformers import GPT2LMHeadModel
from transformers import MarianMTModel
from utils import get_layer_list,  drop_layers, load_model


def main(args):
    model = load_model(args.model_type)
    orig_param_count = sum(p.numel() for p in model.parameters())
    enc_indices = None
    dec_indices = None
    
    if args.encoder_layers != None:
        encoder_layer_idxs = get_layer_list(args.encoder_layers)
        enc_indices = encoder_layer_idxs

    if args.decoder_layers != None:
        decoder_layer_idxs = get_layer_list(args.decoder_layers)
        dec_indices = decoder_layer_idxs

    model = drop_layers(model, enc_indices, dec_indices, args.model_type)
    new_param_count = sum(p.numel() for p in model.parameters())
    print(f'ratio: {new_param_count/orig_param_count}')
    
    torch.save(model.state_dict(), args.output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute merged feed-forward layers')
    parser.add_argument('--model-type', type=str, required=True, default='mt', help='Type of model, use this or model')
    parser.add_argument('--encoder-layers', type=str, required=False, help='layer string')
    parser.add_argument('--decoder-layers', type=str, required=False, help='layer string')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')

    args = parser.parse_args()
    main(args)

    
