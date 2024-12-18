import argparse 
import torch
import scipy
import json
import numpy as np
from transformers import ViTForImageClassification
from transformers import GPT2LMHeadModel
from transformers import MarianMTModel
from utils import get_layer_list, load_model, model_param_names

'''
Computes the merge matrices, based off of the correlation matrices. 
Includes assigment problem to find the optimal merge.
'''
def compute_merge_matrix(correlation_matrices, layer_list, ref_type, reference_only=False):
    merges = {}
    costs = None
    if ref_type == 'first':
        reference = int(layer_list[0])
    elif ref_type == 'last':
        reference = int(layer_list[-1])
    elif ref_type == 'middle':
        reference = int(layer_list[len(layer_list)//2])
    
    for i in layer_list:
        if int(i) == reference:
            merges[int(i)] = torch.eye(correlation_matrices[int(i)][int(i)].shape[0])
        else:
            if reference_only:
                corr_mat = correlation_matrices[reference][int(i)]
            else:
                if int(i) < reference:
                    corr_mat = correlation_matrices[int(i)][reference].T
                else:
                    corr_mat = correlation_matrices[reference][int(i)]
            dim = corr_mat.shape[0]
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mat, maximize=True)
            cost = float(corr_mat[row_ind, col_ind].sum())
            new_mat = torch.eye(dim)[col_ind]         
            merges[int(i)] = new_mat
    return merges, costs

'''
Applies the merge matrices to the model.
Applies "unmerge" matrices as well via transpose. 
'''
def apply_merges(model, merges, layer_list, layer_idxs, model_type, transpose=False):
    fc1_name = model_param_names[model_type]['fc1']
    fc2_name = model_param_names[model_type]['fc2']
    for idx, layer in zip(layer_idxs, layer_list):
        merge = merges[int(idx)]
        if transpose == True:
            model[layer + f'{fc1_name}.weight'] = (merge @ model[layer + f'{fc1_name}.weight'].T).T
            model[layer + f'{fc2_name}.weight'] =  (model[layer + f'{fc2_name}.weight'].T @ merge.T).T
        else:
            model[layer + f'{fc1_name}.weight'] = merge @ model[layer + f'{fc1_name}.weight']
            model[layer + f'{fc2_name}.weight'] =  model[layer + f'{fc2_name}.weight'] @ merge.T
        model[layer + f'{fc1_name}.bias'] = merge @ model[layer + f'{fc1_name}.bias']
    return model

'''
After application of merge matrices, FFs are averaged in this step. 
Replacement of old FFs by merged FF occurs here as well. 
'''
def merge_ffs(model, layer_list, model_type):
    # layer_list is a list of dictionaries
    weights = [f"{model_param_names[model_type]['fc1']}.weight", f"{model_param_names[model_type]['fc2']}.weight"]
    weights.extend([f"{model_param_names[model_type]['fc1']}.bias", f"{model_param_names[model_type]['fc2']}.bias"])
    # go through weights 
    for weight in weights:
        sum = None
        for layer in layer_list:        
            tensor = model[layer + weight]
            sum = sum + tensor if sum is not None else tensor
        avg = sum / len(layer_list)
    # replace layers
        for layer in layer_list:
            model[layer + weight]= avg
    return model



def main(args):
    model = load_model(args.model_type)
    model_dict = model.state_dict()

    if args.encoder_layers != None:
        encoder_layer_idxs = get_layer_list(args.encoder_layers)
        encoder_layer_list = []
        for idx in encoder_layer_idxs:
            enc_prefix = model_param_names[args.model_type]['encoder_prefix']
            encoder_layer_list.append(f'{enc_prefix}.{idx}.')

    if args.decoder_layers != None:
        decoder_layer_idxs = get_layer_list(args.decoder_layers)
        decoder_layer_list = []
        for idx in decoder_layer_idxs:
            dec_prefix = model_param_names[args.model_type]['decoder_prefix']
            decoder_layer_list.append(f'{dec_prefix}.{idx}.')


    if args.corrs is not None:
        corrs = torch.load(args.corrs)
        if args.encoder_layers != None:
            corrs_enc = corrs['enc']
            encoder_layer_idxs = get_layer_list(args.encoder_layers)
            merge_matrices_enc, _ = compute_merge_matrix(corrs_enc, encoder_layer_idxs, ref_type=args.ref_type, reference_only=args.reference_only)
            model_dict = apply_merges(model_dict, merge_matrices_enc, encoder_layer_list, encoder_layer_idxs, args.model_type, transpose=args.transpose)
        if args.decoder_layers != None: 
            corrs_dec = corrs['dec']
            decoder_layer_idxs = get_layer_list(args.decoder_layers)
            merge_matrices_dec, _ = compute_merge_matrix(corrs_dec, decoder_layer_idxs, ref_type=args.ref_type,  reference_only=args.reference_only)
            model_dict = apply_merges(model_dict, merge_matrices_dec, decoder_layer_list, decoder_layer_idxs, args.model_type, transpose=args.transpose)

    if args.encoder_layers != None:
        model_dict = merge_ffs(model_dict, encoder_layer_list, args.model_type)
    if args.decoder_layers != None:
        model_dict = merge_ffs(model_dict, decoder_layer_list, args.model_type)


    torch.save(model_dict, args.output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute merged feed-forward layers')
    parser.add_argument('--model-type', type=str, required=True, default='vit', help='Type of model, use this or model')
    parser.add_argument('--encoder-layers', type=str, required=False, help='layer string')
    parser.add_argument('--decoder-layers', type=str, required=False, help='layer string')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    parser.add_argument('--corrs', type=str, required=False, help='Path to the correlation matrices')
    parser.add_argument('--ref-type', type=str, required=False, help='Type of reference layer (first, last, mixed)', default='first')
    parser.add_argument('--reference-only', action='store_true', help='Whether the corr dict is just reference based')
    parser.add_argument('--transpose', action='store_true', help='Whether to transpose the weight matrices')
    

    args = parser.parse_args()
    main(args)

    
