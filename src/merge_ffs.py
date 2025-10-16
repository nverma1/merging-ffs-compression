import argparse 
import torch
import scipy
from utils import get_layer_list, load_model, model_param_names

"""
Computes the permutation matrices, based off of the correlation matrices. 
Includes assigment problem to find the optimal permutation.
"""

def compute_permutation_matrix(correlation_matrices, layer_list, ref_type, reference_only=False, scaling=None):
    permutations = {}
    costs = {}
    if ref_type == 'first':
        reference = int(layer_list[0])
    elif ref_type == 'last':
        reference = int(layer_list[-1])
    elif ref_type == 'middle':
        # if even number of layers, take the lower middle
        reference = int(layer_list[len(layer_list)//2])
    
    scale = None
    # TODO(neha): if scaling, can make corrs absval since negative scale will be handled by scales
    for i in layer_list:
        if int(i) == reference:
            # do not change the reference layer
            cost = 0
            permutations[int(i)] = torch.eye(correlation_matrices[int(i)][int(i)].shape[0])
        else:
            if reference_only:
                corr_mat = correlation_matrices[reference][int(i)]
                if scaling is not None:
                    scale = scaling[reference][int(i)]
            else:
                # here, we need to get the correct corr matrix because only upper triangle is stored
                if int(i) < reference:
                    corr_mat = correlation_matrices[int(i)][reference].T
                    if scaling is not None:
                        scale = scaling[int(i)][reference]
                else:
                    corr_mat = correlation_matrices[reference][int(i)]
                    if scaling is not None:
                        scale = scaling[reference][int(i)]
            dim = corr_mat.shape[0]
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mat, maximize=True) #reference neuron → neuron in layer i.
            cost = float(corr_mat[row_ind, col_ind].sum()) 
            new_mat = torch.eye(dim)[col_ind] # transforms layer i to reference layer        
            if scale is not None:
                new_mat = new_mat * scale
            permutations[int(i)] = new_mat
        costs[i] = cost
    return permutations, costs
 
'''
Applies the permutation matrices to the model.
Applies "unpermutation" matrices as well via transpose. 
'''
def apply_permutations(model, permutations, layer_list, layer_idxs, model_type, transpose=False):
    fc1_name = model_param_names[model_type]['fc1']
    has_gate = False
    if 'fc1_gate' in model_param_names[model_type]:
        has_gate = True
        fc1_gate_name = model_param_names[model_type]['fc1_gate']
    fc2_name = model_param_names[model_type]['fc2']
    for idx, layer in zip(layer_idxs, layer_list):
        perm = permutations[int(idx)]
        unperm = torch.where(perm != 0, 1.0 / perm, torch.zeros_like(perm)).T
        assert torch.allclose(perm @ unperm, torch.eye(perm.shape[0]), atol=1e-5)
        if transpose == True:
            model[layer + f'{fc1_name}.weight'] = (perm @ model[layer + f'{fc1_name}.weight'].T).T
            model[layer + f'{fc2_name}.weight'] =  (model[layer + f'{fc2_name}.weight'].T @ unperm).T
            if has_gate:
                model[layer + f'{fc1_gate_name}.weight'] = (perm @ model[layer + f'{fc1_gate_name}.weight'].T).T
        else:
            model[layer + f'{fc1_name}.weight'] = perm @ model[layer + f'{fc1_name}.weight']
            model[layer + f'{fc2_name}.weight'] =  model[layer + f'{fc2_name}.weight'] @ unperm
            if has_gate:
                model[layer + f'{fc1_gate_name}.weight'] = perm @ model[layer + f'{fc1_gate_name}.weight']
        if model_param_names[model_type]['has_bias'] == True:
            model[layer + f'{fc1_name}.bias'] = perm @ model[layer + f'{fc1_name}.bias']
    return model

def get_ff_weights(model_type):
    weights = [f"{model_param_names[model_type]['fc1']}.weight", f"{model_param_names[model_type]['fc2']}.weight"]
    if 'fc1_gate' in model_param_names[model_type]:
        weights.append(f"{model_param_names[model_type]['fc1_gate']}.weight")
    if model_param_names[model_type]['has_bias'] == True:
        weights.extend([f"{model_param_names[model_type]['fc1']}.bias", f"{model_param_names[model_type]['fc2']}.bias"])
    return weights

def get_weight_norms(model, layer_list, model_type):
    weights = get_ff_weights(model_type)
    norms = {}
    for weight in weights:
        for layer in layer_list:
            tensor = model[layer + weight]
            norms[layer + weight] = tensor.norm()
    return norms
'''
After application of merge matrices, FFs are averaged in this step. 
Replacement of old FFs by merged FF occurs here as well. 
'''

def merge_ffs(model, layer_list, model_type, norms=None):
    # layer_list is a list of dictionaries
    weights = get_ff_weights(model_type)
    # go through the name of the weights in FF layers  
    for weight in weights:
        sum = None
        for layer in layer_list:        
            tensor = model[layer + weight]
            sum = sum + tensor if sum is not None else tensor
        avg = sum / len(layer_list)
        avg_norm = None
        if norms is not None:
            avg_norm = avg.norm()
        # replace layers
        for layer in layer_list:
            if norms is not None: 
                prev_norm = norms[layer + weight]
                tmp_avg = avg * (prev_norm / avg_norm)
            else:
                tmp_avg = avg
            model[layer + weight] = tmp_avg
    return model


def main(args):
    model = load_model(args.model_type)
    model_dict = model.state_dict()
    starting_param_count = sum([model_dict[key].numel() for key in model_dict.keys()])

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

    all_scales = None  
    if args.scaling is not None:
        all_scales = torch.load(args.scaling)

    if args.corrs is not None:
        corrs = torch.load(args.corrs)
        if args.encoder_layers != None:
            corrs_enc = corrs['enc']
            enc_scales = None
            if args.scaling is not None:
                enc_scales = all_scales['enc']
            encoder_layer_idxs = get_layer_list(args.encoder_layers)
            merge_matrices_enc, _ = compute_permutation_matrix(corrs_enc, encoder_layer_idxs, ref_type=args.ref_type, reference_only=args.reference_only, scaling=enc_scales)
            model_dict = apply_permutations(model_dict, merge_matrices_enc, encoder_layer_list, encoder_layer_idxs, args.model_type, transpose=args.transpose)
        if args.decoder_layers != None: 
            corrs_dec = corrs['dec']
            dec_scales = None
            if args.scaling is not None:
                dec_scales = all_scales['dec']
            decoder_layer_idxs = get_layer_list(args.decoder_layers)
            merge_matrices_dec, _ = compute_permutation_matrix(corrs_dec, decoder_layer_idxs, ref_type=args.ref_type,  reference_only=args.reference_only, scaling=dec_scales)
            model_dict = apply_permutations(model_dict, merge_matrices_dec, decoder_layer_list, decoder_layer_idxs, args.model_type, transpose=args.transpose)

    if args.save_permuted:
        model.load_state_dict(model_dict)
        model.save_pretrained(args.output)
        return 

    prev_norms = None
    if args.normalize:
        if args.encoder_layers != None:
            prev_norms = get_weight_norms(model_dict, encoder_layer_list, args.model_type)
        if args.decoder_layers != None:
            prev_norms = get_weight_norms(model_dict, decoder_layer_list, args.model_type)
    
    if args.encoder_layers != None: 
        model_dict = merge_ffs(model_dict, encoder_layer_list, args.model_type, norms=prev_norms)
    if args.decoder_layers != None:
        model_dict = merge_ffs(model_dict, decoder_layer_list, args.model_type, norms=prev_norms)

    end_param_count = sum([model_dict[key].numel() for key in model_dict.keys()])
    print('ratio of parameters:', end_param_count/starting_param_count)

    model.load_state_dict(model_dict)
    model.save_pretrained(args.output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute merged feed-forward layers')
    parser.add_argument('--model-type', type=str, required=True, default='vit', help='Type of model')
    parser.add_argument('--encoder-layers', type=str, required=False, help='layer string')
    parser.add_argument('--decoder-layers', type=str, required=False, help='layer string')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    parser.add_argument('--corrs', type=str, required=False, help='Path to the similarity matrices')
    parser.add_argument('--ref-type', type=str, required=False, help='Type of reference layer (first, last, mixed)', default='first')
    parser.add_argument('--reference-only', action='store_true', help='Whether the corr dict is just reference based')
    parser.add_argument('--transpose', action='store_true', help='Whether to transpose the weight matrices')
    parser.add_argument('--scaling', type=str, help='Path to the scaling factors', default=None)
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize the weight matrices')
    parser.add_argument('--save-permuted', action='store_true', help='Whether to save the permuted model')

    args = parser.parse_args()
    main(args)
