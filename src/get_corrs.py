import torch
import argparse
import os
import numpy as np

from tqdm import tqdm

from utils import (
    cov_to_corr, 
    load_model, 
    load_tokenizer, 
    load_wikitext, 
    get_num_layers,
    prepare_tatoeba_dataloader,
    load_imagenet,
    has_encoder,
    has_decoder,
    get_layers_by_path,
)
from utils import (
    dim_map,
    model_param_names
)

outer_prods = {}
means = {}
total_tokens = 0
TOL=1e-5


def instantiate_trackers(dimension=2048, sides=['dec'], enc_layers=0, dec_layers=0, device='cpu', layer_range=None, reference=None):
    global outer_prods
    global means
    for side in sides:
        side_layers = enc_layers if side == 'enc' else dec_layers
        outer_prods[side] = {}
        means[side] = {}
        if layer_range is not None:
            for j in layer_range:
                outer_prods[side][j] = {}
                means[side][j] = torch.zeros(dimension).to(device)
                for k in range(j, layer_range[-1]+1):
                    outer_prods[side][j][k] = torch.zeros(dimension, dimension).to(device) 
        elif reference is not None:
            outer_prods[side][reference] = {}
            means[side][reference] = torch.zeros(dimension).to(device)
            for j in range(side_layers):
                if j != reference:
                    outer_prods[side][j] = {}
                    means[side][j] = torch.zeros(dimension).to(device)
                    outer_prods[side][reference][j] = torch.zeros(dimension, dimension).to(device) 
                outer_prods[side][j][j] = torch.zeros(dimension, dimension).to(device)


def finalize_correlations(outdir, max_toks, layer_range=None, reference=None, sides=['dec']):
    global outer_prods
    global means
    corrs = {}
    covs = {}
    stds = {}
    for side in sides:
        corrs[side] = {}
        stds[side] = {}
        covs[side] = {}

        # compute means 
        if layer_range is not None:
            for j in layer_range:
                means[side][j] = means[side][j].div(total_tokens)
        elif reference is not None:
            for j in range(len(outer_prods[side][reference])):
                means[side][j] = means[side][j].div(total_tokens)

        # compute covariances
        if layer_range is not None:
            for j in layer_range:
                covs[side][j] = {}
                for k in range(j, layer_range[-1]+1):
                    outer_prods[side][j][k] = outer_prods[side][j][k].div(total_tokens)
                    cov = outer_prods[side][j][k] - torch.outer(means[side][j], means[side][k])
                    covs[side][j][k] = cov
        elif reference is not None:
            covs[side][reference] = {}
            for j in range(len(outer_prods[side][reference])):
                outer_prods[side][reference][j] = outer_prods[side][reference][j].div(total_tokens)
                cov = outer_prods[side][reference][j] - torch.outer(means[side][reference], means[side][j])
                covs[side][reference][j] = cov
            for j in range(len(outer_prods[side][reference])):
                if j != reference:
                    covs[side][j] = {}
                    outer_prods[side][j][j] = outer_prods[side][j][j].div(total_tokens)
                    cov = outer_prods[side][j][j] - torch.outer(means[side][j], means[side][j])
                    covs[side][j][j] = cov
        # compute standard deviations
        if layer_range is not None:
            for j in layer_range:
                stds[side][j] = torch.sqrt(torch.diag(covs[side][j][j]))
        elif reference is not None:
            for j in range(len(outer_prods[side][reference])):
                stds[side][j] = torch.sqrt(torch.diag(covs[side][j][j]))

        if layer_range is not None:
            for j in layer_range:
                corrs[side][j] = {}
                for k in range(j, layer_range[-1]+1):
                    print(side, j, k)
                    corrs[side][j][k] = cov_to_corr(covs[side][j][k].cpu().numpy(), 
                                                stds[side][j].cpu().numpy(), 
                                                stds[side][k].cpu().numpy())
                    try:
                        assert np.all(np.abs(corrs[side][j][k]) < 1 + TOL)
                    except AssertionError:
                        breakpoint()
            torch.save(corrs, os.path.join(outdir, f'corrs_{str(max_toks)}_{layer_range[0]}-{layer_range[-1]}.pt'))
            print('done')
        elif reference is not None:
            
            corrs[side][reference] = {}
            for j in range(len(outer_prods[side][reference])):
                corrs[side][reference][j] = cov_to_corr(covs[side][reference][j].cpu().numpy(), 
                                                stds[side][reference].cpu().numpy(), 
                                                stds[side][j].cpu().numpy())
                try:
                    assert np.all(np.abs(corrs[side][reference][j]) < 1 + TOL)
                except AssertionError:  
                    breakpoint()
            torch.save(corrs, os.path.join(outdir, f'corrs_{str(max_toks)}_ref_{args.reference}.pt'))
            print('done')


def add_hooks(model,activations_dict, model_name='vit'):
    def activation_hook(name, side):
        def hook(model, input, output):
            activations_dict[side][name] = output
        return hook
    if has_encoder(model_name):
        encoder_layers = get_layers_by_path(model, model_param_names[model_name]['encoder_prefix'])
        for i, block in enumerate(encoder_layers):
            parts = model_param_names[model_name]['fc1'].split('.')
            for part in parts:
                block = getattr(block, part)
            block.register_forward_hook(activation_hook(f'{i}', 'enc'))
    if has_decoder(model_name):
        decoder_layers = get_layers_by_path(model, model_param_names[model_name]['decoder_prefix'])
        for i, block in enumerate(decoder_layers):
            parts = model_param_names[model_name]['fc1'].split('.')
            for part in parts:
                block = getattr(block, part)
            block.register_forward_hook(activation_hook(f'{i}', 'dec'))



def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args.model_name).to(device)
    tokenizer = load_tokenizer(args.model_name)

    # Load dataloader
    if args.model_name == 'gpt2-large':
        dataloader = load_wikitext(tokenizer, args.batch_size)
        os.environ['HF_HOME'] = args.hf_cache
    elif args.model_name == 'opusmt':
        dataloader = prepare_tatoeba_dataloader(tokenizer, args.batch_size)
        os.environ['HF_HOME'] = args.hf_cache
    elif args.model_name == 'vit':
        token_string = open(args.hf_token).read().strip()
        dataloader = load_imagenet(tokenizer, token_string, args.batch_size)

    # instantiate dict and add hooks
    activations_dict = {}
    sides = []
    if has_encoder(args.model_name):
        sides.append('enc')
        activations_dict['enc'] = {}
    if has_decoder(args.model_name):
        sides.append('dec')
        activations_dict['dec'] = {}
    (n_enc_layers, n_dec_layers) = get_num_layers(model, args.model_name)
    
    inner_dim = dim_map[args.model_name]

    if args.layer_range is not None:
        first = int(args.layer_range.split('-')[0])
        last = int(args.layer_range.split('-')[1])
        layer_range = list(range(first, last+1))
    else:
        layer_range = None

    if layer_range is not None:
        instantiate_trackers(dimension=inner_dim, sides=sides, enc_layers=n_enc_layers, dec_layers=n_dec_layers, device='cpu', layer_range=layer_range)
    elif args.reference is not None:
        instantiate_trackers(dimension=inner_dim, sides=sides, enc_layers=n_enc_layers, dec_layers=n_dec_layers,  device='cpu', reference=args.reference)

    if layer_range is not None:
        for i in layer_range:
            for side in sides:
                activations_dict[side][str(i)] = []
    else: #reference based 
        for side in sides:
            n_layers = n_enc_layers if side == 'enc' else n_dec_layers
            for i in range(n_layers):
                activations_dict[side][str(i)] = []

    add_hooks(model, activations_dict, model_name=args.model_name )

    print('added hooks')
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # vit is (list of tensors, labels)
            # gpt2 is dict of tensors
            # opusmt is dict of tensors
            if args.model_name == 'vit':
                images, _ = batch
                _ = model(images.to(device), interpolate_pos_encoding=True )
            elif args.model_name == 'opusmt':
                # put all values on device
                batch.to(device)
                _ = model(attention_mask=batch['attention_mask'], input_ids=batch['input_ids'], labels=batch['labels'])
            else:
                batch.to(device)
                _ = model(**batch)

            global outer_prods
            global means
            global total_tokens 

            for side in sides:
                if layer_range is not None:
                    for i in layer_range:
                        tmp = activations_dict[side][str(i)].reshape(-1, activations_dict[side][str(i)].shape[-1])
                        if i == layer_range[0] and side == sides[0]:
                            total_tokens += tmp.shape[0]
                        tmp_sum = tmp.sum(dim=0)
                        means[side][i] += tmp_sum.detach().cpu()
                        for j in range(i, layer_range[-1]+1):
                            tmp2 = activations_dict[side][str(j)].reshape(-1, activations_dict[side][str(j)].shape[-1])
                            outer = tmp.T @ tmp2
                            outer_prods[side][i][j] += outer.detach().cpu()
                else:
                    tmp_reference = activations_dict[side][str(args.reference)].reshape(-1, activations_dict[side][str(args.reference)].shape[-1])
                    for i in range(n_layers):
                        tmp = activations_dict[side][str(i)].reshape(-1, activations_dict[side][str(i)].shape[-1])
                        if i == 0 and side == sides[0]:
                            total_tokens += tmp.shape[0]
                        tmp_sum = tmp.sum(dim=0)
                        means[side][i] += tmp_sum.detach().cpu()
                        # get cross
                        outer = tmp_reference.T @ tmp
                        outer_prods[side][args.reference][i] += outer.detach().cpu()
                        # get self
                        outer = tmp.T @ tmp
                        outer_prods[side][i][i] += outer.detach().cpu()
                
            print(total_tokens)
            if total_tokens > args.max_toks:
                break
    
    if args.layer_range is not None:
        finalize_correlations(args.outdir, args.max_toks, layer_range=layer_range, sides=sides)
    elif args.reference is not None:
        finalize_correlations(args.outdir, args.max_toks, reference=args.reference, sides=sides)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-name',
        type=str, 
        required=True, 
        default='vit', 
        help='Type of model, use this or model'
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--outdir",
        default=".",
    )
    parser.add_argument(
        "--max-toks",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--reference",
        type=int,
        help='Reference layer'
    )
    parser.add_argument(
        '--layer-range',
        type=str,
        help='Layer range'
    )
    parser.add_argument(
        '--hf-cache',
        type=str,
    )
    parser.add_argument(
        '--hf-token',
        type=str,
    )
    args = parser.parse_args()
    main(args)


