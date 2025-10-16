from __future__ import annotations
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm



from utils import (
    cov_to_corr,
    load_model,
    load_tokenizer,
    load_wikitext,
    load_dolma,
    get_num_layers,
    prepare_tatoeba_dataloader,
    load_imagenet,
    has_encoder,
    has_decoder,
    get_layers_by_path,
    dim_map,
    model_param_names,
)


TOL=1e-5

class CorrTracker:
    def __init__(
        self, 
        dim: int = 512,
        sides: list = ['dec'],
        enc_layers: int = 0,
        dec_layers: int = 0,
        device='cpu',
        layer_range: list | None = None,
        reference: int | None = None,
    ):
        self.dim = dim
        self.sides = sides
        self.n_enc_layers = enc_layers
        self.n_dec_layers = dec_layers
        self.layer_range = layer_range
        self.reference = reference
        self.device = device

        self.means = {}
        self.outer_prods = {}
        self.total_tokens = 0

        assert (layer_range is not None) or (reference is not None), "Please provide either layer_range or reference"
        assert not (layer_range is not None and reference is not None), "Please provide either layer_range or reference, not both"

        self._initialize_trackers()

    def _initialize_trackers(self):
        for side in self.sides:
            self.means[side] = {}
            self.outer_prods[side] = {}

            if self.layer_range is not None:
                # Mode 1, correlate layers j to k where j<=k
                for j in self.layer_range:
                    self.means[side][j] = torch.zeros(self.dim).to(self.device)
                    self.outer_prods[side][j] = {}
                    for k in range(j, self.layer_range[-1]+1):
                        self.outer_prods[side][j][k] = torch.zeros(self.dim, self.dim).to(self.device) 
            elif self.reference is not None:
                # Mode 2: Correlating all layers 'j' with the reference layer 'ref', plus self-correlations (j, j)
                n_layers = self.n_enc_layers if side == 'enc' else self.n_dec_layers

                # Initialize means for all layers
                for j in range(n_layers):
                    self.means[side][j] = torch.zeros(self.dim).to(self.device)
                
                # Initialize outer products for (reference, j) and (j, j)
                self.outer_prods[side][self.reference] = {}
                for j in range(n_layers):
                    # Cross-product 
                    self.outer_prods[side] [self.reference][j] = torch.zeros(self.dim, self.dim).to(self.device)

                    # Self-product
                    if not j in self.outer_prods[side]:
                        self.outer_prods[side][j] = {}
                    self.outer_prods[side][j][j] = torch.zeros(self.dim, self.dim).to(self.device)
    
    def update(self, activations_dict):
        """
        Updates the means and outer products with activations from a single batch.
        
        Args:
            activations_dict (dict): Dictionary mapping side ('enc'/'dec') -> layer index (str) -> activation tensor.
        """
        inner_dim = self.dim
        n_layers = {'enc': self.n_enc_layers, 'dec': self.n_dec_layers}
        current_batch_tokens = 0
        
        for side in self.sides:
            if self.layer_range is not None:
                for i in self.layer_range:
                    # Reshape activation tensor: (batch_size, seq_len, dim) -> (N, dim)
                    tmp = activations_dict[side][str(i)].reshape(-1, inner_dim).cpu()
                    
                    # only update batch tokens on first side and first layer
                    if side == self.sides[0] and i == self.layer_range[0]:
                        current_batch_tokens = tmp.shape[0]
                        
                    self.means[side][i] += tmp.sum(0)
                    
                    for j in range(i, self.layer_range[-1]+1):
                        tmp2 = activations_dict[side][str(j)].reshape(-1, inner_dim).cpu()
                        self.outer_prods[side][i][j] += (tmp.T @ tmp2)
            
            elif self.reference is not None:
                # Get activation for the reference layer
                tmp_reference = activations_dict[side][str(self.reference)].reshape(-1, inner_dim).cpu()
                
                for i in range(n_layers[side]):
                    # Get activation for layer i
                    tmp = activations_dict[side][str(i)].reshape(-1, inner_dim).cpu()
                    
                    if side == self.sides[0] and i == 0:
                        current_batch_tokens = tmp.shape[0]
                        
                    # Update mean for layer i
                    self.means[side][i] += tmp.sum(0)
                    
                    # get cross-product M(ref, i)
                    self.outer_prods[side][self.reference][i] += (tmp_reference.T @ tmp)
                    
                    # get self-product M(i, i)
                    self.outer_prods[side][i][i] += (tmp.T @ tmp)

        self.total_tokens += current_batch_tokens
        return current_batch_tokens


    def finalize_correlations(
        self,
        outdir: str,
        scaling: bool = False, 
        post_act: bool = False
    ):
        corrs = {}
        covs = {}
        stds = {}
        scales = {}

        ref_str = ""
        # Determine parameters for file naming
        if self.layer_range is not None:
            range_str = f"{self.layer_range[0]}-{self.layer_range[-1]}"
        elif self.reference is not None:
            range_str = None
            ref_str = f"ref_{self.reference}"
        
        for side in self.sides:
            corrs[side] = {}
            stds[side] = {}
            covs[side] = {}
            scales[side] = {}
            # --- 1. Compute means --- 
            if self.layer_range is not None:
                for j in self.layer_range:
                    self.means[side][j] = self.means[side][j].div(self.total_tokens)
            elif self.reference is not None:
                for j in self.means[side]:
                    self.means[side][j] = self.means[side][j].div(self.total_tokens)

            # --- 2. Compute scaling (if requested) ---
            if scaling:
                if self.layer_range is not None:
                    for j in self.layer_range:
                        scales[side][j] = {}
                        for k in range(j, self.layer_range[-1]+1):
                            scales[side][j][k] = self.outer_prods[side][j][k] / (self.outer_prods[side][k][k].diag() + 1e-10)
                            scales[side][j][k] = torch.where(scales[side][j][k] < 0, torch.ones_like(scales[side][j][k]), scales[side][j][k])
                    torch.save(scales, os.path.join(outdir, f'scales_{str(self.total_tokens)}_{range_str}.pt'))
                elif self.reference is not None:
                    scales[side][self.reference] = {}
                    for j in self.outer_prods[side][self.reference]:
                        scales[side][self.reference][j] = self.outer_prods[side][self.reference][j] / (self.outer_prods[side][j][j].diag() + 1e-10)
                    torch.save(scales, os.path.join(outdir, f'scales_{str(self.total_tokens)}_{ref_str}.pt'))
                    
            # --- 3. Compute covariances ---
            if self.layer_range is not None:
                for j in self.layer_range:
                    covs[side][j] = {}
                    for k in range(j, self.layer_range[-1]+1):
                        self.outer_prods[side][j][k] = self.outer_prods[side][j][k].div(self.total_tokens)
                        cov = self.outer_prods[side][j][k] - torch.outer(self.means[side][j], self.means[side][k])
                        covs[side][j][k] = cov
            elif self.reference is not None:
                covs[side][self.reference] = {}
                # Cross-covariances Cov(ref, j)
                for j in self.outer_prods[side][self.reference]: 
                    self.outer_prods[side][self.reference][j] = self.outer_prods[side][self.reference][j].div(self.total_tokens)
                    cov = self.outer_prods[side][self.reference][j] - torch.outer(self.means[side][self.reference], self.means[side][j])
                    covs[side][self.reference][j] = cov
                
                # Self-covariances Cov(j, j)
                for j in self.means[side]:
                    if j != self.reference:
                        if j not in covs[side]:
                            covs[side][j] = {}
                        self.outer_prods[side][j][j] = self.outer_prods[side][j][j].div(self.total_tokens)
                        cov = self.outer_prods[side][j][j] - torch.outer(self.means[side][j], self.means[side][j])
                        covs[side][j][j] = cov
                        
            # --- 4. Compute standard deviations (from self-covariances) ---
            if self.layer_range is not None:
                for j in self.layer_range:
                    stds[side][j] = torch.sqrt(torch.diag(covs[side][j][j]))
            elif self.reference is not None:
                for j in range(len(self.outer_prods[side][self.reference])):
                    stds[side][j] = torch.sqrt(torch.diag(covs[side][j][j]))
    
            # --- 5. Compute correlations and save ---
            if self.layer_range is not None:
                for j in self.layer_range:
                    corrs[side][j] = {}
                    for k in range(j, self.layer_range[-1]+1):
                        print(side, j, k)
                        corrs[side][j][k] = cov_to_corr(covs[side][j][k].cpu().numpy(), 
                                                    stds[side][j].cpu().numpy(), 
                                                    stds[side][k].cpu().numpy())
                        try:
                            assert np.all(np.abs(corrs[side][j][k]) < 1 + TOL)
                        except AssertionError:
                            breakpoint()         
                
            elif self.reference is not None:
                corrs[side][self.reference] = {}
                for j in self.outer_prods[side][self.reference]:
                    corrs[side][self.reference][j] = cov_to_corr(covs[side][self.reference][j].cpu().numpy(), 
                                                            stds[side][self.reference].cpu().numpy(), 
                                                            stds[side][j].cpu().numpy())
                    try:
                        assert np.all(np.abs(corrs[side][self.reference][j]) < 1 + TOL)
                    except AssertionError:  
                        breakpoint()
        if self.layer_range is not None:
            torch.save(corrs, os.path.join(outdir, f'corrs_{str(self.total_tokens)}_{range_str}_post_{post_act}.pt'))
            print('done')
        else:
            torch.save(corrs, os.path.join(outdir, f'corrs_{str(self.total_tokens)}_{ref_str}_post_{post_act}.pt'))
            print('done')

"""
Given an empty activations dict, add_hooks will populate it with activations during forward passes
"""
def add_hooks(model, activations_dict, model_name='vit'):
    def activation_hook(name, side):
        def hook(model, input, output):
            activations_dict[side][name] = output.to(dtype=torch.float32)
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

def add_pre_hooks_fc2(model, activations_dict, model_name='vit'):
    def activation_hook(name, side):
        def hook(module, input):
            # input is a tuple; we store the first element
            activations_dict[side][name] = input[0]
        return hook
    if has_encoder(model_name):
        encoder_layers = get_layers_by_path(model, model_param_names[model_name]['encoder_prefix'])
        for i, block in enumerate(encoder_layers):
            parts = model_param_names[model_name]['fc2'].split('.')
            target = block
            for part in parts:
                target = getattr(target, part)
            target.register_forward_pre_hook(activation_hook(f'{i}', 'enc'))
    if has_decoder(model_name):
        decoder_layers = get_layers_by_path(model, model_param_names[model_name]['decoder_prefix'])
        for i, block in enumerate(decoder_layers):
            parts = model_param_names[model_name]['fc2'].split('.')
            target = block
            for part in parts:
                target = getattr(target, part)
            target.register_forward_pre_hook(activation_hook(f'{i}', 'dec'))

def create_random_batches(model_dim, seq_len, batch_size):
    while True:
        yield torch.randn(batch_size, seq_len, model_dim)

def get_model_specific_dataloader(model_name, tokenizer, batch_size, hf_token=None):
    if model_name == 'gpt2-large':
        return load_wikitext(tokenizer, batch_size)
    elif model_name == "olmo1b":
        return load_dolma(tokenizer, batch_size)
    elif model_name == 'opusmt':
        return prepare_tatoeba_dataloader(tokenizer, batch_size)
    elif model_name == 'vit':
        token_string = open(hf_token).read().strip()
        return load_imagenet(tokenizer, token_string, batch_size)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def get_activations_dict(
    sides: list = ['dec'],
    layer_range: list | None = None,
    reference: int | None = None,
    n_enc_layers: int = 0,
    n_dec_layers: int = 0,
):
    activations_dict = {}
    for side in sides:
        activations_dict[side] = {}
        if layer_range is not None:
            for i in layer_range:
                activations_dict[side][str(i)] = []
        elif reference is not None:
            n_layers = n_enc_layers if side == 'enc' else n_dec_layers
            for i in range(n_layers):
                activations_dict[side][str(i)] = []
    return activations_dict

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_name).to(device)
    if args.model_name == 'olmo1b':
        model = model.half()
    tokenizer = load_tokenizer(args.model_name)
    # os.environ['HF_HOME'] = args.hf_cache

    dataloader = get_model_specific_dataloader(
        args.model_name, tokenizer, args.batch_size, args.hf_token
    )

    # instantiate dict and add hooks
    activations_dict = {}
    sides = []
    if has_encoder(args.model_name):
        sides.append('enc')
    if has_decoder(args.model_name):
        sides.append('dec')
    (n_enc_layers, n_dec_layers) = get_num_layers(model, args.model_name)
    
    inner_dim = dim_map[args.model_name]

    # get the range of layers that will be correlated 
    if args.layer_range is not None:
        first = int(args.layer_range.split('-')[0])
        last = int(args.layer_range.split('-')[1])
        layer_range = list(range(first, last+1))
    else:
        layer_range = None

    corr_tracker = CorrTracker(
        dim=inner_dim,
        sides=sides,
        enc_layers=n_enc_layers,
        dec_layers=n_dec_layers, 
        device='cpu',
        layer_range=layer_range,
        reference=args.reference,
    )

    # set up structure of activations dict
    activations_dict = get_activations_dict(sides, layer_range, args.reference, n_enc_layers, n_dec_layers)

    if args.post_act:
        add_pre_hooks_fc2(model, activations_dict, model_name=args.model_name )
    else:
        add_hooks(model, activations_dict, model_name=args.model_name )
    print('added hooks')

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if args.model_name == 'vit':
                # vit is (list of tensors, labels)
                images, _ = batch
                _ = model(images.to(device), interpolate_pos_encoding=True)
            elif args.model_name == 'opusmt':
                # opusmt is dict of tensors
                batch.to(device)
                _ = model(attention_mask=batch['attention_mask'], input_ids=batch['input_ids'], labels=batch['labels'])
            elif args.model_name == 'olmo1b':
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # gpt2 is dict of tensors
                batch.to(device)
                _ = model(**batch)

            # Update the tracker with the collected activations
            corr_tracker.update(activations_dict)

            # clear activations_dict after processing batch to free VRAM
            for side in sides:
                for layer_key in activations_dict[side]:
                    activations_dict[side][layer_key] = None

            if corr_tracker.total_tokens > args.max_toks:
                break
    
    corr_tracker.finalize_correlations(
        args.outdir, scaling=args.scaling, post_act=args.post_act
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, default="vit")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--max-toks", type=int, default=10000)
    parser.add_argument("--reference", type=int, help="Reference layer")
    parser.add_argument("--layer-range", type=str, help="Layer range (e.g., 2-6)")
    parser.add_argument("--hf-cache", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--scaling", action='store_true')
    parser.add_argument("--post-act", action='store_true', help='Whether to use post-activation reps')
    args = parser.parse_args()
    main(args)
