import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

'''
This file is used to plot the CKA scores between layers of a model, given an output from the CKA analysis script.
'''

def plot_cka_map(sims, layer_type='ff', model_name='gpt2'):
    if layer_type == 'ff':
        expanded_name = 'Feed-Forward'
    elif layer_type == 'attn':
        expanded_name = 'Attention'

    n_layers = len(sims)
    values = np.zeros((n_layers,n_layers))
    for i in sims:
        # symmetry 
        for j in sims[i]:
            values[int(i)][int(j)] = sims[i][j][layer_type]
            values[int(j)][int(i)] = values[int(i)][int(j)]
    # diagonal filling
    for i in range(0,n_layers):
        values[i][i] = 1
    
    plt.figure(figsize=(5, 3))
    plt.imshow(values, cmap='viridis')
    plt.colorbar(label='CKA Score')
    plt.xlabel(f'{expanded_name} Layer i')
    plt.ylabel(f'{expanded_name} Layer j')

    plt.savefig(f'{model_name}_map_{layer_type}.pdf', format='pdf', bbox_inches='tight',)


def main(args):
    with open(args.file, 'r') as f:
        sims = json.load(f)
    plot_cka_map(sims, layer_type=args.component, model_name=args.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CKA')
    parser.add_argument('--file', type=str, help='file with results')
    parser.add_argument('--component', type=str, help='component to plot, ff or attn')
    parser.add_argument('--model-name', type=str, help='model name, used for outfile naming')
    args = parser.parse_args()
    main(args)