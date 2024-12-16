#!/bin/bash

outdir=$1

mkdir -p $outdir/gpt2-large
mkdir -p $outdir/vit
mkdir -p $outdir/opusmt

for model in gpt2-large vit; do 
    mkdir -p $outdir/$model
    python -m sim_analysis.compute_cka --batch-size 16 --outdir $outdir/$model --max-toks 10000 --model-type $model --cka linear
done

# we use batch size of 1 to avoid padding tokens, can afford because capping around 10k. 

for model in opusmt; do 
    mkdir -p $outdir/$model
    python -m sim_analysis.compute_cka --batch-size 1 --outdir $outdir/$model --max-toks 10000 --model-type $model --cka linear
done