#!/bin/bash


outdir=$1
toks=$2
start=$3
end=$4
split=$5

save_dir=$outdir/gpt2

mkdir -p $save_dir


python ../../src/get_corrs.py --outdir $save_dir --max-toks $toks \
    --reference $start --model-name gpt2-large --hf-cache $HF_HOME

python ../../src/merge_ffs.py --model-type gpt2-large \
    --decoder-layers ${start}-${end} --output $save_dir --corrs $save_dir/corrs_${toks}_ref_${start}.pt \
    --reference-only --ref-type first --transpose 

python ../../src/eval/eval_ppl.py --model $save_dir --split $split 