#!/bin/bash


outdir=$1
toks=$2
start=$3
end=$4
split=$5

save_dir=$outdir/vit

mkdir -p $save_dir

if [ -e $save_dir/corrs_${toks}_0-11.pt ]; then     
    echo "correlations exist, skipping corr computation..."
else
    python ../../src/get_corrs.py --outdir $save_dir --max-toks $toks \
        --layer-range 0-11 --model-name vit --hf-cache $HF_HOME  --hf-token $HF_TOK
fi

python ../../src/merge_ffs.py --model-type vit \
    --encoder-layers ${start}-${end} --output $save_dir --corrs $save_dir/corrs_${toks}_0-11.pt \
    --ref-type first  

python ../../src/eval/eval_vit.py --model $save_dir --split $split --hf-token $HF_TOK