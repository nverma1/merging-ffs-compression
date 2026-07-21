import torch
import argparse 
import numpy as np
import glob
import os 

from datasets import load_dataset, load_metric
from transformers import TrainingArguments
from torch.optim.lr_scheduler import LambdaLR

from transformers import AdamW
from transformers import ViTImageProcessor, ViTForImageClassification

'''
We use streaming datasets to load the imagenet dataset.
'''
from helper import CustomTrainer

#Adapted from https://huggingface.co/blog/fine-tune-vit

def find_latest_checkpoint(output_dir):
    # List all directories that match the checkpoint pattern
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None

    # Sort directories by creation time and return the latest one
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint

def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]
    
# metric
metric = load_metric('accuracy', trust_remote_code=True)
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def tie_weights(model, layers_to_tie):
    reference = layers_to_tie[0]
    for layer in layers_to_tie: 
        if layer != reference:
            model.vit.encoder.layer[layer].intermediate.dense.weight = model.vit.encoder.layer[reference].intermediate.dense.weight
            model.vit.encoder.layer[layer].intermediate.dense.bias = model.vit.encoder.layer[reference].intermediate.dense.bias
            model.vit.encoder.layer[layer].output.dense.bias = model.vit.encoder.layer[reference].output.dense.bias
            model.vit.encoder.layer[layer].output.dense.weight = model.vit.encoder.layer[reference].output.dense.weight

def transform(batch, image_processor):
    inputs = image_processor([x.convert('RGB') for x in batch['image']], return_tensors='pt')
    inputs['labels'] = batch['label']
    return inputs

def remove_layers(model, layers):
    model.vit.encoder.layer = torch.nn.ModuleList([model.vit.encoder.layer[i] for i in range(len(model.vit.encoder.layer)) if i not in layers])
    return model
    
def prepare_imagenet_dataset(image_processor, split='train'):
    # Load imagenet valid dataset, create dataloader
    streaming = True
    imagenet_dataset = load_dataset('ILSVRC/imagenet-1k', split=split, streaming=streaming, token=os.getenv('HF_TOKEN'), trust_remote_code=True)


    prepared_ds = imagenet_dataset.map(lambda x: transform(x, image_processor), batched=True)

    return prepared_ds

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([item['image'] for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch])
    }


def custom_lr_lambda(current_step, num_training_steps, min_lr):
    # Linear decay from initial LR to min_lr over the first 20k steps
    if current_step < 20000:
        lr = (1 - current_step / 20000) * (1 - min_lr) + min_lr
        if lr > min_lr:
            return lr
        else:
            return min_lr
    # After 20k steps, learning rate stays at min_lr
    return min_lr

# Custom scheduler
def get_custom_scheduler(optimizer, num_training_steps, min_lr):
    lr_lambda = lambda step: custom_lr_lambda(step, num_training_steps, min_lr)
    return LambdaLR(optimizer, lr_lambda)

def main(args):
    model_name = 'google/vit-base-patch16-224'
    model = ViTForImageClassification.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)


    
    if not args.baseline:
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict)


    model_dict = torch.load(args.model)
    model.load_state_dict(model_dict)
    orig_params = model.num_parameters()
    print(f'Original params {orig_params}')

    # tie feed-forward layers:
    # tie feed-forward layers:
    if args.tie_ff:
        layers_to_tie = get_layer_list(args.tie_ff)
        tie_weights(model, layers_to_tie)
    if args.drop_layers:
        layer_list = get_layer_list(args.drop_layers)
        model = remove_layers(model, layer_list)
            
    compressed_params = model.num_parameters()
    print(f'Compressed params {compressed_params}')
    print(f'ratio {compressed_params / orig_params}')

    train_ds = prepare_imagenet_dataset(image_processor, split='train')
    val_ds = prepare_imagenet_dataset(image_processor, split='validation')

    if args.dropout:
        model.config.hidden_dropout_prob = args.dropout
        model.config.attention_probs_dropout_prob = args.dropout

    training_args = TrainingArguments(
        output_dir=args.outdir,          # output directory
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=256,   # batch size for evaluation
        fp16=True,                       # use mixed precision training
        do_train=True,
        do_eval=True,
        logging_dir='./logs',
        eval_strategy='steps',   # directory for storing logs
        save_steps=3000,
        eval_steps=3000,
        max_steps=args.num_updates,
        logging_steps=100,
        save_safetensors=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        max_grad_norm=1.0, # adding gradient clipping 9.16
    )

    


    #     def get_eval_dataloader(self, eval_dataset=None):
    #         # Reinitialize the streaming dataset
    #         eval_dataset = prepare_imagenet_dataset(self.tokenizer, split='validation')
    #         return super().get_eval_dataloader(eval_dataset)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0)
    lr_scheduler = get_custom_scheduler(
        optimizer=optimizer, 
        num_training_steps=args.num_updates, 
        min_lr=0.02 # this is the fraction of the original LR, the actual LR here is 1e-6 
    )



    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        optimizers=(optimizer, lr_scheduler)
    )
 
    latest_checkpoint = find_latest_checkpoint(args.outdir)
    if latest_checkpoint:
        print('resuming from checkpoint')
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print('from scratch')
        train_results = trainer.train()



    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    metrics = trainer.evaluate(val_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune ViT models on classification')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
    parser.add_argument('--num-updates', type=int, default=20000, help='Number of updates')
    parser.add_argument('--tie-ff', help='Comma sep list of which ff to tie')
    parser.add_argument('--outdir', type=str, help='Output directory')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--drop-layers', type=str, help='Layer indices to drop')

    args = parser.parse_args()



    main(args)
