import argparse
from datasets import Dataset
from transformers import AutoTokenizer

'''
This file is used to convert raw data from the OPUS dataset into a HuggingFace dataset. 
'''

# load raw data into HF dataset
def load_opus_from_file(filepath, split, dataset=None):
    with open(filepath + '/train.trg', 'r') as f:
        train_zh = f.readlines()
    with open(filepath + '/train.src', 'r') as f:
        train_en = f.readlines()
    assert len(train_zh) == len(train_en)

    data_dict = {'translation': [{'zh': zh.strip(), 'en': en.strip()} for zh, en in zip(train_zh, train_en)]}
    return Dataset.from_dict(data_dict)

# tokenize data with src and tgt tokenizers
def preprocess_function(examples, tokenizer, split='train'):
    if split == 'train':
        inputs= [example['zh'] for example in examples['translation']]
        targets = [example['en'] for example in examples['translation']]
    else:
        inputs = [example for example in examples['targetString']]
        targets = [example for example in examples['sourceString']]
    model_inputs = tokenizer(text=inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# load, tokenize, and save training data
def main(args):
    train_dataset = load_opus_from_file(args.path, 'train')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
    processed_train = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer, split='train'), batched=True)
    processed_train.save_to_disk(f'{args.path}/opus-zh-en-full.hf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train', help='path to the raw data downloaded and opened')
    args = parser.parse_args()

    main(args)