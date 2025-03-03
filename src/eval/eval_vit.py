import argparse 
import os
import torch
from tqdm import tqdm 
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, BitsAndBytesConfig

model_name = 'google/vit-base-patch16-224'

def load_quantized(args):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if args.baseline:
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', quantization_config=quantization_config,
    device_map="auto")
    elif os.path.isdir(args.model):
        model = ViTForImageClassification.from_pretrained(args.model, load_in_8bit=True)
    elif os.path.isfile(args.model):
        return NotImplementedError('Please give the parent directory of the model file')
    return model

def remove_layers(model, layers):
    print(layers)
    model.vit.encoder.layer = torch.nn.ModuleList([model.vit.encoder.layer[i] for i in range(len(model.vit.encoder.layer)) if i not in layers])
    return model

def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]

def main(args):
    # load model and processor
    model_name = 'google/vit-base-patch16-224'
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.quantize:
        model = load_quantized(args)
    else:
        model = ViTForImageClassification.from_pretrained(model_name)
        if not args.baseline:
            state_dict = torch.load(args.model)
            model.load_state_dict(state_dict)
       
    if args.drop_layers:
        params_orig = sum(p.numel() for p in model.parameters())
        layer_list = get_layer_list(args.drop_layers)
        model = remove_layers(model, layer_list)
        params_new = sum(p.numel() for p in model.parameters())
        print(f'ratio: {params_new/params_orig}')
        
        
    model.to(device)
    
    print('Model loaded')

    # Load imagenet valid dataset, create dataloader
    token_string = open(args.hf_token).read().strip()
    if args.split == 'validation':
        imagenet_eval = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True, token=token_string, trust_remote_code=True)
    elif args.split == 'test':
        imagenet_eval = load_dataset('ILSVRC/imagenet-1k', split='test', streaming=True, token=token_string, trust_remote_code=True)
    
    def collate_fn(batch):
        # Extract images and labels
        images = [item['image'].convert('RGB')  for item in batch]
        labels = [item['label'] for item in batch]
        
        # Preprocess images
        inputs = image_processor(images, return_tensors='pt')
        
        # Return images and labels as tensors
        return inputs['pixel_values'], torch.tensor(labels)

    dataloader = DataLoader(imagenet_eval, batch_size=args.batch_size, collate_fn=collate_fn)
    print('Dataloader created')

    # get model ready for evaluation
    model.eval()

    # evaluate model
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'{total} images processed')
            print(f'Accuracy: {correct / total}')
    
    # get accuracy
    acc = correct / total
    print(f'Accuracy: {acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ViT models on classification')
    parser.add_argument('--model', type=str, required=False, help='Path to model')
    parser.add_argument('--baseline', action='store_true', help='Whether to use the baseline model')
    parser.add_argument('--batch-size', type=int, default=64, required=False, help='Batch size')
    parser.add_argument('--quantize', action='store_true', help='Whether to quantize the model')
    parser.add_argument('--split', default='validation', type=str, help='Dataset split to evaluate on')
    parser.add_argument('--drop-layers', type=str, help='layers to drop')
    parser.add_argument('--hf-token', type=str)
    
    args = parser.parse_args()
    main(args)