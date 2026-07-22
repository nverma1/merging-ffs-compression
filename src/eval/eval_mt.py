from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, BitsAndBytesConfig, MarianConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from accelerate.utils import BnbQuantizationConfig
from accelerate.utils import load_and_quantize_model
import torch
import os
from accelerate import init_empty_weights
import evaluate

def load_quantized(args):
    quantization_config =   BitsAndBytesConfig(load_in_8bit=True)
    model_name='Helsinki-NLP/opus-mt-zh-en'
    if args.baseline:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quantization_config)
    elif os.path.isdir(args.model):
        model_config = MarianConfig.from_pretrained(args.model)
        with init_empty_weights():
            #empty_model = AutoModelForSeq2SeqLM.from_config(model_config)
            empty_model = MarianMTModel(config=model_config)
        bnb_quant_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
        model = load_and_quantize_model(empty_model, weights_location='./fixed_weights.pt', bnb_quantization_config=bnb_quant_config,)
    elif os.path.isfile(args.model):
        raise NotImplementedError('Loading from file not implemented currently')
    return model

def get_layer_list(string_input):
    if '-' in string_input:
        start, end = string_input.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    else:
        return [int(i) for i in string_input.split(',')]


def remove_layers(model, layers):
    model.model.encoder.layers = torch.nn.ModuleList([model.model.encoder.layers[i] for i in range(len(model.model.encoder.layers)) if i not in layers])
    model.model.decoder.layers = torch.nn.ModuleList([model.model.decoder.layers[i] for i in range(len(model.model.decoder.layers)) if i not in layers])
    return model

def main(args):
    # Step 1: Load the model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    if args.quantize:
        model = load_quantized(args)
    else:
        if args.baseline:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif os.path.isdir(args.model):
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        elif os.path.isfile(args.model):
            print('loading from file')
            state_dict = torch.load(args.model)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.load_state_dict(state_dict, strict=False)
        if args.drop_layers:
            layer_list = get_layer_list(args.drop_layers)
            model = remove_layers(model, layer_list)
       

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move the model to the GPU
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.quantize is False:
        model.to(device)

    # Step 2: Load the Tatoeba eng-zho dataset
    dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-zho", split=args.split)

    # Step 3: Prepare the DataLoader for batching
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Step 4: Translate and evaluate
    predictions = []
    references = []

    torch.set_flush_denormal(True)
    bleu = evaluate.load("bleu")

    for batch in tqdm(data_loader):
        # Tokenize the batch of chinese texts
        max_length_in_batch = max(len(t) for t in batch["targetString"])
        inputs = tokenizer(text=batch["targetString"], return_tensors="pt", padding=True, truncation=True, max_length=max_length_in_batch)
        # Move inputs to GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # Generate translations
        outputs = model.generate(**inputs)

        # Decode the translations
        with tokenizer.as_target_tokenizer():
            batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Store predictions and references
        predictions.extend(batch_translations)
        print(batch_translations)
        references.extend([[ref] for ref in batch["sourceString"]])  # BLEU expects a list of lists
        print([ref] for ref in batch["sourceString"])

        bleu_score = bleu.compute(predictions=predictions, references=references)
        print(f"BLEU score so far: {bleu_score['bleu']:.2f}")


    # Step 5: Compute evaluation metrics (e.g., BLEU score)

    # write preds to file:
    # get final path from args.model

    print('writing to file')
    os.makedirs('outputs/', exist_ok=True)
    with open(f'outputs/preds_{args.outfile}.txt', 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')
    with open(f'outputs/refs_{args.outfile}.txt', 'w') as f:
        for ref in references:
            f.write(ref[0] + '\n')
    print('finished writing')
    

    # We recommend to use the cmdline sacrebleu on the saved files
   

    # Compute BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU score: {bleu_score['bleu']:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for translation")
    parser.add_argument("--baseline", action="store_true", help="Use baseline model")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate on")
    parser.add_argument("--outfile", default="baseline", help="Output file name")
    parser.add_argument('--quantize', action='store_true', help='Whether to quantize the model')
    parser.add_argument("--drop-layers", default=None, type=str, help="Layer indices to drop")

    args = parser.parse_args()

    main(args)


