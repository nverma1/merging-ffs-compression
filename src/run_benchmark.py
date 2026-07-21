import argparse
import torch
import utils
import os

USE_DYNAMIC_CACHE = {
    "gpt2-large": False,
    "opusmt": False,
    "olmo": True,
    "olmo2-1b": True,
}

def uses_dynamic_cache(model_name: str) -> bool:
    for key in USE_DYNAMIC_CACHE:
        if key in model_name.lower():
            return USE_DYNAMIC_CACHE[key]
    return False
  
def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)


def benchmark_vit(model, model_name, input_batch, device) -> dict:
    """Benchmark ViT model's latency and throughput on a given image batch."""

    model.eval()
    model.to(device)

    input_tensor, labels = input_batch  # Shape: (batch_size, 3, H, W)
    input_tensor = input_tensor.to(device)
    batch_size = input_tensor.shape[0]
   
    with torch.no_grad():
        # Warmup pass (optional but can help avoid cold start effects)
        _ = model(input_tensor)

def benchmark_mt(model, model_name, batch, device, tokenizer=None) -> dict:
    """Benchmark the model's latency and throughput on the given input batch."""
    print(f" batch size {batch['input_ids'].shape}")
    with torch.no_grad():
        _ = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
        # outputs = model.generate(
        #     inputs=batch["input_ids"].to(device),
        #     attention_mask=batch["attention_mask"].to(device),
        #     max_new_tokens=30,     # <- decode exactly 30 tokens
        #     num_beams=4,           # beam width
        #     early_stopping=False, # force decode full 30 steps
        #     return_dict_in_generate=True,
        #     num_return_sequences=4,
        # )
        breakpoint()
        return

# updating with transformers versioning
def benchmark(model, model_name, input_batch, device) -> dict:
    with torch.no_grad():
        """Benchmark the model's latency and throughput on the given input batch."""
        _ = model(input_batch["input_ids"].to(device), attention_mask=input_batch["attention_mask"].to(device))
        # outputs = model.generate(
        #     input_ids=input_batch["input_ids"].to(device),
        #     attention_mask=input_batch["attention_mask"].to(device),
        #     max_new_tokens=128,
        #     do_sample=False,  # greedy decoding for consistency
        #     num_beams=4,  
        #     use_cache=True,
        # )

   

def benchmark_olmo(model, model_name, input_batch, device) -> dict:
    _ = model(input_batch["input_ids"].to(device), attention_mask=input_batch["attention_mask"].to(device))

def benchmarking_main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running benchmarking of a sliced model.")
    print(f"PyTorch device: {device}")
    print(f"Number of available cuda devices: {torch.cuda.device_count()}")

    tokenizer = utils.load_tokenizer(args.model_type)
    if args.model_path:
        model = utils.load_model(args.model_type, model_path=args.model_path)
    else:
        model = utils.load_model(args.model_type)


    print(f"num_params: {sum(p.numel() for p in model.parameters())}")
    if args.tie_ff:
        layers_to_tie = utils.get_layer_list(args.tie_ff)
        utils.tie_weights(model, layers_to_tie, model_type=args.model_type)

    print(f"num_params after tying: {sum(p.numel() for p in model.parameters())}")


    model.eval()

    model.to(device)

    if args.model_type == "gpt2-large" or args.model_type == "opusmt":
        dataloader = utils.get_dataloader(args.model_type, tokenizer, args.batch_size, token_string=args.hf_token, seq_len=args.ntokens)
    else:
        dataloader = utils.get_dataloader(args.model_type, tokenizer, args.batch_size, token_string=args.hf_token)

    if args.model_type == "vit":
        results = benchmark_vit(model, args.model_type, next(iter(dataloader)), device)
        print('done!')
        return
    elif args.model_type == 'gpt2-large':
        results = benchmark(model, args.model_type, next(iter(dataloader)), device)
        print('done!')
        return
    elif args.model_type == "opusmt":
        results = benchmark_mt(model, args.model_type, next(iter(dataloader)), device, tokenizer=tokenizer)
        print('done!')
        return

    print(f"Median time per batch: {results['median_time']} s/batch.")
    print(f"Throughput: {results['throughput']} token/s.")
    print(f"Latency: {results['latency']} s/token.")

def benchmarking_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default='vit',
        help="Model to load",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type to use.",
        choices=["fp32", "fp16"],
        default="fp16"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="Dataset to evaluate on.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument("--tie-ff", help='Comma sep list of which ff to tie')
    parser.add_argument("--ntokens", type=int, help="Number of tokens to benchmark over.", default=128)
    parser.add_argument('--distribute-model', action='store_true', help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN'))

    return parser.parse_args() 

if __name__ == "__main__":
    benchmarking_args = benchmarking_arg_parser()
    benchmarking_main(benchmarking_args)