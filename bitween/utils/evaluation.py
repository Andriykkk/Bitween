import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="test", eval_samples=200, verbose=True, **kwargs):
    """
    Calculates the perplexity of a model on a given dataset.
    """
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text_samples = dataset['text'][:eval_samples]
    
    total_neg_log_likelihood = 0
    total_tokens = 0
    
    # Use tqdm only if verbose
    iterator = tqdm(text_samples, desc=f"Calculating perplexity on {eval_samples} samples") if verbose else text_samples
    
    for text in iterator:
        if not text:
            continue
            
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss * input_ids.size(1)

        total_neg_log_likelihood += neg_log_likelihood.item()
        total_tokens += input_ids.size(1)
        
    if total_tokens == 0:
        return float('inf')
        
    avg_neg_log_likelihood = total_neg_log_likelihood / total_tokens
    perplexity = torch.exp(torch.tensor(avg_neg_log_likelihood)).item()
    
    if verbose:
        print(f"Perplexity calculated: {perplexity:.4f}")
    return perplexity

def calculate_kl_divergence(original_model, quantized_model, tokenizer, 
                            dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", 
                            split="test", eval_samples=200, verbose=True, **kwargs):
    """
    Calculates both batch-mean KL and per-token KL divergence between the outputs of two models.
    Evaluates models sequentially to minimize GPU memory usage.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store original device of the original model
    original_device = next(original_model.parameters()).device
    
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text_samples = dataset['text'][:eval_samples]

    if verbose:
        print(f"Calculating KL-Divergence on {eval_samples} samples...")
    
    # Prepare input data once
    input_data = []
    for text in text_samples:
        if not text.strip():
            continue
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_data.append(encodings.input_ids)
    
    # Step 1: Get outputs from original model
    original_model.eval().to(device)
    original_outputs = []
    
    with torch.no_grad():
        iterator = tqdm(input_data, desc="Original model") if verbose else input_data
        for input_ids in iterator:
            input_ids = input_ids.to(device)
            logits1 = original_model(input_ids).logits
            original_outputs.append(logits1.cpu())  # Store on CPU
    
    # Move original model back to CPU and clear GPU cache
    original_model.to(original_device)
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Step 2: Get outputs from quantized model
    quantized_model.eval().to(device)
    
    total_kl_div = 0.0
    num_batches = 0
    total_kl = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        iterator = zip(input_data, original_outputs)
        if verbose:
            iterator = tqdm(list(iterator), desc="Quantized model")
            
        for input_ids, logits1 in iterator:
            input_ids = input_ids.to(device)
            logits1 = logits1.to(device)
            logits2 = quantized_model(input_ids).logits

            min_len = min(logits1.shape[1], logits2.shape[1])
            logits1 = logits1[:, :min_len, :]
            logits2 = logits2[:, :min_len, :]

            log_prob1 = F.log_softmax(logits1, dim=-1)
            prob2 = F.softmax(logits2, dim=-1)

            # Batch mean KL (for backward compatibility)
            kl_div = F.kl_div(log_prob1, prob2, reduction='batchmean', log_target=False)
            total_kl_div += kl_div.item()
            num_batches += 1

            # Per-token KL divergence
            per_token_kl = F.kl_div(log_prob1, prob2, reduction='none', log_target=False)
            per_token_kl = per_token_kl.sum(dim=-1)
            total_kl += per_token_kl.sum().item()
            total_tokens += per_token_kl.numel()

    avg_kl_div = total_kl_div / num_batches if num_batches > 0 else 0.0
    avg_kl_per_token = total_kl / total_tokens if total_tokens > 0 else 0.0

    if verbose:
        print(f"Average KL-Divergence (batchmean): {avg_kl_div:.6f}")
        print(f"Average KL-Divergence per token  : {avg_kl_per_token:.6f}")

    return avg_kl_div, avg_kl_per_token

def print_report(original_ppl, quantized_ppl, kl_div, token_kl_div):
    """
    Prints a formatted report comparing the performance metrics.
    """
    ppl_diff = quantized_ppl - original_ppl
    ppl_diff_percent = (ppl_diff / original_ppl) * 100 if original_ppl != 0 else float('inf')
    
    print("\n--- Quantization Performance Report ---")
    print("\nPerplexity (PPL) - Lower is better")
    print(f"  - Original Model PPL:   {original_ppl:.4f}")
    print(f"  - Quantized Model PPL:  {quantized_ppl:.4f}")
    print(f"  - Difference (Quant-Orig): {ppl_diff:+.4f} ({ppl_diff_percent:+.2f}%)")
    
    print("\nKL-Divergence - Lower means more similar to original")
    print(f"  - Average KL-Divergence: {kl_div:.6f}")
    print(f"  - Average per-token KL-Divergence: {token_kl_div:.6f}")
    print("\n--- End of Report ---")
