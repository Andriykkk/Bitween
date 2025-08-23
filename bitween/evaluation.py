import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="test", num_samples=200, **kwargs):
    """
    Calculates the perplexity of a model on a given dataset.
    """
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text_samples = dataset['text'][:num_samples]
    
    total_neg_log_likelihood = 0
    total_tokens = 0
    
    print(f"Calculating perplexity on {num_samples} samples...")
    for text in tqdm(text_samples):
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
    
    print(f"Perplexity calculated: {perplexity:.4f}")
    return perplexity

def calculate_kl_divergence(original_model, quantized_model, tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="test", num_samples=200, **kwargs):
    """
    Calculates the average KL-Divergence between the output distributions of two models.
    """
    original_model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    quantized_model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text_samples = dataset['text'][:num_samples]
    
    total_kl_div = 0
    num_batches = 0
    
    print(f"Calculating KL-Divergence on {num_samples} samples...")
    for text in tqdm(text_samples):
        if not text:
            continue
            
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(original_model.device)
        
        with torch.no_grad():
            logits1 = original_model(input_ids).logits
            logits2 = quantized_model(input_ids).logits

        if logits1.ndim == 2:
            logits1 = logits1.unsqueeze(1)
        if logits2.ndim == 2:
            logits2 = logits2.unsqueeze(1)

        min_len = min(logits1.shape[1], logits2.shape[1])
        logits1 = logits1[:, :min_len, :]
        logits2 = logits2[:, :min_len, :]

        prob2 = F.softmax(logits2, dim=-1)
        log_prob1 = F.log_softmax(logits1, dim=-1)
        
        kl_div = F.kl_div(log_prob1, prob2, reduction='batchmean', log_target=False)
        
        total_kl_div += kl_div.item()
        num_batches += 1
        
    avg_kl_div = total_kl_div / num_batches if num_batches > 0 else 0
    print(f"Average KL-Divergence calculated: {avg_kl_div:.6f}")
    return avg_kl_div

def print_report(original_ppl, quantized_ppl, kl_div):
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
    print("\n--- End of Report ---")
