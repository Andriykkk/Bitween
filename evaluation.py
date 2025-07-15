import torch
import math
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

from model import GPT, GPTConfig
from utils import load_model_for_analysis

# --- Evaluation Settings ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'model_final.pt'
tokenized_dataset_path = "tokenized_wikitext"
# <<< CHANGE THIS TO TEST DIFFERENT MODELS >>>
QUANTIZATION_MODE = 'bf16_partial' # Options: 'none', 'bf16_partial'
batch_size = 2

# --- Model Configuration (should match training) ---
config = GPTConfig(
    vocab_size=50257,
    block_size=256,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.1
)

# --- Load Model and Data ---
model = load_model_for_analysis(
    model_config=config,
    model_path=model_path,
    quantization_type=QUANTIZATION_MODE,
    device=device
)
model.eval()

print("Loading validation dataset...")
tokenized_datasets = load_from_disk(tokenized_dataset_path)
val_dataset = tokenized_datasets['validation']
val_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
print("Dataset loaded.")

# --- Initialize Metrics ---
f1_metric = MulticlassF1Score(num_classes=config.vocab_size, average='macro').to(device)
total_loss = 0
total_batches = 0

# Determine if we need to use autocast for mixed precision
use_amp = any(p.dtype in [torch.bfloat16, torch.float16] for p in model.parameters())

# --- Evaluation Loop ---
print("\nStarting evaluation...")
with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, loss = model(input_ids, targets=labels)
        
        if loss is not None:
            total_loss += loss.item()
            total_batches += 1

        # Get predictions for F1 score
        preds = torch.argmax(logits, dim=-1)
        
        # Update F1 metric (flatten batch and sequence dimensions)
        f1_metric.update(preds.view(-1), labels.view(-1))

# --- Calculate and Print Results ---
print("\n--- Evaluation Results ---")

# Perplexity
if total_batches > 0:
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity  : {perplexity:.4f}")
else:
    print("Could not calculate perplexity (no loss computed).")

# F1 Score
f1_score = f1_metric.compute()
print(f"F1 Score (Macro): {f1_score.item():.4f}")
print("--------------------------")
print("\nNote: F1 score treats each token prediction as a classification task.")
print("Perplexity measures how well the probability distribution of the model predicts a sample.")
