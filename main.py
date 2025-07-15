import torch
import logging
import os
import math
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import GPT, GPTConfig
import matplotlib.pyplot as plt

# --- Configuration ---
config = GPTConfig(
    vocab_size=50257,  # GPT-2 tokenizer's vocab size
    block_size=256,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.1
)

# --- Hyperparameters ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16 
# LR Scheduler Settings
max_lr = 1e-4
min_lr = 1e-5
warmup_steps = 200
max_steps = 20000 # Total training steps

gradient_accumulation_steps = 2 # Number of steps to accumulate gradients for
eval_interval = 100
log_interval = 10
val_steps = 50
save_interval = 1000
tokenized_dataset_path = "tokenized_wikitext"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataset and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained('gpt2')

if os.path.exists(tokenized_dataset_path):
    logging.info(f"Loading tokenized dataset from {tokenized_dataset_path}")
    lm_datasets = load_from_disk(tokenized_dataset_path)
else:
    logging.info("Tokenizing and caching dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=config.block_size)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // config.block_size) * config.block_size
        result = {
            k: [t[i : i + config.block_size] for i in range(0, total_length, config.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    lm_datasets.save_to_disk(tokenized_dataset_path)

train_dataset = lm_datasets['train']
val_dataset = lm_datasets['validation']

train_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Calculate steps per epoch
steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps))
logging.info(f"Steps per epoch: {steps_per_epoch}")


# --- Model and Optimizer ---
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

# --- LR Scheduler ---
def get_lr(step):
    # 1) linear warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) if step > max_steps, return min learning rate
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# --- Training Loop ---
train_losses = []
val_losses = []
val_loss_steps = []
learning_rates = []

train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)

logging.info("Starting training...")
model.train()
optimizer.zero_grad()

for step in range(max_steps):
    # Set learning rate
    lr = get_lr(step)
    learning_rates.append(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Gradient accumulation loop
    for _ in range(gradient_accumulation_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits, loss = model(input_ids, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    train_losses.append(loss.item() * gradient_accumulation_steps)

    if step > 0 and step % log_interval == 0:
        logging.info(f"Step {step}, Train Loss: {train_losses[-1]:.4f}, LR: {lr:.6f}")
    
    if step > 0 and step % eval_interval == 0:
        model.eval()
        current_val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_batch = next(val_iter)
                
                input_ids = val_batch['input_ids'].to(device)
                labels = val_batch['labels'].to(device)
                _, v_loss = model(input_ids, labels)
                current_val_loss += v_loss.item()

        avg_val_loss = current_val_loss / val_steps
        val_losses.append(avg_val_loss)
        val_loss_steps.append(step)
        logging.info(f"Step {step}, Val Loss: {avg_val_loss:.4f}")
        model.train()

    if step % save_interval == 0 and step > 0:
        torch.save(model.state_dict(), f'model_step_{step}.pt')

torch.save(model.state_dict(), 'model_final.pt')
logging.info("Training finished.")

# --- Plotting ---
# Loss Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(val_loss_steps, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)

# Learning Rate Plot
plt.subplot(1, 2, 2)
plt.plot(learning_rates)
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plots.png')
logging.info("Saved training plots to training_plots.png")
