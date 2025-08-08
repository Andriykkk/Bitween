# !pip install huggingface_hub
# !pip install bitsandbytes
import torch
import logging
import os
import math
import time
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F

# !hf auth login

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = self.n_embd // self.n_head
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / self.head_size**0.5)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


# --- Configuration ---
config = GPTConfig(
    vocab_size=50257,  # GPT-2 tokenizer's vocab size
    block_size = 256,
    n_embd = 384,
    n_head=6,
    n_layer=6,
    dropout=0.1
)

# --- Hyperparameters ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8 
# LR Scheduler Settings
max_lr = 1e-4
min_lr = 1e-5
warmup_steps = 200
max_steps = 10000 # Total training steps

gradient_accumulation_steps = 4 # Number of steps to accumulate gradients for
eval_interval = 100
log_interval = 10
val_steps = 50
save_interval = 5000
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
        return tokenizer(examples['text'])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=6, remove_columns=["text"])

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

import bitsandbytes as bnb

# --- Model and Optimizer ---
model = GPT(config).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
# Use 8-bit Adam from bitsandbytes to save memory
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=max_lr)
logging.info("Using 8-bit AdamW optimizer from bitsandbytes.")

# number of parameters
num_parameters = sum(p.numel() for p in model.parameters())
logging.info(f"Number of parameters: {num_parameters / 1e6:.2f}M")

use_amp = True

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
log_file_path = "training_log.txt"
log_lines = ["Step\tTrain Loss\tVal Loss\tLearning Rate"]
train_losses = []
val_losses = []
val_loss_steps = []
learning_rates = []
val_loss_display = ""
start_time = time.time()

train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)

logging.info("Starting training...")
model.train()
optimizer.zero_grad()

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

        with autocast(dtype=torch.bfloat16, enabled=use_amp):
            logits, loss = model(input_ids, labels)

        loss = loss / gradient_accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    train_loss = loss.item() * gradient_accumulation_steps
    train_losses.append(loss.item() * gradient_accumulation_steps)
    if step > 0 and step % log_interval == 0:
        logging.info(f"Step {step}, Train Loss: {train_losses[-1]:.4f}, LR: {lr:.6f}")
    
    if step == 0 or step % eval_interval == 0 or step == max_steps:
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
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    _, v_loss = model(input_ids, labels)
                current_val_loss += v_loss.item()

        avg_val_loss = current_val_loss / val_steps
        val_losses.append(avg_val_loss)
        val_loss_steps.append(step)
        val_loss_display = f"{avg_val_loss:.4f}"
        logging.info(f"Step {step}, Val Loss: {val_loss_display}")
        model.train()

    log_lines.append(f"{step}\t{train_loss:.4f}\t{val_loss_display}\t{lr:.6f}")
    if step % save_interval == 0 and step > 0:
        torch.save(model.state_dict(), f'model_step_{step}.pt')

torch.save(model.state_dict(), 'model_final.pt')
logging.info("Training finished.")

# print time for training
end_time = time.time()
elapsed_time = end_time - start_time

formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print(f"Total training time: {formatted_time}")
logging.info(f"Total training time: {formatted_time}")

log_lines.append(f"Total training time: {formatted_time}")

with open("training_log.txt", "w") as f:
    f.write("\n".join(log_lines))

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
