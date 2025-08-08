import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
import os
import math
import time
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import bitsandbytes as bnb

# --- Configuration ---
# Using a smaller config for faster experimentation
class CompressorConfig:
    vocab_size = 50257  # GPT-2 tokenizer's vocab size
    block_size = 256
    n_embd = 384
    n_head = 6
    n_layer = 4 # Reduced layers for speed
    dropout = 0.1
    compression_ratio = 2 # e.g., 2 means compress from 256 to 128

config = CompressorConfig()

# --- Hyperparameters ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
max_lr = 1e-4
min_lr = 1e-5
warmup_steps = 200
max_steps = 5000 # Reduced steps for initial run
gradient_accumulation_steps = 4
eval_interval = 100
log_interval = 10
val_steps = 50
save_interval = 1000
tokenized_dataset_path = "tokenized_wikitext"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Definition ---

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, encoder_output):
        B, T, C = x.size()
        B_e, T_e, C_e = encoder_output.size()

        q = self.c_attn_q(x)
        k, v = self.c_attn_kv(encoder_output).split(self.n_embd, dim=2);

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B_e, T_e, self.n_head, C_e // self.n_head).transpose(1, 2)
        v = v.view(B_e, T_e, self.n_head, C_e // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # This is self-attention, but non-causal
        self.attn = CausalSelfAttention(config) 
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        # Modify the attention to be non-causal
        self.attn.register_buffer("bias", None)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, encoder_output):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), encoder_output)
        x = x + self.mlp(self.ln_3(x))
        return x

class CompressorAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shared components
        self.transformer_wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer_wpe = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        
        # Compression Layer
        self.compressed_len = config.block_size // config.compression_ratio
        self.compress_layer = nn.Linear(config.block_size * config.n_embd, self.compressed_len * config.n_embd)

        # Decoder
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer_wte.weight = self.lm_head.weight # Weight tying

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # --- ENCODER ---
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer_wte(idx)
        pos_emb = self.transformer_wpe(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        # --- COMPRESSION ---
        x_flat = x.view(B, -1)
        compressed_flat = self.compress_layer(x_flat)
        compressed_seq = compressed_flat.view(B, self.compressed_len, self.config.n_embd)

        # --- DECODER ---
        # The decoder input is the original sequence (teacher forcing)
        # In a real inference scenario, this would be generated token by token
        dec_input = x # Using the original embeddings as input to the decoder
        
        y = dec_input
        for block in self.decoder_blocks:
            y = block(y, compressed_seq)
        
        y = self.ln_f(y)
        logits = self.lm_head(y)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# --- Dataset and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained('gpt2')

if os.path.exists(tokenized_dataset_path):
    logging.info(f"Loading tokenized dataset from {tokenized_dataset_path}")
    lm_datasets = load_from_disk(tokenized_dataset_path)
else:
    logging.info("Tokenizing and caching dataset...")
    # Using wikitext for a more general compression task
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples['text'])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // config.block_size) * config.block_size
        result = {
            k: [t[i : i + config.block_size] for i in range(0, total_length, config.block_size)]
            for k, t in concatenated_examples.items()
        }
        # For an autoencoder, the labels are the same as the input
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

# --- Model and Optimizer ---
model = CompressorAutoencoder(config).to(device)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=max_lr)
logging.info("Using 8-bit AdamW optimizer from bitsandbytes.")

num_parameters = sum(p.numel() for p in model.parameters())
logging.info(f"Number of parameters: {num_parameters / 1e6:.2f}M")

use_amp = True

# --- LR Scheduler ---
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# --- Training Loop ---
log_file_path = "compressor_training_log.txt"
log_lines = ["Step\tTrain Loss\tVal Loss\tLearning Rate"]
train_losses = []
val_losses = []
val_loss_steps = []
learning_rates = []
val_loss_display = ""
start_time = time.time()

train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)

logging.info("Starting training for CompressorAutoencoder...")
model.train()
optimizer.zero_grad()

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for step in range(max_steps):
    lr = get_lr(step)
    learning_rates.append(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

        scaler.scale(loss).backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    train_loss = loss.item() * gradient_accumulation_steps
    train_losses.append(train_loss)
    
    if step > 0 and step % log_interval == 0:
        logging.info(f"Step {step}, Train Loss: {train_loss:.4f}, LR: {lr:.6f}")
    
    if step == 0 or step % eval_interval == 0 or step == max_steps -1:
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
                with autocast(dtype=torch.bfloat16, enabled=use_amp):
                    _, v_loss = model(input_ids, labels)
                current_val_loss += v_loss.item()

        avg_val_loss = current_val_loss / val_steps
        val_losses.append(avg_val_loss)
        val_loss_steps.append(step)
        val_loss_display = f"{avg_val_loss:.4f}"
        logging.info(f"Step {step}, Val Loss: {val_loss_display}")
        model.train()

    log_lines.append(f"{step}\t{train_loss:.4f}\t{val_loss_display}\t{lr:.6f}")
    if step > 0 and step % save_interval == 0:
        torch.save(model.state_dict(), f'compressor_model_step_{step}.pt')

torch.save(model.state_dict(), 'compressor_model_final.pt')
logging.info("Training finished.")

end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
logging.info(f"Total training time: {formatted_time}")
log_lines.append(f"Total training time: {formatted_time}")

with open(log_file_path, "w") as f:
    f.write("\n".join(log_lines))

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(val_loss_steps, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Compressor Training and Validation Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(learning_rates)
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
plt.savefig('compressor_training_plots.png')
logging.info("Saved training plots to compressor_training_plots.png")