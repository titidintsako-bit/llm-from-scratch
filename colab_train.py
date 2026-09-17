"""
colab_train.py — Complete GPT Training Script for Google Colab (v2)
Copy this entire file into a Colab notebook cell and run it.
Set Runtime -> Change runtime type -> GPU (T4).

v2 changes (after analyzing our first training run):
  1. DROPOUT added — v1 had none, so the model memorized the training set
     (val loss fell to 1.54 by step 800, then climbed to 4.17 by step 4900)
  2. EARLY STOPPING — stops when val loss stops improving
  3. Fewer steps (2000) + smaller batch (32) — v1 wasted ~4200 steps overfitting
  4. PORTABLE checkpoints — config saved as a plain dict, loads anywhere
  5. Best-checkpoint logic kept: we always ship the lowest-val-loss model
"""

# STEP 1: Install Dependencies
print("=" * 60)
print("STEP 1: Installing dependencies...")
print("=" * 60)
import subprocess, sys
for package in ['torch', 'numpy', 'tqdm']:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
print("Dependencies installed!")

# STEP 2: Download Shakespeare Dataset
print("\n" + "=" * 60)
print("STEP 2: Downloading Shakespeare dataset...")
print("=" * 60)
import urllib.request, os
os.makedirs("data", exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "data/shakespeare.txt")
with open("data/shakespeare.txt", "r") as f:
    text = f.read()
print(f"Shakespeare downloaded: {len(text):,} characters")

# STEP 3: Define the Model (WITH dropout — the fix for our overfitting)
print("\n" + "=" * 60)
print("STEP 3: Defining the GPT model...")
print("=" * 60)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, asdict

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2   # v1 had 0.0 -> the model memorized the training set

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # Randomly zero some attention weights during training (regularization)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.dropout(self.c_proj(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

print("Model defined! (dropout=0.2 this time)")

# STEP 4: Training Functions
print("\n" + "=" * 60)
print("STEP 4: Defining training functions...")
print("=" * 60)

def load_data(filepath, block_size, batch_size, device):
    with open(filepath, "r") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(tokens):,} chars, vocab size: {vocab_size}")
    def get_batch(split_tokens):
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y
    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    return get_train, get_val, vocab_size, stoi, itos

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss(model, get_train_batch, get_val_batch, eval_iters=20):
    """Average loss over several batches — less noisy than a single batch."""
    model.eval()
    out = {}
    for name, get_batch in [("train", get_train_batch), ("val", get_val_batch)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch()
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate(model, prompt, stoi, itos, max_new_tokens=200, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return "".join([itos[i] for i in idx[0].tolist()])

print("Training functions defined!")

# STEP 5: Train the Model (with early stopping this time)
print("\n" + "=" * 60)
print("STEP 5: Training the GPT model...")
print("=" * 60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

block_size = 256
batch_size = 32          # v1 used 64 — 32 is plenty and trains faster per step
max_steps = 2000         # v1 used 5000 — we overfit long before that
eval_interval = 100
patience = 10            # stop after 10 evals (1000 steps) with no improvement
max_lr = 1e-3
min_lr = max_lr * 0.1
warmup_steps = 100

get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
    "data/shakespeare.txt", block_size, batch_size, device
)
config = GPTConfig(vocab_size=vocab_size, block_size=block_size)
model = GPT(config).to(device)
print(f"Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}D, "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, dropout={config.dropout}")

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
loss_log = {"steps": [], "train": [], "val": []}
best_val_loss = float('inf')
best_step = 0
evals_without_improvement = 0

from tqdm import tqdm
import json

print("\nStarting training...")
pbar = tqdm(range(max_steps), desc="Training")
for step in pbar:
    if step % eval_interval == 0:
        losses = estimate_loss(model, get_train_batch, get_val_batch)
        val_loss = losses["val"]
        loss_log["steps"].append(step)
        loss_log["train"].append(losses["train"])
        loss_log["val"].append(val_loss)
        tqdm.write(f"Step {step:5d} | train: {losses['train']:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            evals_without_improvement = 0
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": asdict(config),  # plain dict = portable
                "stoi": stoi, "itos": itos,
                "val_loss": val_loss,
            }, "checkpoint_best.pt")
            tqdm.write(f"  -> new best ({val_loss:.4f}), checkpoint saved")
        else:
            evals_without_improvement += 1
            tqdm.write(f"  -> no improvement for {evals_without_improvement}/{patience} evals")

        if evals_without_improvement >= patience:
            tqdm.write(f"\nEarly stopping: val loss hasn't improved in {patience} evals.")
            tqdm.write(f"Best val loss {best_val_loss:.4f} was at step {best_step}.")
            break

    lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_train_batch()
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

with open("loss_log.json", "w") as f:
    json.dump(loss_log, f)
print(f"\nTraining complete! Best val loss: {best_val_loss:.4f} (step {best_step})")

# STEP 6: Generate Text from the BEST checkpoint
print("\n" + "=" * 60)
print("STEP 6: Generating Shakespeare text...")
print("=" * 60)
checkpoint = torch.load("checkpoint_best.pt", map_location=device, weights_only=False)
config = GPTConfig(**checkpoint["config"])   # portable dict -> config
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

for prompt in ["To be or not", "ROMEO:", "Roses are red"]:
    torch.manual_seed(42)
    output = generate(model, prompt, stoi, itos, max_new_tokens=200, temperature=0.8)
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    print(output)

# STEP 7: Plot Loss Curves
print("\n" + "=" * 60)
print("STEP 7: Plotting loss curves...")
print("=" * 60)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(loss_log["steps"], loss_log["train"], alpha=0.3, label="Train Loss")
    plt.plot(loss_log["steps"], loss_log["val"], label="Val Loss", marker='o')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves (v2: with dropout + early stopping)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
except ImportError:
    print("matplotlib not installed, skipping plot")

print("\n" + "=" * 60)
print("DONE! Best model saved as checkpoint_best.pt")
print("Download it with:  files.download('checkpoint_best.pt')")
print("=" * 60)
