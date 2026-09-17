"""
train.py — Complete training pipeline for our GPT
This is the improved version of what you ran in Colab. Changes from v1:
  1. Dropout (fixes the overfitting we saw: val loss 1.54 -> 4.17)
  2. Early stopping (stop when val loss stops improving)
  3. Better hyperparameters (batch 32, fewer steps, warmup 5%, floor 10%)
  4. Portable checkpoints (config saved as a plain dict)
  5. SMOKE_TEST flag for a 2-minute sanity run before the real training
"""
import math
import os
import time
import json
import urllib.request
import dataclasses

import torch
from tqdm import tqdm

from model import GPT, GPTConfig

# ============================================================================
# CONFIGURATION — edit these to control the whole training run
# ============================================================================
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"  # or flip to True manually
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join("data", "shakespeare.txt")
CHECKPOINT_PATH = "checkpoint_best.pt"

# If SMOKE_TEST, use a tiny config so we can verify everything works in minutes
if SMOKE_TEST:
    N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE, BATCH_SIZE, MAX_STEPS, EVAL_INTERVAL = 2, 2, 128, 128, 8, 200, 50
else:
    N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE, BATCH_SIZE, MAX_STEPS, EVAL_INTERVAL = 6, 6, 384, 256, 32, 2000, 100

DROPOUT = 0.2             # The fix for our overfitting problem
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
PATIENCE = 10             # Stop after this many evals without val-loss improvement
GRAD_CLIP = 1.0

# ============================================================================
# DATA LOADING
# ============================================================================
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    print(f"Downloading Shakespeare to {DATA_PATH}...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)

n = int(0.9 * len(tokens))  # 90% train, 10% val
train_tokens, val_tokens = tokens[:n], tokens[n:]
print(f"Dataset: {len(tokens):,} chars | vocab: {vocab_size} | train: {len(train_tokens):,} | val: {len(val_tokens):,}")

def get_batch(split_tokens):
    ix = torch.randint(len(split_tokens) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([split_tokens[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([split_tokens[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# ============================================================================
# SETUP
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"Device: {device}")

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    dropout=DROPOUT,
)
model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D, {n_params / 1e6:.1f}M params, dropout={DROPOUT}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate schedule: linear warmup (5% of steps) then cosine decay to 10% of max
warmup_steps = max(10, int(0.05 * MAX_STEPS))
min_lr = LEARNING_RATE * 0.1

def get_lr(step):
    if step < warmup_steps:
        return LEARNING_RATE * (step + 1) / warmup_steps
    if step >= MAX_STEPS:
        return min_lr
    progress = (step - warmup_steps) / (MAX_STEPS - warmup_steps)
    return min_lr + 0.5 * (LEARNING_RATE - min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss():
    """Average loss over eval batches — less noisy than a single batch."""
    model.eval()
    out = {}
    for name, split in [("train", train_tokens), ("val", val_tokens)]:
        losses = torch.zeros(20)
        for k in range(20):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out

# ============================================================================
# GENERATION (for progress samples during training)
# ============================================================================
@torch.no_grad()
def generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    model.eval()
    tokens_out = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens_out], dtype=torch.long, device=device)
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
    model.train()
    return "".join([itos[i] for i in idx[0].tolist()])

# ============================================================================
# TRAINING LOOP with early stopping
# ============================================================================
loss_log = {"steps": [], "train": [], "val": []}
best_val_loss = float("inf")
best_step = 0
evals_without_improvement = 0
t0 = time.time()

pbar = tqdm(range(MAX_STEPS), desc="Training")
for step in pbar:
    # ---- Periodic evaluation + checkpointing ----
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
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
                "config": dataclasses.asdict(config),   # plain dict = portable
                "stoi": stoi,
                "itos": itos,
                "val_loss": val_loss,
            }, CHECKPOINT_PATH)
            tqdm.write(f"  -> new best ({val_loss:.4f}), checkpoint saved")
        else:
            evals_without_improvement += 1
            tqdm.write(f"  -> no improvement for {evals_without_improvement}/{PATIENCE} evals")

    # ---- Early stopping ----
    if evals_without_improvement >= PATIENCE:
        tqdm.write(f"\nEarly stopping at step {step}: val loss hasn't improved in {PATIENCE} evals.")
        tqdm.write(f"Best val loss {best_val_loss:.4f} was at step {best_step}.")
        break

    # ---- Learning rate schedule ----
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # ---- One optimization step ----
    x, y = get_batch(train_tokens)
    x, y = x.to(device), y.to(device)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

elapsed = time.time() - t0
print(f"\nTraining finished in {elapsed / 60:.1f} min | best val loss: {best_val_loss:.4f} (step {best_step})")

with open("loss_log.json", "w") as f:
    json.dump(loss_log, f)

# ============================================================================
# SAMPLE FROM THE BEST MODEL
# ============================================================================
print("\n" + "=" * 60)
print("Loading best checkpoint and generating samples...")
print("=" * 60)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
best_config = GPTConfig(**ckpt["config"])
best_model = GPT(best_config).to(device)
best_model.load_state_dict(ckpt["model_state_dict"])

for prompt in ["To be or not", "ROMEO:", "The world is"]:
    torch.manual_seed(42)
    output = generate(best_model, prompt)
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    print(output)
