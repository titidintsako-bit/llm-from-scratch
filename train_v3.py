"""
train_v3.py — "GPT-2-lite": the upgrade from v2.

Why v3 exists (the lesson): v2 plateaued at val loss ~1.48. That number looks
like a code problem, but it is a DATA + TOKENIZER problem:
  - v2 reads text 1 character at a time (65-token vocab). The phrase " think"
    costs 6 tokens. GPT-2 reads it as 1 token out of 50,257.
  - v2 saw ~1 MB of Shakespeare. GPT-2 saw ~40 GB of web text.
  - Every 256-token window covers ~256 characters (~a sentence). With BPE at
    4096 vocab it covers ~16,000 characters (~15 paragraphs). Context becomes
    meaningful.

What changes from v2:
  1. BPE tokenizer trained from scratch on the corpus (tokenizers lib),
     4096 merges — vocabulary learned, not hand-listed.
  2. Corpus: wikitext-103 (~100M chars of curated encyclopedic English,
     public domain). Streams via `datasets`; no manual download.
  3. Bigger model (~25M params) + block_size 512.
  4. Mixed precision (bf16 if available, fp16 fallback) — ~2x faster on GPU.
  5. The tokenizer is saved INSIDE every checkpoint (portable: model +
     tokenizer + config travel together).
  6. Same early stopping that rescued v2 — because it works.

Estimated: 1.5–2.5 h on a free T4 GPU with early stopping.
"""
import math
import os
import time
import json
import dataclasses

import torch
from tqdm import tqdm

from model import GPT, GPTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
VOCAB_SIZE = 4096        # BPE merges learned from the corpus
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
CHECKPOINT_PATH = "checkpoint_v3.pt"

if SMOKE_TEST:
    # Tiny config: 2-min CPU run that exercises the full pipeline.
    N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE = 2, 2, 128, 128
    BATCH_SIZE, MAX_STEPS, EVAL_INTERVAL, PATIENCE = 4, 60, 10, 2
    NUM_PROC = 1
    CORPUS_LIMIT = 200_000       # chars — smoke test only
else:
    N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE = 8, 8, 512, 512
    BATCH_SIZE, MAX_STEPS, EVAL_INTERVAL, PATIENCE = 16, 10000, 200, 15
    NUM_PROC = 2                 # T4 vCPU count; harmless on CPU

DROPOUT = 0.1
LEARNING_RATE = 6e-4           # bigger model -> lower LR than v2's 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
GRAD_ACCUM = 4                 # effective batch 64 sequences x 512 tokens
CORPUS_CHARS = 100_000_000     # ~100M chars of wiki text
PAD_BY_NEWLINE = True          # docs are joined with newlines (GPT-2 style)

# ============================================================================
# 1) LOAD THE CORPUS (streaming; no big download to disk)
# ============================================================================
print(f"Loading corpus: {DATASET_NAME}/{DATASET_CONFIG} ...")
from datasets import load_dataset

ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)


def corpus_iterator():
    cap = CORPUS_LIMIT if SMOKE_TEST else CORPUS_CHARS
    n_chars = 0
    for row in ds:
        line = row["text"]
        if PAD_BY_NEWLINE and line and not line.endswith("\n"):
            line += "\n"
        n_chars += len(line)
        yield line
        if n_chars >= cap:
            break


# ============================================================================
# 2) TRAIN A BPE TOKENIZER ON THE CORPUS (the big idea of v3)
# ============================================================================
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

TOKENIZER_PATH = "tokenizer_v3.json"
if os.path.exists(TOKENIZER_PATH) and not SMOKE_TEST:
    print(f"Reusing existing tokenizer: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
else:
    print(f"Training BPE tokenizer: vocab_size={VOCAB_SIZE} ...")
    tokenizer = Tokenizer(BPE(unk_token=None))      # byte-level fallback: no UNK ever
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=ByteLevel.alphabet(),      # all 256 bytes as base tokens
        show_progress=True,
    )
    # The GPT-2 recipe: byte-level pre-tokenization means spaces live INSIDE tokens
    # ("Ġthe" = " the"), and the matching decoder reassembles text exactly.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
    tokenizer.save(TOKENIZER_PATH)

actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
print(f"Tokenizer ready: {actual_vocab} tokens")
for s in [" the", " think", " king", "Fantastical"]:
    ids = tokenizer.encode(s).ids
    print(f"  {s!r:15} -> {len(ids)} token(s) {ids}")

# ============================================================================
# 3) COLLECT THE CORPUS TEXT (second stream pass) AND SPLIT
# ============================================================================
text_parts = []
for line in corpus_iterator():                     # same source the tokenizer saw
    text_parts.append(line)
text_all = "".join(text_parts)

print(f"Corpus: {len(text_all):,} chars -> encoding ...")
ids = tokenizer.encode(text_all).ids
tokens = torch.tensor(ids, dtype=torch.long)
n = int(0.99 * len(tokens))                        # 99% train, 1% val (big data!)
train_tokens, val_tokens = tokens[:n], tokens[n:]
print(f"Tokens: {len(tokens):,} | vocab: {actual_vocab} | train: {len(train_tokens):,} | val: {len(val_tokens):,}")
print(f"(v2 had ~1.1M char-tokens. We now have ~{len(tokens) / 1_100_000:.0f}x more signal.)")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"Device: {device}")


def get_batch(split_tokens):
    ix = torch.randint(len(split_tokens) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([split_tokens[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([split_tokens[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================================
# 4) MODEL + OPTIMIZER + AMP
# ============================================================================
config = GPTConfig(
    vocab_size=actual_vocab,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    dropout=DROPOUT,
)
model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D/{BLOCK_SIZE}ctx, {n_params / 1e6:.1f}M params, dropout={DROPOUT}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Mixed precision: bf16 needs no loss scaler, fp16 does; fp32 fallback on CPU.
use_amp = device == "cuda"
amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
print(f"AMP: {use_amp} ({amp_dtype})")

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
    model.eval()
    out = {}
    for name, split in [("train", train_tokens), ("val", val_tokens)]:
        losses = torch.zeros(10)
        for k in range(10):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(model, prompt, max_new_tokens=150, temperature=0.8, top_k=40):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    eot = tokenizer.token_to_id("<|endoftext|>")
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float() / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if eot is not None and next_token.item() == eot:
            break
        idx = torch.cat([idx, next_token], dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist())


# ============================================================================
# 5) TRAINING LOOP (v2's proven logic: eval -> checkpoint-if-best -> early stop)
# ============================================================================
loss_log = {"steps": [], "train": [], "val": []}
best_val_loss = float("inf")
best_step = 0
evals_without_improvement = 0
t0 = time.time()


def save_checkpoint(step, val_loss):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(config),
        "tokenizer_json": tokenizer.to_str(),       # tokenizer travels with the model
        "tokenizer_format": "tokenizers.BPE.v3",
        "val_loss": val_loss,
    }, CHECKPOINT_PATH)


pbar = tqdm(range(MAX_STEPS), desc="Training v3")
for step in pbar:
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
            save_checkpoint(step, val_loss)
            tqdm.write(f"  -> new best ({val_loss:.4f}), checkpoint saved")
        else:
            evals_without_improvement += 1
            tqdm.write(f"  -> no improvement for {evals_without_improvement}/{PATIENCE} evals")

    if evals_without_improvement >= PATIENCE:
        tqdm.write(f"\nEarly stopping at step {step}: no val improvement in {PATIENCE} evals.")
        tqdm.write(f"Best val loss {best_val_loss:.4f} at step {best_step}.")
        break

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Gradient accumulation: effective batch = BATCH_SIZE * GRAD_ACCUM
    optimizer.zero_grad(set_to_none=True)
    for micro in range(GRAD_ACCUM):
        x, y = get_batch(train_tokens)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()

    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

elapsed = time.time() - t0
print(f"\nTraining finished in {elapsed / 60:.1f} min | best val loss: {best_val_loss:.4f} (step {best_step})")

with open("loss_log_v3.json", "w") as f:
    json.dump(loss_log, f)

# ============================================================================
# 6) SAMPLES FROM THE BEST MODEL
# ============================================================================
print("\n" + "=" * 60)
print("Loading best checkpoint and generating samples...")
print("=" * 60)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
best_model = GPT(GPTConfig(**ckpt["config"])).to(device)
best_model.load_state_dict(ckpt["model_state_dict"])

for prompt in ["The history of Rome begins", "Once upon a time", "The most important thing"]:
    torch.manual_seed(42)
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    print(generate(best_model, prompt))

# Some tokenizer/pyarrow builds leave a background thread whose teardown prints a
# scary-but-harmless "Fatal Python error: PyGILState_Release" at interpreter exit.
# Flush everything first, then exit hard to keep the log clean for the student.
import sys
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
