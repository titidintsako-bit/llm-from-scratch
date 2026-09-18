# 🧠 LLM From Scratch — Build a GPT Language Model

A hands-on workshop where you write every piece of a GPT training pipeline yourself, understanding what each component does and why.

## 🎯 What You'll Build

A working GPT model trained from scratch on Shakespeare, capable of generating Shakespeare-like text. You'll write:

- **Tokenizer** — turning text into numbers the model can process
- **Model architecture** — the transformer: embeddings, attention, feed-forward layers
- **Training loop** — forward pass, loss, backprop, optimizer, learning rate scheduling
- **Text generation** — sampling from your trained model

## 📚 Learning Resources

| File | What It Covers |
|------|----------------|
| `model.py` | GPT transformer architecture with detailed comments |
| `train.py` | Complete training pipeline with validation and checkpointing |
| `generate.py` | Text generation with temperature and top-k sampling |
| `attention_deep_dive.py` | Interactive explanation of self-attention |
| `backprop_deep_dive.py` | Interactive explanation of backpropagation |

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. Set GPU: **Runtime → Change runtime type → GPU (T4)**
4. Copy-paste the contents of `colab_train.py` into a cell
5. Run the cell — training takes ~15 minutes on free T4 GPU

### Option 2: Local Setup

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd llm-from-scratch
uv sync

# Train the model
cd scratchpad
python train.py
```

## 🚀 v3 — NtsakoGPT (the "GPT-2-lite" upgrade)

| Version | Tokenizer | Data | Params | Best val loss |
|---------|-----------|------|--------|---------------|
| v2 | char-level (65 tokens) | tiny Shakespeare (~1 MB) | 10.8M | 1.48 |
| v3 | **BPE (4096, trained from scratch)** | **wikitext-103 (~100 MB)** | **27.6M** | 3.21 (step 3600, run #1) |

What v3 adds: a real subword tokenizer (words become 1 token instead of 4–6),
a 100× bigger corpus, a bigger model with 512-token context, mixed-precision
training, and checkpoints that carry the tokenizer inside them.

```bash
# Train v3 — GPU required. ~5–6 h on a free Colab T4 = TWO sessions with
# checkpoints in Drive (CHECKPOINT_DIR) + RESUME=1 — see DEPLOY.md
pip install torch numpy tqdm datasets tokenizers
CHECKPOINT_DIR=. python train_v3.py
# ...if interrupted:
RESUME=1 CHECKPOINT_DIR=. python train_v3.py

# Generate — works with BOTH v2 char-level and v3 BPE checkpoints
python generate.py checkpoint_v3.pt --prompt "The history of Rome begins"
```

Deploy a public web demo with `space/` + `DEPLOY.md` (Hugging Face Spaces, free tier).

## 🏗️ Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│   TOKENIZER     │  "hello" → [20, 43, 50, 50, 53]
└────────┬────────┘
    ▼
┌─────────────────┐
│ Token Embed +   │  token IDs → vectors (384 dimensions)
│ Position Embed  │  + positional information
└────────┬────────┘
    ▼
┌─────────────────┐
│  TRANSFORMER    │  × 6 layers
│  BLOCK:         │
│  ┌────────────┐ │
│  │ LayerNorm  │ │
│  │ Self-Attn  │ │  6 parallel attention heads
│  │ + Residual │ │
│  ├────────────┤ │
│  │ LayerNorm  │ │
│  │ MLP (FFN)  │ │  expand 4x, GELU, project back
│  │ + Residual │ │
│  └────────────┘ │
└────────┬────────┘
    ▼
┌─────────────────┐
│ LayerNorm       │
│ Linear → logits │  65 outputs (probability over next char)
└─────────────────┘
```

## 📊 Model Configs

| Config | Params | n_layer | n_head | n_embd | Train Time (T4 GPU) |
|--------|--------|---------|--------|--------|---------------------|
| Tiny | ~0.5M | 2 | 2 | 128 | ~3 min |
| Small | ~4M | 4 | 4 | 256 | ~8 min |
| **Medium (default)** | **~10M** | **6** | **6** | **384** | **~15 min** |

## 📈 Training Progress

Watch the model learn to write Shakespeare:

**Step 100** (val loss: ~3.2) — Random characters:
```
To be or notis p ce mei odorethleedetire'ilethed ye m arkesothir fnon b tigb'i.
```

**Step 500** (val loss: ~2.2) — Words forming:
```
To be or not men, and my lord.

ROMEO:
Thou sir, do content the he, stray, there ir;
```

**Step 1000** (val loss: 1.64) — Coherent phrases:
```
To be or nothing are good men,
The profent of little, our actory.

CORIOLANUS:
Is it now of your many death?
```

**Step 2400** (val loss: ~1.60) — Peak quality:
```
To be or not to be some of you shall know
That everlature by Romeo: what news,
Which you had knock'd my part to speak
```

## 🔬 Key Concepts

### Tokenization
- Convert text → numbers the model can process
- Character-level: 65 unique chars in Shakespeare
- Simple but effective for small datasets

### Self-Attention
- Lets every token look at every other token
- Formula: `Attention(Q,K,V) = softmax(QK^T / √d_k) @ V`
- Causal masking prevents looking at future tokens

### Backpropagation
- Computes how each weight affects the loss
- Uses chain rule to trace error backwards
- Gradient descent updates weights: `new = old - lr × gradient`

## 📁 Project Structure

```
llm-from-scratch/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── colab_train.py           # Complete script for Google Colab
├── scratchpad/
│   ├── model.py             # GPT transformer architecture
│   ├── train.py             # Training loop
│   ├── generate.py          # Text generation
│   ├── attention_deep_dive.py   # Attention explanation
│   └── backprop_deep_dive.py    # Backpropagation explanation
│   ├── train_v3.py          # v3: BPE tokenizer + wikitext-103 (NtsakoGPT)
│   └── space/               # Hugging Face Space demo (app.py + README.md)
└── data/
    └── shakespeare.txt      # Training dataset
```

## 🎓 Learning Path

1. **Start with the README** — understand the big picture
2. **Read `model.py`** — learn the transformer architecture
3. **Read `train.py`** — understand how training works
4. **Read `generate.py`** — see how text generation works
5. **Run `attention_deep_dive.py`** — deep dive into attention
6. **Run `backprop_deep_dive.py`** — deep dive into backpropagation
7. **Train the model** — watch it learn to write Shakespeare!

## 🏆 Competition: Best AI Poet

Once you've trained a model, try the competition:
1. Find a poetry dataset (Poetry Foundation, Project Gutenberg)
2. Train a model on the poetry data
3. Generate the best poem you can
4. Submit your checkpoint and generation command

## 📚 Further Reading

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Minimal GPT training in ~300 lines of PyTorch
- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — 4-hour video building GPT-2
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — The original transformer paper
- [GPT-2 Paper (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Language models as unsupervised learners

## 🤝 Contributing

This is a learning project! Feel free to:
- Add more documentation
- Improve the code
- Add new training experiments
- Share your results

## 📄 License

MIT License — feel free to use this for learning!
