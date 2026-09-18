# TESTING.md — Break the model yourself

A model you haven't poked is a model you don't understand. Every experiment here
runs on your machine (no GPU needed) using the smoke checkpoint. Before each run,
make a prediction — then check it. That prediction-check loop *is* machine learning
intuition training.

```bash
cd llm-with-scratch/scratchpad
PY=.venv/bin/python
```

---

## A. Tokenizer experiments — "the model's eyes"

**Q1: is the tokenizer truly lossless?** Prediction: the decode should exactly
equal the input for any string, even emoji.

```bash
$PY - <<'EOF'
import torch
ckpt = torch.load("checkpoint_v3.pt", weights_only=False)
from tokenizers import Tokenizer
tok = Tokenizer.from_str(ckpt["tokenizer_json"])
for s in ["Roses are red", "Sawubona Mntungwa!", "naïve café ☕", "  spaces  preserved  "]:
    ids = tok.encode(s).ids
    print(f"{s!r:30} -> {len(ids):3d} tokens -> roundtrip ok: {tok.decode(ids) == s}")
EOF
```

**Q2: why does the BPE model "think" faster?** Compare token counts. Prediction:
the BPE line should be several times shorter.

```bash
$PY - <<'EOF'
import torch
ckpt = torch.load("checkpoint_v3.pt", weights_only=False)
from tokenizers import Tokenizer
tok = Tokenizer.from_str(ckpt["tokenizer_json"])
s = "The history of Rome begins in the ancient city"
bpe = len(tok.encode(s).ids)
chars = len(s)
print(f"sentence: {s!r}")
print(f"BPE tokens: {bpe}   |   char tokens (v2 style): {chars}")
print(f"-> each BPE token covers ~{chars/bpe:.1f} chars; a 512-token window sees ~{512*chars/bpe:.0f} chars")
EOF
```

**Q3: what does a token ID look like inside?** Prediction: merged pairs will
decode to multi-character fragments like "ing" or " the".

```bash
$PY - <<'EOF'
import torch
ckpt = torch.load("checkpoint_v3.pt", weights_only=False)
from tokenizers import Tokenizer
tok = Tokenizer.from_str(ckpt["tokenizer_json"])
ids = tok.encode("The history of Rome begins").ids
for i in ids:
    print(f"id {i:4d} -> {tok.decode([i])!r}")
EOF
```

---

## B. Sampling experiments — the generation dials

**B1: temperature.** Run each command twice. Low temp = repetitive/safe, high
temp = adventurous/incoherent. (Output will be nonsense words — the toy model
only trained 20 seconds — but watch the *texture* change.)

```bash
$PY generate.py checkpoint_v3.pt --prompt "The history of" --temperature 0.3 --max_new_tokens 40 --seed 1
$PY generate.py checkpoint_v3.pt --prompt "The history of" --temperature 1.5 --max_new_tokens 40 --seed 1
```

**B2: seed reproducibility.** Prediction: same seed + same temperature = byte-identical output.

```bash
$PY generate.py checkpoint_v3.pt --prompt "Test" --seed 123 --max_new_tokens 30
$PY generate.py checkpoint_v3.pt --prompt "Test" --seed 123 --max_new_tokens 30
```

**B3: top-k.** With `--top_k 1` the model must always pick its #1 choice —
sampling becomes deterministic. Prediction: two runs differ, but each is
"greedy-ish" and loops.

```bash
$PY generate.py checkpoint_v3.pt --prompt "The" --top_k 1 --max_new_tokens 40
$PY generate.py checkpoint_v3.pt --prompt "The" --top_k 1 --max_new_tokens 40
```

**B4: prompt sensitivity.** The model continues *style*, not meaning. Try:

```bash
$PY generate.py checkpoint_v3.pt --prompt "Once upon a time" --max_new_tokens 60
$PY generate.py checkpoint_v3.pt --prompt "def main():" --max_new_tokens 60
```

---

## C. Checkpoint anatomy — what's actually in the .pt file?

```bash
$PY - <<'EOF'
import torch
ckpt = torch.load("checkpoint_v3.pt", weights_only=False)
print("keys:", list(ckpt.keys()))
print("step:", ckpt["step"], "| val_loss:", ckpt["val_loss"])
print("config:", ckpt["config"])
sd = ckpt["model_state_dict"]
print(f"\n{len(sd)} tensors. Shapes of the first layer:")
for k in list(sd)[:6]:
    print(f"  {k:45} {tuple(sd[k].shape)}")
n = sum(v.numel() for v in sd.values())
print(f"\ntotal parameters: {n:,}")
# The tied-weights trick: wte and lm_head share storage
print("embedding == lm_head (weight tying):", torch.equal(sd["transformer.wte.weight"], sd["lm_head.weight"]))
EOF
```

Watch for: `transformer.wte.weight` with shape (4096, 128) — one learned vector
per token — and the tied `lm_head.weight`. That's the embedding table you
revised in the quiz, live.

> **Trap for experts:** summing `state_dict` gives 1,461,760 params but the model
> only *has* 937,472. Weight tying means `wte.weight` and `lm_head.weight` are the
> SAME tensor counted twice (4096×128 = 524,288 shared numbers). One parameter,
> two jobs: it reads tokens in AND projects logits out. GPT-2 does this too.

---

## D. The adversarial test — try to make it fail

**D1: out-of-distribution prompt.** The model only knows Wikipedia-ish English.
Prediction: input from a totally different domain degrades gracefully (byte-level
means no crash, just confusion).

```bash
$PY generate.py checkpoint_v3.pt --prompt "    = = = " --max_new_tokens 40
$PY generate.py checkpoint_v3.pt --prompt "SELECT * FROM" --max_new_tokens 40
```

**D2: empty prompt.** `generate.py` substitutes "The". What does a bare start
feel like?

```bash
$PY generate.py checkpoint_v3.pt --prompt "" --max_new_tokens 60
```

**D3: context overflow.** Feed a prompt longer than block_size (128 for the toy
checkpoint). Prediction: no crash — the loop crops `idx[:, -block_size:]` and
only the tail is remembered.

```bash
$PY - <<'EOF'
import torch
from generate import load_model_and_tokenizer, generate
model, encode, decode, kind = load_model_and_tokenizer("checkpoint_v3.pt")
long_prompt = "The history of Rome begins " * 20          # ~140 tokens > block_size 128
print(f"prompt tokens: {len(encode(long_prompt))} vs block_size={model.config.block_size}")
out = generate(model, encode, decode, long_prompt, max_new_tokens=15)
print("model survived, continued with:", repr(out[len(long_prompt):][:80]))
EOF
```

---

## What you're looking for

| Experiment | Skill it builds |
|---|---|
| A1 roundtrip | Trust but verify: tokenizers must be lossless |
| A2 counts | Feel for why subword tokenization wins |
| B1/B3 dials | The sample-quality axes you'll tune in the competition |
| B2 seeds | Reproducibility = science |
| C anatomy | Checkpoints are dicts; tensors have *meaningful shapes* |
| D adversarial | Find the edges before your users do |

When you're done: run the real thing. `DEPLOY.md` → Colab cells → the 25M model.
The toy speaks nonsense; the real one speaks English.
