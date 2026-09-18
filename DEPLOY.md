# Deploying NtsakoGPT — the final lap

Two milestones left: **train v3 in Colab**, then **deploy to a HF Space**.

> Reality check from run #1: the full config needs ~5–6 h of T4 time. Colab free
> tier gives you ~3–4 h before quota runs out — so **plan on 2 sessions**, and
> make every checkpoint land in Google Drive so nothing is ever lost to a
> disconnect again.

## Step 1 — Train v3 in Colab (Drive-backed, resumable)

Use the GPU runtime (**Runtime → Change runtime type → T4 GPU**), then:

```python
# Cell 1 — one-time setup: clone, install, mount Drive
!rm -rf llm-from-scratch
!git clone https://github.com/titidintsako-bit/llm-from-scratch.git
%cd llm-from-scratch
!pip install -q datasets tokenizers

from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/NtsakoGPT
```

```python
# Cell 2 — train (checkpoints go to Drive, survive disconnects)
!CHECKPOINT_DIR=/content/drive/MyDrive/NtsakoGPT python train_v3.py
```

Watch the log: `Step 0` should print val ≈ 8.41 (= ln(4096), the random-guess
loss for this vocab). It fell to **3.21 by step 3600** in run #1 — healthy.

**If quota/disconnect hits mid-run:** just reconnect (GPU again), re-run Cell 1
and Cell 2, then:

```python
# Cell 3 — resume exactly where it stopped
!RESUME=1 CHECKPOINT_DIR=/content/drive/MyDrive/NtsakoGPT python train_v3.py
```

Resume restores the model **and** the AdamW optimizer state, so the loss curve
continues seamlessly. Only catch: the tokenizer/corpus pass (~2 min) re-runs
before training continues — that's normal.

## Step 2 — Bring the model home (the step we missed last time!)

From Drive, or straight from a finished session:

```python
from google.colab import files
files.download("/content/drive/MyDrive/NtsakoGPT/checkpoint_v3.pt")  # ~320MB
```

Put it in `scratchpad/` on your machine, then verify locally before deploying:

```bash
python3 generate.py checkpoint_v3.pt --prompt "The history of Rome begins" --max_new_tokens 200
```

## Step 3 — Create the Hugging Face Space

1. Create an account at huggingface.co, then: **New → Space**
2. Space name: `NtsakoGPT` · SDK: **Gradio** · Hardware: **CPU basic (free)** · Public
3. Upload these 5 files into the Space (via the web UI "Files" tab, or git):
   - `space/README.md` → rename to just `README.md` at the Space root
   - `space/app.py`
   - `space/requirements.txt`
   - `model.py` (copy from this folder)
   - `checkpoint_v3.pt` (your trained model)

Easiest upload path (needs `pip install huggingface_hub`, one `huggingface-cli login`):

```bash
huggingface-cli upload <your-username>/NtsakoGPT . --repo-type space
# run from a folder containing exactly: README.md (the space one), app.py,
# requirements.txt, model.py, checkpoint_v3.pt
```

The first build takes ~5 min (it installs torch). Your Space then lives at
`https://huggingface.co/spaces/<your-username>/NtsakoGPT` — a public URL anyone can use.

## Notes

- If the Space build fails on `sdk_version`, create the Space first and copy the
  `sdk_version` value it puts in its generated `README.md` (newest Gradio changes often).
- Generation on free CPU: ~1–2 s per token for the 27M model — fine for a demo.
- The Space never needs the `datasets` library — the tokenizer rides inside the checkpoint.
