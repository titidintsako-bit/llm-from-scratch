# Deploying NtsakoGPT — the final lap

Two milestones left: **train v3 in Colab**, then **deploy to a HF Space**.

## Step 1 — Train v3 in Colab (one-time, ~1.5–2.5 h on free T4)

Use the free GPU runtime (**Runtime → Change runtime type → T4 GPU**), then:

```python
# Cell 1 — clone and train
!rm -rf llm-from-scratch
!git clone https://github.com/titidintsako-bit/llm-from-scratch.git
%cd llm-from-scratch
!pip install -q datasets tokenizers
!python train_v3.py
```

Watch the log: the tokenizer training takes a few minutes, then
`Step    0 | train: 8.3xxx | val: 8.3xxx` — yes, ~8.3 = ln(4096), the random-guess
loss for the new vocab. Expect it to fall below 4.0 within the first few hundred steps.
Early stopping will end the run on its own.

```python
# Cell 2 — bring the model home (the step we missed last time!)
from google.colab import files
files.download("checkpoint_v3.pt")
```

## Step 2 — Verify locally before deploying

Drop `checkpoint_v3.pt` in `scratchpad/`, then:

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
- Generation on free CPU: ~1–2 s per token for the 25M model — fine for a demo.
- The Space never needs the `datasets` library — the tokenizer rides inside the checkpoint.
