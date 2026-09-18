"""
app.py — Gradio demo for NtsakoGPT (v3), for Hugging Face Spaces.

Loads the checkpoint (which contains the model weights, the config, AND the
BPE tokenizer) once at startup, then serves generation requests.

Space files needed: this file, model.py, checkpoint_v3.pt, requirements.txt, README.md
"""
import os

import gradio as gr
import torch

from model import GPT, GPTConfig

CKPT_PATH = os.environ.get("CHECKPOINT", "checkpoint_v3.pt")

# ---------------------------------------------------------------------------
# Load once at startup
# ---------------------------------------------------------------------------
checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
config = GPTConfig(**checkpoint["config"])
model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])

val_loss = checkpoint.get("val_loss", float("nan"))
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {CKPT_PATH}: {n_params / 1e6:.1f}M params, val_loss={val_loss:.4f}, device={device}")


@torch.no_grad()
def generate_text(prompt, max_new_tokens, temperature, top_k):
    if not prompt.strip():
        prompt = "The"          # empty prompt -> model free-associates
    idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    for _ in range(int(max_new_tokens)):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return tokenizer.decode(idx[0].tolist())


with gr.Blocks(title="NtsakoGPT") as demo:
    gr.Markdown(
        f"""# NtsakoGPT 🤖
A GPT trained **from scratch** — {n_params / 1e6:.1f}M parameters, trained on wikitext-103
with a BPE tokenizer learned from the data. *(best val loss: {val_loss:.3f})*
Not GPT-4. Not even GPT-2. But every parameter here was learned by this model itself."""
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="The history of Rome begins...", value="The history of Rome begins")
            with gr.Row():
                max_new = gr.Slider(32, 512, value=200, step=16, label="New tokens")
                temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.05, label="Temperature (higher = wilder)")
                top_k = gr.Slider(0, 200, value=40, step=10, label="Top-k (0 = off)")
            btn = gr.Button("Generate", variant="primary")
        output = gr.Textbox(label="Output", lines=14)
    btn.click(generate_text, [prompt, max_new, temperature, top_k], output)
    prompt.submit(generate_text, [prompt, max_new, temperature, top_k], output)

    gr.Examples(
        examples=[
            ["The history of Rome begins"],
            ["Once upon a time"],
            ["The most important discovery in science was"],
        ],
        inputs=[prompt],
    )

if __name__ == "__main__":
    demo.launch()
