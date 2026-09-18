"""
generate.py — Text Generation from a Trained GPT (v2 char-level AND v3 BPE)

The checkpoint knows what it needs:
  - v3 checkpoints contain "tokenizer_json" -> we rebuild a BPE tokenizer from it
  - v2/v1 checkpoints contain "stoi"/"itos"  -> char-level, as before
"""
import torch
from model import GPT, GPTConfig


def load_model_and_tokenizer(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not isinstance(config, GPTConfig):          # new checkpoints store a plain dict
        config = GPTConfig(**config)
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "tokenizer_json" in checkpoint:             # v3: BPE tokenizer travels inside
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])

        def encode(prompt):
            return tokenizer.encode(prompt).ids

        def decode(ids):
            return tokenizer.decode(ids)

        kind = "BPE (v3)"
    else:                                          # v1/v2: character-level
        stoi, itos = checkpoint["stoi"], checkpoint["itos"]

        def encode(prompt):
            return [stoi[c] for c in prompt if c in stoi]

        def decode(ids):
            return "".join(itos[i] for i in ids)

        kind = "char-level (v1/v2)"
    return model, encode, decode, kind


@torch.no_grad()
def generate(model, encode, decode, prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
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
    return decode(idx[0].tolist())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT")
    parser.add_argument("checkpoint", help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", default="To be or not", help="Starting text")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, encode, decode, kind = load_model_and_tokenizer(args.checkpoint)
    print(f"[{kind} | {model.config.n_layer}L/{model.config.n_head}H/{model.config.n_embd}D]")

    output = generate(model, encode, decode, args.prompt,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k)
    print(output)
