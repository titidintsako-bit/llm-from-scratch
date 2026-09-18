---
title: NtsakoGPT
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
suggested_hardware: cpu-basic
short_description: A GPT trained from scratch (BPE tokenizer + wikitext-103)
tags:
  - gpt
  - transformer
  - from-scratch
  - education
---

# NtsakoGPT 🤖

A GPT-style transformer **built and trained from scratch** — no pretrained weights, no fine-tuning. Every parameter was learned by the model during its own training run (~25M params, 8-layer transformer, byte-level BPE tokenizer trained on wikitext-103).

Built as an educational project: character-level GPT → v2 with dropout + early stopping → v3 with a real tokenizer and corpus.
