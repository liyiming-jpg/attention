# Attention Is All You Need — Minimal Reproduction

This repository provides a compact, end-to-end reproduction of the Transformer architecture from the paper "Attention Is All You Need".

It includes:
- Multi-head self-attention
- Positional encoding
- Feed-forward blocks
- Encoder and decoder stacks
- A small synthetic training loop that demonstrates the model learns a copy task

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## What this reproduces

The implementation follows the standard paper structure:
- scaled dot-product attention
- residual connections + layer normalization
- encoder-decoder Transformer design
- teacher-forced training for a toy seq2seq problem

## Files

- `train.py` — entry point
- `transformer_repro.py` — model implementation
- `requirements.txt` — Python dependencies
