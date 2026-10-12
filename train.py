"""Train Kylo's tiny character-level GPT from random initialization."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch

from model import TinyGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", default="checkpoints/kylo.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Training file not found: {data_path}. Create it using text you are allowed to use.")

    text = data_path.read_text(encoding="utf-8")
    if len(text) < args.block_size + 2:
        raise SystemExit(f"Training text is too short; provide at least {args.block_size + 2} characters.")

    # Character-level vocabulary keeps this first prototype dependency-light.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    if len(encoded) <= args.block_size + 1:
        raise SystemExit("Training text must be longer than block_size + 1.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT(vocab_size=len(chars), block_size=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(encoded) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([encoded[s : s + args.block_size] for s in starts]).to(device)
        y = torch.stack([encoded[s + 1 : s + args.block_size + 1] for s in starts]).to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step}/{args.steps} | loss {loss.item():.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "config": {
                "vocab_size": len(chars),
                "block_size": args.block_size,
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 128,
            },
            "itos": chars,
            "stoi": stoi,
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
