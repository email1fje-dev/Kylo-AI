"""Train Kylo's tiny character-level GPT from random initialization.

This is an educational prototype, not a production chat model. Training data
must be supplied by the user and must be licensed/owned for this use.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from model import TinyGPT


def save_checkpoint(output_path, model, chars, stoi, block_size, step, elapsed):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "config": {
                "vocab_size": len(chars),
                "block_size": block_size,
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 128,
            },
            "itos": chars,
            "stoi": stoi,
            "training": {"step": step, "elapsed_seconds": elapsed},
        },
        output_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--minutes", type=float, default=0, help="Train for this many minutes instead of a fixed step count; 0 disables timed mode.")
    parser.add_argument("--save-every", type=int, default=100, help="Save a checkpoint every N steps.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", default="checkpoints/kylo.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps < 1 or args.minutes < 0 or args.save_every < 1:
        raise SystemExit("--steps and --save-every must be positive; --minutes cannot be negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(
            f"Training file not found: {data_path}. Add your permitted Persian/English text first."
        )

    text = data_path.read_text(encoding="utf-8")
    if len(text) < args.block_size + 2:
        raise SystemExit(f"Training text is too short; provide at least {args.block_size + 2} characters.")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | characters: {len(chars)} | dataset chars: {len(encoded)}")
    if device == "cpu" and args.minutes >= 60:
        print("Notice: a 60-minute CPU run may be slow and will not guarantee useful language ability.")

    model = TinyGPT(vocab_size=len(chars), block_size=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_path = Path(args.output)
    start_time = time.monotonic()
    deadline = start_time + args.minutes * 60 if args.minutes else None
    step = 0
    model.train()

    while True:
        if deadline is not None and step > 0 and time.monotonic() >= deadline:
            break
        if deadline is None and step >= args.steps:
            break

        starts = torch.randint(0, len(encoded) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([encoded[s : s + args.block_size] for s in starts]).to(device)
        y = torch.stack([encoded[s + 1 : s + args.block_size + 1] for s in starts]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step == 1 or step % 25 == 0:
            elapsed = time.monotonic() - start_time
            print(f"step {step} | loss {loss.item():.4f} | elapsed {elapsed / 60:.1f} min")

        if step % args.save_every == 0:
            save_checkpoint(output_path, model, chars, stoi, args.block_size, step, time.monotonic() - start_time)
            print(f"Checkpoint saved: {output_path}")

    save_checkpoint(output_path, model, chars, stoi, args.block_size, step, time.monotonic() - start_time)
    print(f"Training finished after {step} steps. Final checkpoint: {output_path}")


if __name__ == "__main__":
    main()
