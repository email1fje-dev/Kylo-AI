# Kylo AI — Kylo 1.0

Kylo is an experimental language-model project. The first milestone is a tiny decoder-only Transformer whose weights are initialized and trained from scratch.

## Status

- [x] Initial project structure
- [x] Minimal Transformer model
- [x] From-scratch training script
- [x] FastAPI service scaffold
- [ ] Train and evaluate a checkpoint
- [ ] Add accounts, chat history, and usage limits
- [ ] Build the website and connect it to the API

**Important:** Until a checkpoint is trained, Kylo is not a useful conversational AI. A randomly initialized model will not produce meaningful answers.

## Files

- `model.py` — tiny decoder-only Transformer
- `train.py` — trains new weights from a plain-text dataset
- `app.py` — API for health checks and text generation
- `requirements.txt` — Python dependencies
- `Procfile` — Railway start command

## Train

Use Python 3.11+ in an environment with PyTorch installed.

```bash
pip install -r requirements.txt
```

Create `data/train.txt` with quality Persian and English text you are allowed to use, then run:

```bash
python train.py --data data/train.txt --steps 1000
```

This prototype uses a character-level vocabulary and starts with randomly initialized weights. Training time and quality depend on the dataset and available hardware. Begin with a small dataset for testing.

## Run the API

After training creates `checkpoints/kylo.pt`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

- `GET /health` — health and checkpoint status
- `POST /generate` — generate text, with JSON such as `{"prompt":"Hello","max_new_tokens":40}`

## Railway

Railway can run the API using the included `Procfile`. Model training and hosting inference are separate resource needs; a free hosted service may not have enough CPU, RAM, or execution time for training. Verify current plan limits before deploying.

## Licensing and security

Review licenses for third-party code and datasets. Do not commit private data, API keys, passwords, training datasets, or model checkpoints to this repository.
