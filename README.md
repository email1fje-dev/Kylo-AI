# Kylo AI — Kylo 1.0

Kylo is an experimental language-model project. Its first model is a small decoder-only Transformer initialized and trained from scratch.

## Current status

- [x] Small Transformer architecture and training script
- [x] FastAPI API with health check and generation endpoint
- [x] Minimal black-and-white chat website served at `/`
- [ ] Prepare a legally usable Persian/English training dataset
- [ ] Train, evaluate, and save a checkpoint
- [ ] Deploy the API and website
- [ ] Add accounts, saved chats, and usage limits

**Important:** The web UI is connected to the API, but Kylo does not have a trained checkpoint yet. Until a checkpoint is trained, it cannot provide meaningful AI answers.

## Run locally

Use Python 3.11+ and install dependencies:

```bash
pip install -r requirements.txt
```

Create `data/train.txt` containing text you own or are licensed to use, then train:

```bash
python train.py --data data/train.txt --steps 1000
```

Training starts from random weights. This prototype is character-level and is intended for learning/testing, not as a capable general-purpose assistant. More steps alone do not guarantee quality; dataset quality and compute matter.

After training, start the site and API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## API

- `GET /` — Kylo chat website
- `GET /health` — service status and whether a checkpoint loaded
- `POST /generate` — generate text, e.g. JSON: `{"prompt":"Hello","max_new_tokens":40}`

## Railway

The included `Procfile` starts the API/web service. Training and hosting are separate resource needs; a free hosted service may not have enough CPU, RAM, or runtime to train a model. Deploying the current repository will host the UI/API, but generation will return a clear not-trained message until a checkpoint is available.

## Data, licensing, and security

Use only datasets you have permission to use. Do not commit private data, API keys, passwords, datasets, or model checkpoints to this repository. Review third-party licenses before reuse.
