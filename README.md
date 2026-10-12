# Kylo AI — Kylo 1.0

Kylo is an experimental language-model project. Its first model is a small decoder-only Transformer initialized and trained from scratch.

## Current status

- [x] Small Transformer architecture and training script
- [x] FastAPI API with health check and generation endpoint
- [x] Minimal black-and-white chat website served at `/`
- [x] Timed training mode and periodic checkpoint saves
- [ ] Prepare a legally usable Persian/English training dataset
- [ ] Run training, evaluate, and verify a checkpoint
- [ ] Deploy the API and website

**Important:** The model is not trained yet. The repository does not contain a training dataset or a checkpoint. This prototype is character-level and is for learning, not a capable general-purpose assistant.

## Train

Use Python 3.11+ and install dependencies:

```bash
pip install -r requirements.txt
```

Create `data/train.txt` with Persian and English text you own or are licensed to use. Then train for at least 60 minutes, saving a checkpoint every 100 steps:

```bash
python train.py --data data/train.txt --minutes 60 --save-every 100
```

Or use a fixed number of steps:

```bash
python train.py --data data/train.txt --steps 1000
```

Training starts from random weights. A one-hour run is only a duration target; it does not guarantee meaningful language ability. Runtime depends on hardware, and CPU training may be slow. Checkpoints are written to `checkpoints/kylo.pt`.

## Run the site and API

After a checkpoint has been trained:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

- `GET /` — Kylo chat website
- `GET /health` — service status and whether a checkpoint loaded
- `POST /generate` — generate text, e.g. JSON: `{"prompt":"Hello","max_new_tokens":40}`

## Railway

The included `Procfile` starts the API/web service. Training and hosting are separate resource needs; a free hosted service may not have enough CPU, RAM, or runtime to train a model. The chat UI can be hosted, but generation will remain unavailable until a checkpoint is supplied.

## Data, licensing, and security

Use only datasets you have permission to use. Do not commit private data, API keys, passwords, datasets, or model checkpoints to this repository. Review third-party licenses before reuse.
