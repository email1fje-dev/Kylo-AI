"""Minimal API and web UI for a trained Kylo checkpoint."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import TinyGPT

app = FastAPI(title="Kylo AI", version="1.0.0")
ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("KYLO_CHECKPOINT", str(ROOT / "checkpoints" / "kylo.pt")))
WEB_PAGE = ROOT / "static" / "index.html"
model = None
itos = None
stoi = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.get("/")
def home():
    if not WEB_PAGE.is_file():
        raise HTTPException(status_code=404, detail="Kylo web UI is missing.")
    return FileResponse(WEB_PAGE)


@app.on_event("startup")
def load_checkpoint():
    global model, itos, stoi
    if not CHECKPOINT.is_file():
        return
    try:
        checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        model = TinyGPT(**cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        itos = checkpoint["itos"]
        stoi = checkpoint["stoi"]
    except Exception as exc:
        raise RuntimeError(f"Could not load Kylo checkpoint: {exc}") from exc


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=40, ge=1, le=120)
    temperature: float = Field(default=0.8, gt=0, le=2)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model": "Kylo 1.0"}


@app.post("/generate")
def generate(request: GenerateRequest):
    if model is None or stoi is None or itos is None:
        raise HTTPException(
            status_code=503,
            detail="Kylo has no trained checkpoint yet. Train a model and provide checkpoints/kylo.pt.",
        )
    unknown = [char for char in request.prompt if char not in stoi]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail="Prompt contains characters missing from Kylo's training vocabulary. Add more diverse training text and retrain.",
        )
    idx = torch.tensor([[stoi[ch] for ch in request.prompt]], dtype=torch.long, device=device)
    output = model.generate(idx, request.max_new_tokens, request.temperature)[0].tolist()
    return {"model": "Kylo 1.0", "text": "".join(itos[i] for i in output)}
