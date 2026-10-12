# Training data

Place your permitted plain-text training corpus in `data/train.txt` (this file is intentionally not committed).

- Use text you wrote yourself or datasets whose licenses permit this use.
- For Persian + English, include both languages and the punctuation/characters you expect prompts to contain.
- Remove private information and secrets.
- Start with a small sample to verify the training pipeline before attempting larger runs.

Then run from the repository root:

```bash
python train.py --data data/train.txt --steps 1000
```

The first model is a tiny character-level educational experiment; it is not a production chat model.