# Training data

A small original Persian-English seed corpus is included at `data/starter_corpus.txt`. It is useful for checking that the training pipeline works, but it is far too small to produce a capable general-purpose assistant.

For a quick test, run from the repository root:

```bash
python train.py --data data/starter_corpus.txt --steps 100
```

For a longer experiment, you can run:

```bash
python train.py --data data/starter_corpus.txt --minutes 60 --save-every 100
```

The timed command runs for up to 60 minutes and saves checkpoints periodically. It still needs to be started in a live Python environment; adding the command to the repository does not train the model by itself.

For better results, place a larger permitted plain-text corpus in `data/train.txt` (this personal dataset file is intentionally not committed).

- Use text you wrote yourself or datasets whose licenses permit this use.
- Include both Persian and English if you want both languages represented.
- Remove private information and secrets.
- Start with a small sample to verify the training pipeline before attempting larger runs.

The first model is a tiny character-level educational experiment; it is not a production chat model.