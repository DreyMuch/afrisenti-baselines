"""Score tweets with the pooled model.

uv run python -m afrisenti_baselines.predict "e dey sweet me die" "ọ dị njọ"
"""

import sys
from pathlib import Path

import joblib

MODEL = Path(__file__).resolve().parents[2] / "models" / "pooled_word_char.joblib"


def main(texts: list[str]) -> None:
    if not MODEL.exists():
        sys.exit("No model yet. Run: uv run python -m afrisenti_baselines.train")
    model = joblib.load(MODEL)
    for text, probs in zip(texts, model.predict_proba(texts)):
        label = model.classes_[probs.argmax()]
        detail = "  ".join(f"{c}={p:.2f}" for c, p in zip(model.classes_, probs))
        print(f"{label:9s} {detail}  | {text}")


if __name__ == "__main__":
    main(sys.argv[1:] or [line.strip() for line in sys.stdin if line.strip()])
