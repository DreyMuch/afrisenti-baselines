"""Train and evaluate every model on every language, then write reports/.

Protocol
- C is tuned on the dev split; test is touched once per final model.
- Headline metric is weighted F1, the official AfriSenti Subtask A metric.
  Macro F1 is reported too because neutral is often the minority class.
- "per-language" models see one language's training data.
  The "pooled" model is trained on all four languages at once and scored per language.

Run: uv run python -m afrisenti_baselines.train
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from . import LABELS, LANGUAGES
from .data import load
from .models import C_GRID, MODELS, build

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODEL_DIR = ROOT / "models"
SEED = 13


def scores(y_true, y_pred) -> dict:
    return {
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def tune(name, train, dev):
    """Pick C on dev. Returns (fitted pipeline, best C, dev weighted F1)."""
    grid = [1.0] if name == "majority" else C_GRID
    best = None
    for C in grid:
        model = build(name, C).fit(train.tweet, train.label)
        f1 = f1_score(dev.label, model.predict(dev.tweet), average="weighted")
        if best is None or f1 > best[2]:
            best = (model, C, f1)
    return best


def learning_curve(name, C, train, test, fractions=(0.1, 0.25, 0.5, 1.0)):
    rng = np.random.default_rng(SEED)
    out = []
    for frac in fractions:
        idx = rng.permutation(len(train))[: max(50, int(frac * len(train)))]
        sub = train.iloc[idx]
        model = build(name, C).fit(sub.tweet, sub.label)
        out.append({"n_train": len(sub), **scores(test.label, model.predict(test.tweet))})
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    data = {lang: {s: load(lang, s) for s in ("train", "dev", "test")} for lang in LANGUAGES}

    report = {"dataset": {}, "per_language": {}, "pooled": {}, "learning_curve": {}}
    for lang, d in data.items():
        report["dataset"][lang] = {
            s: {"n": len(df), **df.label.value_counts(normalize=True).round(3).to_dict()}
            for s, df in d.items()
        }

    # 1. one model per language
    best_per_lang = {}
    for lang, d in data.items():
        report["per_language"][lang] = {}
        for name in MODELS:
            model, C, dev_f1 = tune(name, d["train"], d["dev"])
            res = {
                "C": C,
                "dev_weighted_f1": round(dev_f1, 4),
                **scores(d["test"].label, model.predict(d["test"].tweet)),
            }
            report["per_language"][lang][name] = res
            print(
                f"{lang:4s} {name:10s} C={C:<4} dev={dev_f1:.3f} test wF1={res['weighted_f1']:.3f} mF1={res['macro_f1']:.3f}"
            )
            if name != "majority" and (lang not in best_per_lang or dev_f1 > best_per_lang[lang][2]):
                best_per_lang[lang] = (name, C, dev_f1, model)

    # 2. pooled multilingual model, best feature set chosen on pooled dev
    pool = lambda s: pd.concat([d[s] for d in data.values()], ignore_index=True)
    pooled, pooled_C, _pooled_dev = tune("word+char", pool("train"), pool("dev"))
    for lang, d in data.items():
        pred = pooled.predict(d["test"].tweet)
        report["pooled"][lang] = {"C": pooled_C, **scores(d["test"].label, pred)}
        print(f"{lang:4s} pooled     test wF1={report['pooled'][lang]['weighted_f1']:.3f}")
    joblib.dump(pooled, MODEL_DIR / "pooled_word_char.joblib")

    # 3. how much data does each language need?
    for lang, d in data.items():
        name, C, _, _ = best_per_lang[lang]
        report["learning_curve"][lang] = {
            "model": name,
            "points": learning_curve(name, C, d["train"], d["test"]),
        }

    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))
    plot_comparison(report)
    plot_learning_curves(report)
    plot_confusions(pooled, data)
    print(f"wrote {REPORTS / 'metrics.json'} and figures/")


def plot_comparison(report):
    langs = list(LANGUAGES)
    series = {name: [report["per_language"][lg][name]["weighted_f1"] for lg in langs] for name in MODELS}
    series["pooled word+char"] = [report["pooled"][lg]["weighted_f1"] for lg in langs]
    x = np.arange(len(langs))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#c4c4c4", "#9db8d9", "#5b8cc7", "#2a5a9e", "#e07b39"]
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=colors[i])
    ax.set_xticks(x, [LANGUAGES[lg] for lg in langs])
    ax.set_ylabel("weighted F1 (test)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_learning_curves(report):
    fig, ax = plt.subplots(figsize=(6, 4))
    for lang, lc in report["learning_curve"].items():
        pts = lc["points"]
        ax.plot(
            [p["n_train"] for p in pts], [p["weighted_f1"] for p in pts], marker="o", label=LANGUAGES[lang]
        )
    ax.set_xscale("log")
    ax.set_xlabel("training tweets (log scale)")
    ax.set_ylabel("weighted F1 (test)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_confusions(model, data):
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 3.6))
    for ax, (lang, d) in zip(axes, data.items()):
        ConfusionMatrixDisplay.from_predictions(
            d["test"].label,
            model.predict(d["test"].tweet),
            labels=LABELS,
            normalize="true",
            values_format=".2f",
            ax=ax,
            colorbar=False,
            cmap="Blues",
        )
        ax.set_title(LANGUAGES[lang])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if ax is axes[0] else "")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_pooled.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
