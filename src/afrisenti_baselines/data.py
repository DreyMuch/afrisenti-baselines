"""Download and load the AfriSenti SemEval-2023 Subtask A splits.

The data isn't stored in this repo. It is fetched from the official AfriSenti
repository the first time it's needed and cached under data/raw/.
"""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from . import LABELS, LANGUAGES

BASE_URL = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/SubtaskA"
SPLITS = ("train", "dev", "test")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def path_for(lang: str, split: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{lang}_{split}.tsv"


def download(lang: str, split: str, data_dir: Path = DATA_DIR) -> Path:
    if lang not in LANGUAGES or split not in SPLITS:
        raise ValueError(f"unknown language/split: {lang}/{split}")
    target = path_for(lang, split, data_dir)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{BASE_URL}/{split}/{lang}_{split}.tsv", timeout=60) as resp:
        target.write_bytes(resp.read())
    return target


def read_tsv(path: Path) -> pd.DataFrame:
    # quoting=3 (QUOTE_NONE): tweets contain stray quote characters
    df = pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False)
    df = df.rename(columns=str.lower)[["id", "tweet", "label"]]
    df["label"] = df["label"].str.strip().str.lower()
    bad = set(df["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"{path.name}: unexpected labels {bad}")
    return df


def load(lang: str, split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = read_tsv(download(lang, split, data_dir))
    df["lang"] = lang
    return df
