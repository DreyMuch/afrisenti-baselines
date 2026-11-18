from pathlib import Path

import pytest

from afrisenti_baselines import LABELS
from afrisenti_baselines.data import read_tsv
from afrisenti_baselines.models import MODELS, build
from afrisenti_baselines.text import normalize


def test_normalize_masks_users_and_urls():
    assert normalize("@Kofi check https://t.co/x NOW") == "@user check url now"


def test_normalize_squashes_repeats_but_keeps_emphasis():
    assert normalize("sweeeeet!!!!") == "sweet!!"


def test_normalize_composes_igbo_diacritics():
    decomposed = "ụ"  # u + combining dot below
    assert normalize(decomposed) == "ụ"  # precomposed ụ


def test_normalize_keeps_twi_letters():
    assert normalize("Wabɔdam ANAA") == "wabɔdam anaa"


def test_read_tsv_rejects_unknown_labels(tmp_path: Path):
    f = tmp_path / "x.tsv"
    f.write_text('ID\ttweet\tlabel\n1\tok "quoted\tmaybe\n')
    with pytest.raises(ValueError, match="unexpected labels"):
        read_tsv(f)


def test_read_tsv_tolerates_stray_quotes(tmp_path: Path):
    f = tmp_path / "x.tsv"
    f.write_text('ID\ttweet\tlabel\n1\the said "no\tNegative\n2\tfine\tneutral\n')
    df = read_tsv(f)
    assert list(df.label) == ["negative", "neutral"]


TOY = [
    "e sweet well well",
    "i love am",
    "dis one sweet",
    "na beta tin",
    "e bad die",
    "i hate am",
    "wahala too much",
    "nonsense tin",
    "di match dey 4pm",
    "i dey house",
    "na monday",
    "we go see",
] * 3
TOY_Y = (["positive"] * 4 + ["negative"] * 4 + ["neutral"] * 4) * 3


@pytest.mark.parametrize("name", MODELS)
def test_every_model_fits_and_predicts_known_labels(name):
    model = build(name).fit(TOY, TOY_Y)
    assert set(model.predict(["i love dis", "e bad"])) <= set(LABELS)


def test_text_model_learns_toy_data():
    model = build("word+char", C=10).fit(TOY, TOY_Y)
    assert model.score(TOY, TOY_Y) > 0.9
