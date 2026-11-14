"""Tests de la partition par identité.

Règle n° 1 du protocole : aucune personne du test n'apparaît à l'entraînement
ni en validation. Ces tests ne doivent JAMAIS être désactivés ou ignorés.

Ils ne lisent que les fichiers versionnés de splits/ : aucune donnée Drive
n'est nécessaire.
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from facegeom.data.splits import IdentityOverlapError, check_disjoint, filter_min_images
from facegeom.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "data.yaml")["splits"]
SPLIT_DIR = ROOT / CFG["output_dir"]
NAMES = list(CFG["names"].values())


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(SPLIT_DIR / f"{name}.csv") for name in NAMES}


# --- Vérifications sur les vrais fichiers ---------------------------------


def test_all_split_files_present(splits):
    assert set(splits) == {"train", "val", "test"}
    for name, df in splits.items():
        assert list(df.columns) == ["image", "identity"], name
        assert len(df) > 0, name


@pytest.mark.parametrize("a, b", list(combinations(["train", "val", "test"], 2)))
def test_no_identity_in_two_splits(splits, a, b):
    shared = set(splits[a].identity) & set(splits[b].identity)
    assert not shared, f"{len(shared)} identités à la fois dans {a} et {b} : {sorted(shared)[:10]}"


@pytest.mark.parametrize("a, b", list(combinations(["train", "val", "test"], 2)))
def test_no_image_in_two_splits(splits, a, b):
    shared = set(splits[a].image) & set(splits[b].image)
    assert not shared, f"{len(shared)} images à la fois dans {a} et {b} : {sorted(shared)[:10]}"


def test_no_duplicate_image_within_split(splits):
    for name, df in splits.items():
        assert not df.image.duplicated().any(), name


def test_min_images_per_identity(splits):
    k = CFG["min_images_per_identity"]
    for name, df in splits.items():
        counts = df.groupby("identity").size()
        assert counts.min() >= k, f"{name} : une identité a {counts.min()} images (< {k})"


def test_counts_match_summary(splits):
    summary = json.loads((SPLIT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["rule"]["min_images_per_identity"] == CFG["min_images_per_identity"]
    for name, df in splits.items():
        expected = summary["after_filter"][name]
        assert df.identity.nunique() == expected["identities"], name
        assert len(df) == expected["images"], name


# --- Le détecteur détecte vraiment (exemples construits à la main) --------


def _toy(rows):
    return pd.DataFrame(rows, columns=["image", "identity", "split"])


def test_check_disjoint_detects_overlap():
    df = _toy([("a.jpg", 1, "train"), ("b.jpg", 2, "train"), ("c.jpg", 1, "test")])
    with pytest.raises(IdentityOverlapError):
        check_disjoint(df)


def test_check_disjoint_accepts_disjoint():
    df = _toy([("a.jpg", 1, "train"), ("b.jpg", 1, "train"), ("c.jpg", 2, "test")])
    check_disjoint(df)  # ne doit rien lever


def test_filter_min_images_removes_small_identities():
    rows = [(f"{i}.jpg", 1, "train") for i in range(4)] + [("x.jpg", 2, "train")]
    kept = filter_min_images(_toy(rows), min_images=4)
    assert set(kept.identity) == {1}
    assert len(kept) == 4
