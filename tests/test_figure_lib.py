import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import figure_lib as fl  # noqa: E402


def test_config_constants_present():
    assert set(fl.DATASET_HASH) == {"kelsic", "stiffler", "giacomelli", "cas12f", "brenan"}
    assert fl.MEDIUM_PAIR == ("facebook/esm2_t30_150M_UR50D", "esmc_300m")
    assert fl.MATCHED_PAIR == ("facebook/esm2_t33_650M_UR50D", "esmc_600m")
    assert fl.ESM2_COLOR == "#A8C4E0" and fl.ESMC_COLOR == "#1F4E79"
    assert fl.PROTEIN_ORDER == ["kelsic", "stiffler", "giacomelli", "cas12f", "brenan"]


def test_held_out_spearman_informative_feature():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = X[:, 0]
    mean, se = fl.held_out_spearman_curve(X, y, train_sizes=[64], seeds=5)
    assert mean[64] > 0.6 and se[64] >= 0.0


def test_plate2_precision_recovers_ranking():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] * 5.0
    mean_count, se = fl.plate2_precision(X, y, plate_size=95, seeds=4)
    assert mean_count > 60


@pytest.mark.skipif(not (fl.CACHE_DIR).exists(), reason="DMS cache absent")
def test_kelsic_train95_matches_comparison_note():
    ids, y, X = fl.load_aligned("kelsic", list(fl.MEDIUM_PAIR), fl.DATASET_HASH["kelsic"])
    e2_mean, _ = fl.held_out_spearman_curve(X[fl.MEDIUM_PAIR[0]], y, [95], seeds=20)
    ec_mean, _ = fl.held_out_spearman_curve(X[fl.MEDIUM_PAIR[1]], y, [95], seeds=20)
    assert abs(e2_mean[95] - 0.337) < 0.02
    assert abs(ec_mean[95] - 0.572) < 0.02
