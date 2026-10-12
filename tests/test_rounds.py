import numpy as np
import pandas as pd
import pytest

from plm_evolve.rounds import (
    _add_activity_columns,
    build_training_table,
    load_round_data,
    validate_coverage,
)

# WT sequence chosen so that M1, K2, V3, A4, A5 match the tokens used below.
WT_SEQ = "MKVAA"


def _write_wt(tmp_path):
    p = tmp_path / "wt.fasta"
    p.write_text(">WT\n" + WT_SEQ + "\n")
    return p


def _write_round_csv(tmp_path, name, rows):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_round_data_valid(tmp_path):
    wt = _write_wt(tmp_path)
    rf = _write_round_csv(
        tmp_path,
        "r1.csv",
        {"Variant": ["M1A", "K2R", "WT"], "activity": [2.0, 0.5, 1.0]},
    )
    df = load_round_data([rf], wt)
    assert list(df["variant"]) == ["M1A", "K2R", "WT"]
    assert list(df["activity"]) == [2.0, 0.5, 1.0]


def test_load_round_data_wt_residue_mismatch_raises(tmp_path):
    wt = _write_wt(tmp_path)
    # Token claims position 1 is 'K' but WT has 'M'.
    rf = _write_round_csv(
        tmp_path, "r1.csv", {"Variant": ["K1A"], "activity": [2.0]}
    )
    with pytest.raises(ValueError) as exc:
        load_round_data([rf], wt)
    assert "K1A" in str(exc.value)


def test_load_round_data_duplicate_raises(tmp_path):
    wt = _write_wt(tmp_path)
    rf = _write_round_csv(
        tmp_path,
        "r1.csv",
        {"Variant": ["M1A", "M1A"], "activity": [2.0, 3.0]},
    )
    with pytest.raises(ValueError) as exc:
        load_round_data([rf], wt)
    assert "Duplicate" in str(exc.value)


def test_activity_binary_and_scaled_correct():
    df = pd.DataFrame({"variant": ["a", "b", "c"], "activity": [0.0, 1.0, 3.0]})
    out = _add_activity_columns(df)
    assert list(out["activity_binary"]) == [0, 1, 1]
    # min=0 max=3 -> scaled 0, 1/3, 1
    np.testing.assert_allclose(out["activity_scaled"], [0.0, 1.0 / 3.0, 1.0])


def test_activity_scaled_constant_no_nan():
    df = pd.DataFrame({"variant": ["a", "b"], "activity": [5.0, 5.0]})
    out = _add_activity_columns(df)
    assert not out["activity_scaled"].isna().any()
    assert (out["activity_scaled"] == 0.0).all()


def test_coverage_warning_fires_on_unmatched(capsys):
    measured = pd.DataFrame({"variant": ["M1A", "K2R"], "activity": [1.0, 2.0]})
    usable = validate_coverage(measured, ["M1A"])  # K2R missing
    captured = capsys.readouterr()
    assert "K2R" in captured.err
    assert "WARNING" in captured.err
    assert list(usable["variant"]) == ["M1A"]


def test_no_coverage_warning_when_all_present(capsys):
    measured = pd.DataFrame({"variant": ["M1A", "K2R"], "activity": [1.0, 2.0]})
    validate_coverage(measured, ["M1A", "K2R", "V3D"])
    captured = capsys.readouterr()
    assert captured.err == ""


def test_build_training_table_alignment_and_measured_flag():
    measured = pd.DataFrame({"variant": ["M1A", "K2R"], "activity": [2.0, 0.0]})
    index = ["WT", "M1A", "K2R", "V3D"]
    table = build_training_table(measured, index)
    assert list(table["variant"]) == index
    assert list(table["measured"]) == [False, True, True, False]
    # Unmeasured rows have NaN activity.
    assert np.isnan(table.loc[table["variant"] == "WT", "activity"].iloc[0])
