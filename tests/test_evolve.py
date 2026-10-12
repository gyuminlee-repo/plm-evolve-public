import numpy as np
import pandas as pd
import pytest

from plm_evolve.cli import build_parser, main
from plm_evolve.evolve import run_round


def _make_embeddings():
    # 6 variants, 4 dims. Make activity roughly linear in dim 0 so the forest
    # can learn a monotone-ish ranking.
    index = ["WT", "M1A", "K2R", "V3D", "A4G", "A5T"]
    data = {
        "WT": [0.0, 0.0, 0.0, 0.0],
        "M1A": [1.0, 0.1, 0.0, 0.0],
        "K2R": [2.0, 0.0, 0.1, 0.0],
        "V3D": [3.0, 0.0, 0.0, 0.1],
        "A4G": [4.0, 0.1, 0.1, 0.0],
        "A5T": [5.0, 0.0, 0.1, 0.1],
    }
    return pd.DataFrame.from_dict(data, orient="index", columns=["d0", "d1", "d2", "d3"]).reindex(index)


def _write_wt(tmp_path):
    p = tmp_path / "wt.fasta"
    p.write_text(">WT\nMKVAA\n")
    return p


def _write_round(tmp_path):
    # Measure WT + two variants. activity tracks dim 0.
    p = tmp_path / "r1.csv"
    pd.DataFrame(
        {"Variant": ["WT", "M1A", "A5T"], "activity": [0.0, 1.0, 5.0]}
    ).to_csv(p, index=False)
    return p


def test_run_round_end_to_end(tmp_path):
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    rf = _write_round(tmp_path)

    result = run_round(emb, [rf], wt, top_n=2)

    # Trained on the 3 measured variants.
    assert result.n_train == 3

    # Ranked table includes ALL candidates (measured + unmeasured).
    assert len(result.ranked) == len(emb)
    assert set(result.ranked["variant"]) == set(emb.index)
    assert result.ranked["measured"].sum() == 3

    # Ranked is sorted descending by predicted.
    preds = result.ranked["predicted"].to_numpy()
    assert np.all(np.diff(preds) <= 1e-12)

    # next_round excludes already-measured variants and WT.
    measured_set = {"WT", "M1A", "A5T"}
    assert len(result.next_round) == 2
    assert measured_set.isdisjoint(set(result.next_round["variant"]))


def test_run_round_injects_wt_into_training_by_default(tmp_path):
    """EVOLVEpro anchors WT in every round's training set; run_round does the
    same by default. A round file WITHOUT a WT row still trains on WT."""
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    p = tmp_path / "r_nowt.csv"
    pd.DataFrame({"Variant": ["M1A", "A5T"], "activity": [1.0, 5.0]}).to_csv(p, index=False)

    # Default: the 2 measured mutants + the auto-injected WT = 3 training rows.
    res = run_round(emb, [p], wt, top_n=2)
    assert res.n_train == 3
    # WT is trained on but never proposed.
    assert "WT" not in set(res.next_round["variant"])

    # Opt-out trains on the measured mutants only.
    res_no = run_round(emb, [p], wt, top_n=2, include_wt_in_training=False)
    assert res_no.n_train == 2

    # No double-count when WT is already a measured variant.
    p2 = tmp_path / "r_wt.csv"
    pd.DataFrame({"Variant": ["WT", "M1A"], "activity": [0.0, 1.0]}).to_csv(p2, index=False)
    assert run_round(emb, [p2], wt, top_n=2).n_train == 2


def test_run_round_writes_outputs(tmp_path):
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    rf = _write_round(tmp_path)
    out = tmp_path / "out"

    run_round(emb, [rf], wt, top_n=2, out_dir=out)
    assert (out / "ranked.csv").exists()
    assert (out / "next_round.csv").exists()


def test_run_round_no_match_raises(tmp_path):
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    # Variant V3D position 3 must be 'V' in WT MKVAA -> valid token, but use a
    # token NOT in the embedding index to force no training overlap.
    p = tmp_path / "r1.csv"
    pd.DataFrame({"Variant": ["V3D"], "activity": [9.0]}).to_csv(p, index=False)
    # Drop V3D from embeddings so nothing matches.
    emb2 = emb.drop("V3D")
    with pytest.raises(ValueError):
        run_round(emb2, [p], wt, top_n=2)


# --------------------------------------------------------------------------
# OPT-IN acquisition (research strategies wired into production)
# --------------------------------------------------------------------------
def test_acquisition_greedy_is_byte_identical_to_default(tmp_path):
    """acquisition='greedy' must reproduce the default path EXACTLY."""
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    rf = _write_round(tmp_path)

    default = run_round(emb, [rf], wt, top_n=3)
    explicit = run_round(emb, [rf], wt, top_n=3, acquisition="greedy")

    pd.testing.assert_frame_equal(default.ranked, explicit.ranked)
    pd.testing.assert_frame_equal(default.next_round, explicit.next_round)
    assert default.n_train == explicit.n_train


@pytest.mark.parametrize("acquisition", ["ucb", "coverage"])
def test_acquisition_nongreedy_valid_next_round(tmp_path, acquisition):
    """ucb/coverage select a valid next_round (count, unmeasured, no WT/dupes)."""
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    rf = _write_round(tmp_path)

    result = run_round(emb, [rf], wt, top_n=2, acquisition=acquisition, seed=0)

    nxt = result.next_round
    # Schema preserved (no extra columns beyond the ranked schema).
    assert list(nxt.columns) == list(result.ranked.columns)
    # Correct count.
    assert len(nxt) == 2
    # All unmeasured, WT excluded, no dupes.
    measured_set = {"WT", "M1A", "A5T"}
    picked = nxt["variant"].tolist()
    assert measured_set.isdisjoint(set(picked))
    assert "WT" not in picked
    assert not nxt["measured"].any()
    assert len(set(picked)) == len(picked)


@pytest.mark.parametrize("acquisition", ["ucb", "coverage", "thompson"])
def test_acquisition_is_deterministic_for_fixed_seed(tmp_path, acquisition):
    """A fixed seed yields an identical next_round across calls."""
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    rf = _write_round(tmp_path)

    a = run_round(emb, [rf], wt, top_n=2, acquisition=acquisition, seed=7)
    b = run_round(emb, [rf], wt, top_n=2, acquisition=acquisition, seed=7)
    pd.testing.assert_frame_equal(a.next_round, b.next_round)


def test_acquisition_unknown_raises(tmp_path):
    """An unknown acquisition fails fast (no silent fallback)."""
    emb = _make_embeddings()
    wt = _write_wt(tmp_path)
    rf = _write_round(tmp_path)
    with pytest.raises(ValueError):
        run_round(emb, [rf], wt, top_n=2, acquisition="bogus")


# --------------------------------------------------------------------------
# CLI: new flags + propose-multi subcommand
# --------------------------------------------------------------------------
def test_parser_accepts_new_rank_flags_default_greedy():
    parser = build_parser()
    # Default rank invocation: acquisition stays greedy.
    args = parser.parse_args(
        ["rank", "--embeddings", "e.csv", "--wt", "w.fa", "--rounds", "r.csv"]
    )
    assert args.acquisition == "greedy"
    assert args.beta == 1.0
    assert args.penalty == 1.0
    assert args.seed == 0
    # Explicit non-greedy parses too.
    args2 = parser.parse_args(
        [
            "rank", "--embeddings", "e.csv", "--wt", "w.fa", "--rounds", "r.csv",
            "--acquisition", "coverage", "--beta", "2.0", "--penalty", "0.5",
            "--seed", "3",
        ]
    )
    assert args2.acquisition == "coverage"
    assert args2.beta == 2.0
    assert args2.penalty == 0.5
    assert args2.seed == 3


def test_parser_accepts_propose_multi_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "propose-multi", "--embeddings", "e.csv", "--wt", "w.fa",
            "--rounds", "r.csv", "--order", "2", "--top-singles", "3",
            "--top", "5", "--out", "p.csv",
        ]
    )
    assert args.command == "propose-multi"
    assert args.order == 2
    assert args.top_singles == 3
    assert args.top == 5
    assert args.func.__name__ == "_cmd_propose_multi"


def test_cli_propose_multi_writes_valid_proposals(tmp_path):
    """End-to-end propose-multi: valid ':'-joined distinct-position doubles,
    ranked by additive score (no model download; synthetic embeddings)."""
    emb = _make_embeddings()
    emb_path = tmp_path / "emb.csv"
    # from_csv expects an index column; to_csv with index writes it.
    emb.to_csv(emb_path)

    wt = _write_wt(tmp_path)
    rf = _write_round(tmp_path)
    out = tmp_path / "proposals.csv"

    rc = main(
        [
            "propose-multi",
            "--embeddings", str(emb_path),
            "--wt", str(wt),
            "--rounds", str(rf),
            "--order", "2",
            "--top", "4",
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()

    proposals = pd.read_csv(out)
    assert list(proposals.columns) == ["variant", "additive_predicted"]
    assert len(proposals) >= 1

    for token in proposals["variant"]:
        parts = str(token).split(":")
        assert len(parts) == 2  # order-2 multi.
        positions = []
        for part in parts:
            mut = part  # token form <WTaa><pos><newaa>
            # distinct positions: extract the numeric middle.
            pos = int("".join(ch for ch in mut if ch.isdigit()))
            positions.append(pos)
        assert len(set(positions)) == 2  # distinct positions.

    # Ranked by additive score, descending.
    scores = proposals["additive_predicted"].to_numpy()
    assert np.all(np.diff(scores) <= 1e-9)
