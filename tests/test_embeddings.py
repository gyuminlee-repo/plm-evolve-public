import numpy as np
import pandas as pd
import pytest

from plm_evolve.embeddings import (
    ESM2Extractor,
    masked_mean_pool,
    mutated_token_pool,
    residue_count_from_masks,
)


def test_masked_mean_pool_excludes_special_tokens():
    # One sequence, seq_len=5, dim=2. Tokens: [CLS, r1, r2, r3, EOS].
    # CLS and EOS hold sentinel values that MUST NOT enter the mean.
    hidden = np.array(
        [
            [
                [999.0, 999.0],  # CLS sentinel
                [1.0, 10.0],     # residue 1
                [2.0, 20.0],     # residue 2
                [3.0, 30.0],     # residue 3
                [-999.0, -999.0],  # EOS sentinel
            ]
        ]
    )
    attention_mask = np.array([[1, 1, 1, 1, 1]])
    special_tokens_mask = np.array([[1, 0, 0, 0, 1]])

    pooled = masked_mean_pool(hidden, attention_mask, special_tokens_mask)
    # Mean over residues 1..3 only.
    np.testing.assert_allclose(pooled, [[2.0, 20.0]])


def test_masked_mean_pool_excludes_padding():
    # seq_len=5: CLS, r1, r2, EOS, PAD.
    hidden = np.array(
        [
            [
                [100.0],
                [4.0],
                [6.0],
                [100.0],
                [123456.0],  # padding sentinel
            ]
        ]
    )
    attention_mask = np.array([[1, 1, 1, 1, 0]])
    special_tokens_mask = np.array([[1, 0, 0, 1, 0]])
    pooled = masked_mean_pool(hidden, attention_mask, special_tokens_mask)
    np.testing.assert_allclose(pooled, [[5.0]])


def test_masked_mean_pool_zero_residues_raises():
    hidden = np.zeros((1, 2, 3))
    attention_mask = np.array([[1, 1]])
    special_tokens_mask = np.array([[1, 1]])  # all special -> no residues
    with pytest.raises(ValueError):
        masked_mean_pool(hidden, attention_mask, special_tokens_mask)


def test_embed_fasta_with_monkeypatched_forward(tmp_path, monkeypatch):
    # Build a tiny FASTA with two records.
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">v1\nMKV\n>v2\nMKA\n")

    extractor = ESM2Extractor(batch_size=8)

    # Skip real model loading.
    monkeypatch.setattr(extractor, "_load", lambda: None)

    def fake_forward(sequences):
        # Emulate tokenizer output: [CLS] + residues + [EOS], dim=2.
        batch, seq_len, dim = len(sequences), 5, 2
        hidden = np.zeros((batch, seq_len, dim))
        attention = np.ones((batch, seq_len))
        special = np.zeros((batch, seq_len))
        for b, seq in enumerate(sequences):
            special[b, 0] = 1  # CLS
            special[b, 4] = 1  # EOS
            hidden[b, 0] = 999.0  # CLS sentinel
            hidden[b, 4] = -999.0  # EOS sentinel
            # residues 1..3 = ordinal of first char so v1/v2 differ.
            base = float(ord(seq[0]))
            hidden[b, 1:4] = np.array([[base, base + 1]] * 3)
        return hidden, attention, special

    monkeypatch.setattr(extractor, "_forward", fake_forward)

    df = extractor.embed_fasta(fasta)
    assert list(df.index) == ["v1", "v2"]
    assert df.shape == (2, 2)
    # Sentinels excluded -> pooled equals the residue value (constant over r1..3).
    base = float(ord("M"))
    np.testing.assert_allclose(df.loc["v1"].to_numpy(), [base, base + 1])


# --- helper tests for the new pure pooling primitives --------------------


def test_residue_count_from_masks_excludes_special_and_padding():
    # Row 0: CLS, r1, r2, r3, EOS  -> 3 residues.
    # Row 1: CLS, r1, EOS, PAD, PAD -> 1 residue.
    attention = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    special = np.array([[1, 0, 0, 0, 1], [1, 0, 1, 0, 0]])
    counts = residue_count_from_masks(attention, special)
    np.testing.assert_array_equal(counts, [3, 1])


def test_mutated_token_pool_picks_token_index():
    # (seq_len=4, dim=2): index 0 CLS, 1 r1, 2 r2, 3 EOS.
    hidden = np.array([[0.0, 0.0], [10.0, 11.0], [20.0, 21.0], [-1.0, -1.0]])
    np.testing.assert_allclose(mutated_token_pool(hidden, 1), [10.0, 11.0])
    np.testing.assert_allclose(mutated_token_pool(hidden, 2), [20.0, 21.0])


# --- shared fake tokenizer/forward for pooling-mode tests ----------------


def _make_fake_forward(seq_to_token_hidden, dim, seqs_len_to_pad=None):
    """Build a fake ``_forward`` emulating [CLS] + residues + [EOS] (+ padding).

    For each sequence, the hidden state at residue token i (1-based) is set by
    ``seq_to_token_hidden(seq, i)`` -> length-``dim`` vector. CLS/EOS carry
    sentinel values that must never leak into mean or mutated-token pooling.
    Right-padding is applied to the batch max length (transformers default).
    """

    def fake_forward(sequences):
        residue_lens = [len(s) for s in sequences]
        max_res = max(residue_lens)
        seq_len = max_res + 2  # CLS + residues + EOS
        batch = len(sequences)
        hidden = np.zeros((batch, seq_len, dim))
        attention = np.zeros((batch, seq_len))
        special = np.zeros((batch, seq_len))
        for b, seq in enumerate(sequences):
            n = len(seq)
            # CLS at 0, residues 1..n, EOS at n+1; pad after.
            attention[b, : n + 2] = 1
            special[b, 0] = 1  # CLS
            special[b, n + 1] = 1  # EOS
            hidden[b, 0] = 777.0  # CLS sentinel
            hidden[b, n + 1] = -777.0  # EOS sentinel
            for i in range(1, n + 1):
                hidden[b, i] = np.asarray(
                    seq_to_token_hidden(seq, i), dtype=np.float64
                )
        return hidden, attention, special

    return fake_forward


def _patch_extractor(monkeypatch, extractor, fake_forward):
    monkeypatch.setattr(extractor, "_load", lambda: None)
    monkeypatch.setattr(extractor, "_forward", fake_forward)


def test_pooling_mean_is_byte_identical(tmp_path, monkeypatch):
    """Default 'mean' must equal masked_mean_pool on the same forward output."""
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A\nAKV\n>K2A\nMAV\n>WT\nMKV\n")

    # Residue token i -> deterministic non-constant vector so the mean is a
    # genuine average over residues (not a degenerate constant).
    def hidden_fn(seq, i):
        base = float(ord(seq[i - 1]))
        return [base + i, base - i]

    fake_forward = _make_fake_forward(hidden_fn, dim=2)

    extractor = ESM2Extractor(batch_size=2)  # default pooling='mean'
    _patch_extractor(monkeypatch, extractor, fake_forward)
    df = extractor.embed_fasta(fasta)

    # Recompute the expected mean per record directly via masked_mean_pool.
    for rid, seq in [("M1A", "AKV"), ("K2A", "MAV"), ("WT", "MKV")]:
        h, a, s = fake_forward([seq])
        expected = masked_mean_pool(h, a, s)[0]
        np.testing.assert_allclose(df.loc[rid].to_numpy(), expected)


def test_pooling_delta_subtracts_wt_and_zeros_wt_row(tmp_path, monkeypatch):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A\nAKV\n>K2A\nMAV\n>WT\nMKV\n")

    def hidden_fn(seq, i):
        base = float(ord(seq[i - 1]))
        return [base + i, base - i]

    fake_forward = _make_fake_forward(hidden_fn, dim=2)

    # WT lands in the SECOND batch (batch_size=2) to exercise cross-batch WT.
    extractor = ESM2Extractor(batch_size=2, pooling="delta")
    _patch_extractor(monkeypatch, extractor, fake_forward)
    df = extractor.embed_fasta(fasta)

    def mean_of(seq):
        h, a, s = fake_forward([seq])
        return masked_mean_pool(h, a, s)[0]

    wt_mean = mean_of("MKV")
    np.testing.assert_allclose(
        df.loc["M1A"].to_numpy(), mean_of("AKV") - wt_mean
    )
    np.testing.assert_allclose(
        df.loc["K2A"].to_numpy(), mean_of("MAV") - wt_mean
    )
    # WT's own delta row is exactly zero.
    np.testing.assert_allclose(df.loc["WT"].to_numpy(), np.zeros(2))


def test_pooling_delta_missing_wt_raises(tmp_path, monkeypatch):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A\nAKV\n>K2A\nMAV\n")  # no WT record

    fake_forward = _make_fake_forward(lambda seq, i: [1.0, 2.0], dim=2)
    extractor = ESM2Extractor(batch_size=8, pooling="delta")
    _patch_extractor(monkeypatch, extractor, fake_forward)
    with pytest.raises(ValueError, match="requires a record with id 'WT'"):
        extractor.embed_fasta(fasta)


def test_pooling_mutated_token_picks_correct_index(tmp_path, monkeypatch):
    # Variant ids encode the mutated 1-based position; assert the picked token
    # hidden state == that position's token (index == position, CLS at 0).
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A\nAKVLI\n>K2A\nMAVLI\n>V3A\nMKALI\n>WT\nMKVLI\n")

    # Distinct per-token vectors keyed by token index so the assertion is sharp.
    def hidden_fn(seq, i):
        return [100.0 * i, 100.0 * i + 1.0]

    fake_forward = _make_fake_forward(hidden_fn, dim=2)
    extractor = ESM2Extractor(batch_size=2, pooling="mutated_token")
    _patch_extractor(monkeypatch, extractor, fake_forward)
    df = extractor.embed_fasta(fasta)

    # token index == 1-based position.
    np.testing.assert_allclose(df.loc["M1A"].to_numpy(), [100.0, 101.0])
    np.testing.assert_allclose(df.loc["K2A"].to_numpy(), [200.0, 201.0])
    np.testing.assert_allclose(df.loc["V3A"].to_numpy(), [300.0, 301.0])
    # WT has no mutated position -> zero vector.
    np.testing.assert_allclose(df.loc["WT"].to_numpy(), np.zeros(2))


def test_pooling_mutated_token_multi_mutant_raises(tmp_path, monkeypatch):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A_K2R\nAKVLI\n")  # multi-mutant id

    fake_forward = _make_fake_forward(lambda seq, i: [float(i), float(i)], dim=2)
    extractor = ESM2Extractor(batch_size=8, pooling="mutated_token")
    _patch_extractor(monkeypatch, extractor, fake_forward)
    with pytest.raises(ValueError, match="multi-mutant"):
        extractor.embed_fasta(fasta)


def test_pooling_mutated_token_position_past_window_raises(tmp_path, monkeypatch):
    # Sequence has 3 residues but the id claims position 50 -> out of window.
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M50A\nMKV\n")

    fake_forward = _make_fake_forward(lambda seq, i: [float(i), float(i)], dim=2)
    extractor = ESM2Extractor(batch_size=8, pooling="mutated_token")
    _patch_extractor(monkeypatch, extractor, fake_forward)
    with pytest.raises(ValueError, match="outside the post-truncation residue"):
        extractor.embed_fasta(fasta)


def test_unknown_pooling_mode_raises():
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        ESM2Extractor(pooling="median")
