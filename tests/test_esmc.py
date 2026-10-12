"""ESM-C backend tests.

The ESM-C model and its forward pass are MONKEYPATCHED: no weights are
downloaded and no network is touched. ``ESMCExtractor._forward`` is documented
to return the SAME ``(hidden_state, attention_mask, special_tokens_mask)`` numpy
tuple convention as ESM-2 (BOS at token index 0, EOS after the last residue,
right-padding), so a small local fake suffices to drive every pooling mode.
"""

import numpy as np
import pytest

from plm_evolve.embeddings import (
    ESM2Extractor,
    ESMCExtractor,
    make_extractor,
    masked_mean_pool,
)


def _make_fake_forward(seq_to_token_hidden, dim):
    """Fake ESM-C ``_forward`` emitting [BOS] + residues + [EOS] (+ right padding).

    Residue token i (1-based) gets ``seq_to_token_hidden(seq, i)`` (length-dim);
    BOS/EOS carry sentinel values that must never leak into pooling. This mirrors
    the ESM-2 _forward tuple convention so _PoolingMixin transfers unchanged.
    """

    def fake_forward(sequences):
        residue_lens = [len(s) for s in sequences]
        max_res = max(residue_lens)
        seq_len = max_res + 2  # BOS + residues + EOS
        batch = len(sequences)
        hidden = np.zeros((batch, seq_len, dim))
        attention = np.zeros((batch, seq_len))
        special = np.zeros((batch, seq_len))
        for b, seq in enumerate(sequences):
            n = len(seq)
            attention[b, : n + 2] = 1
            special[b, 0] = 1  # BOS
            special[b, n + 1] = 1  # EOS
            hidden[b, 0] = 555.0  # BOS sentinel
            hidden[b, n + 1] = -555.0  # EOS sentinel
            for i in range(1, n + 1):
                hidden[b, i] = np.asarray(
                    seq_to_token_hidden(seq, i), dtype=np.float64
                )
        return hidden, attention, special

    return fake_forward


def _patch(monkeypatch, extractor, fake_forward):
    monkeypatch.setattr(extractor, "_load", lambda: None)
    monkeypatch.setattr(extractor, "_forward", fake_forward)


def test_esmc_mean_is_masked_residue_mean(tmp_path, monkeypatch):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A\nAKV\n>K2A\nMAV\n>WT\nMKV\n")

    def hidden_fn(seq, i):
        base = float(ord(seq[i - 1]))
        return [base + i, base - i]

    fake_forward = _make_fake_forward(hidden_fn, dim=2)
    extractor = ESMCExtractor(batch_size=2)  # default pooling='mean'
    _patch(monkeypatch, extractor, fake_forward)
    df = extractor.embed_fasta(fasta)

    for rid, seq in [("M1A", "AKV"), ("K2A", "MAV"), ("WT", "MKV")]:
        h, a, s = fake_forward([seq])
        expected = masked_mean_pool(h, a, s)[0]
        np.testing.assert_allclose(df.loc[rid].to_numpy(), expected)


def test_esmc_mutated_token_picks_correct_index(tmp_path, monkeypatch):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A\nAKVLI\n>K2A\nMAVLI\n>V3A\nMKALI\n>WT\nMKVLI\n")

    # Distinct per-token vectors keyed by token index so the assertion is sharp.
    def hidden_fn(seq, i):
        return [100.0 * i, 100.0 * i + 1.0]

    fake_forward = _make_fake_forward(hidden_fn, dim=2)
    extractor = ESMCExtractor(batch_size=2, pooling="mutated_token")
    _patch(monkeypatch, extractor, fake_forward)
    df = extractor.embed_fasta(fasta)

    # token index == 1-based position (BOS at 0 -> residue 1 at index 1).
    np.testing.assert_allclose(df.loc["M1A"].to_numpy(), [100.0, 101.0])
    np.testing.assert_allclose(df.loc["K2A"].to_numpy(), [200.0, 201.0])
    np.testing.assert_allclose(df.loc["V3A"].to_numpy(), [300.0, 301.0])
    # WT has no mutated position -> zero vector.
    np.testing.assert_allclose(df.loc["WT"].to_numpy(), np.zeros(2))


def test_esmc_delta_subtracts_wt_and_zeros_wt_row(tmp_path, monkeypatch):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A\nAKV\n>K2A\nMAV\n>WT\nMKV\n")

    def hidden_fn(seq, i):
        base = float(ord(seq[i - 1]))
        return [base + i, base - i]

    fake_forward = _make_fake_forward(hidden_fn, dim=2)
    # WT lands in the SECOND batch (batch_size=2) to exercise cross-batch WT.
    extractor = ESMCExtractor(batch_size=2, pooling="delta")
    _patch(monkeypatch, extractor, fake_forward)
    df = extractor.embed_fasta(fasta)

    def mean_of(seq):
        h, a, s = fake_forward([seq])
        return masked_mean_pool(h, a, s)[0]

    wt_mean = mean_of("MKV")
    np.testing.assert_allclose(df.loc["M1A"].to_numpy(), mean_of("AKV") - wt_mean)
    np.testing.assert_allclose(df.loc["K2A"].to_numpy(), mean_of("MAV") - wt_mean)
    # WT's own delta row is exactly zero.
    np.testing.assert_allclose(df.loc["WT"].to_numpy(), np.zeros(2))


def test_esmc_mutated_token_multi_mutant_raises(tmp_path, monkeypatch):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">M1A_K2R\nAKVLI\n")  # multi-mutant id

    fake_forward = _make_fake_forward(lambda seq, i: [float(i), float(i)], dim=2)
    extractor = ESMCExtractor(batch_size=8, pooling="mutated_token")
    _patch(monkeypatch, extractor, fake_forward)
    with pytest.raises(ValueError, match="multi-mutant"):
        extractor.embed_fasta(fasta)


def test_esmc_forward_assembles_esm2_tuple_from_encode_one(monkeypatch):
    """_forward (the one path tests normally bypass) must right-pad _encode_one
    outputs into the ESM-2 (hidden, attention, special) tuple convention."""
    extractor = ESMCExtractor(batch_size=8)
    monkeypatch.setattr(extractor, "_load", lambda: None)

    # Fake _encode_one returns (Li+2, width): BOS row, Li residue rows, EOS row.
    width = 3

    def fake_encode_one(seq):
        n = len(seq)
        emb = np.zeros((n + 2, width))
        emb[0] = 9.0  # BOS sentinel
        emb[n + 1] = -9.0  # EOS sentinel
        for i in range(1, n + 1):
            emb[i] = float(i)
        return emb

    monkeypatch.setattr(extractor, "_encode_one", fake_encode_one)

    # Two sequences of different length -> right-padding to batch max.
    hidden, attention, special = extractor._forward(["AKV", "MK"])
    # Batch max token length = max(3+2, 2+2) = 5.
    assert hidden.shape == (2, 5, width)
    # Row 0 (len 3): BOS, r1..r3, EOS -> 5 real tokens, no pad.
    np.testing.assert_array_equal(attention[0], [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(special[0], [1, 0, 0, 0, 1])
    # Row 1 (len 2): BOS, r1, r2, EOS, PAD -> last slot padded.
    np.testing.assert_array_equal(attention[1], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(special[1], [1, 0, 0, 1, 0])
    # EOS sits at tlen-1 (index 3 for the short seq), padding row stays zero.
    np.testing.assert_allclose(hidden[1, 3], [-9.0, -9.0, -9.0])
    np.testing.assert_allclose(hidden[1, 4], [0.0, 0.0, 0.0])


def test_make_extractor_dispatches_by_model_tag():
    esmc_300 = make_extractor("esmc_300m", pooling="mean")
    assert isinstance(esmc_300, ESMCExtractor)
    assert esmc_300.model_name == "esmc_300m"

    esmc_600 = make_extractor("esmc_600m")
    assert isinstance(esmc_600, ESMCExtractor)

    esm2 = make_extractor("facebook/esm2_t12_35M_UR50D", pooling="mutated_token")
    assert isinstance(esm2, ESM2Extractor)
    assert esm2.pooling == "mutated_token"

    # Arbitrary/unknown HF id falls back to ESM-2 (not ESM-C).
    assert isinstance(make_extractor("some/other-model"), ESM2Extractor)
