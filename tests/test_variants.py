import pytest

from plm_evolve.variants import (
    AA_ALPHABET,
    generate_single_mutant_fasta,
    parse_variant,
)


def test_parse_valid_single_mutant():
    mut = parse_variant("M1A")
    assert mut.wt_aa == "M"
    assert mut.position == 1
    assert mut.new_aa == "A"
    assert mut.token == "M1A"


def test_parse_valid_multidigit_position():
    mut = parse_variant("K123R")
    assert mut.position == 123
    assert mut.token == "K123R"


def test_parse_multi_mutant_raises_with_token_named():
    with pytest.raises(ValueError) as exc:
        parse_variant("M1A_K5R")
    assert "M1A_K5R" in str(exc.value)
    assert "multi-mutant" in str(exc.value).lower()


def test_parse_non_numeric_position_raises():
    with pytest.raises(ValueError):
        parse_variant("MXA")


def test_parse_invalid_residue_raises():
    with pytest.raises(ValueError):
        parse_variant("Z1A")
    with pytest.raises(ValueError):
        parse_variant("M1Z")


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_variant("")


def test_generate_single_mutant_fasta_count_and_headers(tmp_path):
    wt_seq = "MKVAA"  # L = 5
    wt_fasta = tmp_path / "wt.fasta"
    wt_fasta.write_text(">WT\n" + wt_seq + "\n")

    out = tmp_path / "mutants.fasta"
    n = generate_single_mutant_fasta(wt_fasta, out)

    # 19 substitutions per position * L + 1 WT record.
    assert n == 19 * len(wt_seq) + 1

    text = out.read_text()
    # The FASTA id is the first whitespace-delimited token after ">".
    headers = [
        line[1:].split()[0] for line in text.splitlines() if line.startswith(">")
    ]
    assert "WT" in headers
    # Every non-WT header is a valid single-mutant token at a real position.
    for h in headers:
        if h == "WT":
            continue
        mut = parse_variant(h)
        assert wt_seq[mut.position - 1] == mut.wt_aa
        assert mut.new_aa != mut.wt_aa
    # No duplicate headers.
    assert len(headers) == len(set(headers))


def test_aa_alphabet_length():
    assert len(AA_ALPHABET) == 20
