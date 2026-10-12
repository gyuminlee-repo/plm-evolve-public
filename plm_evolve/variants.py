"""Single-mutant variant tokens and FASTA generation.

Variant token convention (matches the embedding index): ``<WTaa><pos><newaa>``
with a 1-based position, e.g. ``M1A`` mutates Met at position 1 to Ala.
A special ``WT`` token denotes the wild-type sequence.

Only single amino-acid substitutions are supported. Multi-mutant tokens
(e.g. ``M1A_K5R``) are explicitly out of scope and raise ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Canonical 20-letter amino-acid alphabet (sorted for deterministic output).
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Sentinel token for the wild-type record.
WT_TOKEN = "WT"


@dataclass(frozen=True)
class Mutation:
    """A parsed single amino-acid substitution.

    Attributes:
        wt_aa: wild-type residue letter.
        position: 1-based residue position.
        new_aa: substituted residue letter.
    """

    wt_aa: str
    position: int
    new_aa: str

    @property
    def token(self) -> str:
        return f"{self.wt_aa}{self.position}{self.new_aa}"


def parse_variant(token: str) -> Mutation:
    """Parse a single-substitution variant token, failing fast on anything else.

    The token must be ``<WTaa><pos><newaa>`` where the middle part is all
    digits. Multi-mutant tokens (containing ``_``, ``,``, ``:``, ``/`` or ``+``)
    are not yet supported and raise a clear error pointing at that gap.

    Args:
        token: variant identifier, e.g. ``"M1A"``.

    Returns:
        Parsed :class:`Mutation`.

    Raises:
        ValueError: token is empty, malformed, or encodes more than one
            substitution. No silent fallback is performed.
    """
    if token is None or not isinstance(token, str) or token.strip() == "":
        raise ValueError(f"Empty or non-string variant token: {token!r}")

    raw = token.strip()

    # Detect multi-mutant separators explicitly so the message is actionable.
    for sep in ("_", ",", ":", "/", "+", ";"):
        if sep in raw:
            raise ValueError(
                f"Variant token {raw!r} looks like a multi-mutant combination "
                f"(separator {sep!r}). Multi-mutant support is not yet "
                f"implemented; only single substitutions <WTaa><pos><newaa> "
                f"(e.g. 'M1A') are supported."
            )

    if len(raw) < 3:
        raise ValueError(
            f"Variant token {raw!r} is too short; expected <WTaa><pos><newaa> "
            f"such as 'M1A'."
        )

    wt_aa, new_aa = raw[0], raw[-1]
    position_part = raw[1:-1]

    # Fail fast: the position must be purely numeric.
    if not position_part.isdigit():
        raise ValueError(
            f"Variant token {raw!r} has a non-numeric position part "
            f"{position_part!r}. Only single substitutions <WTaa><pos><newaa> "
            f"(e.g. 'M1A') are supported; multi-mutant combinations are not "
            f"yet implemented."
        )

    if wt_aa not in AA_ALPHABET:
        raise ValueError(
            f"Variant token {raw!r} has an invalid wild-type residue {wt_aa!r} "
            f"(not in {AA_ALPHABET})."
        )
    if new_aa not in AA_ALPHABET:
        raise ValueError(
            f"Variant token {raw!r} has an invalid substituted residue "
            f"{new_aa!r} (not in {AA_ALPHABET})."
        )

    position = int(position_part)
    if position < 1:
        raise ValueError(
            f"Variant token {raw!r} has a non-positive position {position}; "
            f"positions are 1-based."
        )

    return Mutation(wt_aa=wt_aa, position=position, new_aa=new_aa)


def read_wt_sequence(wt_fasta: Union[str, Path]) -> str:
    """Read the single wild-type sequence from a FASTA file."""
    record = SeqIO.read(str(wt_fasta), "fasta")
    return str(record.seq)


def iter_single_mutants(wt_sequence: str) -> Iterator[Mutation]:
    """Yield every 19 x L single-substitution mutation for a WT sequence."""
    for i, wt_aa in enumerate(wt_sequence):
        if wt_aa not in AA_ALPHABET:
            raise ValueError(
                f"Wild-type sequence has a non-standard residue {wt_aa!r} at "
                f"position {i + 1}; cannot enumerate substitutions."
            )
        for new_aa in AA_ALPHABET:
            if new_aa != wt_aa:
                yield Mutation(wt_aa=wt_aa, position=i + 1, new_aa=new_aa)


def generate_single_mutant_fasta(
    wt_fasta: Union[str, Path], output_fasta: Union[str, Path]
) -> int:
    """Write all single-substitution sequences plus a WT record to FASTA.

    The header ids are ``<WTaa><pos><newaa>`` (1-based) plus one ``WT`` record.

    Args:
        wt_fasta: path to the wild-type FASTA (a single record).
        output_fasta: path to write the mutant FASTA.

    Returns:
        Number of records written (19 * L + 1).
    """
    wt_sequence = read_wt_sequence(wt_fasta)

    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    # Empty description so the FASTA header is exactly ">WT" (the id used as
    # the embedding index key); Biopython would otherwise append the text.
    records = [SeqRecord(Seq(wt_sequence), id=WT_TOKEN, description="")]
    for mut in iter_single_mutants(wt_sequence):
        i = mut.position - 1
        mutated = wt_sequence[:i] + mut.new_aa + wt_sequence[i + 1 :]
        records.append(SeqRecord(Seq(mutated), id=mut.token, description=""))

    with open(output_fasta, "w") as handle:
        SeqIO.write(records, handle, "fasta")

    return len(records)
