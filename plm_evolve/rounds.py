"""Load measured-round data and align it to the embedding index.

A round file (Excel or CSV) has a ``Variant`` column (tokens of the form
``<WTaa><pos><newaa>`` or ``WT``) and an activity column. This module aligns
measured variants to the embedding index, validates fail-fast, and produces
the labels table used to train the top-layer regressor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .variants import WT_TOKEN, parse_variant, read_wt_sequence

DEFAULT_VARIANT_COLUMN = "Variant"
DEFAULT_ACTIVITY_COLUMN = "activity"


def _read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read an Excel (.xlsx/.xls) or CSV round file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported round file format {suffix!r} for {path}; use .xlsx or .csv."
    )


def _resolve_column(df: pd.DataFrame, preferred: str, kind: str) -> str:
    """Find a column case-insensitively, failing fast if absent."""
    if preferred in df.columns:
        return preferred
    lowered = {c.lower(): c for c in df.columns}
    if preferred.lower() in lowered:
        return lowered[preferred.lower()]
    raise ValueError(
        f"Round file is missing a {kind} column (looked for {preferred!r}); "
        f"available columns: {list(df.columns)}."
    )


def load_round_data(
    round_files: Sequence[Union[str, Path]],
    wt_fasta: Union[str, Path],
    variant_column: str = DEFAULT_VARIANT_COLUMN,
    activity_column: str = DEFAULT_ACTIVITY_COLUMN,
) -> pd.DataFrame:
    """Load and concatenate measured rounds, validating variant tokens.

    Each non-WT variant token is parsed (single-substitution fail-fast) and
    its wild-type residue is checked against the WT sequence at that position
    using ``wt_fasta``. Mismatches raise immediately rather than silently
    mislabelling activity.

    Args:
        round_files: one or more round files (Excel/CSV).
        wt_fasta: wild-type FASTA used to validate variant WT residues.
        variant_column: name of the variant column (default ``"Variant"``).
        activity_column: name of the activity column (default ``"activity"``).

    Returns:
        DataFrame with columns ``variant`` and ``activity`` (one row per
        measured variant across all rounds, in file order).

    Raises:
        ValueError: malformed token, WT-residue mismatch, or duplicate variant.
    """
    wt_sequence = read_wt_sequence(wt_fasta)

    frames: List[pd.DataFrame] = []
    for rf in round_files:
        raw = _read_table(rf)
        vcol = _resolve_column(raw, variant_column, "variant")
        acol = _resolve_column(raw, activity_column, "activity")

        variants: List[str] = []
        activities: List[float] = []
        for token, activity in zip(raw[vcol].astype(str), raw[acol]):
            token = token.strip()
            if token == WT_TOKEN:
                variants.append(WT_TOKEN)
                activities.append(float(activity))
                continue

            mut = parse_variant(token)  # fail-fast on multi-mutant / malformed
            pos = mut.position
            if pos > len(wt_sequence):
                raise ValueError(
                    f"Variant {token!r} position {pos} exceeds WT length "
                    f"{len(wt_sequence)}."
                )
            seq_aa = wt_sequence[pos - 1]
            if seq_aa != mut.wt_aa:
                raise ValueError(
                    f"Variant {token!r} declares WT residue {mut.wt_aa!r} at "
                    f"position {pos}, but the WT sequence has {seq_aa!r}. "
                    f"Check the variant token against the WT FASTA."
                )
            variants.append(mut.token)
            activities.append(float(activity))

        frames.append(pd.DataFrame({"variant": variants, "activity": activities}))

    combined = pd.concat(frames, ignore_index=True)

    # Fail fast on duplicates (do NOT return None like the reference does).
    dup_mask = combined["variant"].duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(combined.loc[dup_mask, "variant"].unique().tolist())
        raise ValueError(
            f"Duplicate variants found across rounds: {dups}. Each variant must "
            f"be measured at most once."
        )

    return combined


def _add_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``activity_binary`` (>=1) and min-max ``activity_scaled``.

    When all measured activities are equal (or there is a single row), the
    min-max range is zero; ``activity_scaled`` is set to 0.0 for every row
    rather than emitting NaN/inf.
    """
    out = df.copy()
    out["activity_binary"] = (out["activity"] >= 1).astype(int)

    amin = out["activity"].min()
    amax = out["activity"].max()
    span = amax - amin
    if span == 0:
        out["activity_scaled"] = 0.0
    else:
        out["activity_scaled"] = (out["activity"] - amin) / span
    return out


def validate_coverage(
    measured: pd.DataFrame, embedding_index: Sequence[str]
) -> pd.DataFrame:
    """Warn (no silent drop) on measured variants absent from the embeddings.

    Args:
        measured: DataFrame with a ``variant`` column.
        embedding_index: variant ids present in the embedding table.

    Returns:
        The subset of ``measured`` whose variants ARE present in the embeddings
        (the rows usable for training). Unmatched rows are named on stderr.
    """
    index_set = set(embedding_index)
    present_mask = measured["variant"].isin(index_set)
    missing = measured.loc[~present_mask, "variant"].tolist()
    if missing:
        print(
            f"[plm-evolve] WARNING: {len(missing)} measured variant(s) have no "
            f"matching embedding row and cannot be used for training: "
            f"{missing}",
            file=sys.stderr,
        )
    return measured.loc[present_mask].reset_index(drop=True)


def build_training_table(
    measured: pd.DataFrame, embedding_index: Sequence[str]
) -> pd.DataFrame:
    """Build the labels table aligned to the embedding index.

    Returns one row per embedding id with columns ``variant``, ``activity``,
    ``activity_binary``, ``activity_scaled`` and ``measured`` (bool). Variants
    without a measurement have NaN activity and ``measured == False``.

    Activity-derived columns are computed over the measured variants that have
    an embedding (coverage-validated first).
    """
    usable = validate_coverage(measured, embedding_index)
    usable = _add_activity_columns(usable)

    table = pd.DataFrame({"variant": list(embedding_index)})
    table = table.merge(usable, on="variant", how="left")
    table["measured"] = table["activity"].notna()
    return table
