"""The in-silico oracle: a fixed landscape mapping variant id -> true activity.

In the real wet-lab loop a "measurement" is an experiment. In simulation the
measurement comes from this oracle instead. The :class:`Landscape` holds a
fixed ``variant_id -> scalar activity`` map; ``measure(variant)`` returns the
true activity for any variant the landscape contains. The active-learning logic
(train -> predict -> rank) is identical between sim and real; only the source of
the measured value differs.

Three ways to build a landscape:

* :func:`Landscape.from_csv` -- a generic ``variant,activity`` CSV.
* :func:`Landscape.from_proteingym` -- a ProteinGym-style substitution DMS CSV
  (``mutant``/``mutated_sequence`` + ``DMS_score``) normalised to the package's
  ``<WTaa><pos><newaa>`` token convention.
* :func:`synthetic_landscape` -- a self-contained additive site-effect model
  (plus optional pairwise epistasis and Gaussian noise) over all single
  mutants, used by the tests (no model download, no network).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..variants import AA_ALPHABET, WT_TOKEN, parse_variant


@dataclass
class Landscape:
    """A fixed oracle map ``variant id -> true scalar activity``.

    Attributes:
        activities: mapping from variant token to true activity. Tokens follow
            the package convention (``<WTaa><pos><newaa>`` or ``WT``).
    """

    activities: Dict[str, float]

    def __post_init__(self) -> None:
        if not self.activities:
            raise ValueError("Landscape is empty; needs at least one variant.")
        # Materialise a stable, sorted id order so downstream selection and
        # metrics are deterministic regardless of dict insertion order.
        self._ids: List[str] = sorted(self.activities.keys())
        self._values = np.array(
            [float(self.activities[v]) for v in self._ids], dtype=float
        )

    # --- access -----------------------------------------------------------
    def variant_ids(self) -> List[str]:
        """All variant ids in the landscape (sorted, stable order)."""
        return list(self._ids)

    def contains(self, variant: str) -> bool:
        return variant in self.activities

    def measure(self, variant: str) -> float:
        """Return the true activity for ``variant`` (fail-fast if absent).

        This is the in-silico replacement for a wet-lab measurement. There is
        no silent fallback: querying a variant outside the landscape is a bug
        in the caller, not a 0.0 default.
        """
        if variant not in self.activities:
            raise KeyError(
                f"Variant {variant!r} is not in the landscape; the oracle can "
                f"only measure variants it contains ({len(self.activities)} total)."
            )
        return float(self.activities[variant])

    def measure_many(self, variants: Sequence[str]) -> np.ndarray:
        """Vectorised :meth:`measure` over a sequence of variant ids."""
        return np.array([self.measure(v) for v in variants], dtype=float)

    # --- summary statistics used by the metrics ---------------------------
    def global_max(self) -> float:
        """The best true activity in the whole landscape."""
        return float(self._values.max())

    def true_values(self) -> np.ndarray:
        """All true activities, aligned to :meth:`variant_ids` order."""
        return self._values.copy()

    def top_k_ids(self, k: int) -> List[str]:
        """The ``k`` variant ids with the highest true activity.

        Ties are broken by the stable id order so the set is deterministic.
        """
        if k <= 0:
            return []
        k = min(k, len(self._ids))
        # argsort is ascending; negate to get descending, stable for ties.
        order = np.argsort(-self._values, kind="stable")[:k]
        return [self._ids[i] for i in order]

    def __len__(self) -> int:
        return len(self._ids)

    # --- loaders ----------------------------------------------------------
    @classmethod
    def from_mapping(cls, mapping: Dict[str, float]) -> "Landscape":
        return cls(activities={str(k): float(v) for k, v in mapping.items()})

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        variant_column: str = "variant",
        activity_column: str = "activity",
    ) -> "Landscape":
        """Load a generic ``variant,activity`` CSV into a Landscape."""
        df = pd.read_csv(path)
        vcol = _resolve_column(df, variant_column, "variant")
        acol = _resolve_column(df, activity_column, "activity")
        mapping = _frame_to_mapping(df, vcol, acol, source=str(path))
        return cls(activities=mapping)

    @classmethod
    def from_proteingym(
        cls,
        path: Union[str, Path],
        score_column: str = "DMS_score",
        mutant_column: str = "mutant",
    ) -> "Landscape":
        """Load a ProteinGym-style substitution DMS CSV into a Landscape.

        ProteinGym substitution assays carry a ``mutant`` column whose tokens
        are already ``<WTaa><pos><newaa>`` (multi-mutants joined by ``:``) and a
        ``DMS_score`` column. The token convention matches this package for
        single substitutions, so they pass straight through ``parse_variant``
        for validation. Multi-mutant rows are out of scope for the single-
        substitution core and raise via ``parse_variant`` -- fail fast rather
        than silently dropping them.

        Args:
            path: path to the DMS CSV.
            score_column: activity score column (default ``"DMS_score"``).
            mutant_column: substitution token column (default ``"mutant"``).
        """
        df = pd.read_csv(path)
        acol = _resolve_column(df, score_column, "activity score")
        mcol = _resolve_column(df, mutant_column, "mutant")

        mapping: Dict[str, float] = {}
        for token, score in zip(df[mcol].astype(str), df[acol]):
            token = token.strip()
            if token in ("", "WT", WT_TOKEN):
                mapping[WT_TOKEN] = float(score)
                continue
            # Validate the token against the package convention (fail-fast on
            # multi-mutants). The normalised token is the canonical id.
            mut = parse_variant(token)
            if mut.token in mapping:
                raise ValueError(
                    f"Duplicate mutant {mut.token!r} in {path}; each variant "
                    f"must appear once."
                )
            mapping[mut.token] = float(score)

        if not mapping:
            raise ValueError(f"No usable rows parsed from ProteinGym CSV {path}.")
        return cls(activities=mapping)


def _resolve_column(df: pd.DataFrame, preferred: str, kind: str) -> str:
    """Find a column case-insensitively, failing fast if absent."""
    if preferred in df.columns:
        return preferred
    lowered = {c.lower(): c for c in df.columns}
    if preferred.lower() in lowered:
        return lowered[preferred.lower()]
    raise ValueError(
        f"CSV is missing a {kind} column (looked for {preferred!r}); "
        f"available columns: {list(df.columns)}."
    )


def _frame_to_mapping(
    df: pd.DataFrame, vcol: str, acol: str, source: str
) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for variant, activity in zip(df[vcol].astype(str), df[acol]):
        variant = variant.strip()
        if variant == "":
            raise ValueError(f"Empty variant id in {source}.")
        if variant in mapping:
            raise ValueError(
                f"Duplicate variant {variant!r} in {source}; each variant must "
                f"appear at most once in a landscape."
            )
        mapping[variant] = float(activity)
    if not mapping:
        raise ValueError(f"No rows parsed from {source}.")
    return mapping


def synthetic_landscape(
    n_positions: int,
    alphabet_subset: Optional[Sequence[str]] = None,
    seed: int = 0,
    epistasis: float = 0.0,
    noise: float = 0.0,
    n_multi_mutants: int = 0,
) -> Landscape:
    """Generate a synthetic additive (+optional epistasis/noise) landscape.

    The model is an additive site-effect model: every position draws a vector
    of per-amino-acid effects, and a variant's activity is the sum of its
    site effects relative to a fixed wild-type residue per position. Optional
    small pairwise epistasis and Gaussian observation noise can be layered on.
    The landscape covers ``WT`` plus all single mutants, and optionally a few
    random multi-mutant *combinations* (kept additive over their sites; these
    use ``:``-joined tokens and are NOT parseable by the single-substitution
    core -- they exist only as extra oracle entries for harness exercising).

    Args:
        n_positions: number of residue positions (sequence length).
        alphabet_subset: amino acids to use (default the full 20-letter set).
        seed: RNG seed -- the same seed yields an identical landscape.
        epistasis: scale of the pairwise epistasis term (0 -> purely additive).
        noise: standard deviation of added Gaussian observation noise.
        n_multi_mutants: number of extra random multi-mutant entries to add.

    Returns:
        A :class:`Landscape` over WT + all single mutants (+ optional multis).
    """
    if n_positions < 1:
        raise ValueError("n_positions must be >= 1.")
    alphabet = list(alphabet_subset) if alphabet_subset else list(AA_ALPHABET)
    if len(alphabet) < 2:
        raise ValueError("alphabet_subset needs at least 2 residues.")
    for aa in alphabet:
        if aa not in AA_ALPHABET:
            raise ValueError(f"Residue {aa!r} not in the canonical alphabet.")

    rng = np.random.default_rng(seed)

    # Wild-type residue per position (drawn from the alphabet).
    wt_idx = rng.integers(0, len(alphabet), size=n_positions)
    wt_sequence = "".join(alphabet[i] for i in wt_idx)

    # Additive per-(position, residue) effect table. WT residue effect is 0 by
    # construction so that WT activity is exactly 0 before noise.
    site_effects = rng.normal(0.0, 1.0, size=(n_positions, len(alphabet)))
    aa_to_col = {aa: j for j, aa in enumerate(alphabet)}
    for p in range(n_positions):
        site_effects[p] -= site_effects[p, wt_idx[p]]

    # Optional symmetric pairwise epistasis term over (pos_i, aa_i, pos_j, aa_j).
    # Kept small and deterministic; only consulted when epistasis > 0.
    if epistasis != 0.0:
        epi = rng.normal(0.0, 1.0, size=(n_positions, len(alphabet),
                                          n_positions, len(alphabet)))
    else:
        epi = None

    def additive_score(mut_positions: Sequence[int], mut_cols: Sequence[int]) -> float:
        score = 0.0
        for p, c in zip(mut_positions, mut_cols):
            score += site_effects[p, c]
        if epi is not None and len(mut_positions) >= 2:
            for a in range(len(mut_positions)):
                for b in range(a + 1, len(mut_positions)):
                    score += epistasis * epi[
                        mut_positions[a], mut_cols[a],
                        mut_positions[b], mut_cols[b],
                    ]
        return score

    activities: Dict[str, float] = {}

    # WT entry (activity 0 before noise).
    activities[WT_TOKEN] = float(0.0 + (rng.normal(0.0, noise) if noise else 0.0))

    # All single mutants.
    for p in range(n_positions):
        wt_aa = alphabet[wt_idx[p]]
        for aa in alphabet:
            if aa == wt_aa:
                continue
            token = f"{wt_aa}{p + 1}{aa}"
            base = additive_score([p], [aa_to_col[aa]])
            if noise:
                base += float(rng.normal(0.0, noise))
            activities[token] = float(base)

    # Optional extra multi-mutant combinations (additive over sites, with the
    # epistasis term if enabled). These use ``:``-joined tokens.
    for _ in range(max(0, int(n_multi_mutants))):
        k = int(rng.integers(2, min(3, n_positions) + 1)) if n_positions >= 2 else 1
        positions = sorted(rng.choice(n_positions, size=k, replace=False).tolist())
        sub_tokens = []
        cols = []
        ok = True
        for p in positions:
            wt_aa = alphabet[wt_idx[p]]
            choices = [aa for aa in alphabet if aa != wt_aa]
            aa = choices[int(rng.integers(0, len(choices)))]
            sub_tokens.append(f"{wt_aa}{p + 1}{aa}")
            cols.append(aa_to_col[aa])
        token = ":".join(sub_tokens)
        if token in activities:
            continue
        base = additive_score(positions, cols)
        if noise:
            base += float(rng.normal(0.0, noise))
        activities[token] = float(base)

    landscape = Landscape(activities=activities)
    # Stash the WT sequence so harness consumers can build matching embeddings
    # / one-hot encodings if they want; harmless metadata.
    landscape.wt_sequence = wt_sequence  # type: ignore[attr-defined]
    return landscape
