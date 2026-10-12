"""Multi-mutant proposal by ADDITIVE stacking (GAP 4 -- a CAPABILITY).

The single-mutant active-learning core (embed -> top-layer RF -> rank singles ->
measure -> iterate) only ever proposes single substitutions. This module adds the
multi-mutant proposal step as a general tool feature, REUSING the package's
variant parsing and oracle latent construction -- it does NOT reimplement the RF
or the active-learning loop.

The design follows an explicit USER DE-SCOPE: epistasis is OUT of the predictor.
MULTI-evolve (Tran 2026) modelled multi-mutant epistasis explicitly; Visani 2026
showed a purely additive multi-mutant predictor reproduces it at r > 0.999 on the
additive-dominated landscapes that matter, consistent with the Kelsic
additive-dominance finding. So the PREDICTOR here is PURELY ADDITIVE: a
multi-mutant's predicted effect is ``baseline + sum of its constituent
single-mutant effects``, with NO pairwise/epistasis term and NO all-pairwise
double screen. The loop just stacks the best singles and refines by feedback.

Pieces:

* :func:`enumerate_multimutants` -- distinct-position combinations of single
  tokens, joined with ``':'`` in a CANONICAL (position-sorted) order so each
  multi has ONE id. Deterministic capping (never silent truncation).
* :class:`AdditiveMultiPredictor` -- the purely additive predictor, sourcing
  single effects from MEASURED singles and falling back to a single-mutant model
  for unmeasured constituents (fail-fast if a constituent is unavailable, never
  silently 0).
* :func:`propose_multimutants` -- enumerate from the best singles, score
  additively, return the top-N distinct multi tokens.
* :func:`make_generative_landscape` -- a GENERATIVE oracle that can score ANY
  single or ``':'``-joined multi token from one shared latent additive
  site-effect table plus an optional pairwise EPISTASIS term, used to test where
  additive stacking breaks. The oracle MAY contain epistasis; the predictor must
  NOT.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..variants import AA_ALPHABET, WT_TOKEN, parse_variant
from .oracle import Landscape


# --------------------------------------------------------------------------
# 1. Enumeration of multi-mutant candidate tokens.
# --------------------------------------------------------------------------
def canonical_multi_token(single_tokens: Sequence[str]) -> str:
    """Join single-substitution tokens into ONE canonical multi-mutant id.

    The components are validated by :func:`~plm_evolve.variants.parse_variant`
    and sorted by INTEGER residue position (not string), so ``M1A`` and ``K10R``
    always order as ``M1A:K10R`` regardless of input order -- one multi, one id.

    Args:
        single_tokens: single-substitution tokens at DISTINCT positions.

    Returns:
        The ``':'``-joined canonical token.

    Raises:
        ValueError: a component is not a valid single substitution, or two
            components share a residue position (a within-multi clash).
    """
    parsed = [(parse_variant(t).position, parse_variant(t).token) for t in single_tokens]
    positions = [p for p, _ in parsed]
    if len(set(positions)) != len(positions):
        raise ValueError(
            f"Multi-mutant components clash on a residue position: {list(single_tokens)}. "
            f"Each component must mutate a DISTINCT position."
        )
    parsed.sort(key=lambda pt: pt[0])  # integer-position canonical order.
    return ":".join(tok for _, tok in parsed)


def enumerate_multimutants(
    single_tokens: Sequence[str],
    order: int = 2,
    top_k: Optional[int] = None,
    max_candidates: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Enumerate distinct-position multi-mutant combinations of single tokens.

    Every combination of ``order`` single-substitution tokens at DISTINCT
    residue positions is joined into a canonical ``':'``-token (see
    :func:`canonical_multi_token`). ``single_tokens`` is treated as already
    PRIORITY-ORDERED (e.g. by a ranking/score): ``top_k`` restricts the
    enumeration to the first ``top_k`` of them.

    If the combination count exceeds ``max_candidates`` the result is sampled
    deterministically via ``rng`` after a canonical sort, and a ``UserWarning``
    is emitted (the cap is reported, never silently applied).

    Args:
        single_tokens: priority-ordered single-substitution tokens.
        order: combination size (``>= 2``; ``2`` -> doubles, ``3`` -> triples).
        top_k: restrict to the first ``top_k`` singles (``None`` -> all).
        max_candidates: cap on the number returned (``None`` -> uncapped).
        rng: RNG for the deterministic cap sample (required if a cap binds).

    Returns:
        Distinct canonical multi tokens (sorted; a deterministic subset if the
        cap binds). Empty if fewer than ``order`` distinct-position singles exist.

    Raises:
        ValueError: ``order < 2``, ``top_k``/``max_candidates`` non-positive, a
            component is invalid, or the cap binds with no ``rng`` provided.
    """
    if order < 2:
        raise ValueError(f"order must be >= 2 (multi-mutant), got {order}.")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 if given, got {top_k}.")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError(f"max_candidates must be >= 1 if given, got {max_candidates}.")

    singles = list(single_tokens)
    if top_k is not None:
        singles = singles[:top_k]

    # Validate components up front (fail-fast) and key each by integer position
    # so distinct-position combinations are cheap to filter.
    position_of: Dict[str, int] = {}
    for tok in singles:
        position_of[tok] = parse_variant(tok).position

    # Combinations of single tokens; keep only those whose positions are all
    # distinct (no within-multi site clash), join canonically -> ONE id each.
    seen: set = set()
    multis: List[str] = []
    for combo in itertools.combinations(singles, order):
        positions = [position_of[t] for t in combo]
        if len(set(positions)) != order:
            continue  # a site clash within the combo -> not a valid multi.
        token = canonical_multi_token(combo)
        if token not in seen:
            seen.add(token)
            multis.append(token)

    multis.sort()  # canonical, input-order-independent base ordering.

    if max_candidates is not None and len(multis) > max_candidates:
        if rng is None:
            raise ValueError(
                f"enumerate_multimutants produced {len(multis)} candidates but "
                f"max_candidates={max_candidates} and no rng was provided to sample "
                f"deterministically. Pass an rng or raise the cap."
            )
        warnings.warn(
            f"enumerate_multimutants capped {len(multis)} candidates to "
            f"{max_candidates} via deterministic rng sampling (not silent "
            f"truncation).",
            UserWarning,
            stacklevel=2,
        )
        idx = rng.choice(len(multis), size=max_candidates, replace=False)
        multis = [multis[i] for i in sorted(idx.tolist())]

    return multis


# --------------------------------------------------------------------------
# 2. The purely additive multi-mutant predictor (NO epistasis term).
# --------------------------------------------------------------------------
SingleEffectSource = Union[Dict[str, float], Callable[[str], float]]


@dataclass
class AdditiveMultiPredictor:
    """Predict a multi-mutant effect = baseline + SUM of constituent singles.

    PURELY ADDITIVE: there is NO pairwise/epistasis term -- that is the explicit
    de-scope. A multi token ``'M1A:K10R'`` is predicted as
    ``baseline + effect('M1A') + effect('K10R')``.

    The single-effect source is layered, MEASURED-first:

    * ``measured_effects`` -- a ``{single_token: effect}`` dict of MEASURED
      singles (used first, no model call).
    * ``model`` -- a fallback for UNMEASURED constituents: either a callable
      ``single_token -> effect`` or a precomputed ``{single_token: effect}`` dict.

    A constituent whose effect is in NEITHER source is a fail-fast error -- the
    predictor never silently substitutes ``0`` for a missing effect.

    Attributes:
        measured_effects: ``{single_token: effect}`` for measured singles.
        model: optional fallback callable or dict for unmeasured constituents.
        baseline: constant added to every prediction (WT activity; default 0).
    """

    measured_effects: Dict[str, float]
    model: Optional[SingleEffectSource] = None
    baseline: float = 0.0

    def single_effect(self, single_token: str) -> float:
        """Effect of ONE single substitution (measured first, then model).

        Raises:
            KeyError: the token is in neither the measured map nor the model,
                so its effect is genuinely unavailable (no silent ``0``).
        """
        token = parse_variant(single_token).token  # validate + canonicalise.
        if token in self.measured_effects:
            return float(self.measured_effects[token])
        if self.model is not None:
            if callable(self.model):
                return float(self.model(token))
            if token in self.model:
                return float(self.model[token])
        raise KeyError(
            f"No effect available for constituent single {token!r}: it is not in "
            f"measured_effects and "
            + (
                "the model dict does not contain it."
                if (self.model is not None and not callable(self.model))
                else "no fallback model was provided."
            )
            + " The additive predictor never silently uses 0 for a missing effect."
        )

    def predict(self, multi_token: str) -> float:
        """Predict a single or multi token's effect = baseline + sum of singles.

        A plain single token returns ``baseline + single_effect(token)``; a
        ``':'``-joined multi returns ``baseline + sum`` over its constituents.
        """
        components = [p.strip() for p in str(multi_token).split(":") if p.strip()]
        if not components:
            raise ValueError(f"Empty multi-mutant token: {multi_token!r}")
        positions = [parse_variant(c).position for c in components]
        if len(set(positions)) != len(positions):
            raise ValueError(
                f"Multi token {multi_token!r} has repeated residue positions; "
                f"a multi-mutant must mutate distinct positions."
            )
        return float(self.baseline + sum(self.single_effect(c) for c in components))

    def predict_many(self, multi_tokens: Sequence[str]) -> np.ndarray:
        """Vectorised :meth:`predict` over many tokens."""
        return np.array([self.predict(t) for t in multi_tokens], dtype=float)


# --------------------------------------------------------------------------
# 3. Propose multi-mutants: enumerate from best singles, score additively.
# --------------------------------------------------------------------------
def propose_multimutants(
    single_effects: Dict[str, float],
    order: int = 2,
    n_propose: int = 10,
    *,
    top_k: Optional[int] = None,
    max_candidates: Optional[int] = None,
    baseline: float = 0.0,
    model: Optional[SingleEffectSource] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Propose the top-N multi-mutants by ADDITIVE stacking of the best singles.

    Steps (all deterministic for fixed inputs):

      1. Rank ``single_effects`` by effect DESCENDING (ties broken by token, so
         the order is stable) -- the "best singles".
      2. :func:`enumerate_multimutants` over those ranked singles (``top_k`` /
         ``max_candidates`` honoured; cap is reported via warning).
      3. Score each candidate with :class:`AdditiveMultiPredictor` (purely
         additive: baseline + sum of constituent singles), using
         ``single_effects`` as the measured source and ``model`` for any
         constituent not present (only relevant if ``model`` injects extra
         singles; by construction every enumerated constituent is in
         ``single_effects``).
      4. Return the top ``n_propose`` distinct multi tokens by additive score
         (descending, ties broken by token for determinism).

    Args:
        single_effects: ``{single_token: effect}`` to stack (the candidate
            singles AND the additive effect source).
        order: multi-mutant order (``>= 2``).
        n_propose: number of multi tokens to return.
        top_k: restrict enumeration to the best ``top_k`` singles.
        max_candidates: cap the enumerated candidate count (deterministic sample).
        baseline: WT activity added to every additive prediction.
        model: optional fallback effect source for constituents not in
            ``single_effects`` (unused when every constituent is a key there).
        rng: RNG for the deterministic enumeration cap (required if it binds).

    Returns:
        Up to ``n_propose`` distinct canonical multi tokens, best-additive first.
    """
    if n_propose < 1:
        raise ValueError(f"n_propose must be >= 1, got {n_propose}.")
    if not single_effects:
        raise ValueError("single_effects is empty; nothing to stack.")

    # Validate every single token up front and rank by effect descending; ties
    # broken by token so the ranking (and the enumerate input order) is stable.
    for tok in single_effects:
        parse_variant(tok)
    ranked_singles = sorted(
        single_effects.keys(),
        key=lambda t: (-float(single_effects[t]), t),
    )

    candidates = enumerate_multimutants(
        ranked_singles,
        order=order,
        top_k=top_k,
        max_candidates=max_candidates,
        rng=rng,
    )
    if not candidates:
        return []

    predictor = AdditiveMultiPredictor(
        measured_effects=dict(single_effects), model=model, baseline=baseline
    )
    scores = predictor.predict_many(candidates)

    # Descending additive score, ties broken by token for determinism.
    ranked = sorted(
        zip(candidates, scores.tolist()),
        key=lambda cs: (-cs[1], cs[0]),
    )
    return [tok for tok, _ in ranked[:n_propose]]


# --------------------------------------------------------------------------
# 4. The generative oracle: score ANY single or multi token.
# --------------------------------------------------------------------------
@dataclass
class _AdditiveLatent:
    """The shared latent: additive site effects + an optional epistasis term.

    This mirrors :func:`~plm_evolve.benchmark.oracle.synthetic_landscape`'s
    latent construction (per-(position, residue) additive effects with the WT
    residue effect pinned to 0, plus an optional symmetric pairwise epistasis
    tensor consulted ONLY for multi-mutants). It is duplicated here -- rather
    than refactored out of ``synthetic_landscape`` -- so the existing landscape's
    RNG draw order (hence every test value that depends on it) is untouched.

    The latent is NOISELESS and pins WT activity to exactly 0, so the additivity
    identity ``score(multi) == sum(score(single_i))`` holds EXACTLY at
    ``epistasis == 0`` and is broken ONLY by the epistasis term for multis.
    """

    alphabet: List[str]
    wt_idx: np.ndarray
    site_effects: np.ndarray  # (n_positions, n_alphabet)
    epi: Optional[np.ndarray]  # (P, A, P, A) or None.
    epistasis: float

    @property
    def n_positions(self) -> int:
        return self.site_effects.shape[0]

    def _col(self, aa: str) -> int:
        return self.alphabet.index(aa)

    def score_components(
        self, positions: Sequence[int], cols: Sequence[int]
    ) -> float:
        """Additive site sum (+ epistasis for >= 2 sites) over 0-based positions."""
        score = 0.0
        for p, c in zip(positions, cols):
            score += float(self.site_effects[p, c])
        if self.epi is not None and len(positions) >= 2:
            for a in range(len(positions)):
                for b in range(a + 1, len(positions)):
                    score += self.epistasis * float(
                        self.epi[positions[a], cols[a], positions[b], cols[b]]
                    )
        return score


def _build_additive_latent(
    n_positions: int,
    alphabet_subset: Optional[Sequence[str]],
    seed: int,
    epistasis: float,
) -> _AdditiveLatent:
    """Construct the shared additive (+optional epistasis) latent (noiseless)."""
    if n_positions < 1:
        raise ValueError("n_positions must be >= 1.")
    alphabet = list(alphabet_subset) if alphabet_subset else list(AA_ALPHABET)
    if len(alphabet) < 2:
        raise ValueError("alphabet_subset needs at least 2 residues.")
    for aa in alphabet:
        if aa not in AA_ALPHABET:
            raise ValueError(f"Residue {aa!r} not in the canonical alphabet.")

    rng = np.random.default_rng(seed)
    wt_idx = rng.integers(0, len(alphabet), size=n_positions)

    site_effects = rng.normal(0.0, 1.0, size=(n_positions, len(alphabet)))
    for p in range(n_positions):
        site_effects[p] -= site_effects[p, wt_idx[p]]  # WT residue effect -> 0.

    if epistasis != 0.0:
        epi = rng.normal(
            0.0, 1.0, size=(n_positions, len(alphabet), n_positions, len(alphabet))
        )
    else:
        epi = None

    return _AdditiveLatent(
        alphabet=alphabet,
        wt_idx=wt_idx,
        site_effects=site_effects,
        epi=epi,
        epistasis=float(epistasis),
    )


def make_generative_landscape(
    n_positions: int,
    alphabet_subset: Optional[Sequence[str]] = None,
    seed: int = 0,
    epistasis: float = 0.0,
) -> Tuple[Landscape, Callable[[str], float]]:
    """A generative oracle that scores ANY single or multi token.

    The fixed-dict :class:`~plm_evolve.benchmark.oracle.Landscape` can only score
    variants it already contains, so it cannot evaluate an arbitrary proposed
    multi-mutant. This builds BOTH:

    * a single-mutant ``Landscape`` over ``WT`` + all single substitutions, and
    * a ``score_token(token)`` callable that scores ANY single or ``':'``-joined
      multi token from ONE shared latent additive site-effect table plus an
      optional pairwise EPISTASIS term,

    from the SAME latent (:func:`_build_additive_latent`). The single-mutant
    Landscape is built BY CALLING ``score_token`` over WT + singles, so
    ``Landscape.measure(single) == score_token(single)`` by construction.

    WT activity is exactly ``0`` and the oracle is NOISELESS, so at
    ``epistasis == 0`` ``score_token(multi)`` equals the SUM of its constituent
    singles exactly; at ``epistasis > 0`` the pairwise term makes the multi
    DIFFER from that sum (the term is live for multis only). The oracle MAY thus
    contain epistasis -- to test where additive stacking breaks -- while the
    additive predictor must NOT.

    Args:
        n_positions: sequence length (number of residue positions).
        alphabet_subset: amino-acid subset (default the full 20-letter set).
        seed: RNG seed -- the same seed yields an identical oracle.
        epistasis: scale of the pairwise epistasis term (0 -> purely additive).

    Returns:
        ``(landscape, score_token)``: the single-mutant Landscape and a callable
        scoring ANY single or multi token.
    """
    latent = _build_additive_latent(
        n_positions=n_positions,
        alphabet_subset=alphabet_subset,
        seed=seed,
        epistasis=epistasis,
    )
    alphabet = latent.alphabet
    aa_to_col = {aa: j for j, aa in enumerate(alphabet)}

    def score_token(token: str) -> float:
        """Score ANY single or ``':'``-joined multi token (noiseless)."""
        raw = str(token).strip()
        if raw in ("", WT_TOKEN):
            return 0.0  # WT activity pinned to exactly 0.
        components = [p.strip() for p in raw.split(":") if p.strip()]
        positions: List[int] = []
        cols: List[int] = []
        for comp in components:
            mut = parse_variant(comp)  # validates the component.
            pos0 = mut.position - 1  # 1-based token -> 0-based latent index.
            if not (0 <= pos0 < latent.n_positions):
                raise ValueError(
                    f"Token {comp!r} position {mut.position} is outside the "
                    f"{latent.n_positions}-position landscape."
                )
            if mut.new_aa not in aa_to_col:
                raise ValueError(
                    f"Token {comp!r} substitutes {mut.new_aa!r}, outside the "
                    f"landscape alphabet {alphabet}."
                )
            positions.append(pos0)
            cols.append(aa_to_col[mut.new_aa])
        if len(set(positions)) != len(positions):
            raise ValueError(
                f"Multi token {raw!r} repeats a residue position; a multi-mutant "
                f"must mutate distinct positions."
            )
        return latent.score_components(positions, cols)

    # Build the single-mutant Landscape BY CALLING score_token (single source of
    # truth): WT + every single substitution.
    activities: Dict[str, float] = {WT_TOKEN: 0.0}
    for p in range(latent.n_positions):
        wt_aa = alphabet[int(latent.wt_idx[p])]
        for aa in alphabet:
            if aa == wt_aa:
                continue
            token = f"{wt_aa}{p + 1}{aa}"
            activities[token] = score_token(token)

    landscape = Landscape(activities=activities)
    return landscape, score_token
