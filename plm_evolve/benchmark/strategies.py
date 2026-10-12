"""Selection strategies for the active-learning simulation.

Every strategy implements one method::

    pick(measured_ids, measured_y, candidate_ids, embeddings, n_select, rng)
        -> list[str]   # selected candidate ids (subset of candidate_ids)

``embeddings`` is a DataFrame indexed by variant id (columns = dimensions),
exactly the table produced by the core embedding step. The strategies never
look at true activities of the candidates -- only at the measured labels and
the (label-free) embeddings -- so the simulation stays honest.

:class:`GreedyTopLayer` is the EVOLVEpro-equivalent: it REUSES the core
:class:`~plm_evolve.regressor.TopLayerRegressor` and the same ranking idiom as
:func:`plm_evolve.evolve.run_round` (fit on measured -> predict all candidates
-> stable-sort descending -> take top-N).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from ..regressor import TopLayerRegressor
from ..variants import parse_variant


def _validate_pick(
    selected: List[str], candidate_ids: Sequence[str], n_select: int
) -> List[str]:
    """Common post-condition guard shared by every strategy."""
    cand_set = set(candidate_ids)
    if not set(selected).issubset(cand_set):
        raise ValueError("Strategy selected ids outside the candidate set.")
    if len(set(selected)) != len(selected):
        raise ValueError("Strategy selected duplicate ids.")
    return selected[:n_select]


class RandomStrategy:
    """Uniform-random selection -- the do-nothing baseline."""

    name = "random"

    def pick(
        self,
        measured_ids: Sequence[str],
        measured_y: Sequence[float],
        candidate_ids: Sequence[str],
        embeddings: pd.DataFrame,
        n_select: int,
        rng: np.random.Generator,
    ) -> List[str]:
        candidates = list(candidate_ids)
        n = min(n_select, len(candidates))
        idx = rng.choice(len(candidates), size=n, replace=False)
        selected = [candidates[i] for i in sorted(idx.tolist())]
        return _validate_pick(selected, candidate_ids, n_select)


class GreedyTopLayer:
    """EVOLVEpro-equivalent greedy selection on the top-layer regressor.

    Fits the core :class:`TopLayerRegressor` on the measured embeddings -> y,
    predicts every candidate, and takes the top-N by predicted activity using
    the SAME stable-descending-sort idiom as :func:`plm_evolve.evolve.run_round`.
    This is the iterative active-learning workhorse.
    """

    name = "greedy"

    def __init__(self, n_estimators: int = 100, random_state: int = 1) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def pick(
        self,
        measured_ids: Sequence[str],
        measured_y: Sequence[float],
        candidate_ids: Sequence[str],
        embeddings: pd.DataFrame,
        n_select: int,
        rng: np.random.Generator,
    ) -> List[str]:
        measured_ids = list(measured_ids)
        candidate_ids = list(candidate_ids)
        if len(measured_ids) == 0:
            # Cold start: no labels to fit on -> fall back to random so the
            # round can still proceed (mirrors needing a seed set first).
            return RandomStrategy().pick(
                measured_ids, measured_y, candidate_ids, embeddings, n_select, rng
            )

        X_train = embeddings.loc[measured_ids].to_numpy()
        y_train = np.asarray(measured_y, dtype=float)

        model = TopLayerRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        model.fit(X_train, y_train)

        X_cand = embeddings.loc[candidate_ids].to_numpy()
        predicted = model.predict(X_cand)

        # Same ranking idiom as evolve.run_round: stable descending sort.
        ranked = pd.DataFrame(
            {"variant": candidate_ids, "predicted": predicted}
        ).sort_values("predicted", ascending=False, kind="stable")
        selected = ranked["variant"].head(n_select).tolist()
        return _validate_pick(selected, candidate_ids, n_select)


class AdditiveLinear:
    """Additive null baseline: ridge regression on one-hot mutation tokens.

    The feature for a variant is a one-hot encoding of its substitution token
    (``WT`` -> the all-zero feature). A ridge linear model is fit on the
    measured one-hots -> y and used to rank candidates. This deliberately
    ignores the PLM embeddings and tests whether a purely additive,
    site-independent model already explains the landscape -- the EVOLVEpro
    paper's "additive" null. If greedy on embeddings does not beat this, the
    PLM is buying nothing.
    """

    name = "additive"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    @staticmethod
    def _feature_names(ids: Sequence[str]) -> List[str]:
        names = set()
        for vid in ids:
            if vid == "WT":
                continue
            # token: <WTaa><pos><newaa> -> one feature per (pos, newaa). For
            # multi-mutant ':'-joined tokens, split and add each component.
            for part in str(vid).split(":"):
                part = part.strip()
                if part and part != "WT":
                    names.add(part)
        return sorted(names)

    def _encode(self, ids: Sequence[str], feature_names: List[str]) -> np.ndarray:
        index = {name: j for j, name in enumerate(feature_names)}
        X = np.zeros((len(ids), len(feature_names)), dtype=float)
        for i, vid in enumerate(ids):
            if vid == "WT":
                continue
            for part in str(vid).split(":"):
                part = part.strip()
                if part in index:
                    X[i, index[part]] = 1.0
        return X

    def pick(
        self,
        measured_ids: Sequence[str],
        measured_y: Sequence[float],
        candidate_ids: Sequence[str],
        embeddings: pd.DataFrame,
        n_select: int,
        rng: np.random.Generator,
    ) -> List[str]:
        from sklearn.linear_model import Ridge

        measured_ids = list(measured_ids)
        candidate_ids = list(candidate_ids)
        if len(measured_ids) == 0:
            return RandomStrategy().pick(
                measured_ids, measured_y, candidate_ids, embeddings, n_select, rng
            )

        # Build the feature vocabulary from BOTH measured and candidate tokens
        # so candidate features align with the fitted coefficients.
        feature_names = self._feature_names(list(measured_ids) + list(candidate_ids))
        X_train = self._encode(measured_ids, feature_names)
        y_train = np.asarray(measured_y, dtype=float)

        model = Ridge(alpha=self.alpha)
        model.fit(X_train, y_train)

        X_cand = self._encode(candidate_ids, feature_names)
        predicted = model.predict(X_cand)

        ranked = pd.DataFrame(
            {"variant": candidate_ids, "predicted": predicted}
        ).sort_values("predicted", ascending=False, kind="stable")
        selected = ranked["variant"].head(n_select).tolist()
        return _validate_pick(selected, candidate_ids, n_select)


# --- Uncertainty-aware acquisition (a CAPABILITY, not a novelty) ----------
# FolDE (Roberts 2025) and ALDE (Wittmann 2025, which found Thompson > UCB >
# greedy in wet-lab) already established uncertainty-aware acquisition for
# PLM-guided directed evolution. These two strategies add it as a general tool
# feature with the SAME ``pick(...)`` contract, REUSING the core
# ``TopLayerRegressor`` (whose ``predict_with_uncertainty`` returns per-tree
# mean and std). They differ from :class:`GreedyTopLayer` ONLY in how the fitted
# (mean, std) becomes a ranking; the fit/cold-start idiom is identical.


class UCBStrategy:
    """Upper-Confidence-Bound acquisition on the top-layer regressor.

    Fits the SAME core :class:`TopLayerRegressor` as :class:`GreedyTopLayer`,
    then ranks candidates by ``acquisition = mean + beta * std`` (descending,
    same stable-sort idiom). ``beta`` trades exploitation (the greedy mean) for
    exploration (high model-disagreement / sparse-data regions).

    Invariant: ``beta = 0`` reduces EXACTLY to :class:`GreedyTopLayer` -- the
    acquisition becomes ``mean + 0 * std``, and ``mean`` is byte-identical to
    ``TopLayerRegressor.predict`` (the greedy ranking signal), so the picks are
    identical for the same measured set / seed.
    """

    name = "ucb"

    def __init__(
        self, beta: float = 1.0, n_estimators: int = 100, random_state: int = 1
    ) -> None:
        self.beta = beta
        self.n_estimators = n_estimators
        self.random_state = random_state

    def pick(
        self,
        measured_ids: Sequence[str],
        measured_y: Sequence[float],
        candidate_ids: Sequence[str],
        embeddings: pd.DataFrame,
        n_select: int,
        rng: np.random.Generator,
    ) -> List[str]:
        measured_ids = list(measured_ids)
        candidate_ids = list(candidate_ids)
        if len(measured_ids) == 0:
            # Same cold-start fallback as GreedyTopLayer: no labels -> random.
            return RandomStrategy().pick(
                measured_ids, measured_y, candidate_ids, embeddings, n_select, rng
            )

        X_train = embeddings.loc[measured_ids].to_numpy()
        y_train = np.asarray(measured_y, dtype=float)

        model = TopLayerRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        model.fit(X_train, y_train)

        X_cand = embeddings.loc[candidate_ids].to_numpy()
        mean, std = model.predict_with_uncertainty(X_cand)
        # beta=0 -> acquisition == mean == predict() byte-for-byte == greedy.
        acquisition = mean + self.beta * std

        ranked = pd.DataFrame(
            {"variant": candidate_ids, "acquisition": acquisition}
        ).sort_values("acquisition", ascending=False, kind="stable")
        selected = ranked["variant"].head(n_select).tolist()
        return _validate_pick(selected, candidate_ids, n_select)


class ThompsonStrategy:
    """Thompson-sampling acquisition on the top-layer regressor.

    Fits the SAME core :class:`TopLayerRegressor` as :class:`GreedyTopLayer`,
    then builds a batch by posterior sampling. Sampling choice (documented and
    deterministic): the per-candidate posterior is approximated as
    ``N(mean, std)`` from :meth:`TopLayerRegressor.predict_with_uncertainty`.
    For a batch of ``n_select`` picks, draw ``n_select`` INDEPENDENT posterior
    samples (one per pick), and for each sample take the argmax over candidates
    not yet picked. All randomness comes from the passed-in ``rng`` ONLY
    (``rng.normal(mean, std)`` per pick), so the selection is reproducible for a
    given seed. Drawing a fresh sample per pick (rather than one shared sample)
    gives the batch its diversity: different posterior draws favour different
    candidates, so the batch spreads over plausible optima instead of collapsing
    onto the single greedy argmax.
    """

    name = "thompson"

    def __init__(self, n_estimators: int = 100, random_state: int = 1) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def pick(
        self,
        measured_ids: Sequence[str],
        measured_y: Sequence[float],
        candidate_ids: Sequence[str],
        embeddings: pd.DataFrame,
        n_select: int,
        rng: np.random.Generator,
    ) -> List[str]:
        measured_ids = list(measured_ids)
        candidate_ids = list(candidate_ids)
        if len(measured_ids) == 0:
            # Same cold-start fallback as GreedyTopLayer: no labels -> random.
            return RandomStrategy().pick(
                measured_ids, measured_y, candidate_ids, embeddings, n_select, rng
            )

        X_train = embeddings.loc[measured_ids].to_numpy()
        y_train = np.asarray(measured_y, dtype=float)

        model = TopLayerRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        model.fit(X_train, y_train)

        X_cand = embeddings.loc[candidate_ids].to_numpy()
        mean, std = model.predict_with_uncertainty(X_cand)

        # Greedily take the argmax of an independent posterior draw per pick.
        n_pick = min(n_select, len(candidate_ids))
        available = np.ones(len(candidate_ids), dtype=bool)
        selected: List[str] = []
        for _ in range(n_pick):
            sample = rng.normal(mean, std)  # passed-in rng ONLY (determinism).
            sample[~available] = -np.inf  # exclude already-picked candidates.
            j = int(np.argmax(sample))
            available[j] = False
            selected.append(candidate_ids[j])
        return _validate_pick(selected, candidate_ids, n_select)


# --- Batch-diversity selection (a CAPABILITY, not a novelty) --------------
# Naive top-N batch selection (GreedyTopLayer) can spend a whole round on
# near-duplicate candidates -- e.g. several substitutions at the SAME residue
# site, or candidates that cluster together in embedding space -- wasting
# measurements that a diverse batch would have spent on distinct hypotheses.
# Two batch strategies below address that, both REUSING the core
# ``TopLayerRegressor`` score machinery (no RF reimplementation):
#
#   * CoverageBatchStrategy -- the DIFFERENTIATED method. Greedy batch build
#     with a MUTATION-SITE coverage penalty, so one round spreads over residue
#     positions instead of piling substitutions onto one site. Generic
#     embedding-space farthest-point diversity is already scooped by FolDE
#     (Roberts 2025) constant-liar, so the differentiation here is the
#     mutation-SPACE (residue-position) coverage, not generic distance.
#
#   * ConstantLiarStrategy -- a NAMED PARITY baseline (NOT ours), implementing
#     the FolDE (Roberts 2025) constant-liar batch construction so the
#     mutation-coverage method can be benchmarked against the established
#     batch-diversity approach on the same footing.


class CoverageBatchStrategy:
    """Mutation-site coverage batch selection on the top-layer regressor.

    This is the DIFFERENTIATED (mutation-coverage) batch method. It fits the
    SAME core :class:`TopLayerRegressor` as :class:`GreedyTopLayer` ONCE on the
    measured set and scores every candidate; the only thing it changes is how a
    BATCH is assembled from those fixed scores.

    Base score (``base="greedy"`` or ``base="ucb"``):

        * ``greedy`` -> ``score(c) = predict(c)`` (the greedy mean), reusing
          :meth:`TopLayerRegressor.predict`.
        * ``ucb`` -> ``score(c) = mean(c) + beta * std(c)``, reusing
          :meth:`TopLayerRegressor.predict_with_uncertainty`.

    Batch rule (documented and exact): pick candidates one at a time by the
    largest EFFECTIVE score::

        effective_score(c) = score(c) - penalty * (n already selected at
                                                    parse_variant(c).position)

    The penalty count is BATCH-LOCAL (only candidates already chosen WITHIN
    this ``pick`` call count toward a position's tally; it is not cumulative
    over previously-measured variants). After each pick the chosen candidate's
    site tally increments, so a second substitution at the same residue is
    discounted by ``penalty`` -- and a third by ``2 * penalty`` -- which steers
    the round to cover distinct sites rather than re-measuring near-duplicate
    same-site variants. ``penalty = 0`` reduces EXACTLY to the base ranking
    (greedy top-N or UCB top-N), so the penalty is the only diversification.

    Ties on the effective score are broken by the candidate's original order
    (stable ``argmax``), so the batch is deterministic for a given measured
    set / base. The passed-in ``rng`` is only consulted for the cold-start
    random fallback (no measured labels yet).
    """

    name = "coverage"

    def __init__(
        self,
        base: str = "greedy",
        beta: float = 1.0,
        penalty: float = 1.0,
        n_estimators: int = 100,
        random_state: int = 1,
    ) -> None:
        if base not in ("greedy", "ucb"):
            raise ValueError(f"base must be 'greedy' or 'ucb', got {base!r}.")
        if penalty < 0:
            raise ValueError("penalty must be >= 0.")
        self.base = base
        self.beta = beta
        self.penalty = penalty
        self.n_estimators = n_estimators
        self.random_state = random_state

    def pick(
        self,
        measured_ids: Sequence[str],
        measured_y: Sequence[float],
        candidate_ids: Sequence[str],
        embeddings: pd.DataFrame,
        n_select: int,
        rng: np.random.Generator,
    ) -> List[str]:
        measured_ids = list(measured_ids)
        candidate_ids = list(candidate_ids)
        if len(measured_ids) == 0:
            # Same cold-start fallback as GreedyTopLayer: no labels -> random.
            return RandomStrategy().pick(
                measured_ids, measured_y, candidate_ids, embeddings, n_select, rng
            )

        X_train = embeddings.loc[measured_ids].to_numpy()
        y_train = np.asarray(measured_y, dtype=float)

        model = TopLayerRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        model.fit(X_train, y_train)

        X_cand = embeddings.loc[candidate_ids].to_numpy()
        if self.base == "ucb":
            mean, std = model.predict_with_uncertainty(X_cand)
            score = mean + self.beta * std
        else:
            score = model.predict(X_cand)
        score = np.asarray(score, dtype=float)

        # Residue site per candidate (parse_variant gives the position). The
        # penalty is keyed on this site, so same-site substitutions compete.
        sites = np.array(
            [parse_variant(c).position for c in candidate_ids], dtype=int
        )

        n_pick = min(n_select, len(candidate_ids))
        available = np.ones(len(candidate_ids), dtype=bool)
        site_count: dict[int, int] = {}
        selected: List[str] = []
        for _ in range(n_pick):
            penalties = np.array(
                [self.penalty * site_count.get(int(s), 0) for s in sites],
                dtype=float,
            )
            effective = score - penalties
            effective[~available] = -np.inf  # exclude already-picked candidates.
            # Stable argmax: ties resolve to the earliest candidate index.
            j = int(np.argmax(effective))
            available[j] = False
            site = int(sites[j])
            site_count[site] = site_count.get(site, 0) + 1
            selected.append(candidate_ids[j])
        return _validate_pick(selected, candidate_ids, n_select)


class ConstantLiarStrategy:
    """FolDE-style constant-liar batch selection (NAMED PARITY baseline).

    Source: FolDE (Roberts 2025) -- the constant-liar batch construction for
    PLM-guided directed evolution. This is a parity baseline so the
    mutation-coverage method can be compared against the established
    batch-diversity approach; it is NOT claimed as novel here.

    Mechanism (documented and exact). Fit the SAME core
    :class:`TopLayerRegressor` on the real measured set, then build the batch
    one pick at a time:

        1. Predict every remaining candidate; take the argmax (greedy pick).
        2. Append that candidate's embedding to a TEMPORARY training set with a
           LIE label and REFIT the regressor before the next pick.

    The lie (``lie="mean"``, the only supported value) is the running mean of
    the ORIGINAL measured ``y`` -- the classic constant-liar CL[mean]. It is a
    single constant fixed at the start of the batch (not the pick's own
    predicted value), so it is pessimistic relative to a high-scoring pick:
    adding a near-duplicate candidate at the constant mean drags the refit
    prediction DOWN around that pick, suppressing its embedding-space neighbours
    and pushing the next argmax toward a DIFFERENT region. The first pick is
    always the plain greedy argmax (no lie yet); the lie only changes picks 2+,
    which is exactly where naive top-N wastes a redundant batch.

    Determinism: the refit uses a fixed ``random_state`` and the picks are
    argmax over predictions, so the batch is reproducible. The passed-in
    ``rng`` is only consulted for the cold-start random fallback.
    """

    name = "constant_liar"

    def __init__(
        self,
        base: str = "greedy",
        lie: str = "mean",
        n_estimators: int = 100,
        random_state: int = 1,
    ) -> None:
        if base != "greedy":
            raise ValueError(
                f"ConstantLiarStrategy only supports base='greedy', got {base!r}."
            )
        if lie != "mean":
            raise ValueError(
                f"ConstantLiarStrategy only supports lie='mean', got {lie!r}."
            )
        self.base = base
        self.lie = lie
        self.n_estimators = n_estimators
        self.random_state = random_state

    def pick(
        self,
        measured_ids: Sequence[str],
        measured_y: Sequence[float],
        candidate_ids: Sequence[str],
        embeddings: pd.DataFrame,
        n_select: int,
        rng: np.random.Generator,
    ) -> List[str]:
        measured_ids = list(measured_ids)
        candidate_ids = list(candidate_ids)
        if len(measured_ids) == 0:
            # Same cold-start fallback as GreedyTopLayer: no labels -> random.
            return RandomStrategy().pick(
                measured_ids, measured_y, candidate_ids, embeddings, n_select, rng
            )

        y_train = np.asarray(measured_y, dtype=float)
        # The constant liar: a single value fixed at the start of the batch.
        lie_value = float(y_train.mean())

        # Running training matrices (real measured, grown with lie-labelled picks).
        X_train = list(embeddings.loc[measured_ids].to_numpy())
        y_list = list(y_train)

        X_cand_all = embeddings.loc[candidate_ids].to_numpy()
        n_pick = min(n_select, len(candidate_ids))
        available = np.ones(len(candidate_ids), dtype=bool)
        selected: List[str] = []

        for _ in range(n_pick):
            model = TopLayerRegressor(
                n_estimators=self.n_estimators, random_state=self.random_state
            )
            model.fit(np.asarray(X_train), np.asarray(y_list))
            predicted = model.predict(X_cand_all)
            predicted[~available] = -np.inf  # exclude already-picked candidates.
            j = int(np.argmax(predicted))
            available[j] = False
            selected.append(candidate_ids[j])
            # Append the pick with the LIE label and refit before the next pick.
            X_train.append(X_cand_all[j])
            y_list.append(lie_value)

        return _validate_pick(selected, candidate_ids, n_select)


# Convenience registry for the CLI / panel runner.
STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "greedy": GreedyTopLayer,
    "additive": AdditiveLinear,
    "ucb": UCBStrategy,
    "thompson": ThompsonStrategy,
    "coverage": CoverageBatchStrategy,
    "constant_liar": ConstantLiarStrategy,
}


def build_strategy(name: str):
    """Instantiate a strategy by its registry name (fail-fast on unknown)."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy {name!r}; available: {sorted(STRATEGY_REGISTRY)}."
        )
    return STRATEGY_REGISTRY[name]()
