"""The EVOLVEpro core loop: train on measured, predict the rest, rank, propose.

One call to :func:`run_round` performs a single active-learning round:
train the top-layer regressor on the measured variants, predict activity for
every variant in the embedding table, rank all candidates by predicted
activity, and propose the top-N not-yet-measured variants for the next
experimental round.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from .regressor import TopLayerRegressor
from .rounds import build_training_table, load_round_data
from .variants import WT_TOKEN


@dataclass
class RoundResult:
    """Outcome of one directed-evolution round.

    Attributes:
        ranked: all candidates sorted by predicted activity (desc) with
            columns ``variant``, ``predicted``, ``measured`` and ``activity``
            (NaN where unmeasured).
        next_round: the top-N not-yet-measured candidates for the next round.
        n_train: number of measured variants used for training.
    """

    ranked: pd.DataFrame
    next_round: pd.DataFrame
    n_train: int


def run_round(
    embeddings_df: pd.DataFrame,
    round_files: Sequence[Union[str, Path]],
    wt_fasta: Union[str, Path],
    top_n: int = 12,
    regressor: Optional[TopLayerRegressor] = None,
    out_dir: Optional[Union[str, Path]] = None,
    activity_column: str = "activity",
    variant_column: str = "Variant",
    acquisition: str = "greedy",
    beta: float = 1.0,
    penalty: float = 1.0,
    seed: int = 0,
    include_wt_in_training: bool = True,
    wt_activity: float = 1.0,
) -> RoundResult:
    """Run one active-learning round.

    Args:
        embeddings_df: DataFrame indexed by variant id, columns = dims.
        round_files: measured-round files (Excel/CSV).
        wt_fasta: wild-type FASTA (for variant validation).
        top_n: number of next-round candidates to propose.
        regressor: optional pre-built :class:`TopLayerRegressor`.
        out_dir: if given, write ``ranked.csv`` and ``next_round.csv`` there.
        acquisition: OPT-IN next-round selection. ``"greedy"`` (DEFAULT) is the
            original behaviour, byte-identical -- top-N by predicted mean.
            ``"ucb"``/``"thompson"``/``"coverage"`` instead select via the
            corresponding benchmark strategy's ``pick()`` over the not-measured,
            non-WT candidates. ``ranked`` is built the SAME way (predicted mean)
            in every case; only ``next_round`` selection changes.
        beta: exploration weight for ``ucb`` and ``coverage`` (ignored by
            ``greedy``/``thompson``).
        penalty: mutation-site coverage penalty for ``coverage`` (ignored
            otherwise).
        seed: RNG seed for the strategy's numpy ``Generator`` (relevant for
            ``thompson`` determinism and cold-start fallbacks; the ``greedy``
            default never consumes it).
        include_wt_in_training: add the WT row (``embeddings_df.loc["WT"]``) to
            the training set with ``wt_activity`` (default ``True``, matching the
            original EVOLVEpro, whose ``create_iteration_dataframes`` anchors WT
            at iteration 0 so it is always trained on). WT lands in embedding-
            index order like every other training row, so the fit reproduces
            EVOLVEpro byte-for-byte given the same embeddings and sklearn. No-op
            if WT is not in the embeddings or is already a measured variant.
            Set ``False`` to train on the measured mutants only.
        wt_activity: activity value for the injected WT row (default ``1.0``,
            the WT-normalized reference EVOLVEpro assumes; only used when
            ``include_wt_in_training``).

    Returns:
        :class:`RoundResult`.

    Raises:
        ValueError: ``acquisition`` is not one of
            ``{"greedy", "ucb", "thompson", "coverage"}``.
    """
    measured = load_round_data(
        round_files,
        wt_fasta,
        variant_column=variant_column,
        activity_column=activity_column,
    )
    # EVOLVEpro anchors WT (activity 1.0, iteration 0) in every round's training
    # set. Add it here so the RF trains on WT too; build_training_table places it
    # in embedding-index order, matching EVOLVEpro byte-for-byte. No-op if WT is
    # absent from the embeddings or already a measured variant.
    if (
        include_wt_in_training
        and WT_TOKEN in embeddings_df.index
        and WT_TOKEN not in set(measured["variant"])
    ):
        measured = pd.concat(
            [measured, pd.DataFrame({"variant": [WT_TOKEN], "activity": [wt_activity]})],
            ignore_index=True,
        )
    table = build_training_table(measured, embeddings_df.index.tolist())

    train_mask = table["measured"].to_numpy()
    # The auto-injected WT is not a real measurement: require at least one
    # non-WT measured variant, else there is nothing meaningful to train on
    # (WT alone is a single anchor point).
    non_wt_measured = (train_mask & (table["variant"].to_numpy() != WT_TOKEN)).sum()
    if non_wt_measured == 0:
        raise ValueError(
            "No measured variant matched the embedding index; nothing to train "
            "on. Check that round variant tokens match the embedding ids."
        )

    train_variants = table.loc[train_mask, "variant"]
    X_train = embeddings_df.loc[train_variants].to_numpy()
    y_train = table.loc[train_mask, "activity"].to_numpy()

    model = regressor if regressor is not None else TopLayerRegressor()
    model.fit(X_train, y_train)

    # Predict on EVERY variant so the ranked table includes measured rows too
    # (mirrors the reference df_all), aligned to the embedding row order.
    X_all = embeddings_df.loc[table["variant"]].to_numpy()
    predicted = model.predict(X_all)

    ranked = pd.DataFrame(
        {
            "variant": table["variant"].to_numpy(),
            "predicted": predicted,
            "measured": table["measured"].to_numpy(),
            "activity": table["activity"].to_numpy(),
        }
    ).sort_values("predicted", ascending=False, kind="stable").reset_index(drop=True)

    # Next round: top-N among NOT-yet-measured candidates, WT excluded.
    candidates = ranked[(~ranked["measured"]) & (ranked["variant"] != WT_TOKEN)]
    if acquisition == "greedy":
        # DEFAULT: byte-identical to the original behaviour. Do NOT touch.
        next_round = candidates.head(top_n).reset_index(drop=True)
    elif acquisition in ("ucb", "thompson", "coverage"):
        # Lazy import: keep the default path's import surface untouched and
        # avoid any benchmark<->evolve import cycle. Reuse the strategy
        # classes (do NOT reimplement UCB/coverage); build_strategy() cannot
        # thread beta/penalty, so construct directly.
        import numpy as np

        from .benchmark.strategies import (
            CoverageBatchStrategy,
            ThompsonStrategy,
            UCBStrategy,
        )

        if acquisition == "ucb":
            strategy = UCBStrategy(beta=beta)
        elif acquisition == "thompson":
            strategy = ThompsonStrategy()
        else:  # "coverage"
            # base="ucb" so the threaded `beta` is LIVE (under base="greedy"
            # beta is silently ignored -- against the no-silent-fallback rule).
            strategy = CoverageBatchStrategy(base="ucb", beta=beta, penalty=penalty)

        # Train the strategy on the SAME measured set run_round trained on
        # (WT included), so its internal fit matches `ranked`.
        measured_ids = train_variants.tolist()
        measured_y = y_train.tolist()
        candidate_ids = candidates["variant"].tolist()
        picked = strategy.pick(
            measured_ids=measured_ids,
            measured_y=measured_y,
            candidate_ids=candidate_ids,
            embeddings=embeddings_df,
            n_select=top_n,
            rng=np.random.default_rng(seed),
        )
        # Rebuild next_round rows from `ranked` (preserve the 4-col schema) in
        # the strategy's pick order.
        picked_rows = ranked.set_index("variant").loc[picked].reset_index()
        next_round = picked_rows[ranked.columns].reset_index(drop=True)
    else:
        raise ValueError(
            f"Unknown acquisition {acquisition!r}; expected one of "
            f"'greedy', 'ucb', 'thompson', 'coverage'."
        )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out_dir / "ranked.csv", index=False)
        next_round.to_csv(out_dir / "next_round.csv", index=False)

    return RoundResult(
        ranked=ranked, next_round=next_round, n_train=int(train_mask.sum())
    )
