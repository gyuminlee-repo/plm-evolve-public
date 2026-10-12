"""Top-layer regressor: a RandomForest over frozen PLM embeddings.

The EVOLVEpro core principle keeps the PLM frozen and trains only a light
top-layer regressor on the handful of measured variants. This package fixes
that regressor to a RandomForest (the single supported model).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class TopLayerRegressor:
    """RandomForest regressor on top of frozen embeddings.

    Args:
        n_estimators: number of trees (default 100, matching the method).
        random_state: RNG seed for reproducibility (default 1).
        criterion: split-quality function passed to ``RandomForestRegressor``
            (default ``"friedman_mse"``, matching the original EVOLVEpro
            top-layer, which sets it explicitly; the sklearn default is
            ``"squared_error"``). The two agree on the best split at a single
            node (variance decomposition) but weight child sizes differently,
            so they diverge on near-tied splits and give slightly different
            rankings; on an in-house round-1 dataset ~1/3 of the top-12 next-round
            proposal changes between them, while held-out Spearman moves only
            ~0.001. ``friedman_mse`` keeps the next-round proposal byte-level
            faithful to a reference EVOLVEpro run on identical embeddings,
            seed, and measured set.
        n_jobs: trees to build in parallel (default None = 1 core). The forest is
            deterministic in ``random_state`` regardless of ``n_jobs``, so setting
            ``n_jobs=-1`` only speeds up fitting and leaves predictions identical.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 1,
        criterion: str = "friedman_mse",
        n_jobs: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion
        self.n_jobs = n_jobs
        self._model: Optional[RandomForestRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TopLayerRegressor":
        """Fit on measured variants. ``X`` is (n, dim), ``y`` is (n,)."""
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            criterion=self.criterion,
            n_jobs=self.n_jobs,
        )
        self._model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean activity for ``X`` (n, dim) -> (n,)."""
        if self._model is None:
            raise RuntimeError("TopLayerRegressor.predict called before fit.")
        return self._model.predict(np.asarray(X))

    def predict_with_uncertainty(self, X: np.ndarray):
        """Predict mean activity and per-prediction epistemic uncertainty.

        For the RandomForest the predictive distribution is approximated by the
        spread across individual trees (``estimators_``): each tree gives one
        prediction per row, the mean over trees is the point estimate and the
        std over trees is the epistemic (model-disagreement) uncertainty, which
        is large where the training data is sparse and small where it is dense.

        The returned ``mean`` is exactly :meth:`predict` (byte-identical), NOT
        the recomputed per-tree mean. sklearn's ``predict`` and a manual
        ``np.mean`` over ``estimators_`` agree to within floating accumulation
        order; using :meth:`predict` keeps an acquisition function such as UCB
        with ``beta=0`` byte-identical to the greedy ranking (a tested
        invariant). The per-tree mean is still checked to match (``allclose``)
        so a backend that diverged would fail fast.

        Args:
            X: feature matrix ``(n, dim)``.

        Returns:
            ``(mean, std)``, both shape ``(n,)``. ``std`` is the population std
            (``ddof=0``) over trees and is ``>= 0`` elementwise.
        """
        if self._model is None:
            raise RuntimeError(
                "TopLayerRegressor.predict_with_uncertainty called before fit."
            )
        X = np.asarray(X)
        mean = self.predict(X)  # byte-identical to greedy's ranking signal.
        per_tree = np.stack(
            [est.predict(X) for est in self._model.estimators_]
        )
        # The per-tree mean must match sklearn's predict() (sanity, fail-fast):
        # a divergence means the std would not correspond to this mean.
        assert np.allclose(per_tree.mean(axis=0), mean), (
            "Per-tree mean diverged from RandomForest.predict(); uncertainty "
            "would not match the point estimate."
        )
        std = per_tree.std(axis=0)  # ddof=0 -> always >= 0.
        return mean, std
