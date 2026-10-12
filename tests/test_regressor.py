import numpy as np
import pytest

from plm_evolve.regressor import TopLayerRegressor


def _toy_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 8))
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2]
    return X, y


def test_fit_predict_shape():
    X, y = _toy_data()
    reg = TopLayerRegressor().fit(X, y)
    preds = reg.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)


def test_determinism_same_seed():
    X, y = _toy_data()
    p1 = TopLayerRegressor(random_state=1).fit(X, y).predict(X)
    p2 = TopLayerRegressor(random_state=1).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TopLayerRegressor().predict(np.zeros((2, 8)))


def test_predict_with_uncertainty_before_fit_raises():
    with pytest.raises(RuntimeError):
        TopLayerRegressor().predict_with_uncertainty(np.zeros((2, 8)))


def test_predict_with_uncertainty_shape_and_mean_matches_predict():
    X, y = _toy_data()
    reg = TopLayerRegressor().fit(X, y)
    mean, std = reg.predict_with_uncertainty(X)
    # Returns (mean, std) of the right shape.
    assert mean.shape == (X.shape[0],)
    assert std.shape == (X.shape[0],)
    # std is the per-tree spread: non-negative elementwise.
    assert np.all(std >= 0.0)
    # mean is byte-identical to predict() (the greedy ranking signal). This is
    # what makes UCB(beta=0) reduce exactly to greedy.
    np.testing.assert_array_equal(mean, reg.predict(X))


def test_predict_with_uncertainty_std_positive_where_data_disagrees():
    # On a non-trivial regression the trees disagree somewhere -> some std > 0.
    X, y = _toy_data(seed=3)
    reg = TopLayerRegressor().fit(X, y)
    _, std = reg.predict_with_uncertainty(X)
    assert np.any(std > 0.0)
