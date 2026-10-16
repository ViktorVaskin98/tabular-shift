import numpy as np
import pytest

from tshift.data import Superconductors
from tshift.metrics import discovery_rate, precision_at_k, rmse, spearman
from tshift.splits import (
    cluster_labels,
    group_random,
    leave_family_out,
    row_random,
    tc_extrapolation,
)


@pytest.fixture
def toy() -> Superconductors:
    rng = np.random.default_rng(0)
    n = 600
    formula = np.array([f"M{i % 200}" for i in range(n)], dtype=object)
    family = np.where(np.arange(n) % 3 == 0, "cuprate", "other").astype(object)
    target = rng.gamma(2.0, 15.0, size=n)
    return Superconductors(
        features=rng.normal(size=(n, 5)).astype(np.float32),
        target=target,
        formula=formula,
        family=family,
        feature_names=[f"f{i}" for i in range(5)],
    )


def test_group_random_never_shares_formulas(toy: Superconductors) -> None:
    split = group_random(toy, seed=1, n_train=10_000, n_test=10_000)
    assert not set(toy.formula[split.train]) & set(toy.formula[split.test])
    assert len(split.train) + len(split.test) == len(toy.target)


def test_row_random_does_share_formulas(toy: Superconductors) -> None:
    split = row_random(toy, seed=1, n_train=10_000, n_test=10_000)
    assert set(toy.formula[split.train]) & set(toy.formula[split.test])


def test_leave_family_out_holds_family(toy: Superconductors) -> None:
    split = leave_family_out(toy, "cuprate", seed=0, n_train=50, n_test=40)
    assert (toy.family[split.test] == "cuprate").all()
    assert (toy.family[split.train] != "cuprate").all()
    assert len(split.train) == 50 and len(split.test) == 40


def test_tc_extrapolation_caps_training_target(toy: Superconductors) -> None:
    split = tc_extrapolation(toy, seed=0, n_train=10_000, n_test=10_000)
    ceiling = np.quantile(toy.target, 0.8)
    assert toy.target[split.train].max() <= ceiling
    assert toy.target[split.test].max() > ceiling
    assert not set(toy.formula[split.train]) & set(toy.formula[split.test])


def test_cluster_labels_are_constant_within_formula(toy: Superconductors) -> None:
    labels = cluster_labels(toy, n_clusters=4)
    for formula in np.unique(toy.formula)[:20]:
        assert len(set(labels[toy.formula == formula])) == 1


def test_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rmse(y, y) == 0.0
    assert spearman(y, y * 10) == pytest.approx(1.0)
    assert precision_at_k(y, y, k=1, quantile=0.8) == 1.0
    assert precision_at_k(y, -y, k=1, quantile=0.8) == 0.0
    assert discovery_rate(y, y, k=2, ceiling=3.5) == 1.0
