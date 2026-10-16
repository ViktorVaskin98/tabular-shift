from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tshift.data import Superconductors

TC_CEILING_QUANTILE = 0.8


@dataclass(frozen=True)
class Split:
    name: str
    seed: int
    train: np.ndarray
    test: np.ndarray


def _subsample(index: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(index, size=min(size, len(index)), replace=False))


def _group_codes(formula: np.ndarray) -> np.ndarray:
    return np.unique(formula, return_inverse=True)[1]


def _groups_to_rows(groups: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(codes, groups))


def row_random(data: Superconductors, seed: int, n_train: int, n_test: int) -> Split:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.target))
    half = len(order) // 2
    return Split(
        "row-random",
        seed,
        _subsample(order[:half], n_train, rng),
        _subsample(order[half:], n_test, rng),
    )


def group_random(data: Superconductors, seed: int, n_train: int, n_test: int) -> Split:
    rng = np.random.default_rng(seed)
    codes = _group_codes(data.formula)
    groups = rng.permutation(codes.max() + 1)
    half = len(groups) // 2
    train = _groups_to_rows(groups[:half], codes)
    test = _groups_to_rows(groups[half:], codes)
    return Split(
        "group-random", seed, _subsample(train, n_train, rng), _subsample(test, n_test, rng)
    )


def leave_family_out(
    data: Superconductors, family: str, seed: int, n_train: int, n_test: int
) -> Split:
    rng = np.random.default_rng(seed)
    held = data.family == family
    return Split(
        f"no-{family}",
        seed,
        _subsample(np.flatnonzero(~held), n_train, rng),
        _subsample(np.flatnonzero(held), n_test, rng),
    )


def tc_extrapolation(data: Superconductors, seed: int, n_train: int, n_test: int) -> Split:
    rng = np.random.default_rng(seed)
    ceiling = np.quantile(data.target, TC_CEILING_QUANTILE)
    codes = _group_codes(data.formula)
    group_max = np.full(codes.max() + 1, -np.inf)
    np.maximum.at(group_max, codes, data.target)
    groups = rng.permutation(codes.max() + 1)
    half = len(groups) // 2
    train_groups = groups[:half][group_max[groups[:half]] <= ceiling]
    train = _groups_to_rows(train_groups, codes)
    test = _groups_to_rows(groups[half:], codes)
    return Split(
        "tc-extrapolation", seed, _subsample(train, n_train, rng), _subsample(test, n_test, rng)
    )


def cluster_labels(data: Superconductors, n_clusters: int = 8, seed: int = 0) -> np.ndarray:
    scaled = StandardScaler().fit_transform(data.features)
    codes = _group_codes(data.formula)
    _, first = np.unique(codes, return_index=True)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(scaled[first])
    return model.labels_[codes]


def leave_cluster_out(
    labels: np.ndarray, cluster: int, seed: int, n_train: int, n_test: int
) -> Split:
    rng = np.random.default_rng(seed)
    held = labels == cluster
    return Split(
        f"cluster-{cluster}",
        seed,
        _subsample(np.flatnonzero(~held), n_train, rng),
        _subsample(np.flatnonzero(held), n_test, rng),
    )


def all_splits(
    data: Superconductors,
    seeds: list[int],
    n_train: int,
    n_test: int,
    clusters: list[int],
) -> Iterator[Split]:
    labels = cluster_labels(data)
    for seed in seeds:
        yield row_random(data, seed, n_train, n_test)
        yield group_random(data, seed, n_train, n_test)
        yield leave_family_out(data, "cuprate", seed, n_train, n_test)
        yield leave_family_out(data, "iron", seed, n_train, n_test)
        yield tc_extrapolation(data, seed, n_train, n_test)
        for cluster in clusters:
            yield leave_cluster_out(labels, cluster, seed, n_train, n_test)
