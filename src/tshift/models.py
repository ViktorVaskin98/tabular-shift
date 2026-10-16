from collections.abc import Callable
from typing import Protocol

import numpy as np
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Regressor(Protocol):
    def fit(self, features: np.ndarray, target: np.ndarray) -> object: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


def _tabicl() -> Regressor:
    from tabicl import TabICLRegressor

    return TabICLRegressor(device="cpu")


def _tabdpt() -> Regressor:
    from tabdpt import TabDPTRegressor

    return TabDPTRegressor(device="cpu", compile=False, verbose=False)


def _catboost() -> Regressor:
    return CatBoostRegressor(iterations=1000, verbose=False, random_seed=0, thread_count=-1)


def _ridge() -> Regressor:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


MODELS: dict[str, Callable[[], Regressor]] = {
    "ridge": _ridge,
    "catboost": _catboost,
    "tabicl": _tabicl,
    "tabdpt": _tabdpt,
}
