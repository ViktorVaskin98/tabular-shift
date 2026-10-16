from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Superconductors:
    features: np.ndarray
    target: np.ndarray
    formula: np.ndarray
    family: np.ndarray
    feature_names: list[str]


def material_family(composition: pd.DataFrame) -> np.ndarray:
    cuprate = (composition["Cu"] > 0) & (composition["O"] > 0)
    iron = (composition["Fe"] > 0) & (
        (composition["As"] > 0) | (composition["Se"] > 0) | (composition["P"] > 0)
    )
    family = np.full(len(composition), "other", dtype=object)
    family[iron.to_numpy()] = "iron"
    family[cuprate.to_numpy()] = "cuprate"
    return family


def load_superconductors(raw_dir: Path) -> Superconductors:
    features = pd.read_csv(raw_dir / "train.csv")
    composition = pd.read_csv(raw_dir / "unique_m.csv")
    if len(features) != len(composition):
        raise ValueError("train.csv and unique_m.csv are not aligned")
    target = features.pop("critical_temp").to_numpy(dtype=float)
    return Superconductors(
        features=features.to_numpy(dtype=np.float32),
        target=target,
        formula=composition["material"].to_numpy(),
        family=material_family(composition),
        feature_names=list(features.columns),
    )
