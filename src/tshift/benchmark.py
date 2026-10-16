import csv
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from tshift.data import Superconductors
from tshift.metrics import discovery_rate, precision_at_k, rmse, spearman
from tshift.models import MODELS
from tshift.splits import TC_CEILING_QUANTILE, Split

FIELDS = [
    "split",
    "seed",
    "model",
    "n_train",
    "n_test",
    "rmse",
    "spearman",
    "precision_at_50",
    "discovery_at_50",
    "max_pred",
    "max_train_target",
    "seconds",
]


def _done(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as source:
        return {(r["split"], int(r["seed"]), r["model"]) for r in csv.DictReader(source)}


def run_benchmark(
    data: Superconductors, splits: Iterable[Split], models: Sequence[str], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out_path)
    ceiling = float(np.quantile(data.target, TC_CEILING_QUANTILE))
    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for split in splits:
            x_train, y_train = data.features[split.train], data.target[split.train]
            x_test, y_test = data.features[split.test], data.target[split.test]
            for name in models:
                if (split.name, split.seed, name) in done:
                    continue
                started = time.time()
                model = MODELS[name]()
                model.fit(x_train, y_train)
                prediction = np.asarray(model.predict(x_test), dtype=float).ravel()
                row = {
                    "split": split.name,
                    "seed": split.seed,
                    "model": name,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "rmse": rmse(y_test, prediction),
                    "spearman": spearman(y_test, prediction),
                    "precision_at_50": precision_at_k(y_test, prediction, 50, 0.9),
                    "discovery_at_50": discovery_rate(y_test, prediction, 50, ceiling),
                    "max_pred": float(prediction.max()),
                    "max_train_target": float(y_train.max()),
                    "seconds": round(time.time() - started, 1),
                }
                writer.writerow(row)
                sink.flush()
                print(
                    f"{split.name:16s} seed={split.seed} {name:8s} "
                    f"rmse={row['rmse']:.2f} rho={row['spearman']:.3f} t={row['seconds']}s",
                    flush=True,
                )
