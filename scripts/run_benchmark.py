import argparse
from pathlib import Path

from tshift.benchmark import run_benchmark
from tshift.data import load_superconductors
from tshift.models import MODELS
from tshift.splits import all_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="TFM vs GBDT under distribution shift")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/raw.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    data = load_superconductors(args.raw)
    splits = all_splits(data, args.seeds, args.n_train, args.n_test, args.clusters)
    run_benchmark(data, splits, args.models, args.out)


if __name__ == "__main__":
    main()
