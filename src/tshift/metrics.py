import numpy as np
from scipy.stats import spearmanr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(spearmanr(y_true, y_pred).statistic)


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, quantile: float) -> float:
    threshold = np.quantile(y_true, quantile)
    top = np.argsort(-y_pred, kind="stable")[:k]
    return float(np.mean(y_true[top] >= threshold))


def discovery_rate(y_true: np.ndarray, y_pred: np.ndarray, k: int, ceiling: float) -> float:
    top = np.argsort(-y_pred, kind="stable")[:k]
    return float(np.mean(y_true[top] > ceiling))
