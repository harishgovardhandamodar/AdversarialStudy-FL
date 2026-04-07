from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class VFLLogisticModel:
    w_a: np.ndarray
    w_b: np.ndarray
    bias: float

    def logits(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        return x_a @ self.w_a + x_b @ self.w_b + self.bias

    def predict_proba(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(x_a, x_b))

    def accuracy(self, x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(x_a, x_b)
        pred = (p >= 0.5).astype(float)
        return float(np.mean(pred == y))


def train_vfl_logistic(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
    seed: int,
) -> VFLLogisticModel:
    rng = np.random.default_rng(seed)
    w_a = rng.normal(0.0, 0.05, size=x_a.shape[1])
    w_b = rng.normal(0.0, 0.05, size=x_b.shape[1])
    bias = 0.0
    n = len(y)

    for _ in range(epochs):
        logits = x_a @ w_a + x_b @ w_b + bias
        p = _sigmoid(logits)
        err = p - y
        grad_a = (x_a.T @ err) / n + l2 * w_a
        grad_b = (x_b.T @ err) / n + l2 * w_b
        grad_bias = float(np.mean(err))

        w_a -= lr * grad_a
        w_b -= lr * grad_b
        bias -= lr * grad_bias

    return VFLLogisticModel(w_a=w_a, w_b=w_b, bias=bias)
