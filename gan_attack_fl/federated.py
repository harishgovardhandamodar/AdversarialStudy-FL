from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gan_attack_fl.data import FLClientData


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LogisticModel:
    w: np.ndarray
    b: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(x @ self.w + self.b)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(x)
        pred = (p >= 0.5).astype(np.int64)
        return float(np.mean(pred == y))


def _local_train(
    model: LogisticModel,
    client: FLClientData,
    lr: float,
    steps: int,
    l2: float,
) -> LogisticModel:
    w = model.w.copy()
    b = float(model.b)
    x, y = client.x, client.y.astype(float)
    n = len(y)
    for _ in range(steps):
        p = _sigmoid(x @ w + b)
        err = p - y
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
    return LogisticModel(w=w, b=b)


def train_fedavg_logistic(
    clients: list[FLClientData],
    rounds: int,
    local_steps: int,
    lr: float,
    l2: float,
    seed: int,
) -> LogisticModel:
    rng = np.random.default_rng(seed)
    model = LogisticModel(w=rng.normal(0.0, 0.05, size=2), b=0.0)
    for _ in range(rounds):
        locals_: list[LogisticModel] = []
        for c in clients:
            locals_.append(_local_train(model, c, lr=lr, steps=local_steps, l2=l2))
        model = LogisticModel(
            w=np.mean([m.w for m in locals_], axis=0),
            b=float(np.mean([m.b for m in locals_])),
        )
    return model
