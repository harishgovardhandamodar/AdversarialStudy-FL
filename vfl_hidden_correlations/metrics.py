from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ReconstructionMetrics:
    mse: float
    mean_feature_corr: float
    sensitive_attribute_accuracy: float


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_reconstruction(x_b_true: np.ndarray, x_b_hat: np.ndarray) -> ReconstructionMetrics:
    mse = float(np.mean((x_b_true - x_b_hat) ** 2))
    corrs = [_corr(x_b_true[:, j], x_b_hat[:, j]) for j in range(x_b_true.shape[1])]
    mean_feature_corr = float(np.mean(corrs))

    # Example "sensitive attribute": sign of first private feature.
    true_sensitive = (x_b_true[:, 0] > 0).astype(int)
    pred_sensitive = (x_b_hat[:, 0] > 0).astype(int)
    sens_acc = float(np.mean(true_sensitive == pred_sensitive))
    return ReconstructionMetrics(
        mse=mse,
        mean_feature_corr=mean_feature_corr,
        sensitive_attribute_accuracy=sens_acc,
    )
