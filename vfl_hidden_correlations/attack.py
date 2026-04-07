from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from vfl_hidden_correlations.model import VFLLogisticModel


@dataclass
class ReconstructionOutput:
    x_b_hat: np.ndarray
    projection_residual_mse: float


def _fit_linear_map(x_a_shadow: np.ndarray, x_b_shadow: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x_a_shadow.T @ x_a_shadow
    reg = ridge * np.eye(xtx.shape[0])
    return np.linalg.solve(xtx + reg, x_a_shadow.T @ x_b_shadow)


def reconstruct_party_b_features(
    model: VFLLogisticModel,
    x_a_target: np.ndarray,
    victim_logits: np.ndarray,
    x_a_shadow: np.ndarray,
    x_b_shadow: np.ndarray,
    ridge: float = 1e-2,
) -> ReconstructionOutput:
    """
    Post-training attack:
    1) Learn hidden correlation mapping x_b ~= x_a @ W from public/shadow data.
    2) Constrain each reconstruction to satisfy observed logit contribution.
    """
    w_corr = _fit_linear_map(x_a_shadow=x_a_shadow, x_b_shadow=x_b_shadow, ridge=ridge)
    x_b_init = x_a_target @ w_corr

    inferred_b_contrib = victim_logits - (x_a_target @ model.w_a + model.bias)
    w_b = model.w_b
    denom = float(np.dot(w_b, w_b) + 1e-12)
    correction = ((inferred_b_contrib - (x_b_init @ w_b)) / denom)[:, None] * w_b[None, :]
    x_b_hat = x_b_init + correction

    residual = inferred_b_contrib - (x_b_hat @ w_b)
    return ReconstructionOutput(
        x_b_hat=x_b_hat,
        projection_residual_mse=float(np.mean(residual**2)),
    )
