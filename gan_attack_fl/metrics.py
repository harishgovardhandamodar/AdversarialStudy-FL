from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gan_attack_fl.federated import LogisticModel


@dataclass
class AttackMetrics:
    mean_distance: float
    covariance_distance: float
    nearest_neighbor_distance: float
    target_confidence_on_generated: float


def _frobenius_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def evaluate_attack(
    generated_x: np.ndarray,
    target_private_x: np.ndarray,
    victim_model: LogisticModel,
    target_class: int,
) -> AttackMetrics:
    g_mean = np.mean(generated_x, axis=0)
    t_mean = np.mean(target_private_x, axis=0)
    mean_distance = float(np.linalg.norm(g_mean - t_mean))

    g_cov = np.cov(generated_x, rowvar=False)
    t_cov = np.cov(target_private_x, rowvar=False)
    cov_distance = _frobenius_norm(g_cov, t_cov)

    # Approximate sample-level similarity by nearest-neighbor distance.
    dists = np.sqrt(np.sum((generated_x[:, None, :] - target_private_x[None, :, :]) ** 2, axis=2))
    nn_dist = float(np.mean(np.min(dists, axis=1)))

    p = victim_model.predict_proba(generated_x)
    if target_class == 0:
        p = 1.0 - p
    conf = float(np.mean(p))
    return AttackMetrics(
        mean_distance=mean_distance,
        covariance_distance=cov_distance,
        nearest_neighbor_distance=nn_dist,
        target_confidence_on_generated=conf,
    )
