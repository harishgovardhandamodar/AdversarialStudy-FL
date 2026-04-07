from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FLClientData:
    x: np.ndarray
    y: np.ndarray


@dataclass
class GANAttackDataset:
    clients: list[FLClientData]
    target_client_id: int
    target_private_class: int
    target_private_x: np.ndarray
    attacker_public_x: np.ndarray
    attacker_public_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def _make_class_samples(
    n: int,
    mean: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.multivariate_normal(mean=mean, cov=cov, size=n)


def make_federated_gan_dataset(
    num_clients: int,
    points_per_client: int,
    test_points: int,
    target_client_id: int,
    target_private_class: int,
    seed: int,
) -> GANAttackDataset:
    rng = np.random.default_rng(seed)

    means = {
        0: np.array([-2.5, 0.0]),
        1: np.array([2.5, 0.0]),
    }
    cov = np.array([[0.9, 0.2], [0.2, 0.9]])

    clients: list[FLClientData] = []
    target_private_x = np.empty((0, 2))
    for cid in range(num_clients):
        if cid == target_client_id:
            # Target client has class imbalance with more of target_private_class.
            n_private = int(points_per_client * 0.8)
            n_other = points_per_client - n_private
            x_private = _make_class_samples(n_private, means[target_private_class], cov, rng)
            x_other = _make_class_samples(n_other, means[1 - target_private_class], cov, rng)
            y_private = np.full(n_private, target_private_class, dtype=np.int64)
            y_other = np.full(n_other, 1 - target_private_class, dtype=np.int64)
            x = np.vstack([x_private, x_other])
            y = np.concatenate([y_private, y_other])
            target_private_x = x_private
        else:
            half = points_per_client // 2
            x0 = _make_class_samples(half, means[0], cov, rng)
            x1 = _make_class_samples(points_per_client - half, means[1], cov, rng)
            y0 = np.zeros(len(x0), dtype=np.int64)
            y1 = np.ones(len(x1), dtype=np.int64)
            x = np.vstack([x0, x1])
            y = np.concatenate([y0, y1])
        idx = rng.permutation(len(x))
        clients.append(FLClientData(x=x[idx], y=y[idx]))

    # Attacker public data is intentionally smaller and slightly shifted.
    shift = np.array([0.4, -0.2])
    pub0 = _make_class_samples(120, means[0] + shift, cov * 1.1, rng)
    pub1 = _make_class_samples(120, means[1] + shift, cov * 1.1, rng)
    attacker_public_x = np.vstack([pub0, pub1])
    attacker_public_y = np.concatenate([np.zeros(len(pub0), dtype=np.int64), np.ones(len(pub1), dtype=np.int64)])

    t0 = _make_class_samples(test_points // 2, means[0], cov, rng)
    t1 = _make_class_samples(test_points - len(t0), means[1], cov, rng)
    test_x = np.vstack([t0, t1])
    test_y = np.concatenate([np.zeros(len(t0), dtype=np.int64), np.ones(len(t1), dtype=np.int64)])
    tidx = rng.permutation(len(test_x))
    test_x, test_y = test_x[tidx], test_y[tidx]

    return GANAttackDataset(
        clients=clients,
        target_client_id=target_client_id,
        target_private_class=target_private_class,
        target_private_x=target_private_x,
        attacker_public_x=attacker_public_x,
        attacker_public_y=attacker_public_y,
        test_x=test_x,
        test_y=test_y,
    )
