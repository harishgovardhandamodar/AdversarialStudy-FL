from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VFLDataBundle:
    x_a_train: np.ndarray
    x_b_train: np.ndarray
    y_train: np.ndarray
    x_a_test: np.ndarray
    x_b_test: np.ndarray
    y_test: np.ndarray
    x_a_shadow: np.ndarray
    x_b_shadow: np.ndarray


def _make_partition(
    n: int,
    d_a: int,
    d_b: int,
    latent_dim: int,
    corr_strength: float,
    noise_std: float,
    a_map: np.ndarray,
    b_map: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    z = rng.normal(0.0, 1.0, size=(n, latent_dim))

    shared_a = corr_strength * (z @ a_map)
    shared_b = corr_strength * (z @ b_map)
    private_a = (1.0 - corr_strength) * rng.normal(0.0, 1.0, size=(n, d_a))
    private_b = (1.0 - corr_strength) * rng.normal(0.0, 1.0, size=(n, d_b))
    x_a = shared_a + private_a + rng.normal(0.0, noise_std, size=(n, d_a))
    x_b = shared_b + private_b + rng.normal(0.0, noise_std, size=(n, d_b))
    return x_a, x_b


def _make_labels(
    x_a: np.ndarray,
    x_b: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    logits = x_a @ wa + x_b @ wb + rng.normal(0.0, 0.2, size=x_a.shape[0])
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (rng.random(size=probs.shape[0]) < probs).astype(np.float64)


def make_vfl_data(
    n_train: int,
    n_test: int,
    n_shadow: int,
    d_a: int,
    d_b: int,
    latent_dim: int,
    corr_strength: float,
    noise_std: float,
    seed: int,
) -> VFLDataBundle:
    rng = np.random.default_rng(seed)
    wa = rng.normal(0.0, 0.8, size=d_a)
    wb = rng.normal(0.0, 0.8, size=d_b)
    a_map = rng.normal(0.0, 1.0, size=(latent_dim, d_a))
    b_map = rng.normal(0.0, 1.0, size=(latent_dim, d_b))
    x_a_train, x_b_train = _make_partition(
        n=n_train,
        d_a=d_a,
        d_b=d_b,
        latent_dim=latent_dim,
        corr_strength=corr_strength,
        noise_std=noise_std,
        a_map=a_map,
        b_map=b_map,
        rng=rng,
    )
    x_a_test, x_b_test = _make_partition(
        n=n_test,
        d_a=d_a,
        d_b=d_b,
        latent_dim=latent_dim,
        corr_strength=corr_strength,
        noise_std=noise_std,
        a_map=a_map,
        b_map=b_map,
        rng=rng,
    )
    x_a_shadow, x_b_shadow = _make_partition(
        n=n_shadow,
        d_a=d_a,
        d_b=d_b,
        latent_dim=latent_dim,
        corr_strength=corr_strength,
        noise_std=noise_std,
        a_map=a_map,
        b_map=b_map,
        rng=rng,
    )
    y_train = _make_labels(x_a_train, x_b_train, wa=wa, wb=wb, rng=rng)
    y_test = _make_labels(x_a_test, x_b_test, wa=wa, wb=wb, rng=rng)
    return VFLDataBundle(
        x_a_train=x_a_train,
        x_b_train=x_b_train,
        y_train=y_train,
        x_a_test=x_a_test,
        x_b_test=x_b_test,
        y_test=y_test,
        x_a_shadow=x_a_shadow,
        x_b_shadow=x_b_shadow,
    )
