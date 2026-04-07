from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gan_attack_fl.federated import LogisticModel


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GANAttackResult:
    generated_x: np.ndarray
    final_disc_loss: float
    final_gen_loss: float


def run_gan_attack(
    victim_model: LogisticModel,
    public_x: np.ndarray,
    target_class: int,
    noise_dim: int,
    attack_steps: int,
    batch_size: int,
    lr_g: float,
    lr_d: float,
    seed: int,
) -> GANAttackResult:
    """
    Toy GAN attack:
    - Discriminator learns real(public target-class) vs generated.
    - Generator fools discriminator and maximizes victim target-class confidence.
    """
    rng = np.random.default_rng(seed)
    d_w = rng.normal(0.0, 0.1, size=2)
    d_b = 0.0

    g_w = rng.normal(0.0, 0.2, size=(noise_dim, 2))
    g_b = rng.normal(0.0, 0.05, size=2)

    real_pool = public_x
    if len(real_pool) == 0:
        raise ValueError("public_x cannot be empty")

    final_d_loss = 0.0
    final_g_loss = 0.0
    target_sign = 1.0 if target_class == 1 else -1.0

    for _ in range(attack_steps):
        # Sample real and fake batches.
        idx = rng.integers(0, len(real_pool), size=batch_size)
        x_real = real_pool[idx]
        z = rng.normal(0.0, 1.0, size=(batch_size, noise_dim))
        x_fake = np.tanh(z @ g_w + g_b)

        # Discriminator update.
        p_real = _sigmoid(x_real @ d_w + d_b)
        p_fake = _sigmoid(x_fake @ d_w + d_b)
        eps = 1e-12
        final_d_loss = float(-np.mean(np.log(p_real + eps) + np.log(1.0 - p_fake + eps)))

        grad_real = (p_real - 1.0)[:, None] * x_real
        grad_fake = p_fake[:, None] * x_fake
        grad_d_w = np.mean(np.vstack([grad_real, grad_fake]), axis=0)
        grad_d_b = float(np.mean(np.concatenate([p_real - 1.0, p_fake])))
        d_w -= lr_d * grad_d_w
        d_b -= lr_d * grad_d_b

        # Generator update (heuristic gradient for compact toy setup).
        z = rng.normal(0.0, 1.0, size=(batch_size, noise_dim))
        x_fake = np.tanh(z @ g_w + g_b)
        p_fake = _sigmoid(x_fake @ d_w + d_b)
        victim_conf = _sigmoid(x_fake @ victim_model.w + victim_model.b)
        if target_class == 0:
            victim_conf = 1.0 - victim_conf

        final_g_loss = float(-np.mean(np.log(p_fake + eps) + 1.5 * np.log(victim_conf + eps)))

        # Approximate gradient through tanh output.
        grad_on_x = ((p_fake - 1.0)[:, None] * d_w[None, :]) + (
            -1.5 * (1.0 - victim_conf)[:, None] * target_sign * victim_model.w[None, :]
        )
        tanh_grad = 1.0 - x_fake**2
        delta = grad_on_x * tanh_grad
        grad_g_w = z.T @ delta / batch_size
        grad_g_b = np.mean(delta, axis=0)
        g_w -= lr_g * grad_g_w
        g_b -= lr_g * grad_g_b

    z_final = rng.normal(0.0, 1.0, size=(400, noise_dim))
    generated = np.tanh(z_final @ g_w + g_b)
    return GANAttackResult(generated_x=generated, final_disc_loss=final_d_loss, final_gen_loss=final_g_loss)
