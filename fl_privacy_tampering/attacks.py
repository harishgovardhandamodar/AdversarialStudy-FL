from __future__ import annotations

import numpy as np


def apply_selective_tampering(
    local_weights: dict[str, np.ndarray],
    target_layers: list[str],
    target_token_ids: list[int],
    scale: float,
    noise_std: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Tamper selected rows/columns to amplify memorization pressure.
    """
    rng = np.random.default_rng(seed)
    tampered = {k: v.copy() for k, v in local_weights.items()}

    for layer in target_layers:
        if layer not in tampered:
            continue
        arr = tampered[layer]
        if layer == "embedding":
            for tok in target_token_ids:
                if 0 <= tok < arr.shape[0]:
                    arr[tok] *= scale
                    if noise_std > 0:
                        arr[tok] += rng.normal(0.0, noise_std, size=arr[tok].shape)
        elif layer == "output":
            for tok in target_token_ids:
                if 0 <= tok < arr.shape[1]:
                    arr[:, tok] *= scale
                    if noise_std > 0:
                        arr[:, tok] += rng.normal(0.0, noise_std, size=arr[:, tok].shape)
        else:
            arr *= scale
        tampered[layer] = arr

    return tampered
