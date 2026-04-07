from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from fl_privacy_tampering.attacks import apply_selective_tampering
from fl_privacy_tampering.model import TinyLanguageModel


@dataclass
class AttackConfig:
    enabled: bool
    attacker_client_ids: list[int]
    target_layers: list[str]
    target_token_ids: list[int]
    scale: float
    noise_std: float


def fedavg(weight_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = weight_list[0].keys()
    avg = {}
    for k in keys:
        avg[k] = np.mean([w[k] for w in weight_list], axis=0)
    return avg


def train_federated(
    model: TinyLanguageModel,
    client_sequences: list[np.ndarray],
    rounds: int,
    local_steps: int,
    lr: float,
    batch_size: int,
    attack: AttackConfig,
    seed: int,
) -> TinyLanguageModel:
    rng = np.random.default_rng(seed)
    global_model = model.clone()

    for _ in range(rounds):
        local_weight_updates: list[dict[str, np.ndarray]] = []
        for cid, seq in enumerate(client_sequences):
            local_model = global_model.clone()
            for _ in range(local_steps):
                local_model.local_sgd_step(tokens=seq, lr=lr, batch_size=batch_size)
            local_w = local_model.get_weights()

            if attack.enabled and cid in attack.attacker_client_ids:
                local_w = apply_selective_tampering(
                    local_weights=local_w,
                    target_layers=attack.target_layers,
                    target_token_ids=attack.target_token_ids,
                    scale=attack.scale,
                    noise_std=attack.noise_std,
                    seed=int(rng.integers(0, 10_000_000)),
                )
            local_weight_updates.append(local_w)

        global_model.set_weights(fedavg(local_weight_updates))
    return global_model
