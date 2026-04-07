from __future__ import annotations

import argparse
import csv
import json
import os
import numpy as np

from fl_privacy_tampering.data import make_synthetic_clients
from fl_privacy_tampering.federated import AttackConfig, train_federated
from fl_privacy_tampering.leakage import (
    evaluate_canary_leakage,
    evaluate_membership_inference_auc,
)
from fl_privacy_tampering.model import TinyLanguageModel


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_result_csv(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(cfg: dict) -> dict:
    seed = int(cfg["seed"])
    np.random.seed(seed)

    data_cfg = cfg["data"]
    bundle = make_synthetic_clients(
        num_clients=int(data_cfg["num_clients"]),
        seq_len=int(data_cfg["seq_len"]),
        samples_per_client=int(data_cfg["samples_per_client"]),
        vocab_size=int(data_cfg["vocab_size"]),
        canary_clients=list(data_cfg["canary_clients"]),
        seed=seed,
    )

    model_cfg = cfg["model"]
    init_model = TinyLanguageModel(
        vocab_size=bundle.vocab_size,
        hidden_dim=int(model_cfg["hidden_dim"]),
        seed=seed,
    )

    attack_cfg = cfg["attack"]
    attack = AttackConfig(
        enabled=bool(attack_cfg["enabled"]),
        attacker_client_ids=list(attack_cfg["attacker_client_ids"]),
        target_layers=list(attack_cfg["target_layers"]),
        target_token_ids=list(attack_cfg["target_token_ids"]),
        scale=float(attack_cfg["scale"]),
        noise_std=float(attack_cfg["noise_std"]),
    )

    train_cfg = cfg["training"]
    trained_model = train_federated(
        model=init_model,
        client_sequences=bundle.client_sequences,
        rounds=int(train_cfg["rounds"]),
        local_steps=int(train_cfg["local_steps"]),
        lr=float(train_cfg["lr"]),
        batch_size=int(train_cfg["batch_size"]),
        attack=attack,
        seed=seed,
    )

    target_client = int(cfg["evaluation"]["target_canary_client"])
    canary_pair = bundle.canaries[target_client]
    control_pair = tuple(cfg["evaluation"]["control_pair"])

    result = evaluate_canary_leakage(
        model=trained_model,
        canary_pair=canary_pair,
        control_pair=(int(control_pair[0]), int(control_pair[1])),
    )

    eval_cfg = cfg["evaluation"]
    nonmember_count = int(eval_cfg.get("nonmember_count", len(bundle.client_sequences)))
    nonmember_bundle = make_synthetic_clients(
        num_clients=nonmember_count,
        seq_len=int(data_cfg["seq_len"]),
        samples_per_client=int(data_cfg["samples_per_client"]),
        vocab_size=int(data_cfg["vocab_size"]),
        canary_clients=[],
        seed=seed + 9973,
    )
    mi_result = evaluate_membership_inference_auc(
        model=trained_model,
        member_sequences=bundle.client_sequences,
        nonmember_sequences=nonmember_bundle.client_sequences,
    )

    metrics = {
        "seed": seed,
        "attack_enabled": bool(attack_cfg["enabled"]),
        "target_client": target_client,
        "canary_src": int(canary_pair[0]),
        "canary_dst": int(canary_pair[1]),
        "control_src": int(control_pair[0]),
        "control_dst": int(control_pair[1]),
        "canary_loss": result.canary_loss,
        "control_loss": result.control_loss,
        "exposure_gap": result.exposure_gap,
        "target_rank": result.target_rank,
        "mi_member_mean_loss": mi_result.member_mean_loss,
        "mi_nonmember_mean_loss": mi_result.nonmember_mean_loss,
        "mi_auc": mi_result.auc,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = run_experiment(cfg)

    print("=== Experiment Result ===")
    print(f"target_client: {metrics['target_client']}")
    print(f"canary_pair: ({metrics['canary_src']}, {metrics['canary_dst']})")
    print(f"control_pair: ({metrics['control_src']}, {metrics['control_dst']})")
    print(f"attack_enabled: {metrics['attack_enabled']}")
    print(f"canary_loss: {metrics['canary_loss']:.6f}")
    print(f"control_loss: {metrics['control_loss']:.6f}")
    print(f"exposure_gap (control - canary): {metrics['exposure_gap']:.6f}")
    print(f"target_rank (lower is leakier): {metrics['target_rank']}")
    print(f"mi_member_mean_loss: {metrics['mi_member_mean_loss']:.6f}")
    print(f"mi_nonmember_mean_loss: {metrics['mi_nonmember_mean_loss']:.6f}")
    print(f"mi_auc: {metrics['mi_auc']:.6f}")

    if args.csv:
        _append_result_csv(args.csv, metrics)
        print(f"results_csv: {args.csv}")


if __name__ == "__main__":
    main()
