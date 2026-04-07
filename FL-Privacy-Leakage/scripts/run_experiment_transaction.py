from __future__ import annotations

import argparse
import csv
import json
import os
import numpy as np

from fl_privacy_tampering.federated import AttackConfig, train_federated
from fl_privacy_tampering.leakage import evaluate_canary_leakage, evaluate_membership_inference_auc
from fl_privacy_tampering.model import TinyLanguageModel
from fl_privacy_tampering.transaction_data import make_transaction_clients


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_result(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run(cfg: dict) -> dict:
    seed = int(cfg["seed"])
    np.random.seed(seed)

    dcfg = cfg["data"]
    ecfg = cfg["evaluation"]
    bundle = make_transaction_clients(
        csv_path=str(dcfg["csv_path"]),
        max_rows=int(dcfg["max_rows"]),
        num_clients=int(dcfg["num_clients"]),
        vocab_size=int(dcfg["vocab_size"]),
        canary_client_id=int(ecfg["target_canary_client"]),
        seed=seed,
    )

    model = TinyLanguageModel(vocab_size=bundle.vocab_size, hidden_dim=int(cfg["model"]["hidden_dim"]), seed=seed)
    acfg = cfg["attack"]
    attack = AttackConfig(
        enabled=bool(acfg["enabled"]),
        attacker_client_ids=list(acfg["attacker_client_ids"]),
        target_layers=list(acfg["target_layers"]),
        target_token_ids=list(acfg["target_token_ids"]),
        scale=float(acfg["scale"]),
        noise_std=float(acfg["noise_std"]),
    )
    tcfg = cfg["training"]
    trained = train_federated(
        model=model,
        client_sequences=bundle.client_sequences,
        rounds=int(tcfg["rounds"]),
        local_steps=int(tcfg["local_steps"]),
        lr=float(tcfg["lr"]),
        batch_size=int(tcfg["batch_size"]),
        attack=attack,
        seed=seed,
    )

    target_client = int(ecfg["target_canary_client"])
    canary_pair = bundle.canaries[target_client]
    control_pair = tuple(ecfg["control_pair"])
    canary = evaluate_canary_leakage(trained, canary_pair=canary_pair, control_pair=(int(control_pair[0]), int(control_pair[1])))

    nonmember_bundle = make_transaction_clients(
        csv_path=str(dcfg["csv_path"]),
        max_rows=int(dcfg["max_rows"]),
        num_clients=int(ecfg["nonmember_count"]),
        vocab_size=int(dcfg["vocab_size"]),
        canary_client_id=0,
        seed=seed + 999,
    )
    mi = evaluate_membership_inference_auc(
        model=trained,
        member_sequences=bundle.client_sequences,
        nonmember_sequences=nonmember_bundle.client_sequences,
    )
    return {
        "seed": seed,
        "attack_enabled": bool(acfg["enabled"]),
        "target_client": target_client,
        "canary_src": int(canary_pair[0]),
        "canary_dst": int(canary_pair[1]),
        "control_src": int(control_pair[0]),
        "control_dst": int(control_pair[1]),
        "canary_loss": canary.canary_loss,
        "control_loss": canary.control_loss,
        "exposure_gap": canary.exposure_gap,
        "target_rank": canary.target_rank,
        "mi_member_mean_loss": mi.member_mean_loss,
        "mi_nonmember_mean_loss": mi.nonmember_mean_loss,
        "mi_auc": mi.auc,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    r = run(load_config(args.config))
    print("=== TX FL Privacy Leakage ===")
    for k, v in r.items():
        print(f"{k}: {v}")
    if args.csv:
        append_result(args.csv, r)
        print(f"results_csv: {args.csv}")


if __name__ == "__main__":
    main()
