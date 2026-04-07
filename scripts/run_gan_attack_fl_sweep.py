from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
from typing import Any

from gan_attack_fl.attack import run_gan_attack
from gan_attack_fl.data import make_federated_gan_dataset
from gan_attack_fl.federated import train_fedavg_logistic
from gan_attack_fl.metrics import evaluate_attack


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def run_once(cfg: dict[str, Any]) -> dict[str, Any]:
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    acfg = cfg["attack"]
    seed = int(cfg["seed"])

    dataset = make_federated_gan_dataset(
        num_clients=int(dcfg["num_clients"]),
        points_per_client=int(dcfg["points_per_client"]),
        test_points=int(dcfg["test_points"]),
        target_client_id=int(dcfg["target_client_id"]),
        target_private_class=int(dcfg["target_private_class"]),
        seed=seed,
    )

    model = train_fedavg_logistic(
        clients=dataset.clients,
        rounds=int(tcfg["rounds"]),
        local_steps=int(tcfg["local_steps"]),
        lr=float(tcfg["lr"]),
        l2=float(tcfg["l2"]),
        seed=seed,
    )
    test_acc = model.accuracy(dataset.test_x, dataset.test_y)

    public_target = dataset.attacker_public_x[
        dataset.attacker_public_y == int(dcfg["target_private_class"])
    ]
    attack_res = run_gan_attack(
        victim_model=model,
        public_x=public_target,
        target_class=int(dcfg["target_private_class"]),
        noise_dim=int(acfg["noise_dim"]),
        attack_steps=int(acfg["attack_steps"]),
        batch_size=int(acfg["batch_size"]),
        lr_g=float(acfg["lr_g"]),
        lr_d=float(acfg["lr_d"]),
        seed=seed + 1000,
    )
    m = evaluate_attack(
        generated_x=attack_res.generated_x,
        target_private_x=dataset.target_private_x,
        victim_model=model,
        target_class=int(dcfg["target_private_class"]),
    )

    return {
        "seed": seed,
        "global_test_accuracy": test_acc,
        "mean_distance_to_target_private": m.mean_distance,
        "covariance_distance_to_target_private": m.covariance_distance,
        "nearest_neighbor_distance": m.nearest_neighbor_distance,
        "target_confidence_on_generated": m.target_confidence_on_generated,
        "final_discriminator_loss": attack_res.final_disc_loss,
        "final_generator_loss": attack_res.final_gen_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out-csv", default="results/gan_attack_fl_sweep.csv")
    args = parser.parse_args()

    base_cfg = load_json(args.config)
    grid_cfg = load_json(args.grid)
    grid = grid_cfg["grid"]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rows: list[dict[str, Any]] = []
    for run_id, values in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, values):
            set_nested(cfg, k, v)
        cfg["seed"] = int(base_cfg["seed"]) + run_id
        metrics = run_once(cfg)
        row = {"run_id": run_id, **{k: v for k, v in zip(keys, values)}, **metrics}
        rows.append(row)
        print(
            f"run={run_id} acc={metrics['global_test_accuracy']:.4f} "
            f"nn={metrics['nearest_neighbor_distance']:.4f} "
            f"conf={metrics['target_confidence_on_generated']:.4f}"
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv: {args.out_csv}")


if __name__ == "__main__":
    main()
