from __future__ import annotations

import argparse
import json

from gan_attack_fl.attack import run_gan_attack
from gan_attack_fl.federated import train_fedavg_logistic
from gan_attack_fl.metrics import evaluate_attack
from gan_attack_fl.transaction_adapter import make_transaction_gan_dataset


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_once(cfg: dict) -> dict:
    tdcfg = cfg["transaction_data"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    acfg = cfg["attack"]

    dataset = make_transaction_gan_dataset(
        csv_path=str(tdcfg["csv_path"]),
        max_rows=int(tdcfg["max_rows"]),
        num_clients=int(dcfg["num_clients"]),
        test_fraction=float(dcfg["test_fraction"]),
        public_fraction=float(dcfg["public_fraction"]),
        target_quantile=float(dcfg["target_quantile"]),
        min_target_count=int(dcfg["min_target_count"]),
        seed=int(cfg["seed"]),
    )

    model = train_fedavg_logistic(
        clients=dataset.clients,
        rounds=int(tcfg["rounds"]),
        local_steps=int(tcfg["local_steps"]),
        lr=float(tcfg["lr"]),
        l2=float(tcfg["l2"]),
        seed=int(cfg["seed"]),
    )
    test_acc = model.accuracy(dataset.test_x, dataset.test_y)

    public_target = dataset.attacker_public_x[
        dataset.attacker_public_y == int(dataset.target_private_class)
    ]
    attack_res = run_gan_attack(
        victim_model=model,
        public_x=public_target,
        target_class=int(dataset.target_private_class),
        noise_dim=int(acfg["noise_dim"]),
        attack_steps=int(acfg["attack_steps"]),
        batch_size=int(acfg["batch_size"]),
        lr_g=float(acfg["lr_g"]),
        lr_d=float(acfg["lr_d"]),
        seed=int(cfg["seed"]) + 1000,
    )
    metrics = evaluate_attack(
        generated_x=attack_res.generated_x,
        target_private_x=dataset.target_private_x,
        victim_model=model,
        target_class=int(dataset.target_private_class),
    )

    return {
        "seed": int(cfg["seed"]),
        "target_client_id": int(dataset.target_client_id),
        "target_private_class": int(dataset.target_private_class),
        "global_test_accuracy": float(test_acc),
        "mean_distance_to_target_private": float(metrics.mean_distance),
        "covariance_distance_to_target_private": float(metrics.covariance_distance),
        "nearest_neighbor_distance": float(metrics.nearest_neighbor_distance),
        "target_confidence_on_generated": float(metrics.target_confidence_on_generated),
        "final_discriminator_loss": float(attack_res.final_disc_loss),
        "final_generator_loss": float(attack_res.final_gen_loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    r = run_once(cfg)
    print("=== GAN Attack on FL (transaction_data.csv) ===")
    for k, v in r.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
