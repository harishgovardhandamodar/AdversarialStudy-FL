from __future__ import annotations

import argparse
import json

from gan_attack_fl.attack import run_gan_attack
from gan_attack_fl.data import make_federated_gan_dataset
from gan_attack_fl.federated import train_fedavg_logistic
from gan_attack_fl.metrics import evaluate_attack


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)

    dcfg = cfg["data"]
    dataset = make_federated_gan_dataset(
        num_clients=int(dcfg["num_clients"]),
        points_per_client=int(dcfg["points_per_client"]),
        test_points=int(dcfg["test_points"]),
        target_client_id=int(dcfg["target_client_id"]),
        target_private_class=int(dcfg["target_private_class"]),
        seed=int(cfg["seed"]),
    )

    tcfg = cfg["training"]
    model = train_fedavg_logistic(
        clients=dataset.clients,
        rounds=int(tcfg["rounds"]),
        local_steps=int(tcfg["local_steps"]),
        lr=float(tcfg["lr"]),
        l2=float(tcfg["l2"]),
        seed=int(cfg["seed"]),
    )
    test_acc = model.accuracy(dataset.test_x, dataset.test_y)

    acfg = cfg["attack"]
    real_target_public = dataset.attacker_public_x[
        dataset.attacker_public_y == int(dcfg["target_private_class"])
    ]
    attack_res = run_gan_attack(
        victim_model=model,
        public_x=real_target_public,
        target_class=int(dcfg["target_private_class"]),
        noise_dim=int(acfg["noise_dim"]),
        attack_steps=int(acfg["attack_steps"]),
        batch_size=int(acfg["batch_size"]),
        lr_g=float(acfg["lr_g"]),
        lr_d=float(acfg["lr_d"]),
        seed=int(cfg["seed"]) + 1000,
    )

    m = evaluate_attack(
        generated_x=attack_res.generated_x,
        target_private_x=dataset.target_private_x,
        victim_model=model,
        target_class=int(dcfg["target_private_class"]),
    )

    print("=== GAN Attack on Federated Learning ===")
    print(f"global_test_accuracy: {test_acc:.6f}")
    print(f"target_client_id: {dcfg['target_client_id']}")
    print(f"target_private_class: {dcfg['target_private_class']}")
    print(f"mean_distance_to_target_private: {m.mean_distance:.6f}")
    print(f"covariance_distance_to_target_private: {m.covariance_distance:.6f}")
    print(f"nearest_neighbor_distance: {m.nearest_neighbor_distance:.6f}")
    print(f"target_confidence_on_generated: {m.target_confidence_on_generated:.6f}")
    print(f"final_discriminator_loss: {attack_res.final_disc_loss:.6f}")
    print(f"final_generator_loss: {attack_res.final_gen_loss:.6f}")


if __name__ == "__main__":
    main()
