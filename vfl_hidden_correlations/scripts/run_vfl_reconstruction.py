from __future__ import annotations

import argparse
import json

from vfl_hidden_correlations.attack import reconstruct_party_b_features
from vfl_hidden_correlations.data import make_vfl_data
from vfl_hidden_correlations.metrics import evaluate_reconstruction
from vfl_hidden_correlations.model import train_vfl_logistic


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    bundle = make_vfl_data(
        n_train=int(data_cfg["n_train"]),
        n_test=int(data_cfg["n_test"]),
        n_shadow=int(data_cfg["n_shadow"]),
        d_a=int(data_cfg["d_a"]),
        d_b=int(data_cfg["d_b"]),
        latent_dim=int(data_cfg["latent_dim"]),
        corr_strength=float(data_cfg["corr_strength"]),
        noise_std=float(data_cfg["noise_std"]),
        seed=int(cfg["seed"]),
    )

    model_cfg = cfg["model"]
    model = train_vfl_logistic(
        x_a=bundle.x_a_train,
        x_b=bundle.x_b_train,
        y=bundle.y_train,
        lr=float(model_cfg["lr"]),
        epochs=int(model_cfg["epochs"]),
        l2=float(model_cfg["l2"]),
        seed=int(cfg["seed"]),
    )

    train_acc = model.accuracy(bundle.x_a_train, bundle.x_b_train, bundle.y_train)
    test_acc = model.accuracy(bundle.x_a_test, bundle.x_b_test, bundle.y_test)

    # Attacker observes per-sample logit for target records (post-training).
    victim_logits = model.logits(bundle.x_a_test, bundle.x_b_test)
    recon = reconstruct_party_b_features(
        model=model,
        x_a_target=bundle.x_a_test,
        victim_logits=victim_logits,
        x_a_shadow=bundle.x_a_shadow,
        x_b_shadow=bundle.x_b_shadow,
        ridge=float(cfg["attack"]["ridge"]),
    )
    m = evaluate_reconstruction(bundle.x_b_test, recon.x_b_hat)

    print("=== VFL Post-Training Reconstruction Attack ===")
    print(f"train_accuracy: {train_acc:.4f}")
    print(f"test_accuracy: {test_acc:.4f}")
    print(f"corr_strength: {data_cfg['corr_strength']}")
    print(f"reconstruction_mse: {m.mse:.6f}")
    print(f"mean_feature_correlation: {m.mean_feature_corr:.6f}")
    print(f"sensitive_attribute_accuracy: {m.sensitive_attribute_accuracy:.6f}")
    print(f"logit_projection_residual_mse: {recon.projection_residual_mse:.10f}")


if __name__ == "__main__":
    main()
