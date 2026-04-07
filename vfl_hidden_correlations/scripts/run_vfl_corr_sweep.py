from __future__ import annotations

import argparse
import copy
import csv
import json
import os

from vfl_hidden_correlations.attack import reconstruct_party_b_features
from vfl_hidden_correlations.data import make_vfl_data
from vfl_hidden_correlations.metrics import evaluate_reconstruction
from vfl_hidden_correlations.model import train_vfl_logistic


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corr-values", default="0.2,0.5,0.8")
    parser.add_argument("--out-csv", default="results/vfl_corr_sweep.csv")
    args = parser.parse_args()

    base = load_config(args.config)
    corr_values = [float(x.strip()) for x in args.corr_values.split(",") if x.strip()]
    rows = []
    for i, corr in enumerate(corr_values):
        cfg = copy.deepcopy(base)
        cfg["seed"] = int(base["seed"]) + i
        cfg["data"]["corr_strength"] = corr
        d = cfg["data"]
        bundle = make_vfl_data(
            n_train=int(d["n_train"]),
            n_test=int(d["n_test"]),
            n_shadow=int(d["n_shadow"]),
            d_a=int(d["d_a"]),
            d_b=int(d["d_b"]),
            latent_dim=int(d["latent_dim"]),
            corr_strength=float(d["corr_strength"]),
            noise_std=float(d["noise_std"]),
            seed=int(cfg["seed"]),
        )
        mcfg = cfg["model"]
        model = train_vfl_logistic(
            x_a=bundle.x_a_train,
            x_b=bundle.x_b_train,
            y=bundle.y_train,
            lr=float(mcfg["lr"]),
            epochs=int(mcfg["epochs"]),
            l2=float(mcfg["l2"]),
            seed=int(cfg["seed"]),
        )
        logits = model.logits(bundle.x_a_test, bundle.x_b_test)
        recon = reconstruct_party_b_features(
            model=model,
            x_a_target=bundle.x_a_test,
            victim_logits=logits,
            x_a_shadow=bundle.x_a_shadow,
            x_b_shadow=bundle.x_b_shadow,
            ridge=float(cfg["attack"]["ridge"]),
        )
        metr = evaluate_reconstruction(bundle.x_b_test, recon.x_b_hat)
        row = {
            "corr_strength": corr,
            "train_accuracy": model.accuracy(bundle.x_a_train, bundle.x_b_train, bundle.y_train),
            "test_accuracy": model.accuracy(bundle.x_a_test, bundle.x_b_test, bundle.y_test),
            "reconstruction_mse": metr.mse,
            "mean_feature_correlation": metr.mean_feature_corr,
            "sensitive_attribute_accuracy": metr.sensitive_attribute_accuracy,
        }
        rows.append(row)
        print(
            f"corr={corr:.2f} mse={row['reconstruction_mse']:.4f} "
            f"feat_corr={row['mean_feature_correlation']:.4f}"
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv: {args.out_csv}")


if __name__ == "__main__":
    main()
