from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
from typing import Any

import matplotlib.pyplot as plt

from scripts.run_experiment import load_config, run_experiment


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def _write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(rows: list[dict[str, Any]], outdir: str) -> None:
    run_ids = [int(r["run_id"]) for r in rows]
    exposure = [float(r["exposure_gap"]) for r in rows]
    mi_auc = [float(r["mi_auc"]) for r in rows]
    attack_enabled = [str(r.get("attack.enabled", "")) for r in rows]
    colors = ["tab:red" if a == "True" else "tab:blue" for a in attack_enabled]

    plt.figure(figsize=(8, 4))
    plt.scatter(run_ids, exposure, c=colors)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Exposure Gap by Run")
    plt.xlabel("run_id")
    plt.ylabel("exposure_gap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "exposure_gap_by_run.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.scatter(run_ids, mi_auc, c=colors)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Membership Inference AUC by Run")
    plt.xlabel("run_id")
    plt.ylabel("mi_auc")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mi_auc_by_run.png"), dpi=160)
    plt.close()

    if "attack.scale" in rows[0]:
        x = [float(r["attack.scale"]) for r in rows]
        plt.figure(figsize=(8, 4))
        plt.scatter(x, exposure, c=colors)
        plt.title("Exposure Gap vs Attack Scale")
        plt.xlabel("attack.scale")
        plt.ylabel("exposure_gap")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "exposure_vs_attack_scale.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.scatter(x, mi_auc, c=colors)
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        plt.title("MI AUC vs Attack Scale")
        plt.xlabel("attack.scale")
        plt.ylabel("mi_auc")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "mi_auc_vs_attack_scale.png"), dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results/sweep")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    grid_cfg = load_config(args.grid)
    grid = grid_cfg["grid"]

    os.makedirs(args.outdir, exist_ok=True)
    combos = _expand_grid(grid)

    rows: list[dict[str, Any]] = []
    for run_id, combo in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        for key, value in combo.items():
            _set_nested(cfg, key, value)
        cfg["seed"] = int(base_cfg["seed"]) + run_id

        metrics = run_experiment(cfg)
        row = {"run_id": run_id, **combo, **metrics}
        rows.append(row)
        print(
            f"run={run_id} attack={row.get('attack.enabled')} "
            f"exposure_gap={row['exposure_gap']:.4f} mi_auc={row['mi_auc']:.4f}"
        )

    csv_path = os.path.join(args.outdir, "sweep_results.csv")
    _write_rows(csv_path, rows)
    _make_plots(rows, args.outdir)

    print("=== Sweep Complete ===")
    print(f"runs: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"plots_dir: {args.outdir}")


if __name__ == "__main__":
    main()
