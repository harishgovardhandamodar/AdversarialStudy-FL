from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
from typing import Any

from scripts.run_gan_attack_fl_transaction import run_once


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out-csv", default="results/gan_attack_fl_transaction_sweep.csv")
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
            f"run={run_id} steps={row.get('attack.attack_steps')} "
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
