from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from typing import Iterable

import numpy as np

from gan_attack_fl.data import FLClientData, GANAttackDataset


@dataclass
class TransactionRecord:
    user_id: str
    number_items: float
    cost_per_item: float
    hour: float

    @property
    def total_value(self) -> float:
        return self.number_items * self.cost_per_item


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_hour(ts: str) -> float:
    # Example: Sat Feb 02 12:50:00 IST 2019
    parts = ts.split()
    if len(parts) >= 4 and ":" in parts[3]:
        hh = parts[3].split(":")[0]
        return _safe_float(hh, 0.0)
    return 0.0


def _iter_records(path: str, max_rows: int) -> Iterable[TransactionRecord]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            uid = str(r.get("UserId", "-1")).strip().strip('"')
            n_items = _safe_float(str(r.get("NumberOfItemsPurchased", "0")).strip().strip('"'))
            cpi = _safe_float(str(r.get("CostPerItem", "0")).strip().strip('"'))
            t = str(r.get("TransactionTime", "")).strip().strip('"')
            if n_items == 0.0 and cpi == 0.0:
                continue
            yield TransactionRecord(
                user_id=uid,
                number_items=n_items,
                cost_per_item=cpi,
                hour=_parse_hour(t),
            )


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd


def make_transaction_gan_dataset(
    csv_path: str,
    max_rows: int,
    num_clients: int,
    test_fraction: float,
    public_fraction: float,
    target_quantile: float,
    min_target_count: int,
    seed: int,
) -> GANAttackDataset:
    rng = np.random.default_rng(seed)
    records = list(_iter_records(csv_path, max_rows=max_rows))
    if len(records) < 1000:
        raise ValueError("Not enough records parsed from transaction CSV")

    # Build 2D features (compatible with existing GAN-FL model).
    # x0: purchase intensity, x1: transaction monetary value.
    x = np.array(
        [
            [
                math.log1p(max(0.0, rec.number_items)),
                math.log1p(max(0.0, rec.total_value)),
            ]
            for rec in records
        ],
        dtype=float,
    )
    x = _standardize(x)

    values = np.array([rec.total_value for rec in records], dtype=float)
    thresh = float(np.quantile(values, target_quantile))
    y = (values >= thresh).astype(np.int64)

    # Split by user hash into clients.
    user_to_client: dict[str, int] = {}
    clients_raw = [[] for _ in range(num_clients)]
    for i, rec in enumerate(records):
        uid = rec.user_id
        if uid not in user_to_client:
            user_to_client[uid] = abs(hash(uid)) % num_clients
        cid = user_to_client[uid]
        clients_raw[cid].append(i)

    clients: list[FLClientData] = []
    for idxs in clients_raw:
        if not idxs:
            clients.append(FLClientData(x=np.empty((0, 2)), y=np.empty((0,), dtype=np.int64)))
            continue
        arr = np.array(idxs, dtype=int)
        clients.append(FLClientData(x=x[arr], y=y[arr]))

    # Pick target client with enough positives.
    best_cid = None
    best_pos = -1
    for cid, c in enumerate(clients):
        pos = int(np.sum(c.y == 1))
        if pos >= min_target_count and pos > best_pos:
            best_cid = cid
            best_pos = pos
    if best_cid is None:
        raise ValueError("No client has enough target class samples; lower min_target_count.")

    target_private_class = 1
    target_private_x = clients[best_cid].x[clients[best_cid].y == target_private_class]

    # Global train/test split.
    all_idx = np.arange(len(x))
    rng.shuffle(all_idx)
    n_test = int(len(all_idx) * test_fraction)
    test_idx = all_idx[:n_test]
    train_idx = all_idx[n_test:]

    test_x = x[test_idx]
    test_y = y[test_idx]

    # Build attacker public samples from non-target-client train data.
    non_target_mask = np.ones(len(x), dtype=bool)
    non_target_mask[np.array(clients_raw[best_cid], dtype=int)] = False
    candidate_idx = np.where(non_target_mask & np.isin(np.arange(len(x)), train_idx))[0]
    pub_n = max(200, int(len(train_idx) * public_fraction))
    pub_pick = rng.choice(candidate_idx, size=min(pub_n, len(candidate_idx)), replace=False)
    attacker_public_x = x[pub_pick]
    attacker_public_y = y[pub_pick]

    # Rebuild per-client train-only data for FL.
    train_set = set(train_idx.tolist())
    clients_train: list[FLClientData] = []
    for cid, idxs in enumerate(clients_raw):
        keep = np.array([i for i in idxs if i in train_set], dtype=int)
        if len(keep) == 0:
            clients_train.append(FLClientData(x=np.empty((0, 2)), y=np.empty((0,), dtype=np.int64)))
        else:
            clients_train.append(FLClientData(x=x[keep], y=y[keep]))

    # Filter out empty clients for stable training.
    clients_train = [c for c in clients_train if len(c.y) > 0]

    return GANAttackDataset(
        clients=clients_train,
        target_client_id=int(best_cid),
        target_private_class=target_private_class,
        target_private_x=target_private_x,
        attacker_public_x=attacker_public_x,
        attacker_public_y=attacker_public_y,
        test_x=test_x,
        test_y=test_y,
    )
