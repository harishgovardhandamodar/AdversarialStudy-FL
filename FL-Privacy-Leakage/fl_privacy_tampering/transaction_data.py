from __future__ import annotations

from dataclasses import dataclass
import csv
import math
import numpy as np

from fl_privacy_tampering.data import DatasetBundle


@dataclass
class _Row:
    user_id: str
    n_items: float
    cost: float
    hour: int


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _parse_hour(ts: str) -> int:
    # e.g. "Sat Feb 02 12:50:00 IST 2019"
    parts = ts.split()
    if len(parts) >= 4 and ":" in parts[3]:
        try:
            return int(parts[3].split(":")[0])
        except ValueError:
            return 0
    return 0


def _tokenize(r: _Row, vocab_size: int) -> list[int]:
    value = max(0.0, r.n_items * r.cost)
    t_items = int(np.clip(math.log1p(max(0.0, r.n_items)) * 12, 0, vocab_size - 1))
    t_value = int(np.clip(math.log1p(value) * 10, 0, vocab_size - 1))
    t_hour = int(np.clip(r.hour, 0, 23)) % vocab_size
    return [t_items, t_value, t_hour]


def make_transaction_clients(
    csv_path: str,
    max_rows: int,
    num_clients: int,
    vocab_size: int,
    canary_client_id: int,
    seed: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    users: dict[str, list[int]] = {str(i): [] for i in range(num_clients)}
    counts = {str(i): 0 for i in range(num_clients)}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            uid = str(row.get("UserId", "-1")).strip().strip('"')
            n_items = _safe_float(str(row.get("NumberOfItemsPurchased", "0")).strip().strip('"'))
            cost = _safe_float(str(row.get("CostPerItem", "0")).strip().strip('"'))
            hour = _parse_hour(str(row.get("TransactionTime", "")).strip().strip('"'))
            rec = _Row(user_id=uid, n_items=n_items, cost=cost, hour=hour)
            tokens = _tokenize(rec, vocab_size=vocab_size)
            cid = str(abs(hash(uid)) % num_clients)
            users[cid].extend(tokens)
            counts[cid] += 1

    client_sequences: list[np.ndarray] = []
    for i in range(num_clients):
        toks = users[str(i)]
        if len(toks) < 4:
            toks = rng.integers(0, vocab_size, size=8).tolist()
        client_sequences.append(np.array(toks, dtype=np.int64))

    # Build canary for chosen client using rare vocab corner.
    src = vocab_size - 3
    dst = vocab_size - 2
    canaries = {canary_client_id: (src, dst)}
    seq = client_sequences[canary_client_id].copy()
    for j in range(0, min(len(seq) - 1, 80), 2):
        seq[j] = src
        seq[j + 1] = dst
    client_sequences[canary_client_id] = seq

    return DatasetBundle(
        client_sequences=client_sequences,
        canaries=canaries,
        vocab_size=vocab_size,
    )
