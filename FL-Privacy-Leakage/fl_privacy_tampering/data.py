from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DatasetBundle:
    client_sequences: list[np.ndarray]
    canaries: dict[int, tuple[int, int]]
    vocab_size: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_synthetic_clients(
    num_clients: int,
    seq_len: int,
    samples_per_client: int,
    vocab_size: int,
    canary_clients: list[int],
    seed: int,
) -> DatasetBundle:
    """
    Creates per-client token streams with optional client-unique canary bigrams.
    Each client gets a flat token sequence used for next-token training.
    """
    generator = _rng(seed)
    client_sequences: list[np.ndarray] = []
    canaries: dict[int, tuple[int, int]] = {}

    for client_id in range(num_clients):
        tokens = generator.integers(low=0, high=vocab_size, size=samples_per_client + 1)
        if client_id in canary_clients:
            src = int(generator.integers(0, vocab_size))
            dst = int(generator.integers(0, vocab_size))
            canaries[client_id] = (src, dst)
            # Inject canary bigram repeatedly into this client's local stream.
            for idx in range(0, min(samples_per_client - 1, seq_len * 2), 2):
                tokens[idx] = src
                tokens[idx + 1] = dst
        client_sequences.append(tokens)
    return DatasetBundle(client_sequences=client_sequences, canaries=canaries, vocab_size=vocab_size)
