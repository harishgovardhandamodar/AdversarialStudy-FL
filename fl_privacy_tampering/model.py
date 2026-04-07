from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


class TinyLanguageModel:
    """
    One-step LM: token embedding -> linear projection -> next-token logits.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = rng.normal(0.0, 0.05, size=(vocab_size, hidden_dim))
        self.output = rng.normal(0.0, 0.05, size=(hidden_dim, vocab_size))

    def clone(self) -> "TinyLanguageModel":
        m = TinyLanguageModel(self.vocab_size, self.hidden_dim, seed=0)
        m.embedding = self.embedding.copy()
        m.output = self.output.copy()
        return m

    def predict_logits(self, token_ids: np.ndarray) -> np.ndarray:
        h = self.embedding[token_ids]
        return h @ self.output

    def average_loss(self, tokens: np.ndarray) -> float:
        x = tokens[:-1]
        y = tokens[1:]
        probs = softmax(self.predict_logits(x))
        return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))

    def local_sgd_step(self, tokens: np.ndarray, lr: float, batch_size: int) -> None:
        if len(tokens) < 3:
            return
        idx = np.random.randint(0, len(tokens) - 1, size=batch_size)
        x = tokens[idx]
        y = tokens[idx + 1]

        h = self.embedding[x]
        logits = h @ self.output
        probs = softmax(logits)
        probs[np.arange(batch_size), y] -= 1.0
        probs /= batch_size

        grad_output = h.T @ probs
        grad_h = probs @ self.output.T

        grad_embedding = np.zeros_like(self.embedding)
        for i, token in enumerate(x):
            grad_embedding[token] += grad_h[i]

        self.output -= lr * grad_output
        self.embedding -= lr * grad_embedding

    def get_weights(self) -> dict[str, np.ndarray]:
        return {"embedding": self.embedding.copy(), "output": self.output.copy()}

    def set_weights(self, w: dict[str, np.ndarray]) -> None:
        self.embedding = w["embedding"].copy()
        self.output = w["output"].copy()
