from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mlp:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @staticmethod
    def create(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 1) -> "Mlp":
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, output_dim)).astype(np.float32)
        b2 = np.zeros(output_dim, dtype=np.float32)
        return Mlp(w1=w1, b1=b1, w2=w2, b2=b2)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden_pre = x @ self.w1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        out = hidden @ self.w2 + self.b2
        return hidden, out

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[None, :]
        return self.forward(x.astype(np.float32))[1]

    def save(self, path: Path, **metadata: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, **metadata)

    @staticmethod
    def load(path: Path) -> "Mlp":
        data = np.load(path, allow_pickle=False)
        return Mlp(w1=data["w1"], b1=data["b1"], w2=data["w2"], b2=data["b2"])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def train_bce(
    model: Mlp,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 16,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 1,
) -> Mlp:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx].astype(np.float32)
            yb = y[idx].astype(np.float32)
            hidden, logits = model.forward(xb)
            prob = sigmoid(logits)
            grad_logits = (prob - yb) / max(1, xb.shape[0])
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ model.w2.T
            grad_hidden[hidden <= 0.0] = 0.0
            grad_w1 = xb.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            model.w2 -= lr * grad_w2.astype(np.float32)
            model.b2 -= lr * grad_b2.astype(np.float32)
            model.w1 -= lr * grad_w1.astype(np.float32)
            model.b1 -= lr * grad_b1.astype(np.float32)
    return model
