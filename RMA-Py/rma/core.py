"""
Relational Memory Accumulator (RMA) - Implémentation Python

Concept original : mémoire relationnelle explicite (STM + LTM)
avec deux versions :
- RelationalMemoryAccumulator : sortie linéaire simple
- RelationalMemoryAccumulatorDeep : sortie via MLP configurable

Compatible Python 3.11 - 3.13
Dépendance unique : numpy
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

class RelationalMemoryAccumulator:
    """Version originale légère du RMA : sortie = W @ combined + b"""

    def __init__(
        self,
        dim: int = 10,
        alpha: float = 0.9,
        beta: float = 0.5,
        k: int = 5,
        max_ltm: int = 100,
    ):
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.max_ltm = max_ltm

        self.stm = np.zeros(dim)
        self.ltm: List[np.ndarray] = []

        # Poids linéaires simples
        self.W = np.random.uniform(-1.0, 1.0, size=(dim, dim))
        self.b = np.random.uniform(-1.0, 1.0, size=(dim,))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _compute_relation(self, x: np.ndarray) -> np.ndarray:
        if not self.ltm:
            return np.ones(self.dim)

        sims = [self._cosine_similarity(x, m) for m in self.ltm]
        top_k_sims = sorted(sims, reverse=True)[:self.k]
        mean_sim = np.mean(top_k_sims)
        return np.full(self.dim, mean_sim)

    def _update_ltm(self, x: np.ndarray):
        if len(self.ltm) < self.max_ltm:
            self.ltm.append(x.copy())
        else:
            sims = [self._cosine_similarity(x, m) for m in self.ltm]
            min_idx = int(np.argmin(sims))
            if sims[min_idx] < 0.4:
                self.ltm[min_idx] = x.copy()

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.shape != (self.dim,):
            raise ValueError(f"Entrée doit être de dimension {self.dim}")

        r_t = self._compute_relation(x)

        self.stm = self.alpha * self.stm + (1 - self.alpha) * (x * r_t)

        self._update_ltm(x)

        ltm_mean = np.mean(self.ltm, axis=0) if self.ltm else np.zeros(self.dim)

        combined = self.stm + self.beta * ltm_mean

        return self.W @ combined + self.b

    def reset(self):
        """Réinitialise STM et LTM"""
        self.stm.fill(0.0)
        self.ltm.clear()

    # Compatibilité avec évaluation
    def get_combined_state(self) -> np.ndarray:
        ltm_mean = np.mean(self.ltm, axis=0) if self.ltm else np.zeros(self.dim)
        return self.stm + self.beta * ltm_mean

    def adjust_weights(self, learning_rate: float, error: float):
        if self.dim != 1:
            raise NotImplementedError("adjust_weights seulement pour dim=1")
        combined = self.get_combined_state()
        self.W[0, 0] -= learning_rate * error * combined[0]
        self.b[0] -= learning_rate * error


class RelationalMemoryAccumulatorDeep:
    """Version puissante avec MLP configurable après la combinaison STM + LTM"""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int = 1,
        alpha: float = 0.95,
        beta: float = 0.6,
        k: int = 8,
        max_ltm: int = 200,
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.max_ltm = max_ltm

        self.stm = np.zeros(input_dim)
        self.ltm: List[np.ndarray] = []

        # Construction du MLP
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            bound = np.sqrt(6.0 / (in_size + out_size))
            W = np.random.uniform(-bound, bound, size=(out_size, in_size))
            b = np.zeros(out_size)
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _compute_relation(self, x: np.ndarray) -> np.ndarray:
        if not self.ltm:
            return np.ones(self.input_dim)

        sims = [self._cosine_similarity(x, m) for m in self.ltm]
        top_k = sorted(sims, reverse=True)[:self.k]
        mean_sim = np.mean(top_k)
        return np.full(self.input_dim, mean_sim)

    def _update_ltm(self, x: np.ndarray):
        if len(self.ltm) < self.max_ltm:
            self.ltm.append(x.copy())
        else:
            sims = [self._cosine_similarity(x, m) for m in self.ltm]
            min_idx = int(np.argmin(sims))
            if sims[min_idx] < 0.4:
                self.ltm[min_idx] = x.copy()

    def _forward_mlp(self, x: np.ndarray) -> np.ndarray:
        act = x
        for i in range(len(self.weights) - 1):
            act = self.weights[i] @ act + self.biases[i]
            act = self._relu(act)
        # Dernière couche linéaire
        act = self.weights[-1] @ act + self.biases[-1]
        return act

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        if x.shape != (self.input_dim,):
            raise ValueError(f"Entrée doit être de dimension {self.input_dim}")

        r_t = self._compute_relation(x)

        self.stm = self.alpha * self.stm + (1.0 - self.alpha) * (x * r_t)

        self._update_ltm(x)

        ltm_mean = np.mean(self.ltm, axis=0) if self.ltm else np.zeros(self.input_dim)

        combined = self.stm + self.beta * ltm_mean

        return self._forward_mlp(combined)

    def reset(self):
        """Réinitialise STM et LTM"""
        self.stm.fill(0.0)
        self.ltm.clear()