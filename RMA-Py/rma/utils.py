"""
Fonctions utilitaires pour le projet RMA
"""

import numpy as np
from typing import Tuple

def generate_sine_data(
    n_samples: int = 300,
    seq_length: int = 10,
    noise_level: float = 0.12,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère des données sinusoïdales avec bruit pour tester les modèles récurrents.
    
    Retourne : (X_train, y_train, X_test, y_test)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_samples + seq_length, n_samples + seq_length)
    data = np.sin(t / 10.0) + noise_level * rng.uniform(-1, 1, size=t.shape)

    X = []
    y = []
    for i in range(n_samples):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)