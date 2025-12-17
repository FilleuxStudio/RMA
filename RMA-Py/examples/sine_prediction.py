"""
Exemple de prédiction sur une sinusoïde avec bruit en utilisant RMA Deep
"""

import numpy as np
import matplotlib.pyplot as plt
from rma.core import RelationalMemoryAccumulatorDeep

def main():
    print("=== Prédiction sinusoïdale avec RMA Deep ===\n")

    # Modèle
    rma = RelationalMemoryAccumulatorDeep(
        input_dim=1,
        hidden_sizes=[64, 32],
        output_dim=1,
        alpha=0.95,
        beta=0.6,
        k=8,
        max_ltm=200
    )

    # Génération de données
    t = np.linspace(0, 20, 200)
    true_values = np.sin(t)
    noisy_input = true_values + 0.1 * np.random.randn(len(t))

    predictions = []

    # Prédiction séquentielle
    for i in range(1, len(t)):
        input_val = noisy_input[i-1]
        pred = rma.step(np.array([input_val]))[0]
        predictions.append(pred)

    predictions = np.array(predictions)

    # Affichage
    plt.figure(figsize=(12, 6))
    plt.plot(t[1:], true_values[1:], label="Vraie sinusoïde", linewidth=2)
    plt.plot(t[1:], noisy_input[1:], label="Entrée bruitée", alpha=0.6, linestyle="--")
    plt.plot(t[1:], predictions, label="Prédiction RMA Deep", linewidth=2)
    plt.legend()
    plt.title("Prédiction de sinusoïde avec RMA Deep")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()