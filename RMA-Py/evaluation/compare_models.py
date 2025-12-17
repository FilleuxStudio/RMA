"""
Comparaison complète entre :
- RNN simple (vanilla)
- RMA Original (léger)
- RMA Deep (puissant)
sur une tâche de prédiction de sinusoïde bruitée
"""

import numpy as np
from rma.core import RelationalMemoryAccumulator, RelationalMemoryAccumulatorDeep
from rma.utils import generate_sine_data, mse

class SimpleRNN:
    """RNN vanilla simple en NumPy pour comparaison"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.hidden_size = hidden_size
        
        # Poids
        self.W_in = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.W_rec = np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        self.W_out = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
        self.b_out = np.zeros(output_size)
        
        self.hidden = np.zeros(hidden_size)

    def forward(self, inputs: np.ndarray) -> float:
        self.hidden = np.zeros(self.hidden_size)
        for x in inputs:
            self.hidden = np.tanh(self.W_in @ x + self.W_rec @ self.hidden)
        return (self.W_out @ self.hidden + self.b_out)[0]

    def reset(self):
        self.hidden = np.zeros(self.hidden_size)

def train_rnn(model: SimpleRNN, X_train, y_train, epochs: int = 100, lr: float = 0.01):
    for epoch in range(epochs):
        total_loss = 0
        for seq, target in zip(X_train, y_train):
            pred = model.forward(seq.reshape(-1, 1))
            error = pred - target
            total_loss += error ** 2
            
            # Backprop simplifiée (dernière étape)
            model.W_out -= lr * error * model.hidden.reshape(1, -1)
            model.b_out -= lr * error
        if epoch % 30 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d} → Loss = {total_loss / len(X_train):.6f}")

def main():
    print("=== Comparaison : RNN vs RMA Original vs RMA Deep ===\n")

    X_train, y_train, X_test, y_test = generate_sine_data(n_samples=240, seq_length=10)

    # === 1. RNN Simple ===
    print("Entraînement du RNN simple (50 unités cachées)...")
    rnn = SimpleRNN(input_size=1, hidden_size=50, output_size=1)
    train_rnn(rnn, X_train.reshape(-1, 10, 1), y_train, epochs=100, lr=0.01)
    rnn_preds = [rnn.forward(seq.reshape(10, 1)) for seq in X_test.reshape(-1, 10, 1)]
    rnn_mse = mse(y_test, np.array(rnn_preds))
    print(f"RNN Test MSE : {rnn_mse:.6f}\n")

    # === 2. RMA Original ===
    print("Entraînement du RMA Original...")
    rma_original = RelationalMemoryAccumulator(dim=1, alpha=0.95, beta=0.6, k=8, max_ltm=200)
    for epoch in range(150):
        total_loss = 0
        rma_original.reset()
        for seq, target in zip(X_train, y_train):
            for j in range(9):
                rma_original.step(np.array([seq[j]]))
            pred = rma_original.step(np.array([seq[9]]))[0]
            error = pred - target
            total_loss += error ** 2
            rma_original.adjust_weights(0.015, error)
        if epoch % 50 == 0 or epoch == 149:
            print(f"  Epoch {epoch:3d} → Loss = {total_loss / len(X_train):.6f}")

    rma_original_preds = []
    rma_original.reset()
    for seq in X_train:
        for val in seq:
            rma_original.step(np.array([val]))
    for seq in X_test:
        for j in range(9):
            rma_original.step(np.array([seq[j]]))
        rma_original_preds.append(rma_original.step(np.array([seq[9]]))[0])
    rma_original_mse = mse(y_test, np.array(rma_original_preds))
    print(f"RMA Original Test MSE : {rma_original_mse:.6f}\n")

    # === 3. RMA Deep ===
    print("Entraînement du RMA Deep...")
    rma_deep = RelationalMemoryAccumulatorDeep(
        input_dim=1,
        hidden_sizes=[128, 64, 32],
        output_dim=1,
        alpha=0.97,
        beta=0.8,
        k=12,
        max_ltm=300
    )

    for epoch in range(100):
        total_loss = 0
        rma_deep.reset()
        for seq, target in zip(X_train, y_train):
            for j in range(9):
                rma_deep.step(np.array([seq[j]]))
            pred = rma_deep.step(np.array([seq[9]]))[0]
            total_loss += (pred - target) ** 2
        if epoch % 20 == 0 or epoch == 99:
            print(f"  Epoch {epoch:3d} → Loss = {total_loss / len(X_train):.6f}")

    rma_deep_preds = []
    rma_deep.reset()
    for seq in X_train:
        for val in seq:
            rma_deep.step(np.array([val]))
    for seq in X_test:
        for j in range(9):
            rma_deep.step(np.array([seq[j]]))
        rma_deep_preds.append(rma_deep.step(np.array([seq[9]]))[0])
    rma_deep_mse = mse(y_test, np.array(rma_deep_preds))
    print(f"RMA Deep Test MSE : {rma_deep_mse:.6f}\n")

    # === Résultats finaux ===
    print("═════════════════════════════════════════════════════════════")
    print("           RÉSULTATS FINAUX SUR LE TEST")
    print("═════════════════════════════════════════════════════════════")
    print(f"RNN simple (50 unités)      : {rnn_mse:.6f}")
    print(f"RMA Original (léger)        : {rma_original_mse:.6f}")
    print(f"RMA Deep (128-64-32 unités) : {rma_deep_mse:.6f}")

    results = sorted([
        ("RNN simple", rnn_mse),
        ("RMA Original", rma_original_mse),
        ("RMA Deep", rma_deep_mse),
    ], key=lambda x: x[1])

    print("\nClassement :")
    for i, (name, score) in enumerate(results):
        print(f"{i+1}. {name} → MSE = {score:.6f}")

if __name__ == "__main__":
    main()