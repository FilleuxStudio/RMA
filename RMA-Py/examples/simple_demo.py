# examples/simple_demo.py
from rma.core import RelationalMemoryAccumulatorDeep

def main():
    print("=== Démo RMA Deep en Python ===\n")

    rma = RelationalMemoryAccumulatorDeep(
        input_dim=3,
        hidden_sizes=[64, 32],
        output_dim=1,
        alpha=0.95,
        beta=0.6,
        k=10,
        max_ltm=200
    )

    metrics = [
        [20.0, 40.0, 0.0],   # Normal
        [70.0, 65.0, 3.0],   # Stress
        [92.0, 78.0, 15.0],  # Critique
    ]

    print("Métriques [CPU, Temp, Err] → Score de risque")
    for m in metrics:
        out = rma.step(m)
        risk = out[0]
        alert = "⚠️ RISQUE ÉLEVÉ" if risk > 3.0 else "Normal"
        print(f"{m} → Risk: {risk:.2f} → {alert}")

if __name__ == "__main__":
    main()