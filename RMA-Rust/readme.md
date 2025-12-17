# Depuis la racine
cargo run --example simple_demo

# Évaluation complète
cd rma-evaluation
cargo run

cargo run --example deep_prediction
cargo bench
cd rma-evaluation && cargo run