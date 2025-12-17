use rma_core::RelationalMemoryAccumulatorDeep;
use std::f64::consts::PI;

fn main() {
    let mut rma = RelationalMemoryAccumulatorDeep::new(
        1, vec![64, 32], 1, 0.95, 0.6, 8, 200
    );

    println!("Prédiction sur sinusoïde avec RMA Deep");
    for i in 0..50 {
        let t = i as f64 / 5.0;
        let input = (t + 10.0).sin(); // Décalage pour "prédire"
        let pred = rma.step(&vec![(t.sin())])[0];
        println!("t={:.2} | réel={:.3} | prédit={:.3}", t, (t + 1.0).sin(), pred);
    }
}