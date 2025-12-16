use rma_core::RelationalMemoryAccumulatorDeep;

fn main() {
    let mut rma = RelationalMemoryAccumulatorDeep::new(
        3, vec![64, 32], 1, 0.95, 0.6, 10, 200,
    );

    let metrics = vec![vec![85.0, 72.0, 8.0], vec![92.0, 78.0, 15.0]];
    for m in metrics {
        let out = rma.step(&m);
        println!("Input: {:?} → Risk: {:.2}", m, out[0]);
    }
}