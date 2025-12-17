use criterion::{criterion_group, criterion_main, Criterion};
use rma_core::RelationalMemoryAccumulatorDeep;

fn bench_step(c: &mut Criterion) {
    let mut rma = RelationalMemoryAccumulatorDeep::new(3, vec![64, 32], 1, 0.95, 0.6, 10, 200);

    c.bench_function("RMA Deep step", |b| {
        b.iter(|| rma.step(&vec![0.5, 0.7, 0.2]));
    });
}

criterion_group!(benches, bench_step);
criterion_main!(benches);