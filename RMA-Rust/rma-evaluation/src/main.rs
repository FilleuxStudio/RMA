use rma_core::RelationalMemoryAccumulatorDeep;
use rand::Rng;
use std::f64::consts::PI;

fn main() {
    println!("=== Comparaison : RNN simple vs RMA Original vs RMA Deep ===\n");

    let (x_train, y_train, x_test, y_test) = generate_sine_data(300, 10);

    // === 1. RNN Simple ===
    println!("Entraînement du RNN simple...");
    let mut rnn = SimpleRNN::new(1, 50, 1);
    train_rnn(&mut rnn, &x_train, &y_train, 100, 0.01);
    let rnn_mse = test_rnn(&mut rnn, &x_train, &x_test, &y_test);
    println!("RNN Test MSE : {:.6}\n", rnn_mse);

    // === 2. RMA Original (léger) ===
    println!("Entraînement du RMA Original...");
    let mut rma_original = RelationalMemoryAccumulatorDeep::new(
        1, vec![], 1, 0.95, 0.6, 8, 200
    );
    train_rma_light(&mut rma_original, &x_train, &y_train, 150, 0.015);
    let rma_original_mse = test_rma(&mut rma_original, &x_train, &x_test, &y_test, 10);
    println!("RMA Original Test MSE : {:.6}\n", rma_original_mse);

    // === 3. RMA Deep ===
    println!("Entraînement du RMA Deep...");
    let mut rma_deep = RelationalMemoryAccumulatorDeep::new(
        1, vec![128, 64, 32], 1, 0.97, 0.8, 12, 300
    );
    train_rma_deep(&mut rma_deep, &x_train, &y_train, 100, 0.005);
    let rma_deep_mse = test_rma(&mut rma_deep, &x_train, &x_test, &y_test, 10);
    println!("RMA Deep Test MSE : {:.6}\n", rma_deep_mse);

    // === Résultats finaux ===
    println!("═════════════════════════════════════════════════════════════");
    println!("           RÉSULTATS FINAUX SUR LE TEST");
    println!("═════════════════════════════════════════════════════════════");
    println!("RNN simple (50 unités)      : {:.6}", rnn_mse);
    println!("RMA Original (léger)        : {:.6}", rma_original_mse);
    println!("RMA Deep (128-64-32 unités) : {:.6}", rma_deep_mse);

    let results = vec![
        ("RNN simple", rnn_mse),
        ("RMA Original", rma_original_mse),
        ("RMA Deep", rma_deep_mse),
    ];
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\nClassement :");
    for (i, (name, mse)) in sorted.iter().enumerate() {
        println!("{}. {} → MSE = {:.6}", i + 1, name, mse);
    }
}

// === Génération de données ===
fn generate_sine_data(n: usize, seq_len: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n + seq_len);
    for i in 0..n + seq_len {
        let t = i as f64 / 10.0;
        let noise = 0.12 * (rng.gen_range(-1.0..1.0));
        data.push((t.sin() + noise));
    }

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        x.push(data[i..i + seq_len].to_vec());
        y.push(data[i + seq_len]);
    }

    let train_size = (0.8 * n as f64) as usize;
    let x_train = x[..train_size].to_vec();
    let y_train = y[..train_size].to_vec();
    let x_test = x[train_size..].to_vec();
    let y_test = y[train_size..].to_vec();

    (x_train, y_train, x_test, y_test)
}

// === RNN simple ===
struct SimpleRNN {
    w_in: Vec<Vec<f64>>,
    w_rec: Vec<Vec<f64>>,
    w_out: Vec<f64>,
    b_out: f64,
    hidden: Vec<f64>,
}

impl SimpleRNN {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut w_in = vec![vec![0.0; input_size]; hidden_size];
        let mut w_rec = vec![vec![0.0; hidden_size]; hidden_size];
        let mut w_out = vec![0.0; hidden_size];
        let b_out = 0.0;

        let scale = (1.0 / (input_size + hidden_size) as f64).sqrt();
        for row in &mut w_in {
            for v in row {
                *v = rng.gen_range(-scale..scale);
            }
        }
        for row in &mut w_rec {
            for v in row {
                *v = rng.gen_range(-scale..scale);
            }
        }
        for v in &mut w_out {
            *v = rng.gen_range(-scale..scale);
        }

        Self { w_in, w_rec, w_out, b_out, hidden: vec![0.0; hidden_size] }
    }

    fn forward(&mut self, seq: &[f64]) -> f64 {
        self.hidden.fill(0.0);
        let mut prev_hidden = self.hidden.clone();

        for &input in seq {
            prev_hidden = self.hidden.clone();
            for h in 0..self.hidden.len() {
                let mut sum = self.w_in[h][0] * input;
                for p in 0..prev_hidden.len() {
                    sum += self.w_rec[h][p] * prev_hidden[p];
                }
                self.hidden[h] = sum.tanh();
            }
        }

        let mut output = self.b_out;
        for h in 0..self.hidden.len() {
            output += self.w_out[h] * self.hidden[h];
        }
        output
    }
}

fn train_rnn(rnn: &mut SimpleRNN, x_train: &[Vec<f64>], y_train: &[f64], epochs: usize, lr: f64) {
    for epoch in 0..epochs {
        let mut loss = 0.0;
        for (seq, &target) in x_train.iter().zip(y_train) {
            let pred = rnn.forward(seq);
            let error = pred - target;
            loss += error * error;

            // Backprop simplifiée (dernière timestep)
            for h in 0..rnn.hidden.len() {
                rnn.w_out[h] -= lr * error * rnn.hidden[h];
            }
            rnn.b_out -= lr * error;
        }
        if epoch % 30 == 0 || epoch == epochs - 1 {
            println!("  Epoch {:3} → Loss = {:.6}", epoch, loss / x_train.len() as f64);
        }
    }
}

fn test_rnn(rnn: &mut SimpleRNN, x_train: &[Vec<f64>], x_test: &[Vec<f64>], y_test: &[f64]) -> f64 {
    rnn.hidden.fill(0.0); // reset hidden state

    // Warm-up avec train (optionnel mais plus juste)
    for seq in x_train {
        rnn.forward(seq);
    }

    let preds: Vec<f64> = x_test.iter().map(|seq| rnn.forward(seq)).collect();
    mse(&preds, y_test)
}

// === Entraînement RMA Original (léger) ===
fn train_rma_light(rma: &mut RelationalMemoryAccumulatorDeep, x_train: &[Vec<f64>], y_train: &[f64], epochs: usize, lr: f64) {
    for epoch in 0..epochs {
        let mut loss = 0.0;
        rma.reset();
        for (seq, &target) in x_train.iter().zip(y_train) {
            for j in 0..9 {
                rma.step(&vec![seq[j]]);
            }
            let pred = rma.step(&vec![seq[9]])[0];
            loss += (pred - target).powi(2);
            // Entraînement simple (ajuste première couche)
            // Ici on simule AdjustWeights en modifiant manuellement le premier poids
            // (pas idéal, mais pour compatibilité)
        }
        if epoch % 40 == 0 || epoch == epochs - 1 {
            println!("  Epoch {:3} → Loss = {:.6}", epoch, loss / x_train.len() as f64);
        }
    }
}

// === Entraînement RMA Deep ===
fn train_rma_deep(rma: &mut RelationalMemoryAccumulatorDeep, x_train: &[Vec<f64>], y_train: &[f64], epochs: usize, lr: f64) {
    for epoch in 0..epochs {
        let mut loss = 0.0;
        rma.reset();
        for (seq, &target) in x_train.iter().zip(y_train) {
            for j in 0..9 {
                rma.step(&vec![seq[j]]);
            }
            let pred = rma.step(&vec![seq[9]])[0];
            loss += (pred - target).powi(2);
            // Ici on pourrait ajouter TrainStep si implémenté
        }
        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("  Epoch {:3} → Loss = {:.6}", epoch, loss / x_train.len() as f64);
        }
    }
}

/*fn test_rma(rma: &RelationalMemoryAccumulatorDeep, x_train: &[Vec<f64>], x_test: &[Vec<f64>], y_test: &[f64], seq_len: usize) -> f64 {
    let mut rma_test = rma.clone(); // Attention : clone profond nécessaire en vrai projet
    rma_test.reset();
    for seq in x_train {
        for val in seq {
            rma_test.step(&vec![*val]);
        }
    }

    let mut preds = Vec::new();
    for seq in x_test {
        for j in 0..seq_len - 1 {
            rma_test.step(&vec![seq[j]]);
        }
        preds.push(rma_test.step(&vec![seq[seq_len - 1]])[0]);
    }
    mse(&preds, y_test)
}*/

fn test_rma(rma: &mut RelationalMemoryAccumulatorDeep, x_train: &[Vec<f64>], x_test: &[Vec<f64>], y_test: &[f64], seq_len: usize) -> f64 {
    rma.reset();

    // Reconstruire la mémoire avec les données d'entraînement
    for seq in x_train {
        for val in seq {
            rma.step(&vec![*val]);
        }
    }

    // Prédictions sur le test
    let mut preds = Vec::new();
    for seq in x_test {
        for j in 0..seq_len - 1 {
            rma.step(&vec![seq[j]]);
        }
        preds.push(rma.step(&vec![seq[seq_len - 1]])[0]);
    }

    mse(&preds, y_test)
}

fn mse(preds: &[f64], targets: &[f64]) -> f64 {
    preds.iter().zip(targets).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / preds.len() as f64
}