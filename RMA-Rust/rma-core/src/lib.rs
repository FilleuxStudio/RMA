//! Relational Memory Accumulator (RMA)
//! 
//! Implémentation en Rust du concept original RMA et RMA Deep.
//! Architecture légère avec mémoire relationnelle explicite.
//! 
//! Auteur : [Ton nom/pseudo]
//! Date : Décembre 2025

use rand::Rng;
use std::f64::consts::SQRT_2;

pub mod original;
pub mod deep;

pub use original::RelationalMemoryAccumulator;
pub use deep::RelationalMemoryAccumulatorDeep;

/*
/// Relational Memory Accumulator - Version Deep (avec MLP configurable)
pub struct RelationalMemoryAccumulatorDeep {
    input_dim: usize,
    hidden_sizes: Vec<usize>,
    output_dim: usize,
    alpha: f64,
    beta: f64,
    k: usize,
    max_ltm: usize,

    stm: Vec<f64>,
    ltm: Vec<Vec<f64>>,

    // Poids du MLP : weights[i] = matrice entre couche i et i+1 (rows = out, cols = in)
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

impl RelationalMemoryAccumulatorDeep {
    /// Constructeur complet
    pub fn new(
        input_dim: usize,
        hidden_sizes: Vec<usize>,
        output_dim: usize,
        alpha: f64,
        beta: f64,
        k: usize,
        max_ltm: usize,
    ) -> Self {
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(&hidden_sizes);
        layer_sizes.push(output_dim);

        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let in_size = layer_sizes[i];
            let out_size = layer_sizes[i + 1];

            // Initialisation Xavier/Glorot
            let bound = (6.0f64).sqrt() / ((in_size + out_size) as f64).sqrt();

            let mut layer_weights = vec![vec![0.0f64; in_size]; out_size];
            let mut layer_bias = vec![0.0f64; out_size];

            for row in &mut layer_weights {
                for w in row {
                    *w = rng.gen_range(-bound..bound);
                }
            }

            weights.push(layer_weights);
            biases.push(layer_bias);
        }

        Self {
            input_dim,
            hidden_sizes,
            output_dim,
            alpha,
            beta,
            k,
            max_ltm,
            stm: vec![0.0; input_dim],
            ltm: Vec::with_capacity(max_ltm),
            weights,
            biases,
        }
    }

    fn relu(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    fn mat_vec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        matrix.iter().map(|row| row.iter().zip(vector).map(|(&w, &v)| w * v).sum()).collect()
    }

    fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-8)
    }

    fn compute_relation(&self, x: &[f64]) -> Vec<f64> {
        if self.ltm.is_empty() {
            return vec![1.0; self.input_dim];
        }

        let mut sims: Vec<(f64, usize)> = self.ltm
            .iter()
            .enumerate()
            .map(|(i, m)| (Self::cosine_similarity(x, m), i))
            .collect();

        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mean_sim = sims.iter().take(self.k).map(|(s, _)| s).sum::<f64>() / self.k as f64;
        vec![mean_sim; self.input_dim]
    }

    fn update_ltm(&mut self, x: &[f64]) {
        if self.ltm.len() < self.max_ltm {
            self.ltm.push(x.to_vec());
        } else {
            let sims: Vec<f64> = self.ltm.iter().map(|m| Self::cosine_similarity(x, m)).collect();
            if let Some(min_idx) = sims.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i) {
                if sims[min_idx] < 0.4 {
                    self.ltm[min_idx] = x.to_vec();
                }
            }
        }
    }

    fn forward_mlp(&self, input: &[f64]) -> Vec<f64> {
        let mut act = input.to_vec();

        for i in 0..self.weights.len() - 1 {
            act = Self::mat_vec_mul(&self.weights[i], &act);
            act = Self::vec_add(&act, &self.biases[i]);
            act = Self::relu(&act);
        }

        // Dernière couche linéaire
        act = Self::mat_vec_mul(self.weights.last().unwrap(), &act);
        act = Self::vec_add(&act, self.biases.last().unwrap());

        act
    }

    /// Étape principale
    pub fn step(&mut self, x: &[f64]) -> Vec<f64> {
        if x.len() != self.input_dim {
            panic!("Dimension d'entrée incorrecte : attendu {}, reçu {}", self.input_dim, x.len());
        }

        let r_t = self.compute_relation(x);

        for i in 0..self.input_dim {
            self.stm[i] = self.alpha * self.stm[i] + (1.0 - self.alpha) * (x[i] * r_t[i]);
        }

        self.update_ltm(x);

        let ltm_mean: Vec<f64> = if self.ltm.is_empty() {
            vec![0.0; self.input_dim]
        } else {
            let count = self.ltm.len() as f64;
            (0..self.input_dim)
                .map(|i| self.ltm.iter().map(|v| v[i]).sum::<f64>() / count)
                .collect()
        };

        let mut combined = vec![0.0; self.input_dim];
        for i in 0..self.input_dim {
            combined[i] = self.stm[i] + self.beta * ltm_mean[i];
        }

        self.forward_mlp(&combined)
    }

    /// Réinitialise les mémoires
    pub fn reset(&mut self) {
        self.stm.fill(0.0);
        self.ltm.clear();
    }
}*/