//! rma-core/src/original.rs
//! Version légère du Relational Memory Accumulator (sortie linéaire simple)

use rand::Rng;

/// Version originale du RMA : sortie = W * combined + b
pub struct RelationalMemoryAccumulator {
    dim: usize,
    alpha: f64,
    beta: f64,
    k: usize,
    max_ltm: usize,

    stm: Vec<f64>,
    ltm: Vec<Vec<f64>>,

    w: Vec<Vec<f64>>,  // Matrice dim x dim
    b: Vec<f64>,       // Biais dim
}

impl RelationalMemoryAccumulator {
    /// Constructeur de la version originale
    pub fn new(dim: usize, alpha: f64, beta: f64, k: usize, max_ltm: usize) -> Self {
        let mut rng = rand::thread_rng();

        let mut w = vec![vec![0.0; dim]; dim];
        let mut b = vec![0.0; dim];

        // Initialisation aléatoire simple [-1, 1]
        for i in 0..dim {
            b[i] = rng.gen_range(-1.0..1.0);
            for j in 0..dim {
                w[i][j] = rng.gen_range(-1.0..1.0);
            }
        }

        Self {
            dim,
            alpha,
            beta,
            k,
            max_ltm,
            stm: vec![0.0; dim],
            ltm: Vec::with_capacity(max_ltm),
            w,
            b,
        }
    }

    /// Calcule la similarité cosinus entre deux vecteurs
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

    /// Calcule le vecteur de relation R_t (moyenne des k meilleures similarités)
    fn compute_relation(&self, x: &[f64]) -> Vec<f64> {
        if self.ltm.is_empty() {
            return vec![1.0; self.dim];
        }

        let mut sims: Vec<f64> = self.ltm
            .iter()
            .map(|m| Self::cosine_similarity(x, m))
            .collect();

        sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mean_sim = sims.iter().take(self.k).sum::<f64>() / self.k as f64;
        vec![mean_sim; self.dim]
    }

    /// Mise à jour de la mémoire longue
    fn update_ltm(&mut self, x: &[f64]) {
        if self.ltm.len() < self.max_ltm {
            self.ltm.push(x.to_vec());
        } else {
            let sims: Vec<f64> = self.ltm.iter().map(|m| Self::cosine_similarity(x, m)).collect();
            if let Some((min_idx, &min_sim)) = sims
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if min_sim < 0.4 {
                    self.ltm[min_idx] = x.to_vec();
                }
            }
        }
    }

    /// Étape principale : traitement d'une entrée et sortie linéaire
    pub fn step(&mut self, x: &[f64]) -> Vec<f64> {
        if x.len() != self.dim {
            panic!("Dimension d'entrée incorrecte : attendu {}, reçu {}", self.dim, x.len());
        }

        let r_t = self.compute_relation(x);

        // Mise à jour STM
        for i in 0..self.dim {
            self.stm[i] = self.alpha * self.stm[i] + (1.0 - self.alpha) * (x[i] * r_t[i]);
        }

        self.update_ltm(x);

        // Moyenne LTM
        let ltm_mean: Vec<f64> = if self.ltm.is_empty() {
            vec![0.0; self.dim]
        } else {
            let count = self.ltm.len() as f64;
            (0..self.dim)
                .map(|i| self.ltm.iter().map(|v| v[i]).sum::<f64>() / count)
                .collect()
        };

        // Combinaison STM + β * LTM_mean
        let mut combined = vec![0.0; self.dim];
        for i in 0..self.dim {
            combined[i] = self.stm[i] + self.beta * ltm_mean[i];
        }

        // Sortie linéaire W * combined + b
        let mut output = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut sum = self.b[i];
            for j in 0..self.dim {
                sum += self.w[i][j] * combined[j];
            }
            output[i] = sum;
        }

        output
    }

    /// Réinitialise les mémoires STM et LTM
    pub fn reset(&mut self) {
        self.stm.fill(0.0);
        self.ltm.clear();
    }

    /// Expose l'état combiné pour l'entraînement (compatibilité évaluation)
    pub fn get_combined_state(&self) -> Vec<f64> {
        let ltm_mean: Vec<f64> = if self.ltm.is_empty() {
            vec![0.0; self.dim]
        } else {
            let count = self.ltm.len() as f64;
            (0..self.dim)
                .map(|i| self.ltm.iter().map(|v| v[i]).sum::<f64>() / count)
                .collect()
        };

        let mut combined = vec![0.0; self.dim];
        for i in 0..self.dim {
            combined[i] = self.stm[i] + self.beta * ltm_mean[i];
        }
        combined
    }

    /// Ajustement simple des poids (pour dim=1, compatibilité évaluation)
    pub fn adjust_weights(&mut self, learning_rate: f64, error: f64) {
        if self.dim != 1 {
            panic!("adjust_weights est implémenté uniquement pour dim=1");
        }
        let combined = self.get_combined_state();
        self.w[0][0] -= learning_rate * error * combined[0];
        self.b[0] -= learning_rate * error;
    }
}