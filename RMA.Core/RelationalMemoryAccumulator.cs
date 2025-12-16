using System;
using System.Collections.Generic;
using System.Linq;

namespace RMA.Core
{
    /// <summary>
    /// Relational Memory Accumulator (RMA)
    /// Un concept original léger pour traiter des séquences temporelles
    /// avec une mémoire à court terme (STM) et à long terme (LTM) basée sur des relations (similarité cosinus).
    /// 
    /// Objectif : "nouveau concept" de réseau de neurones artificiels façon plus simple et interprétable,
    /// tout en gardant une capacité à "se souvenir" des patterns passés via des relations.
    /// 
    /// Applications possibles :
    /// - Prédiction de pannes à partir de métriques SNMP/WMI
    /// - Correction d'équilibre d'un robot humanoïde (gyro + accéléromètre)
    /// - Détection d'anomalies dans des séries temporelles
    /// - Prédiction légère de séquences numériques ou embeddings textuels
    /// </summary>
    public class RelationalMemoryAccumulator
    {
        // Dimensions du vecteur d'entrée (ex: 3 pour [CPU, température, erreurs])
        private readonly int dim;

        // Facteur de rétention pour la mémoire courte (0.9 = oublie lentement)
        private readonly double alpha;

        // Poids de la mémoire longue dans le calcul final
        private readonly double beta;

        // Nombre de souvenirs les plus similaires à considérer pour les relations
        private readonly int k;

        // Taille maximale de la mémoire longue (évite la croissance infinie)
        private readonly int maxLtm;

        // Mémoire à court terme (état caché récurrent)
        private double[] stm;

        // Mémoire à long terme : liste de vecteurs passés jugés importants
        private List<double[]> ltm;

        // Matrice de poids pour transformer l'état combiné en sortie
        private double[,] W;

        // Biais pour la sortie
        private double[] b;

        // Générateur aléatoire pour initialiser les poids (peut être remplacé par un apprentissage)
        private static readonly Random rand = new Random();

        /// <summary>
        /// Constructeur du RMA
        /// </summary>
        /// <param name="dim">Dimension des vecteurs d'entrée</param>
        /// <param name="alpha">Facteur de dégradation STM (0.8-0.99 recommandé)</param>
        /// <param name="beta">Poids de la LTM dans la sortie (0.3-0.7 recommandé)</param>
        /// <param name="k">Nombre de souvenirs les plus similaires pour calculer la relation</param>
        /// <param name="maxLtm">Taille maximale de la mémoire longue</param>
        public RelationalMemoryAccumulator(int dim = 10,
                                           double alpha = 0.9,
                                           double beta = 0.5,
                                           int k = 5,
                                           int maxLtm = 100)
        {
            this.dim = dim;
            this.alpha = alpha;
            this.beta = beta;
            this.k = k;
            this.maxLtm = maxLtm;

            // Initialisation
            stm = new double[dim];
            ltm = new List<double[]>();

            // Initialisation aléatoire des poids (à affiner avec un apprentissage si besoin)
            W = new double[dim, dim];
            b = new double[dim];
            InitializeWeights();
        }

        /// <summary>
        /// Initialise les poids W et le biais b avec des valeurs aléatoires
        /// </summary>
        private void InitializeWeights()
        {
            for (int i = 0; i < dim; i++)
            {
                b[i] = rand.NextDouble() * 2 - 1; // [-1, 1]
                for (int j = 0; j < dim; j++)
                {
                    W[i, j] = rand.NextDouble() * 2 - 1; // [-1, 1]
                }
            }
        }

        /// <summary>
        /// Calcule la similarité cosinus entre deux vecteurs
        /// </summary>
        private double CosineSimilarity(double[] a, double[] b)
        {
            double dot = 0.0, normA = 0.0, normB = 0.0;
            for (int i = 0; i < dim; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            return dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-8);
        }

        /// <summary>
        /// Calcule le vecteur de relation R_t :
        /// Moyenne des similarités avec les k souvenirs les plus proches dans LTM
        /// </summary>
        private double[] ComputeRelation(double[] x)
        {
            if (ltm.Count == 0)
                return Enumerable.Repeat(1.0, dim).ToArray(); // Pas encore de mémoire → relation neutre

            // Calcul des similarités avec tous les éléments de LTM
            var similarities = ltm.Select(m => CosineSimilarity(x, m)).ToList();

            // Sélection des k plus grandes similarités
            var topKSimilarities = similarities
                .Select((sim, idx) => new { sim, idx })
                .OrderByDescending(p => p.sim)
                .Take(k)
                .Select(p => p.sim)
                .ToList();

            double meanSimilarity = topKSimilarities.Average();

            // Retourne un vecteur constant (on pourrait le moduler par dimension si besoin)
            return Enumerable.Repeat(meanSimilarity, dim).ToArray();
        }

        /// <summary>
        /// Met à jour la mémoire longue (LTM)
        /// Ajoute le vecteur si espace, sinon remplace le moins similaire si assez différent
        /// </summary>
        private void UpdateLtm(double[] x)
        {
            if (ltm.Count < maxLtm)
            {
                ltm.Add((double[])x.Clone());
            }
            else
            {
                // Trouve le souvenir le moins similaire
                var similarities = ltm.Select(m => CosineSimilarity(x, m)).ToArray();
                int minIdx = Array.IndexOf(similarities, similarities.Min());

                // Seuil : on remplace seulement si le nouveau est assez différent
                if (similarities[minIdx] < 0.4)
                {
                    ltm[minIdx] = (double[])x.Clone();
                }
            }
        }

        /// <summary>
        /// Traite une nouvelle entrée et retourne la sortie/prédiction/correction
        /// </summary>
        /// <param name="x">Vecteur d'entrée actuel</param>
        /// <returns>Vecteur de sortie (même dimension)</returns>
        public double[] Step(double[] x)
        {
            if (x.Length != dim)
                throw new ArgumentException($"L'entrée doit avoir {dim} dimensions.");

            // 1. Calcul de la relation avec le passé
            double[] R_t = ComputeRelation(x);

            // 2. Mise à jour de la mémoire courte (accumulation pondérée + modulation par relation)
            for (int i = 0; i < dim; i++)
            {
                stm[i] = alpha * stm[i] + (1 - alpha) * (x[i] * R_t[i]);
            }

            // 3. Mise à jour de la mémoire longue
            UpdateLtm(x);

            // 4. Calcul de la moyenne de la mémoire longue
            double[] ltmMean = new double[dim];
            if (ltm.Count > 0)
            {
                for (int i = 0; i < dim; i++)
                {
                    ltmMean[i] = ltm.Average(vec => vec[i]);
                }
            }

            // 5. Combinaison STM + LTM et projection linéaire
            double[] combined = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                combined[i] = stm[i] + beta * ltmMean[i];
            }

            // 6. Sortie finale
            double[] output = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < dim; j++)
                {
                    sum += W[i, j] * combined[j];
                }
                output[i] = sum + b[i];
            }

            return output;
        }

        /// <summary>
        /// Réinitialise les mémoires (utile pour recommencer une nouvelle séquence)
        /// </summary>
        public void Reset()
        {
            stm = new double[dim];
            ltm.Clear();
        }

        /// <summary>
        /// Expose l'état combiné (STM + β * LTM_mean) pour l'entraînement
        /// </summary>
        public double[] GetCombinedState()
        {
            double[] ltmMean = new double[dim];
            if (ltm.Count > 0)
            {
                for (int i = 0; i < dim; i++)
                {
                    ltmMean[i] = ltm.Average(vec => vec[i]);
                }
            }

            double[] combined = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                combined[i] = stm[i] + beta * ltmMean[i];
            }

            return combined;
        }

        /// <summary>
        /// Ajuste les poids W et b avec descente de gradient (simple, pour dim=1)
        /// </summary>
        public void AdjustWeights(double learningRate, double error)
        {
            if (dim != 1)
                throw new InvalidOperationException("AdjustWeights est implémenté uniquement pour dim=1 dans cette évaluation.");

            double[] combined = GetCombinedState();
            W[0, 0] -= learningRate * error * combined[0];
            b[0] -= learningRate * error;
        }
    }
}