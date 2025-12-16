using System;
using System.Collections.Generic;
using System.Linq;

namespace RMA.Core
{
    /// <summary>
    /// Relational Memory Accumulator - Version Deep
    /// 
    /// Architecture récurrente légère avec mémoire relationnelle (STM + LTM)
    /// enrichie d'un réseau fully-connected (MLP) configurable en sortie.
    /// 
    /// Permet d'améliorer fortement les performances sur des tâches complexes
    /// (prédiction de séries temporelles, détection d'anomalies, robotique...)
    /// tout en conservant l'interprétabilité et la légèreté du concept original.
    /// </summary>
    public class RelationalMemoryAccumulatorDeep
    {
        // Paramètres du cœur RMA
        private readonly int inputDim;      // Dimension des vecteurs d'entrée
        private readonly double alpha;      // Facteur de rétention mémoire courte (0.8-0.99)
        private readonly double beta;       // Importance de la mémoire longue dans la combinaison
        private readonly int k;             // Nombre de souvenirs les plus similaires pour la relation
        private readonly int maxLtm;        // Taille maximale de la mémoire longue

        // Architecture du MLP (réseau de sortie)
        private readonly int[] hiddenSizes; // Tailles des couches cachées (ex: {64, 32})
        private readonly int outputDim;     // Dimension de sortie (souvent 1 pour régression)

        // États internes du RMA
        private double[] stm;               // Short-Term Memory (mémoire à court terme)
        private List<double[]> ltm;         // Long-Term Memory (liste de vecteurs mémorisés)

        // Poids et biais du MLP (réseau profond)
        private List<double[,]> weights;    // weights[i] = matrice entre couche i et i+1
        private List<double[]> biases;      // biases[i] = biais de la couche i+1

        private static readonly Random rand = new Random();

        /// <summary>
        /// Constructeur principal permettant de définir l'architecture complète du modèle.
        /// </summary>
        /// <param name="inputDim">Dimension des données d'entrée (ex: 1 pour scalaire, 3 pour métriques serveur)</param>
        /// <param name="hiddenSizes">Tableau des tailles des couches cachées. Ex: new int[] {64, 32} pour 2 couches</param>
        /// <param name="outputDim">Dimension de sortie (1 pour prédire une valeur, inputDim pour reconstruction)</param>
        /// <param name="alpha">Facteur de rétention STM (plus proche de 1 = oublie moins vite)</param>
        /// <param name="beta">Poids de la mémoire longue dans l'état combiné</param>
        /// <param name="k">Nombre de souvenirs les plus similaires utilisés pour calculer la relation</param>
        /// <param name="maxLtm">Capacité maximale de la mémoire longue</param>
        public RelationalMemoryAccumulatorDeep(
            int inputDim,
            int[] hiddenSizes,
            int outputDim = 1,
            double alpha = 0.95,
            double beta = 0.6,
            int k = 8,
            int maxLtm = 200)
        {
            this.inputDim = inputDim;
            this.hiddenSizes = hiddenSizes ?? Array.Empty<int>();
            this.outputDim = outputDim;
            this.alpha = alpha;
            this.beta = beta;
            this.k = k;
            this.maxLtm = maxLtm;

            // Initialisation des mémoires
            stm = new double[inputDim];
            ltm = new List<double[]>();

            // Construction de l'architecture complète du MLP
            var layerSizes = new List<int> { inputDim };
            if (hiddenSizes.Length > 0)
                layerSizes.AddRange(hiddenSizes);
            layerSizes.Add(outputDim);

            weights = new List<double[,]>();
            biases = new List<double[]>();

            // Initialisation des poids avec Xavier/Glorot (meilleure convergence)
            for (int i = 0; i < layerSizes.Count - 1; i++)
            {
                int inSize = layerSizes[i];
                int outSize = layerSizes[i + 1];

                var w = new double[outSize, inSize];
                var b = new double[outSize];

                double bound = Math.Sqrt(6.0 / (inSize + outSize));
                for (int r = 0; r < outSize; r++)
                {
                    b[r] = 0.0; // Biais à zéro
                    for (int c = 0; c < inSize; c++)
                    {
                        w[r, c] = rand.NextDouble() * 2 * bound - bound;
                    }
                }

                weights.Add(w);
                biases.Add(b);
            }
        }

        /// <summary>
        /// Fonction d'activation ReLU (Rectified Linear Unit)
        /// </summary>
        private double[] Relu(double[] x)
        {
            return x.Select(v => Math.Max(0, v)).ToArray();
        }

        /// <summary>
        /// Multiplication matrice-vecteur (couche fully-connected)
        /// </summary>
        private double[] MatrixVectorMultiply(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[] result = new double[rows];

            for (int r = 0; r < rows; r++)
            {
                double sum = 0.0;
                for (int c = 0; c < cols; c++)
                {
                    sum += matrix[r, c] * vector[c];
                }
                result[r] = sum;
            }
            return result;
        }

        /// <summary>
        /// Addition vecteur + vecteur (ajout du biais)
        /// </summary>
        private double[] VectorAdd(double[] a, double[] b)
        {
            return a.Zip(b, (x, y) => x + y).ToArray();
        }

        /// <summary>
        /// Calcule la similarité cosinus entre deux vecteurs
        /// </summary>
        private double CosineSimilarity(double[] a, double[] b)
        {
            double dot = 0.0, normA = 0.0, normB = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            return dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-8);
        }

        /// <summary>
        /// Calcule le vecteur de relation R_t basé sur les k souvenirs les plus similaires dans LTM
        /// </summary>
        private double[] ComputeRelation(double[] x)
        {
            if (ltm.Count == 0)
                return Enumerable.Repeat(1.0, inputDim).ToArray(); // Pas de mémoire → relation neutre

            var similarities = ltm.Select(m => CosineSimilarity(x, m)).ToList();

            var topKSimilarities = similarities
                .OrderByDescending(s => s)
                .Take(k);

            double meanSimilarity = topKSimilarities.Average();

            return Enumerable.Repeat(meanSimilarity, inputDim).ToArray();
        }

        /// <summary>
        /// Met à jour la mémoire longue : ajoute ou remplace selon la similarité
        /// </summary>
        private void UpdateLtm(double[] x)
        {
            if (ltm.Count < maxLtm)
            {
                ltm.Add((double[])x.Clone());
            }
            else
            {
                var similarities = ltm.Select(m => CosineSimilarity(x, m)).ToArray();
                int minIdx = Array.IndexOf(similarities, similarities.Min());

                // Remplacer seulement si le nouveau vecteur est suffisamment différent
                if (similarities[minIdx] < 0.4)
                {
                    ltm[minIdx] = (double[])x.Clone();
                }
            }
        }

        /// <summary>
        /// Propagation avant à travers le MLP (réseau profond)
        /// </summary>
        private double[] ForwardMLP(double[] input)
        {
            double[] activation = input;

            // Couches cachées : ReLU
            for (int i = 0; i < weights.Count - 1; i++)
            {
                activation = MatrixVectorMultiply(weights[i], activation);
                activation = VectorAdd(activation, biases[i]);
                activation = Relu(activation);
            }

            // Couche de sortie : linéaire
            activation = MatrixVectorMultiply(weights[^1], activation);
            activation = VectorAdd(activation, biases[^1]);

            return activation;
        }

        /// <summary>
        /// Étape principale du modèle : traite une nouvelle entrée et retourne la prédiction
        /// </summary>
        /// <param name="x">Vecteur d'entrée actuel</param>
        /// <returns>Vecteur de sortie (prédiction, correction, score de risque...)</returns>
        public double[] Step(double[] x)
        {
            if (x.Length != inputDim)
                throw new ArgumentException($"L'entrée doit avoir exactement {inputDim} dimensions.");

            // 1. Calcul de la relation avec les souvenirs passés
            double[] R_t = ComputeRelation(x);

            // 2. Mise à jour de la mémoire courte (accumulation leaky modifiée par la relation)
            for (int i = 0; i < inputDim; i++)
            {
                stm[i] = alpha * stm[i] + (1.0 - alpha) * (x[i] * R_t[i]);
            }

            // 3. Mise à jour de la mémoire longue
            UpdateLtm(x);

            // 4. Calcul de la moyenne des souvenirs en mémoire longue
            double[] ltmMean = new double[inputDim];
            if (ltm.Count > 0)
            {
                for (int i = 0; i < inputDim; i++)
                {
                    ltmMean[i] = ltm.Average(vec => vec[i]);
                }
            }

            // 5. Combinaison finale : STM + β × moyenne LTM
            double[] combined = new double[inputDim];
            for (int i = 0; i < inputDim; i++)
            {
                combined[i] = stm[i] + beta * ltmMean[i];
            }

            // 6. Passage dans le réseau profond (MLP) pour la sortie finale
            return ForwardMLP(combined);
        }

        /// <summary>
        /// Entraînement simple d'une étape : ajuste tous les poids du MLP avec descente de gradient
        /// Utilise la perte MSE et une approximation (pas de BPTT complète sur la mémoire)
        /// </summary>
        /// <param name="input">Vecteur d'entrée (avant Step)</param>
        /// <param name="target">Valeur cible</param>
        /// <param name="learningRate">Taux d'apprentissage (ex: 0.001 à 0.01)</param>
        public void TrainStep(double[] input, double target, double learningRate)
        {
            // 1. Forward normal
            double[] prediction = Step(input);
            double error = prediction[0] - target;

            // 2. Backprop simple : on propage l'erreur à travers le MLP
            // (on ignore la dépendance temporelle de la mémoire pour simplifier)

            // Commencer par la dernière couche
            List<double[]> activations = new List<double[]> { GetCombinedState() }; // entrée du MLP
            double[] act = activations[0];

            for (int i = 0; i < weights.Count; i++)
            {
                act = MatrixVectorMultiply(weights[i], act);
                act = VectorAdd(act, biases[i]);
                if (i < weights.Count - 1) act = Relu(act); // ReLU sauf dernière couche
                activations.Add(act);
            }

            // Gradient de sortie
            double[] grad = new double[outputDim];
            grad[0] = error; // deriv MSE

            // Backprop couche par couche (à l'envers)
            for (int i = weights.Count - 1; i >= 0; i--)
            {
                double[] prevAct = activations[i];

                // Gradient sur biais
                for (int j = 0; j < biases[i].Length; j++)
                {
                    biases[i][j] -= learningRate * grad[j];
                }

                // Gradient sur poids
                for (int j = 0; j < weights[i].GetLength(0); j++)
                {
                    for (int k = 0; k < weights[i].GetLength(1); k++)
                    {
                        weights[i][j, k] -= learningRate * grad[j] * prevAct[k];
                    }
                }

                if (i > 0)
                {
                    // Propager le gradient vers la couche précédente
                    double[] newGrad = new double[prevAct.Length];
                    for (int j = 0; j < prevAct.Length; j++)
                    {
                        double sum = 0;
                        for (int outJ = 0; outJ < grad.Length; outJ++)
                        {
                            sum += weights[i][outJ, j] * grad[outJ];
                        }
                        newGrad[j] = sum * (i < weights.Count - 1 ? (prevAct[j] > 0 ? 1 : 0) : 1); // deriv ReLU ou linéaire
                    }
                    grad = newGrad;
                }
            }
        }

        /// <summary>
        /// Réinitialise les mémoires STM et LTM.
        /// Utile pour commencer une nouvelle séquence ou un nouveau test.
        /// </summary>
        public void Reset()
        {
            stm = new double[inputDim];
            ltm.Clear();
        }

        /// <summary>
        /// Expose l'état combiné (STM + β * LTM_mean) pour l'entraînement
        /// Nécessaire pour compatibilité avec l'ancien code d'évaluation (version originale du RMA)
        /// </summary>
        /// <returns>L'état combiné avant le MLP</returns>
        public double[] GetCombinedState()
        {
            double[] ltmMean = new double[inputDim];
            if (ltm.Count > 0)
            {
                for (int i = 0; i < inputDim; i++)
                {
                    ltmMean[i] = ltm.Average(vec => vec[i]);
                }
            }

            double[] combined = new double[inputDim];
            for (int i = 0; i < inputDim; i++)
            {
                combined[i] = stm[i] + beta * ltmMean[i];
            }

            return combined;
        }

        /// <summary>
        /// Ajuste les poids W et b avec descente de gradient (simple, pour dim=1)
        /// Cette méthode ajuste UNIQUEMENT la première couche du MLP (entre combined et première couche cachée)
        /// pour rester compatible avec l'ancien entraînement simple de la version originale.
        /// </summary>
        /// <param name="learningRate">Taux d'apprentissage</param>
        /// <param name="error">Erreur (prédiction - cible)</param>
        public void AdjustWeights(double learningRate, double error)
        {
            if (inputDim != 1 || outputDim != 1)
                throw new InvalidOperationException("AdjustWeights est implémenté uniquement pour inputDim=1 et outputDim=1 (compatibilité évaluation simple).");

            double[] combined = GetCombinedState();

            // Ajuste uniquement le poids de la première couche (weights[0][0,0]) et le biais correspondant
            // Si pas de couche cachée, c'est la couche de sortie directe
            int layerIndex = 0;
            weights[layerIndex][0, 0] -= learningRate * error * combined[0];
            biases[layerIndex][0] -= learningRate * error;
        }
    }
}