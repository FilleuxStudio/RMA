using System;
using System.Linq;
using RMA.Core;  // Ta DLL

namespace RMA.Evaluation
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== Comparaison : RNN vs RMA vs RMA Deep ===\n");

            // === Génération des données ===
            Random rand = new Random(42);
            int n = 300;
            double[] data = Enumerable.Range(0, n)
                .Select(i => Math.Sin(i / 10.0) + 0.12 * (rand.NextDouble() * 2 - 1))
                .ToArray();

            int seqLength = 10;
            int numSamples = n - seqLength;

            double[][] X = new double[numSamples][];
            double[] y = new double[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                X[i] = data[i..(i + seqLength)];
                y[i] = data[i + seqLength];
            }

            int trainSize = (int)(0.8 * numSamples);
            double[][] XTrain = X[..trainSize];
            double[][] XTest = X[trainSize..];
            double[] yTrain = y[..trainSize];
            double[] yTest = y[trainSize..];  // Important : yTest est bien défini ici

            // === 1. RNN Simple (50 unités cachées) ===
            Console.WriteLine("Entraînement du RNN simple...");
            var rnn = new SimpleRNN(hiddenSize: 50);
            TrainRNN(rnn, XTrain, yTrain, epochs: 100, learningRate: 0.01);
            double rnnMse = TestRNN(rnn, XTrain, XTest, yTest);
            Console.WriteLine($"RNN Test MSE : {rnnMse:F6}\n");

            // === 2. RMA Original (léger) ===
            Console.WriteLine("Entraînement du RMA Original...");
            var rmaOriginal = new RelationalMemoryAccumulator(dim: 1, alpha: 0.95, beta: 0.6, k: 8, maxLtm: 200);
            TrainRMAOriginal(rmaOriginal, XTrain, yTrain, epochs: 150, learningRate: 0.015);
            double rmaOriginalMse = TestRMA(rmaOriginal, XTrain, XTest, yTest, seqLength);
            Console.WriteLine($"RMA Original Test MSE : {rmaOriginalMse:F6}\n");

            // === 3. RMA Deep (puissant) ===
            Console.WriteLine("Test du RMA Deep (apprentissage par exposition séquentielle)...");
            var rmaDeep = new RelationalMemoryAccumulatorDeep(
                 inputDim: 1,
                 hiddenSizes: new int[] { 128, 64, 32 },
                 outputDim: 1,
                 alpha: 0.97,
                 beta: 0.8,
                 k: 12,
                 maxLtm: 300);

            double learningRateDeep = 0.005;
            int deepEpochs = 100;

            for (int epoch = 0; epoch < deepEpochs; epoch++)
            {
                double totalLoss = 0;
                rmaDeep.Reset();

                for (int i = 0; i < XTrain.Length; i++)
                {
                    // 1. Construire la mémoire avec les 9 premières valeurs
                    for (int j = 0; j < seqLength - 1; j++)
                    {
                        rmaDeep.Step(new double[] { XTrain[i][j] });
                    }

                    // 2. Prédiction sur la 10e valeur
                    double[] predArray = rmaDeep.Step(new double[] { XTrain[i][seqLength - 1] });
                    double pred = predArray[0];
                    double error = pred - yTrain[i];
                    totalLoss += error * error;

                    // 3. Entraînement UNIQUEMENT sur cette dernière étape
                    rmaDeep.TrainStep(new double[] { XTrain[i][seqLength - 1] }, yTrain[i], learningRateDeep);
                }

                if (epoch % 20 == 0 || epoch == deepEpochs - 1)
                    Console.WriteLine($"  Epoch {epoch,3} → Train Loss = {totalLoss / XTrain.Length:F6}");
            }

            // Test normal
            double rmaDeepMse = TestRMA(rmaDeep, XTrain, XTest, yTest, seqLength);
            Console.WriteLine($"RMA Deep Test MSE : {rmaDeepMse:F6}\n");

            // === Résultats finaux ===
            Console.WriteLine("═" + new string('═', 60));
            Console.WriteLine("           RÉSULTATS FINAUX SUR LE TEST");
            Console.WriteLine("═" + new string('═', 60));
            Console.WriteLine($"RNN simple (50 unités)      : {rnnMse:F6}");
            Console.WriteLine($"RMA Original (léger)        : {rmaOriginalMse:F6}");
            Console.WriteLine($"RMA Deep (128-64-32 unités) : {rmaDeepMse:F6}");

            var results = new (string name, double mse)[]
            {
                ("RNN simple", rnnMse),
                ("RMA Original", rmaOriginalMse),
                ("RMA Deep", rmaDeepMse)
            };

            var ordered = results.OrderBy(r => r.mse).ToArray();
            Console.WriteLine("\nClassement :");
            for (int i = 0; i < ordered.Length; i++)
            {
                Console.WriteLine($"{i + 1}. {ordered[i].name} → MSE = {ordered[i].mse:F6}");
            }

            Console.WriteLine("\nAppuie sur une touche pour quitter...");
            Console.ReadKey();
        }

        // === Méthodes d'aide ===

        static void TrainRNN(SimpleRNN rnn, double[][] XTrain, double[] yTrain, int epochs, double learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                rnn.Reset();
                for (int i = 0; i < XTrain.Length; i++)
                {
                    double pred = rnn.Forward(XTrain[i]);
                    double error = pred - yTrain[i];
                    totalLoss += error * error;
                    rnn.Backward(error, learningRate);
                }
                if (epoch % 30 == 0 || epoch == epochs - 1)
                    Console.WriteLine($"  Epoch {epoch,3} → Loss = {totalLoss / XTrain.Length:F6}");
            }
        }

        static double TestRNN(SimpleRNN rnn, double[][] XTrain, double[][] XTest, double[] yTest)
        {
            rnn.Reset();
            foreach (var seq in XTrain) rnn.Forward(seq);

            double[] preds = XTest.Select(seq => rnn.Forward(seq)).ToArray();
            return CalculateMSE(yTest, preds);
        }

        static void TrainRMAOriginal(RelationalMemoryAccumulator rma, double[][] XTrain, double[] yTrain, int epochs, double learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                rma.Reset();
                for (int i = 0; i < XTrain.Length; i++)
                {
                    for (int j = 0; j < 9; j++)
                        rma.Step(new double[] { XTrain[i][j] });

                    double[] pred = rma.Step(new double[] { XTrain[i][9] });
                    double error = pred[0] - yTrain[i];
                    totalLoss += error * error;

                    rma.AdjustWeights(learningRate, error);
                }
                if (epoch % 40 == 0 || epoch == epochs - 1)
                    Console.WriteLine($"  Epoch {epoch,3} → Loss = {totalLoss / XTrain.Length:F6}");
            }
        }

        static double TestRMA<T>(T model, double[][] XTrain, double[][] XTest, double[] yTest, int seqLength) where T : class
        {
            dynamic rma = model;
            rma.Reset();

            // Reconstruire la mémoire avec les données d'entraînement
            foreach (var seq in XTrain)
            {
                foreach (var val in seq)
                {
                    rma.Step(new double[] { val });
                }
            }

            // Prédictions sur le test
            double[] preds = new double[XTest.Length];
            for (int i = 0; i < XTest.Length; i++)
            {
                for (int j = 0; j < seqLength - 1; j++)
                {
                    rma.Step(new double[] { XTest[i][j] });
                }
                preds[i] = rma.Step(new double[] { XTest[i][seqLength - 1] })[0];
            }

            return CalculateMSE(yTest, preds);
        }

        static double CalculateMSE(double[] trueValues, double[] preds)
        {
            return trueValues.Zip(preds, (t, p) => (t - p) * (t - p)).Average();
        }

        // === Classe SimpleRNN (inchangée) ===
        class SimpleRNN
        {
            private readonly int hiddenSize;
            private double[,] inputWeights;
            private double[,] recurrentWeights;
            private double[] outputWeights;
            private double outputBias;
            private double[] hidden;
            private double[] prevHidden;
            private static readonly Random rand = new Random();

            public SimpleRNN(int hiddenSize)
            {
                this.hiddenSize = hiddenSize;
                inputWeights = new double[1, hiddenSize];
                recurrentWeights = new double[hiddenSize, hiddenSize];
                outputWeights = new double[hiddenSize];
                hidden = new double[hiddenSize];
                prevHidden = new double[hiddenSize];
                InitializeWeights();
            }

            private void InitializeWeights()
            {
                for (int i = 0; i < hiddenSize; i++)
                {
                    inputWeights[0, i] = rand.NextDouble() * 0.2 - 0.1;
                    outputWeights[i] = rand.NextDouble() * 0.2 - 0.1;
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        recurrentWeights[i, j] = rand.NextDouble() * 0.2 - 0.1;
                    }
                }
                outputBias = rand.NextDouble() * 0.2 - 0.1;
            }

            public double Forward(double[] sequence)
            {
                Array.Clear(hidden, 0, hidden.Length);
                for (int t = 0; t < sequence.Length; t++)
                {
                    Array.Copy(hidden, prevHidden, hiddenSize);
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        double sum = inputWeights[0, h] * sequence[t];
                        for (int p = 0; p < hiddenSize; p++)
                            sum += recurrentWeights[h, p] * prevHidden[p];
                        hidden[h] = Math.Tanh(sum);
                    }
                }
                double output = outputBias;
                for (int h = 0; h < hiddenSize; h++)
                    output += outputWeights[h] * hidden[h];
                return output;
            }

            public void Backward(double error, double lr)
            {
                for (int h = 0; h < hiddenSize; h++)
                    outputWeights[h] -= lr * error * hidden[h];
                outputBias -= lr * error;

                double[] hiddenGrad = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    hiddenGrad[h] = error * outputWeights[h] * (1 - hidden[h] * hidden[h]);

                for (int h = 0; h < hiddenSize; h++)
                {
                    for (int p = 0; p < hiddenSize; p++)
                        recurrentWeights[h, p] -= lr * hiddenGrad[h] * prevHidden[p];
                }
            }

            public void Reset()
            {
                Array.Clear(hidden, 0, hidden.Length);
                Array.Clear(prevHidden, 0, hiddenSize);
            }
        }
    }
}