using System;
using RMA.Core;

public class Program
{
    static void Main(string[] args)
    {
        // Création du RMA avec 3 métriques d'entrée
        var rma = new RelationalMemoryAccumulator(
            dim: 3,          // [CPU %, Temp °C, Nb erreurs disque]
            alpha: 0.93,     // Bonne rétention courte
            beta: 0.6,       // Importance de l'historique long
            k: 8,            // Considère les 8 patterns passés les plus similaires
            maxLtm: 200);    // Garde jusqu'à 200 souvenirs importants

        // Simulation de métriques collectées toutes les 10 secondes
        double[][] metricsStream = new double[][]
        {
            new double[] { 30, 45, 0 },  // Normal
            new double[] { 35, 48, 0 },
            new double[] { 60, 55, 1 },  // Début de stress
            new double[] { 85, 68, 3 },  // Critique
            new double[] { 92, 72, 7 },  // Très critique → alerte attendue
            // ... suite en temps réel
        };

        Console.WriteLine("Métriques         | Sortie RMA (risque/correction)");
        Console.WriteLine("-----------------------------------------------");

        foreach (var metrics in metricsStream)
        {
            double[] output = rma.Step(metrics);

            // Ici, on interprète la première dimension comme un score de risque (ex: > 5 = alerte)
            double riskScore = output[0];

            Console.WriteLine($"[{string.Join(", ", metrics)}]  →  Risque: {riskScore:F2} {(riskScore > 5 ? "!!! ALERTE PANNE PROBABLE !!!" : "")}");
        }

        Console.ReadKey();
    }
}