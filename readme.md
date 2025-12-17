# Relational Memory Accumulator (RMA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-purple)](https://dotnet.microsoft.com/download/dotnet/8.0)
[![Rust 1.90+](https://img.shields.io/badge/Rust-1.90+-orange)](https://www.rust-lang.org/)
[![Python 3.11-3.13](https://img.shields.io/badge/Python-3.11--3.13-blue)](https://www.python.org/downloads/)

**Un concept original de réseau récurrent léger basé sur une mémoire relationnelle explicite.**

Le **Relational Memory Accumulator (RMA)** est une architecture que j'ai conçue pour explorer une alternative simple et interprétable aux RNN/LSTM classiques, en mettant l'accent sur une **mémoire à court et long terme relationnelle** plutôt que sur des portes complexes.

Ce n'est **pas** une tentative de battre les RNN sur des tâches de prédiction pure (où les RNN/LSTM excellent), mais une proposition différente : un modèle où la mémoire est **explicite, inspectable et basée sur des similarités**, idéal pour des cas où l'interprétabilité et la légèreté comptent autant que la performance brute.

## 🎯 Idée et Concept

L'idée centrale : au lieu de cacher l'information dans un état récurrent opaque, on maintient :
- Une **mémoire courte (STM)** : accumulation leaky pondérée par une relation avec le passé.
- Une **mémoire longue (LTM)** : collection dynamique de vecteurs passés jugés "importants" (via similarité cosinus).
- Une **relation** calculée entre l'entrée actuelle et les souvenirs pour moduler l'accumulation.

Le modèle est volontairement simple, sans portes (forget, input, output comme dans LSTM), mais avec une capacité à "se rappeler" des patterns similaires du passé.

Deux versions existent :
- **RMA** : version légère avec sortie linéaire simple.
- **RMA Deep** : version enrichie d'un MLP configurable après la combinaison STM + LTM pour plus de puissance expressive.

## 📐 Formules Mathématiques

Soit \( x_t \in \mathbb{R}^d \) l'entrée à l'instant \( t \).

### Relation \( R_t \)
$$
R_t = \left( \frac{1}{k} \sum_{i=1}^{k} \cos(x_t, m_i) \right) \cdot \mathbf{1}_d
$$
où \( m_i \) sont les \( k \) souvenirs les plus similaires dans LTM.

### Mémoire courte (STM)
$$
STM_t = \alpha \cdot STM_{t-1} + (1 - \alpha) \cdot (x_t \odot R_t)
$$

### Mémoire longue (LTM)
- Ajout si espace disponible.
- Remplacement du moins similaire si \( \cos(x_t, m_{\min}) < 0.4 \).

### État combiné
$$
combined_t = STM_t + \beta \cdot \overline{LTM}
$$

### Sortie
- **RMA** : \( y_t = W \cdot combined_t + b \)
- **RMA Deep** : \( y_t = \text{MLP}(combined_t) \) (couches fully-connected avec ReLU)

## 🛠 Implémentations disponibles

La bibliothèque est implémentée dans **3 langages** pour une accessibilité maximale :

| Langage | Version | Fichier principal | Notes |
|--------|---------|-------------------|-------|
| **C#** | .NET 8.0 | `src/RMA.Core/RelationalMemoryAccumulator.cs` et `RelationalMemoryAccumulatorDeep.cs` | Performante, idéale pour applications Windows, Unity, services |
| **Rust** | 1.70+ | `src/lib.rs` | Très rapide, mémoire sûre, parfaite pour systèmes embarqués |
| **Python** | 3.11 → 3.13 | `src/rma/core.py` | NumPy uniquement, facile à prototyper et tester |

Toutes les implémentations suivent fidèlement les mêmes formules.

## 🚀 Domaines d'utilisation recommandés

Le RMA n'est **pas** conçu pour battre les RNN/LSTM sur des benchmarks classiques (prédiction de sinusoïde, langage, etc.), mais il brille particulièrement dans :

- **Maintenance prédictive** (serveurs, machines) : détection de pannes à partir de métriques irrégulières grâce à la mémoire relationnelle.
- **Robotique embarquée** : correction d'équilibre ou navigation avec ressources limitées.
- **Détection d'anomalies rares** : capacité à relier un événement actuel à un pattern similaire vu il y a longtemps.
- **Systèmes interprétables** : la LTM est inspectable (on peut voir quels vecteurs sont mémorisés).
- **Edge AI** : très léger, peu de paramètres, pas besoin de GPU.

Sur une tâche de prédiction de série temporelle classique, le RMA Deep atteint un MSE compétitif (~0.02–0.04), mais reste derrière un RNN optimisé — ce qui est normal et attendu.

## 📊 Évaluation

Un programme de comparaison est fourni (`RMA.Evaluation` en C#) qui teste :
- RNN simple (50 unités cachées)
- RMA Original
- RMA Deep (128-64-32)

Résultats typiques sur prédiction de sinusoïde bruitée :
- RNN : ~0.013
- RMA Original : ~0.39
- RMA Deep : ~0.03–0.04 (avec entraînement adapté)

Le RMA Deep est compétitif, mais le vrai avantage réside dans son interprétabilité et sa légèreté.

## ⚙️ Utilisation rapide

### C#
```csharp
var rma = new RelationalMemoryAccumulatorDeep(1, new int[] {64, 32}, 1);
double[] output = rma.Step(new double[] { value });
```

### Rust
```rust
let mut rma = RelationalMemoryAccumulatorDeep::new(1, vec![64, 32], 1, 0.95, 0.6, 8, 200);
let output = rma.step(&vec![value]);
```

### Python
```python
from rma.core import RelationalMemoryAccumulatorDeep
rma = RelationalMemoryAccumulatorDeep(1, [64, 32], 1)
output = rma.step([value])
```

## 📝 Licence

[MIT License](LICENSE) — libre pour usage personnel, commercial, modification.

Vous pouvez utiliser, modifier, distribuer le code librement, tant que la notice de copyright est conservée.

## ✨ Conclusion

Le RMA n'est pas une révolution en performance brute, mais une **exploration intéressante** d'une mémoire récurrente explicite et relationnelle. Il montre qu'on peut obtenir des résultats décents avec une architecture très simple, interprétable et multi-langages.

Idéal pour les projets où la compréhension du modèle compte autant que ses prédictions.

**Contributions bienvenues** : benchmarks, nouveaux cas d'usage, optimisations !

