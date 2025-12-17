pip install numpy
python -m examples.simple_demo

pip install numpy matplotlib  # pour les graphiques
python -m examples.sine_prediction
python -m evaluation.compare_models


# Relational Memory Accumulator (RMA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11-3.13](https://img.shields.io/badge/Python-3.11--3.13-blue)](https://www.python.org/downloads/)
[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-purple)](https://dotnet.microsoft.com/download/dotnet/8.0)
[![Rust 1.70+](https://img.shields.io/badge/Rust-1.70+-orange)](https://www.rust-lang.org/)

**Un concept original de réseau récurrent léger basé sur une mémoire relationnelle explicite.**

Le **Relational Memory Accumulator (RMA)** est une architecture que j'ai conçue pour explorer une alternative simple et interprétable aux RNN/LSTM classiques. L'idée centrale : plutôt que de cacher l'information dans un état récurrent opaque, on maintient une **mémoire à court terme (STM)** et une **mémoire à long terme (LTM)** basées sur des similarités cosinus.

Ce n'est **pas** une tentative de battre les RNN sur des benchmarks classiques (prédiction de séries temporelles pures, langage, etc.), mais une proposition différente pour les cas où **l'interprétabilité, la légèreté et la mémoire explicite** sont plus importantes que la performance brute.