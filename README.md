

[English](#english) | [Português](#portuguese) | [Français](#french)

---

<a name="english"></a>

# Aristotle NLP Model: Semantic Analysis of Classical Texts

This project combines modern Natural Language Processing (NLP) and Word Embeddings with classical philosophy to analyze monumental works like the *Mahabharata* and texts by *Plato/Aristotle*.

The primary goal is to map the vocabulary of these works onto **Aristotle's 10 Categories**, allowing for a quantitative visualization of how abstract concepts manifest within the text.

### 🚀 Key Features
- **Custom Word Embeddings**: Generate word vectors based on co-occurrence matrices and PCA.
- **Aristotelian Mapping**: Automated classification of words into categories such as *Substance, Quality, Quantity, Place, Time, Action*, etc.
- **Topological Visualization**: High-density 2D maps using t-SNE and PCA.

---

<a name="portuguese"></a>

# Aristotle NLP Model: Análise Semântica de Textos Clássicos

Este projeto utiliza técnicas modernas de Processamento de Linguagem Natural (NLP) e Word Embeddings para analisar obras clássicas (como o *Mahabharata* e textos de *Platão/Aristóteles*) sob uma lente filosófica.

O objetivo principal é mapear o vocabulário dessas obras para as **10 Categorias de Aristóteles**, permitindo visualizar como conceitos abstratos se manifestam quantitativamente no texto.

### 🚀 Funcionalidades
- **Embeddings Customizados**: Geração de vetores de palavras baseados em matrizes de co-ocorrência e PCA.
- **Mapeamento Aristotélico**: Classificação de palavras em categorias como *Substância, Qualidade, Quantidade, Lugar, Tempo, Ação*, etc.
- **Visualização Topológica**: Gráficos em 2D de alta densidade usando t-SNE e PCA.

---

<a name="french"></a>

# Aristotle NLP Model: Analyse Sémantique des Textes Classiques

Ce projet combine le traitement moderne du langage naturel (NLP) et les structures de vecteurs de mots (Word Embeddings) avec la philosophie classique pour analyser des œuvres monumentales telles que le *Mahabharata* et les textes de *Platon/Aristote*.

L'objectif principal est de cartographier le vocabulaire de ces œuvres sur les **10 Catégories d'Aristote**, permettant une visualisation quantitative de la manière dont les concepts abstraits se manifestent dans le texte.

### 🚀 Caractéristiques Principales
- **Embeddings Personnalisés** : Génération de vecteurs de mots basés sur des matrices de cooccurrence et l'analyse en composantes principales (PCA).
- **Cartographie Aristotélicienne** : Classification automatisée des mots dans des catégories telles que *Substance, Qualité, Quantité, Lieu, Temps, Action*, etc.
- **Visualisation Topologique** : Cartes 2D haute densité utilisant t-SNE et PCA pour visualiser les clusters sémantiques.

---

## 📂 Project Structure / Estrutura / Structure

```text
.
├── data/               # Raw text files / Textos brutos / Textes sources
├── src/
│   ├── core/           # Base processing / Processamento / Traitement de base
│   ├── philosophy/     # Philosophy & Profiling / Filosofia / Philosophie
│   └── visualization/  # Visualization / Visualização / Visualisation
├── utils/              # Tools / Ferramentas / Outils
├── plots/              # PNG Plots / Gráficos / Graphiques
└── requirements.txt    # Dependencies / Dependências / Dépendances
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📖 Usage / Como Usar / Utilisation

### 1. Process Embeddings
```bash
python src/core/process_book.py
```

### 2. Word Profile
```bash
python src/philosophy/word_profile.py
```

### 3. t-SNE Visualization
```bash
python src/visualization/tsne_visualization.py
```

---

**Author:** [Your Name/GitHub]  
**License:** MIT
