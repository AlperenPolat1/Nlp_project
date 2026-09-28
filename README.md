# NLP Final Project — Authorship Attribution

**Student:** Alperen Polat (220709038)

This project implements and compares three text classification models to identify the author of a given 500-word passage. It was developed as a final project for a Natural Language Processing course.

---

## Problem Definition

Given a passage of text, determine which of the following four authors wrote it:

- Mark Twain
- Jane Austen
- Arthur Conan Doyle
- H.G. Wells

---

## Dataset

Books were downloaded from [Project Gutenberg](https://www.gutenberg.org/). Five books were used per author, each split into non-overlapping 500-word chunks. The dataset was balanced to 200 passages per author (800 total) and split as follows:

| Split      | Size |
|------------|------|
| Train      | 560  |
| Validation | 80   |
| Test       | 160  |

---

## Models

### 1. Vector Space Model (VSM)
Represents each document as a TF-IDF vector. Computes a centroid vector for each author class from the training set, then classifies new documents using cosine similarity.

### 2. N-gram Language Model
A bigram language model with add-k smoothing (k=0.1) and linear interpolation between unigram and bigram probabilities. The interpolation weight (lambda) is tuned on the validation set. Classification is performed using Bayes' theorem in log space.

### 3. Multilayer Perceptron (MLP)
A feedforward neural network trained on TF-IDF vectors using PyTorch. Architecture: `Linear(19541 → 128) → ReLU → Linear(128 → 4)`. Trained with Adam optimizer and Cross-Entropy loss for 20 epochs.

---

## Results

| Model                   | Val Accuracy | Test Accuracy | Training Time |
|-------------------------|-------------|--------------|---------------|
| Vector Space Model      | 98.75%      | 97.50%       | ~1.5s         |
| N-gram Language Model   | 98.75%      | 98.75%       | ~0.6s         |
| Multilayer Perceptron   | 100.00%     | 99.38%       | ~3.9s         |

The MLP achieves the highest test accuracy. The N-gram model offers a strong accuracy-to-speed trade-off. H.G. Wells was the most difficult author to classify across all models.

---

## Project Structure

```
Alperen_Polat_NLP/
├── NLP_Final_Project.ipynb   # Main notebook (Google Colab)
└── nlp_project/
    ├── data/
    │   ├── passages.pkl      # Raw passage dictionary {author: [texts]}
    │   ├── train.pkl         # Training split (corpus, labels)
    │   ├── val.pkl           # Validation split
    │   └── test.pkl          # Test split
    ├── models/
    │   ├── vsm.pkl           # VSM centroids and IDF scores
    │   ├── ngram.pkl         # N-gram counts and best lambda
    │   └── mlp.pt            # Trained MLP model (PyTorch)
    └── vocab/
        ├── word2idx.pkl      # Word-to-index mapping (VSM/MLP)
        └── lm_vocab.pkl      # Vocabulary set for language model
```

---

## Requirements

The project runs on Google Colab. Key dependencies:

- Python 3
- PyTorch
- NumPy
- NLTK (`punkt`, `punkt_tab`)
- scikit-learn
- pandas

---

## How to Run

1. Upload `NLP_Final_Project.ipynb` to Google Colab.
2. Mount your Google Drive when prompted.
3. Run all cells in order. The notebook downloads the books, trains all three models, and prints the final comparison table.
