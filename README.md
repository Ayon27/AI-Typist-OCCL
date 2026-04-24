# Catching AI Typists with Contrastive Learning

This repository contains the source code for the paper **"Catching AI Typists with Contrastive Learning"**, which proposes a One-Class Contrastive Learning (OCCL) framework to detect AI-generated keystroke sequences.

## 🚀 Quick Start: End-to-End Pipeline

To reproduce the results of the paper end-to-end, simply run the main training and evaluation pipelines sequentially from the project root:

```bash
# 1. Run the entire training pipeline
# This will train the Hybrid Encoder, TypeNet, and TSFN models, 
# and then automatically fit the downstream OC-SVM / OC-GMM classifiers.
python -m src.train.main

# 2. Run the entire evaluation pipeline
# This will calculate EER, ROC-AUC, AUPRC, etc., and generate all figures 
# (ROC curves, Loss curves, PR curves, and UMAP latent space projections).
python -m src.evaluate.main
```

---

## 🛠️ Running Specific Portions of the Code

If you want to run only specific parts of the pipeline without doing a full end-to-end run, you can use the targeted entry points below.

### 1. Training Specific Models

To train the models individually, you can invoke their specific training scripts:

**Train the Proposed Hybrid CNN-LSTM Encoder:**
```bash
python -m src.train.model_training.hybrid.train
```

**Train Baseline Models (TypeNet or TSFN):**
```bash
# Train TypeNet baseline
python -m src.train.model_training.baseline.train --model typenet

# Train TSFN baseline
python -m src.train.model_training.baseline.train --model tsfn
```

### 2. Fitting Downstream Classifiers

If you already have trained model checkpoints and just want to fit the downstream One-Class SVM and One-Class GMM classifiers on the extracted embeddings:

```bash
python -m src.train.model_training.ocsvm
```

### 3. Evaluation & Figure Generation

You can independently trigger metric calculations and figure generation using the scripts in the `evaluate` module:

**Calculate Numerical Metrics (EER, ROC-AUC):**
```bash
python -m src.evaluate.metrics
```

**Generate Figures and Plots:**
```bash
# Generates ROC curves, PR curves, Loss curves, and UMAP visualisations
python -m src.evaluate.visualize
```

---

## 📁 Directory Structure Overview

*   `src/data/`: Data loading and preprocessing pipelines. Handles the strictly quarantined zero-day subject splits.
*   `src/models/`: PyTorch model definitions for the Hybrid Encoder, TypeNet, TSFN, and the custom OCCL loss function.
*   `src/train/`: Training logic, hyperparameter configuration, and downstream classifier fitting.
*   `src/evaluate/`: Metric computation and figure generation (matplotlib/seaborn).
*   `results/` *(Generated)*: Checkpoints (`.pt`), numpy embeddings, and logs.
*   `paper/` *(Generated)*: LaTeX source code and compiled PDF figures.

## 📦 Requirements
Dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```
