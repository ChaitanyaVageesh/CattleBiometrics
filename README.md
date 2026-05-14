# 🐄 Cattle Biometrics Identification Pipeline

An end-to-end Machine Learning Operations (MLOps) pipeline designed to identify individual cattle using their unique muzzle prints. Muzzle ridges and beads function similarly to human fingerprints, providing a tamper-proof biometric signature.

This project implements a complete data and model versioning lifecycle using **DVC (Data Version Control)** and **Git**, transitioning from baseline traditional machine learning models to state-of-the-art Deep Transfer Learning architectures.

## 🗂 Project Structure

    ├── data/                   # DVC-tracked dataset directories (raw, processed, augmented)
    ├── models/                 # DVC-tracked trained model files (.pkl, .keras)
    ├── plots/                  # DVC-generated evaluation plots (Confusion Matrix)
    ├── scripts/                # Python pipeline scripts
    │   ├── prepare.py          # Train/Test splitting
    │   ├── transform.py        # CLAHE preprocessing and offline augmentation
    │   ├── train.py            # Dynamic model training logic
    │   └── evaluate.py         # Model evaluation and metric generation
    ├── dvc.yaml                # DVC pipeline definition
    ├── params.yaml             # Centralized hyperparameter configuration
    ├── metrics.json            # Final Macro F1-Score output
    └── README.md


## ⚙️ Data Engineering & Preprocessing
Given the fine-grained nature of muzzle ridges, aggressive data-centric strategies were implemented:
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied to restrict noise amplification while mathematically highlighting the unique biometric ridges of the muzzle prior to augmentation.
* **Offline Augmentation:** Synthetic data generation using rotation, shifting, shearing, zooming, and brightness adjustments to scale the limited raw dataset and prevent overfitting.

## 🧠 Modeling Progression
To establish baselines and iteratively improve performance, the pipeline dynamically supports multiple architectures governed by `params.yaml`:
1. **Traditional ML Baseline:** Support Vector Machine (SVM) using Histogram of Oriented Gradients (HOG) feature extraction.
2. **Shallow Neural Network:** Multi-Layer Perceptron (MLP).
3. **Deep Transfer Learning (Final):** DenseNet169 / ResNet50. Unfreezing the top layers for fine-tuning allowed the model to map specific cattle ridges, while `label_smoothing=0.1` was utilized in the loss function to act as a powerful regularizer against overconfidence.

## 🚀 How to Reproduce
This project strictly enforces reproducibility. The entire data engineering, training, and evaluation lifecycle is tracked via `dvc.yaml`.

### 1. Install Dependencies
    pip install -r requirements.txt

### 2. Configure Parameters
Adjust model selection, batch sizes, learning rates, and augmentation settings directly inside `params.yaml`.

### 3. Execute the Pipeline
    dvc repro

### 4. View Metrics
To view the performance of the current run:

    dvc metrics show

To compare historical experiments and dataset versions:

    dvc metrics show --all-tags
    dvc plots diff $(git tag)

