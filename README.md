# Fusion of Handcrafted and Deep Temporal Features for EEG-Based Depression Detection

A deep learning framework for **EEG-based Major Depressive Disorder (MDD) detection** that combines handcrafted EEG features with a CNN–LSTM hybrid architecture. The model integrates domain-specific signal processing techniques with deep temporal feature learning to improve classification performance while emphasizing **subject-aware evaluation** to prevent data leakage.

> **Associated Paper:** *Fusion of Handcrafted and Deep Temporal Features for EEG-Based Depression Detection Using a CNN–LSTM Hybrid Architecture*

---

## Overview

Electroencephalography (EEG) signals contain valuable biomarkers for detecting Major Depressive Disorder (MDD). Traditional machine learning approaches rely on handcrafted features, whereas modern deep learning methods learn representations directly from raw EEG data.

This project combines both approaches through a **dual-branch architecture**:

- **Handcrafted Feature Branch**
  - Graph Fourier Transform (GFT)
  - Wavelet Energy
  - Shannon Entropy
  - Band Power
  - Phase Locking Value (PLV)
  - Coherence
  - Hjorth Parameters
  - Higuchi Fractal Dimension

- **Deep Learning Branch**
  - 1D CNN
  - LSTM
  - Mid-level feature fusion
  - Dense classifier

The repository contains the complete preprocessing pipeline, feature extraction code, model implementation, and training scripts.

---

## Proposed Architecture

```
                Raw EEG Signals
                      │
        ┌─────────────┴─────────────┐
        │                           │
        │                           │
 Handcrafted Features         CNN → LSTM
        │                           │
        └────────── Feature Fusion ─┘
                     │
                 Dense Layers
                     │
              Depression Detection
```

---

## Features

- EEG preprocessing pipeline
- ICA-based artifact removal
- Sliding window segmentation
- Handcrafted EEG feature extraction
- CNN–LSTM hybrid architecture
- Mid-level feature fusion
- Subject-aware cross-validation
- Data leakage prevention
- Feature selection using SelectKBest
- Class imbalance handling
- Evaluation at both segment and subject levels

---

## Datasets

The experiments were performed using two publicly available EEG depression datasets.

### Dataset A
- 33 subjects
- 19 EEG channels
- 256 Hz sampling frequency

### Dataset B
- 64 subjects
- 19 EEG channels
- 256 Hz sampling frequency

> **Note:** The datasets are **not included** in this repository. Please download them from their original sources before running the code.

---

## EEG Preprocessing Pipeline

1. 50 Hz Notch Filter
2. 1 Hz High-pass Filter
3. 45 Hz Low-pass Filter
4. ICA artifact removal
5. Sliding window segmentation
   - Window length: **4 seconds**
   - Overlap: **75%**
6. Feature extraction
7. Standardization
8. Feature selection
9. Model training

---

## Handcrafted Features

The handcrafted branch extracts several physiologically meaningful EEG descriptors.

- Graph Fourier Transform (GFT)
- Wavelet Energy
- Shannon Entropy
- Phase Locking Value (PLV)
- Spectral Coherence
- Band Power
- Hjorth Parameters
- Higuchi Fractal Dimension

---

## Deep Learning Model

The temporal branch consists of:

- Conv1D
- MaxPooling
- Batch Normalization
- Dropout
- LSTM
- Dense Layers

The learned temporal representation is fused with handcrafted features before final classification.

---

## Repository Structure

```
project/
│
├── preprocessing/
│   ├── filtering.py
│   ├── segmentation.py
│   ├── ica.py
│
├── feature_extraction/
│   ├── bandpower.py
│   ├── gft.py
│   ├── plv.py
│   ├── coherence.py
│   ├── wavelet.py
│   ├── complexity.py
│
├── models/
│   ├── cnn_lstm.py
│   ├── fusion_model.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

*(Modify the folder names to match your repository.)*

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/your-repository.git
```

Move into the project directory

```bash
cd your-repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Preprocess EEG data

```bash
python preprocessing/preprocess.py
```

### 2. Extract handcrafted features

```bash
python feature_extraction/extract_features.py
```

### 3. Train the model

```bash
python training/train.py
```

### 4. Evaluate

```bash
python training/evaluate.py
```

---

## Evaluation Strategy

Unlike many previous studies, this project uses:

- Subject-aware cross-validation
- StratifiedGroupKFold
- No subject overlap between train and test
- Data leakage prevention

This provides a more realistic estimate of real-world performance.

---

## Results

The proposed framework demonstrated strong performance by combining handcrafted EEG descriptors with deep temporal representations.

Key observations include:

- Improved performance through feature fusion
- Better generalization with subject-aware evaluation
- Significant reduction of inflated accuracy caused by data leakage
- Around **88% accuracy** under strict subject-level evaluation

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- SciPy
- Scikit-learn
- MNE
- Pandas
- Matplotlib

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{madan2026fusion,
  title={Fusion of Handcrafted and Deep Temporal Features for EEG-Based Depression Detection Using a CNN--LSTM Hybrid Architecture},
  author={Madan, Shreya and Jain, Kavya and Malhotra, Aaditi and Jain, Minni and Jain, Amita},
  year={2026}
}
```

---

## Future Work

- Transformer-based architectures
- Explainable AI for EEG
- Multi-class mental health classification
- Cross-dataset domain adaptation
- Real-time depression detection

---

## Authors

**Shreya Madan**

Department of Computer Science, IGDTUW

---

## License

This project is released for academic and research purposes.

Please cite the associated paper if you use this work in your research.
