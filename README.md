# MolCross
# Introduction
In this study, we proposed a deep learning algorithm, named MolCross, which combines implicit feature interaction with explicit feature to improve the accuracy of prediction of anti-cancer drug synergy score.

# Dataset
Cancer Cell Line Encyclopedia (CCLE)
Large-scale oncology screen produced by Merck & Co.

# Data
lables_ABBA.csv: Drug name, cell line name, and collaborative scoring data
drugs.csv: Drug feature expression data
exp_mmNoname.csv: Cell line gene expression
emb.csv: Sparse feature of cell line types, cell lines, and subtypes

# Version
Python 3.6
TensorFlow 1.12
Keras 2.2

# Usage
Use the maximum minimum normalization method to normalize drugs.csv and exp_mmNoname.csv
Use the deep autoencoder to extract features and reduce dimensions
Run mainKfold.py to train and evaluate the model through five-fold cross validation
