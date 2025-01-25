# Transformer-Based Peptide Prediction

This repository contains a Python implementation of a Transformer-based regression model for predicting physicochemical properties of peptides. The code includes data preprocessing, model training, and performance evaluation.

## Features

- Encodes peptide sequences into numeric formats for machine learning.
- Implements a Transformer-based neural network for regression tasks.
- Includes training, validation, and testing pipelines.
- Evaluates model performance using metrics such as MSE, MAE, R², and explained variance.
- Visualizes training and validation loss curves.


## Requirements

The following core Python packages are required to run the script:

- `torch==2.0.1`: For building and training the Transformer-based regression model.
- `numpy==1.25.2`: For numerical computations.
- `pandas==2.2.3`: For data manipulation and analysis.
- `scikit-learn==1.3.0`: For data preprocessing, scaling, and evaluation metrics.

You can install these packages using:
```bash
pip install torch==2.0.1 numpy==1.25.2 pandas==2.2.3 scikit-learn==1.3.0 matplotlib==3.9.2 seaborn==0.13.2

