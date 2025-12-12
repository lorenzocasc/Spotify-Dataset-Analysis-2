# Music Analysis: Time Series & Tabular Data Mining

## Project Overview

This repository contains the code and documentation for the "Data Mining 2" project (Academic Year 2023/2024) at the University of Pisa. The project focuses on the extensive analysis of music data, employing advanced data mining techniques on both tabular datasets (track features, artist info) and time-series data (spectral centroids).

The analysis covers the entire data mining pipeline, including data preparation, motif discovery, clustering, anomaly detection, imbalanced learning, and advanced classification/regression tasks using Deep Learning and Gradient Boosting machines.

## Authors

* Lorenzo Cascone
* Emiliano Marrale
* Salvatore Puccio

## Datasets

The project utilizes three distinct datasets:
1. **Tracks Dataset:** Extended Spotify dataset containing over 100,000 records with audio features (danceability, energy, loudness, etc.).
2. **Artists Dataset:** Information on over 30,000 artists, including popularity and followers.
3. **Time-Series Dataset:** 10,000 time series representing the spectral centroids of audio tracks (length: 1280 timepoints), categorized into 20 distinct genres.

## Methodology & Analysis Modules

### 1. Data Understanding and Preparation
* Cleaning of duplicated IDs and song names.
* Dimensionality reduction and feature correlation analysis.
* Trend detection and removal for time-series data using moving averages.

### 2. Time Series Analysis
* **Motifs & Discords:** Utilized Matrix Profile (window size=50) to identify recurring patterns (motifs) and anomalies (discords) within genres.
* **Clustering:** Applied K-Means (with PAA approximation) and Hierarchical Clustering (Ward linkage).
* **Classification:** Compared K-Nearest Neighbors (using DTW vs. Euclidean distance) and Shapelet-based classifiers.
* **Sequential Pattern Mining:** utilized SAX approximation and PrefixSpan to find frequent rhythmic patterns across genres.

### 3. Anomaly Detection
Identified the top 1% of outliers in the tabular dataset using a majority voting ensemble of 6 methods:
* Density-based (LOF).
* Angle-based (ABOD).
* Ensemble-based (Feature Bagging, Isolation Forest).
* Model-based methods.

### 4. Imbalanced Learning
addressed class imbalance in the "Explicit" feature using:
* **Undersampling:** Random Undersampler, Cluster Centroids, Tomek Links, CNN, ENN.
* **Oversampling:** SMOTE, ADASYN.
* **Mixed Methods:** SMOTEENN, SMOTETomek.
* **Result:** Cluster Centroids with KNN provided the best balance between Recall and Precision.

### 5. Advanced Time Series Classification
Implemented Deep Learning approaches to classify music genres based on spectral centroids:
* **Convolutional Neural Network (CNN):** Custom 1D-CNN architecture with Batch Normalization and LeakyReLU.
* **MiniRocket:** Fast time series classification method.
* **Performance:** Both models achieved approximately 41% accuracy on the test set.

### 6. Advanced Tabular Classification
Classified songs into macro-genres (Pop, Rock, Metal, Electronic, National) using:
* Logistic Regression & SVM.
* Random Forest.
* Gradient Boosting Machines: XGBoost, LightGBM, CatBoost.
* **Kolmogorov-Arnold Network (KAN):** Experimental implementation of this novel neural network architecture.
* **Best Result:** XGBoost and CatBoost achieved ~68% accuracy with high AUC scores (>0.90 for some classes).

### 7. Advanced Tabular Regression
Predicted track popularity using Random Forest and XGBoost.
* **Optimization:** Merging track data with artist data significantly improved performance.
* **Best Result:** XGBoost achieved an R2 score of approximately 0.61.

### 8. Explainable AI (XAI)
Interpreted the "Black Box" models (specifically CatBoost) using:
* **SHAP:** Global feature importance (e.g., identifying Danceability and Acousticness as key predictors).
* **LORE:** Local rule-based explanations.
* **LIME:** Local probabilistic explanations for specific instances.

## Key Results Summary

| Task | Best Model | Metric | Score |
| :--- | :--- | :---: | :---: |
| **TS Binary Class.** | KNN (DTW) | Accuracy | 0.78 |
| **TS Multi Class.** | CNN / MiniRocket | Accuracy | ~0.41 |
| **Tabular Class.** | CatBoost / XGBoost | Accuracy | 0.68 |
| **Tabular Reg.** | XGBoost (Merged Data) | R2 Score | 0.61 |

## Repository Structure

```text
.
├── Data/
│   ├── tracks_dataset.csv
│   ├── artists_dataset.csv
│   └── time_series_data/
├── Notebooks/
│   ├── 01_Data_Prep_and_TS_Analysis.ipynb
│   ├── 02_Anomaly_Detection.ipynb
│   ├── 03_Imbalanced_Learning.ipynb
│   ├── 04_Advanced_TS_Classification.ipynb
│   ├── 05_Tabular_Classification_and_KAN.ipynb
│   ├── 06_Tabular_Regression.ipynb
│   └── 07_Explainable_AI.ipynb
├── Report/
│   └── Project_Report.pdf
└── README.md
