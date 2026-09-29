# Machine_Learning_project
# E-Commerce Purchasing Intention & Data Imputation (Machine Learning Pipeline)

## Problem Overview & Objective
This project addresses a multi-stage machine learning challenge using the **UCI Online Shoppers Purchasing Intention** dataset. The primary objective was to predict whether an online session results in a purchase (`Revenue` classification), while resolving synthetic data corruption in a key continuous feature (`Exit Rate`).

📌 **[View Full Jupyter Notebook with code, analysis, and visualizations](./Machine_Learning_Homework.ipynb)**

## Methodology Steps
1. **Data Imputation via Regression:**
   - Handled missing/corrupted values in the `Exit Rate` variable by training robust regression models (comparing algorithms via cross-validation and feature selection).
   - Reconstructed the missing `Exit Rate` data for both training and test sets.
2. **Classification & Feature Selection:**
   - Built and benchmarked classification models to predict user purchase intent (`Revenue`).
   - Evaluated the impact of data imputation by comparing models trained with vs. without the recovered `Exit Rate` feature.
3. **Clustering Analysis:**
   - Applied unsupervised clustering algorithms to uncover latent user browsing patterns and compared their performance against supervised classification benchmarks.

## Tools & Environment
- **Language:** Python 3 (Jupyter Notebook)
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
