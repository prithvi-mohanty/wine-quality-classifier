# Wine Quality Classification

## Problem Statement

Predict wine quality based on physicochemical properties. This is a binary classification task where wines with quality rating ≥ 6 are classified as "Good" (1) and wines with quality < 6 are classified as "Bad" (0).

## Dataset Description

**Source:** Wine Quality Dataset from UCI Machine Learning Repository

The dataset contains chemical analysis of red and white Portuguese "Vinho Verde" wines.

- **Total Samples:** 6,497
- **Features:** 12
- **Classes:** 2 (Binary)

| Feature | Description |
|---------|-------------|
| fixed acidity | Tartaric acid content |
| volatile acidity | Acetic acid content |
| citric acid | Citric acid content |
| residual sugar | Remaining sugar after fermentation |
| chlorides | Salt content |
| free sulfur dioxide | Free SO2 |
| total sulfur dioxide | Total SO2 |
| density | Wine density |
| pH | Acidity level |
| sulphates | Sulfate content |
| alcohol | Alcohol percentage |
| wine_type | Red (0) or White (1) |

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.7392 | 0.8057 | 0.7333 | 0.7392 | 0.7329 | 0.4214 |
| Decision Tree | 0.7638 | 0.7836 | 0.7622 | 0.7638 | 0.7629 | 0.4880 |
| KNN | 0.7408 | 0.8004 | 0.7363 | 0.7408 | 0.7373 | 0.4308 |
| Naive Bayes | 0.6831 | 0.7445 | 0.6714 | 0.6831 | 0.6710 | 0.2861 |
| Random Forest (Ensemble) | 0.8392 | 0.9020 | 0.8377 | 0.8392 | 0.8375 | 0.6490 |
| XGBoost (Ensemble) | 0.8285 | 0.8825 | 0.8269 | 0.8285 | 0.8272 | 0.6265 |

## Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Provides a reasonable baseline with 73.9% accuracy. Linear decision boundary limits its ability to capture complex patterns. Good AUC (0.81) indicates decent ranking capability. |
| Decision Tree | Achieves 76.4% accuracy with interpretable rules. Captures non-linear relationships but lower AUC suggests weaker probability estimates compared to other models. |
| KNN | Similar performance to Logistic Regression (74.1% accuracy). Sensitive to feature scaling and choice of k. Works well for local patterns in data. |
| Naive Bayes | Lowest performance (68.3% accuracy) due to independence assumption. Wine features are correlated, which violates this assumption. Fast training but poor predictions. |
| Random Forest (Ensemble) | **Best model** with 83.9% accuracy and highest AUC (0.90). Ensemble of trees reduces variance and handles feature interactions well. Strong MCC shows balanced performance. |
| XGBoost (Ensemble) | Second best with 82.9% accuracy. Gradient boosting minimizes errors effectively. Slightly lower than Random Forest, possibly due to dataset size. Good regularization. |
