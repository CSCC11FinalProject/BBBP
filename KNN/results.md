# KNN Classifier — Test Results

## Model Configuration

| Item | Value |
|---|---|
| Model | KNeighborsClassifier (scikit-learn) |
| n_neighbors | 43 |
| weights | uniform |
| metric | manhattan |
| Feature dim | 2055 (2048 Morgan FP + 7 chemical descriptors) |
| Scaler | StandardScaler (fit on train only) |
| Random seed | 67 |

## Dataset Split

| Set | Samples | Positive (BBB+) | Ratio |
|---|---|---|---|
| Train | 1631 | 1248 | 76.5% |
| Val | 204 | — | — |
| Test | 204 | — | — |

## Hyperparameter Tuning (Optuna, 50 trials)

- Optimization target: Val AUC-ROC (maximize)
- Best Val AUC-ROC: **0.9412**

## Test Set Metrics

| Metric | Value |
|---|---|
| **AUC-ROC** | **0.9328** |
| **F1 Score** | **0.8807** |
| Precision | 0.7908 |
| Recall (Sensitivity) | 0.9936 |
| Specificity | 0.1458 |

## Correlation Analysis

Highly correlated descriptor pairs (|r| > 0.85):

| Pair | Pearson r |
|---|---|
| TPSA ↔ HBA | 0.919 |
| TPSA ↔ HBD | 0.857 |

All 7 descriptors were retained in the final model. Future work could investigate removing HBA or HBD to reduce redundancy.

## False Positive Analysis

- Total false positives: **41** (actual BBB− predicted as BBB+)

| Feature | False Positives (mean) | True Negatives (mean) |
|---|---|---|
| LogP | 0.831 | 0.397 |
| TPSA | 133.18 | 164.63 |
| MW | 415.03 | 529.55 |

False positives tend to have higher LogP, lower TPSA, and lower MW compared to correctly classified negatives — making them chemically closer to BBB+ molecules, which explains the misclassification.

## Observations

- **High Recall (0.9936)**: The model identifies nearly all BBB+ molecules, missing almost none.
- **Low Specificity (0.1458)**: The model struggles to correctly identify BBB− molecules, predicting most as BBB+.
- This bias is driven by the class imbalance (76.5% positive) — KNN with uniform weights tends to vote majority class in dense neighborhoods.
- The AUC-ROC (0.9328) remains strong, indicating the model's probability ranking is effective even if the default 0.5 threshold is suboptimal.

## Generated Artifacts

| File | Description |
|---|---|
| `best_params.json` | Optuna best hyperparameters |
| `checkpoints/knn_model.joblib` | Trained KNN model |
| `checkpoints/scaler.joblib` | Fitted StandardScaler |
| `plots/correlation_matrix.png` | Descriptor correlation heatmap |
| `plots/confusion_matrix.png` | Test set confusion matrix |
| `plots/roc_curve.png` | Test set ROC curve |
| `plots/false_positives.csv` | Full false positive details |
| `plots/false_positive_summary.csv` | FP vs TN descriptor comparison |
| `plots/false_positives_structures.png` | FP molecular structures |
