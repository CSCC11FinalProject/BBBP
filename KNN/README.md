# KNN Model

This folder contains a K-Nearest Neighbors (KNN) model for predicting blood-brain barrier (BBB) permeability on the BBBP dataset.

## Features

The model uses the following 7 chemical descriptors:

- LogP
- TPSA
- MW
- HBA
- HBD
- RotatableBonds
- Charge

The label is `p_np`, which indicates BBB permeability.

## Method
1. Load the BBBP dataset.
2. Remove missing values and duplicate rows.
3. Split the data into training, validation, and test sets (75/10/15).
4. Apply `StandardScaler` to standardize the features.
5. Tune the hyperparameter `k` using validation ROC-AUC.
6. Train the final KNN model on the combined training and validation sets.
7. Evaluate the model on the test set.

## Evaluation

The model reports:

- F1 score
- ROC-AUC
- Recall
- Precision

It also generates:

- ROC curve
- Confusion matrix

## Output

Visualized Plots are saved to:

- `KNN/plots/roc_curve.png`
- `KNN/plots/confusion_matrix.png`

## Run

From the project root, run:

```bash
python KNN/KNN_model.py
```

## Results

- F1 Score: 0.9032

- AUC: 0.8495

- Recall: 0.9573

