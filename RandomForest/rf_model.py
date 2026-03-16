import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# Load dataset
dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir, '..', 'dataset', 'BBBP.csv')
df = pd.read_csv(data_path)

df = df.dropna()
df = df.drop_duplicates()

# Extract target label and features
y = df['p_np']

morgan_cols = [col for col in df.columns if col.startswith('morgan_')]
extra_cols = ['LogP', 'TPSA']
X = df[morgan_cols + extra_cols]

# Split dataset
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=10/85, random_state=0, stratify=y_dev)

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

test_fold = np.concatenate([
    np.full(len(X_train), -1),
    np.full(len(X_val), 0)
])
ps = PredefinedSplit(test_fold)


# RF Model
rf = RandomForestClassifier(random_state=67, class_weight='balanced')

# # Run 1
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# Run 2
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 30, 40],
    'min_samples_split': [3, 5, 7]
}




# Hyperparameter Tuning

print("\nHyperparameter Tuning")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1', cv=ps, n_jobs=-1, verbose=1)
grid_search.fit(X_train_val, y_train_val)

best_rf = grid_search.best_estimator_

# Model Evaluation
y_hat = best_rf.predict(X_test)
y_hat_proba = best_rf.predict_proba(X_test)[:, 1] # Probabilities for AUC-ROC

print("\nModel Evaluation")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_hat_proba):.4f}")
print(f"F1-Score: {f1_score(y_test, y_hat):.4f}")

print(classification_report(y_test, y_hat))