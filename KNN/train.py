# train.py - train KNN with best hyperparameters, save model + preprocessors

import os
import json
import numpy as np
import joblib  # type: ignore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score

from preprocess import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "best_params.json")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

DEFAULT_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "minkowski",
    "p": 2,
}


if __name__ == "__main__":
    data = load_and_preprocess()
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # Load hyperparameters
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            params = json.load(f)
        print(f"Loaded tuned params from {PARAMS_PATH}:")
    else:
        params = DEFAULT_PARAMS
        print("No tuned params found, using defaults:")
    print(f"  {params}")

    # Merge train + val now that hyperparams are fixed
    X_final = np.vstack([X_train, X_val])
    y_final = np.concatenate([y_train, y_val])

    knn = KNeighborsClassifier(**params)
    knn.fit(X_final, y_final)
    print(f"\nKNN fitted on {len(X_final)} samples (train + val).")

    # Save model and preprocessing objects
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, "knn_model.joblib")
    joblib.dump(knn, model_path)
    print(f"Model saved to {model_path}")

    for key in ("variance_threshold", "pca", "fp_scaler", "desc_scaler"):
        obj_path = os.path.join(CHECKPOINT_DIR, f"{key}.joblib")
        joblib.dump(data[key], obj_path)
        print(f"  {key} saved to {obj_path}")

    # Quick test-set check
    y_prob = knn.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, y_prob)
    test_f1 = f1_score(y_test, y_pred)
    print(f"\nTest AUC-ROC: {test_auc:.4f} | Test F1: {test_f1:.4f}")
