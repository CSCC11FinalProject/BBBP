# train.py — Train KNN with optimal hyperparameters and save the model
#
# Workflow:
#   1. Load tuning results from best_params.json (use defaults if file does not exist)
#   2. Fit KNeighborsClassifier on the training set
#   3. Save the model (joblib) and scaler to checkpoints/
#   4. Quickly evaluate on the test set, print AUC-ROC and F1 score

import os
import json
import joblib  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import roc_auc_score, f1_score  # type: ignore

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
    # 1. Load preprocessed data
    data = load_and_preprocess()
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    scaler = data["scaler"]

    # 2. Load hyperparameters
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            params = json.load(f)
        print(f"Loaded tuned params from {PARAMS_PATH}:")
    else:
        params = DEFAULT_PARAMS
        print("No tuned params found, using defaults:")
    print(f"  {params}")

    # 3. Train KNN
    knn = KNeighborsClassifier(**params)
    knn.fit(X_train, y_train)
    print(f"\nKNN fitted on {len(X_train)} training samples.")

    # 4. Save model and scaler
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, "knn_model.joblib")
    scaler_path = os.path.join(CHECKPOINT_DIR, "scaler.joblib")
    joblib.dump(knn, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    # 5. Quick test evaluation
    y_prob = knn.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, y_prob)
    test_f1 = f1_score(y_test, y_pred)
    print(f"\nTest AUC-ROC: {test_auc:.4f} | Test F1: {test_f1:.4f}")
