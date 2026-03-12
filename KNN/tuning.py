# tuning.py — Using Optuna for KNN hyperparameter tuning
#
# Search space:
#   - n_neighbors (k): odd numbers between [1, 50]
#   - weights: uniform / distance
#   - metric: euclidean / manhattan / minkowski
#   - p (only for minkowski): [1, 5]
#
# Optimization objective: Validation set AUC-ROC (direction="maximize", consistent with MPNN)
# Output: Best parameters saved to best_params.json

import os
import json
import optuna  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore

from preprocess import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "best_params.json")
N_TRIALS = 50  # KNN is very fast to train; 50 trials are sufficient


def objective(
    trial: optuna.Trial,
    X_train, X_val, y_train, y_val,
) -> float:
    """Single trial: sample hyperparameters → train KNN → return val AUC-ROC."""

    # Restrict k to be odd to avoid tie votes
    n_neighbors = trial.suggest_int("n_neighbors", 1, 49, step=2)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])

    params: dict = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
    }

    # p parameter for minkowski distance (p=1 is manhattan, p=2 is euclidean)
    if metric == "minkowski":
        p = trial.suggest_int("p", 1, 5)
        params["p"] = p

    knn = KNeighborsClassifier(**params)
    knn.fit(X_train, y_train)

    # Use predicted probabilities to compute AUC-ROC (more informative than hard labels)
    y_prob = knn.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    return auc


if __name__ == "__main__":
    # ── Load preprocessed data ──
    data = load_and_preprocess()
    X_train, X_val = data["X_train"], data["X_val"]
    y_train, y_val = data["y_train"], data["y_val"]

    # ── Optuna search ──
    study = optuna.create_study(direction="maximize", study_name="knn_bbbp")
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ── Output results ──
    print(f"\nBest trial (out of {N_TRIALS}):")
    print(f"  Val AUC-ROC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # ── Save best parameters ──
    with open(PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest params saved to {PARAMS_PATH}")
