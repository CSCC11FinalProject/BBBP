# tuning.py - Optuna hyperparameter search for KNN
# Search space: n_neighbors (odd 1-49), weights, metric, p (for minkowski)
# Objective: validation AUC-ROC

import os
import json
import optuna  # type: ignore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from preprocess import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "best_params.json")
N_TRIALS = 50


def objective(
    trial: optuna.Trial,
    X_train, X_val, y_train, y_val,
) -> float:
    """Sample hyperparameters, train KNN, return val AUC-ROC."""

    n_neighbors = trial.suggest_int("n_neighbors", 1, 49, step=2)  # odd only
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])

    params: dict = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
    }

    # p=1 and p=2 already covered by manhattan/euclidean
    if metric == "minkowski":
        p = trial.suggest_int("p", 3, 5)
        params["p"] = p

    knn = KNeighborsClassifier(**params)
    knn.fit(X_train, y_train)

    y_prob = knn.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    return auc


if __name__ == "__main__":
    data = load_and_preprocess()
    X_train, X_val = data["X_train"], data["X_val"]
    y_train, y_val = data["y_train"], data["y_val"]

    study = optuna.create_study(direction="maximize", study_name="knn_bbbp")
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\nBest trial (out of {N_TRIALS}):")
    print(f"  Val AUC-ROC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    with open(PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest params saved to {PARAMS_PATH}")
