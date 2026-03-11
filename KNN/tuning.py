# tuning.py — 使用 Optuna 进行 KNN 超参数调优
#
# 搜索空间：
#   - n_neighbors (k): [1, 50] 之间的奇数
#   - weights: uniform / distance
#   - metric: euclidean / manhattan / minkowski
#   - p (仅 minkowski): [1, 5]
#
# 优化目标：验证集 AUC-ROC（direction="maximize"，与 MPNN 一致）
# 输出：最佳参数保存至 best_params.json

import os
import json
import optuna  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore

from preprocess import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "best_params.json")
N_TRIALS = 50  # KNN 训练极快，50 次 trial 绰绰有余


def objective(
    trial: optuna.Trial,
    X_train, X_val, y_train, y_val,
) -> float:
    """单次 trial：采样超参数 → 训练 KNN → 返回 val AUC-ROC。"""

    # k 值限制为奇数，避免平票
    n_neighbors = trial.suggest_int("n_neighbors", 1, 49, step=2)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])

    params: dict = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
    }

    # minkowski 距离的 p 参数（p=1 等价于 manhattan，p=2 等价于 euclidean）
    if metric == "minkowski":
        p = trial.suggest_int("p", 1, 5)
        params["p"] = p

    knn = KNeighborsClassifier(**params)
    knn.fit(X_train, y_train)

    # 用预测概率计算 AUC-ROC（比硬标签更能区分模型好坏）
    y_prob = knn.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    return auc


if __name__ == "__main__":
    # ── 加载预处理数据 ──
    data = load_and_preprocess()
    X_train, X_val = data["X_train"], data["X_val"]
    y_train, y_val = data["y_train"], data["y_val"]

    # ── Optuna 搜索 ──
    study = optuna.create_study(direction="maximize", study_name="knn_bbbp")
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ── 输出结果 ──
    print(f"\nBest trial (out of {N_TRIALS}):")
    print(f"  Val AUC-ROC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # ── 保存最佳参数 ──
    with open(PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest params saved to {PARAMS_PATH}")
