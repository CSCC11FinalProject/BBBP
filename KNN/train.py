# train.py — 使用最优超参数训练 KNN 并保存模型
#
# 流程：
#   1. 从 best_params.json 加载调优结果（若不存在则用默认值）
#   2. 在训练集上 fit KNeighborsClassifier
#   3. 保存模型（joblib）和 scaler 到 checkpoints/
#   4. 在测试集上快速评估，输出 AUC-ROC 和 F1

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
    # ── 1. 加载预处理数据 ──
    data = load_and_preprocess()
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    scaler = data["scaler"]

    # ── 2. 加载超参数 ──
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            params = json.load(f)
        print(f"Loaded tuned params from {PARAMS_PATH}:")
    else:
        params = DEFAULT_PARAMS
        print("No tuned params found, using defaults:")
    print(f"  {params}")

    # ── 3. 训练 KNN ──
    knn = KNeighborsClassifier(**params)
    knn.fit(X_train, y_train)
    print(f"\nKNN fitted on {len(X_train)} training samples.")

    # ── 4. 保存模型和 scaler ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, "knn_model.joblib")
    scaler_path = os.path.join(CHECKPOINT_DIR, "scaler.joblib")
    joblib.dump(knn, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    # ── 5. 快速测试评估 ──
    y_prob = knn.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, y_prob)
    test_f1 = f1_score(y_test, y_pred)
    print(f"\nTest AUC-ROC: {test_auc:.4f} | Test F1: {test_f1:.4f}")
