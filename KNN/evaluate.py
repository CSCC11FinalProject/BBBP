# evaluate.py — 测试集全面评估与可视化
#
# 参照 MPNN/evaluate.py，产出以下内容：
#   1. 核心指标：AUC-ROC, F1, Precision, Recall, Specificity
#   2. 混淆矩阵热力图 → plots/confusion_matrix.png
#   3. ROC 曲线 → plots/roc_curve.png
#   4. False Positive 误分类分析 → plots/false_positives.csv, false_positive_summary.csv
#   5. False Positive 分子结构图 → plots/false_positives_structures.png

import os
import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import (  # type: ignore
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)

from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore

from preprocess import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def load_model():
    """从 checkpoint 加载已训练的 KNN 模型。"""
    path = os.path.join(CHECKPOINT_DIR, "knn_model.joblib")
    return joblib.load(path)


def investigate_false_positives(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """分析 False Positives：实际 BBB-（不可渗透）但被预测为 BBB+（可渗透）。

    与 MPNN/evaluate.py 中的同名函数保持一致的输出格式。
    """
    df = test_df.copy()
    df["prob"] = y_prob
    df["pred"] = y_pred
    df["true"] = y_test

    fps = df[(df["true"] == 0) & (df["pred"] == 1)]
    tns = df[(df["true"] == 0) & (df["pred"] == 0)]

    print(f"\nAnalyzing {len(fps)} false positives...")
    if fps.empty:
        print("No false positives found.")
        return

    print(fps[["name", "smiles", "LogP", "TPSA", "MW", "prob"]])

    # 对比 FP 与 TN 在关键化学描述符上的均值差异
    comparison = pd.DataFrame({
        "Feature": ["LogP", "TPSA", "MW"],
        "False Positives (Mistakes)": [
            fps["LogP"].mean(), fps["TPSA"].mean(), fps["MW"].mean(),
        ],
        "True Negatives (Correct)": [
            tns["LogP"].mean(), tns["TPSA"].mean(), tns["MW"].mean(),
        ],
    })
    print("\n", comparison.to_string(index=False))

    # 保存到 CSV
    fps.to_csv(os.path.join(PLOTS_DIR, "false_positives.csv"), index=False)
    comparison.to_csv(os.path.join(PLOTS_DIR, "false_positive_summary.csv"), index=False)

    # 绘制 FP 分子结构图（最多展示 9 个）
    mols = [Chem.MolFromSmiles(s) for s in fps["smiles"]]
    mols = [m for m in mols if m is not None]
    if mols:
        legends = [f"Prob: {p:.2f}" for p in fps["prob"]]
        img = Draw.MolsToGridImage(
            mols[:9], molsPerRow=3, subImgSize=(300, 300), legends=legends[:9],
        )
        img.save(os.path.join(PLOTS_DIR, "false_positives_structures.png"))
        print(f"False positive structures saved to {PLOTS_DIR}/false_positives_structures.png")


def evaluate_on_test() -> None:
    """主评估流程：加载模型和数据 → 计算指标 → 绘图 → 误分类分析。"""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 加载数据和模型 ──
    data = load_and_preprocess()
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_df = data["test_df"]

    knn = load_model()

    # ── 预测 ──
    y_prob = knn.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ── 核心指标 ──
    test_auc = roc_auc_score(y_test, y_prob)
    test_f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(
        f"Test AUC-ROC: {test_auc:.4f} | "
        f"Test F1: {test_f1:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"Specificity: {specificity:.4f}"
    )

    # ── 混淆矩阵 ──
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["BBB-", "BBB+"], yticklabels=["BBB-", "BBB+"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── ROC 曲线 ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to {PLOTS_DIR}/")

    # ── False Positive 分析 ──
    investigate_false_positives(test_df, y_test, y_pred, y_prob)


if __name__ == "__main__":
    evaluate_on_test()
