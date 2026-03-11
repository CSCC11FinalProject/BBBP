# preprocess.py — 数据预处理与特征工程
#
# 职责：
#   1. 加载已预处理的 BBBP.csv（化学描述符已由 dataset/process.py 生成）
#   2. 将 SMILES 转为 2048 维 Morgan Fingerprint（调用 dataset/utils.py）
#   3. 拼接 Morgan FP + 化学描述符 → 完整特征矩阵
#   4. 相关矩阵分析（仅针对 7 个描述符），辅助特征选择决策
#   5. StandardScaler 标准化（仅在训练集上 fit，防止数据泄露）
#   6. 按 80/10/10 划分 train/val/test（stratified，SEED=67）
#
# 对外接口：
#   load_and_preprocess() → 返回划分好的数据、scaler、test_df 等

import os
import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

# 将项目根目录加入 path，以便导入 dataset 包
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset.utils import get_morgan_fingerprint  # type: ignore

# ── 常量 ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE_DIR, "..", "dataset", "BBBP.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SEED = 67

# CSV 中已有的 7 个化学描述符列
DESCRIPTOR_COLS = ["LogP", "TPSA", "MW", "HBA", "HBD", "RotatableBonds", "Charge"]


def _generate_fingerprints(smiles_series: pd.Series) -> tuple[np.ndarray, list[bool]]:
    """将 SMILES 序列批量转为 2048-bit Morgan Fingerprint。

    Returns:
        fp_matrix: shape (n_valid, 2048)
        valid_mask: 长度与原始序列相同的布尔列表，标记转换成功的行
    """
    fps = []
    valid_mask = []
    for smiles in smiles_series:
        fp = get_morgan_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    return np.array(fps, dtype=np.float32), valid_mask


def _analyze_correlations(df: pd.DataFrame, save_plot: bool = True) -> pd.DataFrame:
    """计算化学描述符的 Pearson 相关矩阵，识别高度相关的特征对。

    仅对 7 个描述符进行分析（Morgan FP 是 2048 维稀疏二值向量，不适合此分析）。
    """
    corr = df[DESCRIPTOR_COLS].corr()

    print("\n=== Chemical Descriptor Correlation Matrix ===")
    print(corr.to_string())

    # 找出 |r| > 0.85 的特征对
    high_corr_pairs = []
    for i in range(len(DESCRIPTOR_COLS)):
        for j in range(i + 1, len(DESCRIPTOR_COLS)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append((DESCRIPTOR_COLS[i], DESCRIPTOR_COLS[j], r))

    if high_corr_pairs:
        print("\nHighly correlated pairs (|r| > 0.85):")
        for f1, f2, r in high_corr_pairs:
            print(f"  {f1} <-> {f2}: r = {r:.3f}")
        print("Consider removing one feature from each pair to reduce redundancy.")
    else:
        print("\nNo feature pairs with |r| > 0.85. All descriptors retained.")

    if save_plot:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Chemical Descriptor Correlation Matrix")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Correlation matrix plot saved to {path}")

    return corr


def load_and_preprocess(
    plot_correlation: bool = False,
) -> dict:
    """加载数据集、生成特征、缩放、划分。

    Returns:
        dict with keys:
            X_train, X_val, X_test       — 标准化后的特征矩阵
            y_train, y_val, y_test        — 标签
            scaler                        — 已 fit 的 StandardScaler
            feature_names                 — 特征列名列表
            test_df                       — 测试集对应的原始 DataFrame（含 smiles、name 等）
            correlation_matrix            — 描述符相关矩阵（plot_correlation=True 时才有）
    """
    # ── 1. 加载数据，丢弃 NaN 行 ──
    df = pd.read_csv(CSV).dropna()
    print(f"Loaded {len(df)} samples from {CSV}")

    # ── 2. SMILES → 2048-bit Morgan Fingerprint ──
    fp_matrix, valid_mask = _generate_fingerprints(df["smiles"])
    df = df[valid_mask].reset_index(drop=True)
    print(f"Valid fingerprints: {len(df)} / {len(valid_mask)} (dropped {sum(not v for v in valid_mask)})")

    # ── 3. 提取已有化学描述符 ──
    descriptors = df[DESCRIPTOR_COLS].values.astype(np.float32)

    # ── 4. 相关矩阵分析（可选） ──
    corr = None
    if plot_correlation:
        corr = _analyze_correlations(df)

    # ── 5. 拼接特征：Morgan FP (2048) + 描述符 (7) = 2055 维 ──
    X = np.hstack([fp_matrix, descriptors])
    y = df["p_np"].values
    feature_names = [f"FP_{i}" for i in range(fp_matrix.shape[1])] + DESCRIPTOR_COLS

    # ── 6. 按索引划分 80/10/10，stratified，保持可复现 ──
    indices = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.1, random_state=SEED, stratify=y,
    )
    # 从剩余 90% 中再分出 val（占总体的 10%，即占 trainval 的 1/9）
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=1 / 9, random_state=SEED, stratify=y[idx_trainval],
    )

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    test_df = df.iloc[idx_test].reset_index(drop=True)

    # ── 7. 标准化：仅在训练集上 fit，防止数据泄露 ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── 汇总 ──
    n_pos_train = int(y_train.sum())
    print(f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Feature dim: {X_train.shape[1]}  (2048 Morgan FP + {len(DESCRIPTOR_COLS)} descriptors)")
    print(f"Train class balance: {n_pos_train}/{len(y_train)} positive "
          f"({n_pos_train / len(y_train):.1%})")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "test_df": test_df,
        "correlation_matrix": corr,
    }


# ── 单独运行：验证预处理流程并输出相关矩阵 ──
if __name__ == "__main__":
    data = load_and_preprocess(plot_correlation=True)
    print("\nPreprocessing complete. Ready for tuning / training.")
