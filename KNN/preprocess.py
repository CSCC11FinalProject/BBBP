# preprocess.py - load BBBP data, generate fingerprints, reduce dimensions, split and scale

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset.utils import get_morgan_fingerprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE_DIR, "..", "dataset", "BBBP.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SEED = 67

DESCRIPTOR_COLS = ["LogP", "TPSA", "MW", "HBA", "HBD", "RotatableBonds", "Charge"]


def _generate_fingerprints(smiles_series: pd.Series) -> tuple[np.ndarray, list[bool]]:
    """Convert SMILES to 2048-bit Morgan Fingerprints.
    Returns (fp_matrix, valid_mask) where valid_mask marks successfully converted rows.
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
    """Pearson correlation matrix for the 7 chemical descriptors."""
    corr = df[DESCRIPTOR_COLS].corr()

    print("\n=== Chemical Descriptor Correlation Matrix ===")
    print(corr.to_string())

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
    n_components: int = 100,
) -> dict:
    """Load dataset, generate features, reduce dimensionality, scale, and split.

    Returns dict with X_train/val/test, y_train/val/test, fitted transformers,
    test_df, feature_names, and optional correlation_matrix.
    """
    df = pd.read_csv(CSV).dropna()
    print(f"Loaded {len(df)} samples from {CSV}")

    # SMILES -> Morgan Fingerprint
    fp_matrix, valid_mask = _generate_fingerprints(df["smiles"])
    df = df[valid_mask].reset_index(drop=True)
    print(f"Valid fingerprints: {len(df)} / {len(valid_mask)} (dropped {sum(not v for v in valid_mask)})")

    descriptors = df[DESCRIPTOR_COLS].values.astype(np.float32)

    corr = None
    if plot_correlation:
        corr = _analyze_correlations(df)

    # Split 75/10/15 before fitting any transformers
    y = df["p_np"].values
    indices = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.15, random_state=SEED, stratify=y,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=2/17, random_state=SEED, stratify=y[idx_trainval],
    )

    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    test_df = df.iloc[idx_test].reset_index(drop=True)

    fp_train, fp_val, fp_test = fp_matrix[idx_train], fp_matrix[idx_val], fp_matrix[idx_test]
    desc_train, desc_val, desc_test = descriptors[idx_train], descriptors[idx_val], descriptors[idx_test]

    # VarianceThreshold: remove nearly constant bits
    vt = VarianceThreshold(threshold=0.01)
    fp_train = vt.fit_transform(fp_train)
    fp_val = vt.transform(fp_val)
    fp_test = vt.transform(fp_test)
    n_after_vt = fp_train.shape[1]

    # PCA on fingerprints
    actual_components = min(n_components, fp_train.shape[1], fp_train.shape[0])
    pca = PCA(n_components=actual_components, random_state=SEED)
    fp_train = pca.fit_transform(fp_train)
    fp_val = pca.transform(fp_val)
    fp_test = pca.transform(fp_test)
    explained_var = pca.explained_variance_ratio_.sum()

    # Standardize FP and descriptors separately
    fp_scaler = StandardScaler()
    fp_train = fp_scaler.fit_transform(fp_train)
    fp_val = fp_scaler.transform(fp_val)
    fp_test = fp_scaler.transform(fp_test)

    desc_scaler = StandardScaler()
    desc_train = desc_scaler.fit_transform(desc_train)
    desc_val = desc_scaler.transform(desc_val)
    desc_test = desc_scaler.transform(desc_test)

    # Concatenate PCA-reduced fingerprints + descriptors
    X_train = np.hstack([fp_train, desc_train])
    X_val = np.hstack([fp_val, desc_val])
    X_test = np.hstack([fp_test, desc_test])

    feature_names = [f"PC_{i}" for i in range(actual_components)] + DESCRIPTOR_COLS

    n_pos_train = int(y_train.sum())
    print(f"Split -- train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Dimensionality reduction: 2048 FP -> {n_after_vt} (VT) -> {actual_components} (PCA)")
    print(f"PCA explained variance: {explained_var:.1%}")
    print(f"Final feature dim: {X_train.shape[1]}  ({actual_components} PCA + {len(DESCRIPTOR_COLS)} descriptors)")
    print(f"Train class balance: {n_pos_train}/{len(y_train)} positive "
          f"({n_pos_train / len(y_train):.1%})")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "fp_scaler": fp_scaler, "desc_scaler": desc_scaler,
        "variance_threshold": vt, "pca": pca,
        "feature_names": feature_names,
        "test_df": test_df,
        "correlation_matrix": corr,
    }


if __name__ == "__main__":
    data = load_and_preprocess(plot_correlation=True)
    print("\nPreprocessing complete. Ready for tuning / training.")
