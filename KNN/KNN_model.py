import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, recall_score, precision_score
import matplotlib.pyplot as plt

SEED = 67

def load_and_explore_data(df):
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    X = df[["LogP", "TPSA", "MW", "HBA", "HBD", "RotatableBonds", "Charge"]]
    y = df["p_np"]
    return X, y

def process_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_train_model, X_val, y_train_model, y_val = train_test_split(X_train, y_train, test_size=10/85, stratify=y_train, random_state=SEED)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_model)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_model, y_val, y_test

def tune_model(X_train_scaled, y_train_model, X_val_scaled, y_val):
    k_values = range(1, 50, 2)
    best_auc = 0
    best_k = 1
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train_model)
        y_prob_val = knn.predict_proba(X_val_scaled)[:, 1]  # 用概率
        auc = roc_auc_score(y_val, y_prob_val)
        if auc > best_auc:
            best_auc = auc
            best_k = k
    print(f"Best AUC: {best_auc:.4f} at k={best_k}")
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(
        np.concatenate((X_train_scaled, X_val_scaled)),
        np.concatenate((y_train_model, y_val))
    )
    return best_k, best_auc, final_model

def analyze_model_performance(y_test, y_pred, y_prob):
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

def visualize_roc_curve(y_test, y_prob, y_pred):
    # plot the roc curve
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.savefig("KNN/plots/roc_curve.png")
    plt.close()
    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["BBB-", "BBB+"], yticklabels=["BBB-", "BBB+"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig("KNN/plots/confusion_matrix.png")
    plt.close()

def main():
    df = pd.read_csv("./dataset/BBBP.csv")
    X, y = load_and_explore_data(df)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_model, y_val, y_test = process_data(X, y)
    best_k, best_auc, final_model = tune_model(X_train_scaled, y_train_model, X_val_scaled, y_val)
    y_pred = final_model.predict(X_test_scaled)
    y_prob = final_model.predict_proba(X_test_scaled)[:, 1]
    analyze_model_performance(y_test, y_pred, y_prob)
    visualize_roc_curve(y_test, y_prob, y_pred)

if __name__ == "__main__":
    main()