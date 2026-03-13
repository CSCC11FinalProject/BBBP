import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

# for correlation analysis:
import matplotlib.pyplot as plt
import seaborn as sns

# load all_predictors dataset, this will also contain Morgan fingerprints data:
original_df = ".../BBBP.csv"
og_df = pd.read_csv(original_df)

# remove rows with NA values:
df = og_df.dropna()
# remove identical rows:
df = df.drop_duplicates()
# remove columns with zero variance:
label = "p_np"
numeric_cols = df.select_dtypes(include=["number"]).columns.drop(label)
zero_var_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
df = df.drop(columns=zero_var_cols)

# remove identical columns with just different names:
# remove TPSA.1:
df = df.drop(columns=["TPSA.1"])
# remove HBD:
df = df.drop(columns=["HBD"])

# reset index:
df = df.reset_index(drop=True)

# split out descriptors and fingerprints after cleaning:
descriptor_cols = df.columns[4:df.columns.get_loc("morgan_0")]
finger_cols = [c for c in df.columns if c.startswith("morgan_")]

X_cont = df[descriptor_cols]
X_fp = df[finger_cols]

# remove fingerprint columns with zero variance
fp_zero_var_cols = [col for col in X_fp.columns if X_fp[col].nunique() <= 1]
if fp_zero_var_cols:
    X_fp = X_fp.drop(columns=fp_zero_var_cols)
    print("Removed zero-variance fingerprint columns:", len(fp_zero_var_cols))

print("Descriptor shape:", X_cont.shape)
print("Fingerprint shape:", X_fp.shape)

# define features and label (use descriptors only for selection)
X = X_cont.copy()
y = df[label]

print("Descriptor Feature Shape:", X.shape)
print("Label Shape:", y.shape)

# remove descriptors with extremely large values
print(X.select_dtypes(include=["number"]).max().sort_values(ascending=False).head(5))
numeric_X = X.select_dtypes(include=["number"])
ext_large_cols = [col for col in numeric_X.columns if numeric_X[col].abs().max() > 1e20]
X = X.drop(columns=ext_large_cols)
print("Removed extremely large descriptor columns:", ext_large_cols)
print(X.select_dtypes(include=["number"]).max().sort_values(ascending=False).head(5))

# 75/10/15 split for train/val/test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=0)
X_train_model, X_val, y_train_model, y_val = train_test_split(X_train, y_train, test_size=10/85, stratify=y_train, random_state=0)

# use random forest's importance ranking to select top 6 predictors, this represents NON-LINEAR:
rf = RandomForestClassifier(n_estimators=400, random_state=0)
rf.fit(X_train_model, y_train_model)
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X_train_model.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top_6_features_rf = feature_importance_df.head(6)["feature"].tolist()
print("Top 6 predictors:", top_6_features_rf)

# Use LassoCV Coefficients to select top 6 predictors, this represents LINEAR:
# we need to scale the features for Lasso:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_model)
lasso = LassoCV(cv=5, random_state=0, max_iter=10000)
lasso.fit(X_scaled, y_train_model)
lasso_coefs = pd.DataFrame({
    "feature": X_train_model.columns,
    "coef": lasso.coef_
})
non_zero_coef = lasso_coefs[lasso_coefs["coef"] != 0].copy()
non_zero_coef["abs_coef"] = non_zero_coef["coef"].abs()
top_6_features_lasso = non_zero_coef.sort_values(by="abs_coef", ascending=False).head(6)["feature"].tolist()
print("Top 6 predictors from LassoCV:", top_6_features_lasso)

# select the union of the top 6 predictors from both methods:
top_refined_features = sorted(set(top_6_features_rf) | set(top_6_features_lasso))
print("Top Refined predictors from both methods:", top_refined_features)

# save only the refined predictors:
X_train_model_with_cor = X_train_model[top_refined_features]
X_val_with_cor = X_val[top_refined_features]
X_test_with_cor = X_test[top_refined_features]

# Correlation heatmap for firstly selected predictors:
corr_mat = X_train_model_with_cor.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Firstly Selected Predictors")
plt.tight_layout()
plt.show()

# by artificially selection, we removed these predictors:
removed_predictors = ["NumHeteroatoms", "NOCount"]
# check the heatmap again after removing these predictors:
best_features = [f for f in top_refined_features if f not in removed_predictors]
X_train_model_refined = X_train_model[best_features]
X_val_refined = X_val[best_features]
X_test_refined = X_test[best_features]
corr_mat_refined = X_train_model_refined.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr_mat_refined, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap after Removing Highly Correlated Predictors")
plt.tight_layout()
plt.show()

# Training Gaussian Naive Bayes Model now with refined dataset:
# Gaussian NB with continuous descriptors:
gnb = GaussianNB()
gnb.fit(X_train_model_refined, y_train_model)

# validation performance
y_val_pred_g = gnb.predict(X_val_refined)
print("Validation Accuracy in GaussianNB:", accuracy_score(y_val, y_val_pred_g))

# validation report:
# confusion matrix:
print("Confusion Matrix in GaussianNB:")
print(confusion_matrix(y_val, y_val_pred_g))

# f1 score for validation:
print("\nF1 Validation Score in GaussianNB:")
print(f1_score(y_val, y_val_pred_g, average='weighted'))

# ROC curve for validation, Gaussian NB:
from sklearn.metrics import roc_curve, auc
y_val_prob_g = gnb.predict_proba(X_val_refined)[:, 1]
fpr_g, tpr_g, _ = roc_curve(y_val, y_val_prob_g)
roc_auc_g = auc(fpr_g, tpr_g)

# Bernoulli NB with fingerprints data:
# split fingerprints using the same indices as descriptor splits
X_fp_train = X_fp.loc[X_train_model.index]
X_fp_val = X_fp.loc[X_val.index]
X_fp_test = X_fp.loc[X_test.index]

bnb = BernoulliNB()
bnb.fit(X_fp_train, y_train_model)

# validation performance
y_val_pred_b = bnb.predict(X_fp_val)
print("Validation Accuracy in BernoulliNB:", accuracy_score(y_val, y_val_pred_b))

# validation report:
# confusion matrix:
print("Confusion Matrix in BernoulliNB:")
print(confusion_matrix(y_val, y_val_pred_b))

# f1 score for validation, Bernoulli NB:
print("\nF1 Validation Score in BernoulliNB:")
print(f1_score(y_val, y_val_pred_b, average='weighted'))

# ROC curve for validation, Bernoulli NB:
y_val_prob_b = bnb.predict_proba(X_fp_val)[:, 1]
fpr_b, tpr_b, _ = roc_curve(y_val, y_val_prob_b)
roc_auc_b = auc(fpr_b, tpr_b)

plt.figure()
plt.plot(fpr_g, tpr_g, color="blue", lw=2, label="GaussianNB (AUC = {:.2f})".format(roc_auc_g))
plt.plot(fpr_b, tpr_b, color="green", lw=2, label="BernoulliNB (AUC = {:.2f})".format(roc_auc_b))
plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: GaussianNB vs BernoulliNB")
plt.legend(loc="lower right")
plt.show()


# We ensemble these two models, as Bernoulli NB will be more accurate when predicting label 0 class;
# we will fully depend on it when it predicts label 0, otherwise we will depend fully on Gaussian NB:
# predictions
y_pred_g = gnb.predict(X_test_refined)
y_pred_b = bnb.predict(X_fp_test)

# probabilities for computing AUROC
prob_g = gnb.predict_proba(X_test_refined)[:, 1]
prob_b = bnb.predict_proba(X_fp_test)[:, 1]

# by our ensemble prediction rule
y_test_pred_ensemble = np.where(y_pred_b == 0, 0, y_pred_g)

# thus the probability will become:
prob_ensemble = np.where(y_pred_b == 0, prob_b, prob_g)

print("Test Accuracy (Ensemble NB Version):", accuracy_score(y_test, y_test_pred_ensemble))
print("Confusion Matrix (Ensemble NB Version):")
print(confusion_matrix(y_test, y_test_pred_ensemble))

print("\nF1 Test Score (Ensemble NB Version):")
print(f1_score(y_test, y_test_pred_ensemble, average="weighted"))

print("\nAUROC Test Score (Ensemble NB Version):")
print(roc_auc_score(y_test, prob_ensemble))


