import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import numpy as np

# load all_predictors dataset:
original_df = ".../BBBP_all_rdkit_descriptors.csv"
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
# reset index:
df = df.reset_index(drop=True)

# define features and label:
X = df.select_dtypes(include=["number"]).drop(columns=[label])
y = df[label]
print("Features Shape:", X.shape)
print("Label Shape:", y.shape)

# remove descriptors with extremely large values
print(X.max().sort_values(ascending=False).head(5))
ext_large_cols = [col for col in X.columns if X[col].abs().max() > 1e20]
df = df.drop(columns=ext_large_cols)
X = df.select_dtypes(include=["number"]).drop(columns=[label])
print(X.max().sort_values(ascending=False).head(5))

# use random forest to select top 5 predictors:
rf = RandomForestClassifier(n_estimators=400, random_state=0)
rf.fit(X, y)
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top_5_features_rf = feature_importance_df.head(5)["feature"].tolist()
print("Top 5 predictors:", top_5_features_rf)

# Use LassoCV to selct top 5 predictors:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lasso = LassoCV(cv=5, random_state=0, max_iter=10000)
lasso.fit(X_scaled, y)
lasso_coefs = pd.DataFrame({
    "feature": X.columns,
    "coef": lasso.coef_
})
non_zero_coef = lasso_coefs[lasso_coefs["coef"] != 0].copy()
non_zero_coef["abs_coef"] = non_zero_coef["coef"].abs()
top_5_features_lasso = non_zero_coef.sort_values(by="abs_coef", ascending=False).head(5)["feature"].tolist()
print("Top 5 predictors from LassoCV:", top_5_features_lasso)

# select the union of the top 5 predictors from both methods:
top_refined_features = list(set(top_5_features_rf) | set(top_5_features_lasso))
print("Top Refined predictors from both methods:", top_refined_features)

# save only the refined predictors and the label to a new CSV file:
top_refined_df = df[top_refined_features + [label]]
top_refined_df.to_csv(".../BBBP_top_predictors.csv", index=False)


