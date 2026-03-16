import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from rdkit import Chem
from rdkit.Chem import Draw


# Load dataset
dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir, '..', 'dataset', 'BBBP.csv')
df = pd.read_csv(data_path)

df = df.dropna()
df = df.drop_duplicates()

# Extract target label and features
y = df['p_np']

morgan_cols = [col for col in df.columns if col.startswith('morgan_')]
extra_cols = ['LogP', 'TPSA']
X = df[morgan_cols + extra_cols]

# Split dataset
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=10/85, random_state=0, stratify=y_dev)

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

test_fold = np.concatenate([
    np.full(len(X_train), -1),
    np.full(len(X_val), 0)
])
ps = PredefinedSplit(test_fold)


# RF Model
rf = RandomForestClassifier(random_state=67, class_weight='balanced')

# # Run 1
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# # Run 2
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 30, 40],
#     'min_samples_split': [3, 5, 7]
# }

# Run 3
param_grid = {
    'n_estimators': [90, 100, 110],
    'max_depth': [None, 50, 60],
    'min_samples_split': [4, 5, 6]
}

# # Run 4
# param_grid = {
#     'n_estimators': [85, 90, 95],
#     'max_depth': [None, 70, 80],
#     'min_samples_split': [4, 5, 6]
# }


# Hyperparameter Tuning

print("\nHyperparameter Tuning")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1', cv=ps, n_jobs=-1, verbose=1)
grid_search.fit(X_train_val, y_train_val)

best_rf = grid_search.best_estimator_

# Model Evaluation
y_hat = best_rf.predict(X_test)
y_hat_proba = best_rf.predict_proba(X_test)[:, 1] # Probabilities for AUC-ROC

print("\nModel Evaluation")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_hat_proba):.4f}")
print(f"F1-Score: {f1_score(y_test, y_hat):.4f}")

print(classification_report(y_test, y_hat))



# False Positive Analysis
fp_indices = y_test[(y_test == 0) & (y_hat == 1)].index
fp_df = df.loc[fp_indices]

fp_csv_path = 'rf_false_positives.csv'
fp_df.to_csv(fp_csv_path, index=False)

summary_cols = ['LogP', 'TPSA'] 
fp_summary = fp_df[summary_cols].describe()
fp_summary.to_csv('rf_false_positive_summary.csv')

smiles_list = fp_df['smiles'].head(8).tolist()
names_list = fp_df['name'].head(8).tolist()


mols = []
valid_names = []
for smile, name in zip(smiles_list, names_list):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        mols.append(mol)
        valid_names.append(str(name))
if mols:
    img = Draw.MolsToGridImage(
        mols, 
        molsPerRow=4, 
        subImgSize=(300, 300), 
        legends=valid_names
    )
    img.save('rf_false_positives_structures.png')


# Confusion Matrix
cm = confusion_matrix(y_test, y_hat)

plt.figure(figsize=(5, 5), dpi=150)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['BBB-', 'BBB+'], 
            yticklabels=['BBB-', 'BBB+'],
            cbar=True)
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', bbox_inches='tight')
plt.show()


# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_hat_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 5), dpi=150)
plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--') # Diagonal dashed line
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('rf_roc_curve.png', bbox_inches='tight')
plt.show()
