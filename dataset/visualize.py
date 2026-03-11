import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('bbbp.csv')
sns.set_theme(style="darkgrid")

# Target Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='p_np', data=df, palette='viridis')
plt.title('Distribution of BBB Permeability (p_np)')
plt.xlabel('Permeability (0: No, 1: Yes)')
plt.ylabel('Count')
plt.savefig('plots/class_distribution.png')
plt.close()

# Distribution of Key Descriptors
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(ax=axes[0], x='p_np', y='LogP', data=df, palette='Set2')
axes[0].set_title('LogP vs Permeability')

sns.boxplot(ax=axes[1], x='p_np', y='TPSA', data=df, palette='Set2')
axes[1].set_title('TPSA vs Permeability')

sns.boxplot(ax=axes[2], x='p_np', y='MW', data=df, palette='Set2')
axes[2].set_title('Molecular Weight vs Permeability')

plt.tight_layout()
plt.savefig('plots/feature_distributions.png')
plt.close()

# Correlation Matrix/Heatmap
plt.figure(figsize=(8, 8))
# Drop non-numerical columns
numerical_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['num'])
corr = numerical_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Chemical Space: LogP vs TPSA
plt.figure(figsize=(8, 6))
sns.scatterplot(x='LogP', y='TPSA', hue='p_np', data=df, alpha=0.6, palette='coolwarm')
plt.axhline(90, color='red', linestyle='--', label='TPSA Threshold (90)')
plt.axvline(1, color='green', linestyle='--', label='LogP Lower')
plt.axvline(5, color='green', linestyle='--', label='LogP Upper')
plt.title('Chemical Space: LogP vs TPSA')
plt.legend()
plt.savefig('plots/chemical_space.png')
plt.close()

print("All visualizations saved to plots/")