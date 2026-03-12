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

print("Saved class distribution and chemical space plots to plots/")