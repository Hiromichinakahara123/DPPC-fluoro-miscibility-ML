"""
main_analysis.py

Feature extraction, multivariate analysis, and Bayesian logistic regression
for miscibility prediction of DPPC/fluorinated compound binary monolayers.

- Input: CSV file with isomeric SMILES for F and DPPC, experimental conditions, miscibility label.
- Output: Feature dataset, PCA and correlation plots, Bayesian regression summary.

Author: Hiromichi Nakahara, Osamu Shibata
2024
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdPartialCharges
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# For Bayesian logistic regression
import pymc as pm
import arviz as az

import os

# ---- Step 1: Load CSV ----
if len(sys.argv) < 2:
    print("Usage: python main_analysis.py <your_data.csv>")
    sys.exit(1)
else:
    csv_path = sys.argv[1]

df = pd.read_csv(csv_path)

# Rename columns for easier handling
df = df.rename(columns={
    'isomeric SMILES(F)': 'smiles_F',
    'isomeric SMILES(DPPC)': 'smiles_DPPC',
    'ionic strength': 'ionic_st',
    'temperature(K)': 'temp_K',
    '(Thermodynamical) Miscibility': 'Miscibility',
})

# ---- Step 2: Define descriptor extraction ----

def count_f_atoms(mol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')

def count_cf3(mol):
    patt = Chem.MolFromSmarts('C(F)(F)F')
    return len(mol.GetSubstructMatches(patt))

def count_cf2(mol):
    patt = Chem.MolFromSmarts('C(F)F')
    return len(mol.GetSubstructMatches(patt))

def f_ratio(mol):
    nF = count_f_atoms(mol)
    n_total = mol.GetNumAtoms()
    return nF / n_total if n_total > 0 else 0

def compute_desc(smiles, error_log, prefix=""):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        error_log.append(smiles)
        return {}
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        return {
            f"{prefix}MolWt": Descriptors.MolWt(mol),
            f"{prefix}LogP": Descriptors.MolLogP(mol),
            f"{prefix}TPSA": Descriptors.TPSA(mol),
            f"{prefix}HBA": Descriptors.NumHAcceptors(mol),
            f"{prefix}HBD": Descriptors.NumHDonors(mol),
            f"{prefix}RotBonds": Descriptors.NumRotatableBonds(mol),
            f"{prefix}nF": count_f_atoms(mol),
            f"{prefix}CF3": count_cf3(mol),
            f"{prefix}CF2": count_cf2(mol),
            f"{prefix}f_ratio": f_ratio(mol),
        }
    except Exception as e:
        error_log.append(f"{smiles} ({e})")
        return {}

F_desc_names = ['F_MolWt', 'F_LogP', 'F_TPSA', 'F_HBA', 'F_HBD', 'F_RotBonds', 'F_nF', 'F_CF3', 'F_CF2', 'F_f_ratio']
DPPC_desc_names = ['DPPC_MolWt', 'DPPC_LogP', 'DPPC_TPSA', 'DPPC_HBA', 'DPPC_HBD', 'DPPC_RotBonds', 'DPPC_nF', 'DPPC_CF3', 'DPPC_CF2', 'DPPC_f_ratio']

# ---- Step 3: Compute descriptors ----

SMILES_F_error_log = []
SMILES_DPPC_error_log = []
F_feats = df['smiles_F'].apply(lambda x: compute_desc(x, SMILES_F_error_log, prefix='F_'))
DPPC_feats = df['smiles_DPPC'].apply(lambda x: compute_desc(x, SMILES_DPPC_error_log, prefix='DPPC_'))

df_F = pd.DataFrame(list(F_feats))
df_DPPC = pd.DataFrame(list(DPPC_feats))

df = pd.concat([df, df_F, df_DPPC], axis=1)

# ---- Step 4: Pairwise (delta/ratio) features ----
df['delta_LogP']    = df['F_LogP'] - df['DPPC_LogP']
df['ratio_LogP']    = df['F_LogP'] / df['DPPC_LogP']
df['delta_MolWt']   = df['F_MolWt'] - df['DPPC_MolWt']
df['ratio_MolWt']   = df['F_MolWt'] / df['DPPC_MolWt']
df['delta_TPSA']    = df['F_TPSA'] - df['DPPC_TPSA']
df['delta_nF']      = df['F_nF'] - df['DPPC_nF']

# ---- Step 5: Select final features ----
feature_cols = [
    'F_MolWt', 'F_LogP', 'F_TPSA', 'F_HBA', 'F_HBD', 'F_RotBonds',
    'F_nF', 'F_CF3', 'F_CF2', 'F_f_ratio',
    'delta_LogP', 'ratio_LogP', 'delta_MolWt', 'ratio_MolWt',
    'delta_TPSA', 'delta_nF',
    'ionic_st', 'pH', 'temp_K', 'Miscibility'
]

df_features = df[feature_cols].copy()

# Drop missing
df_features = df_features.dropna()
df_features.reset_index(drop=True, inplace=True)

# ---- Step 6: Prepare X, y ----
X = df_features.drop(['Miscibility'], axis=1)
y_bin = df_features['Miscibility'].apply(lambda x: 1 if x == 'M' else 0) # 1: miscible, 0: non-miscible

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Step 7: PCA plot ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_bin.map({0: 'P/I', 1: 'M'}),
                palette='Set1', s=80, alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA plot of feature space (M vs P/I)')
plt.legend(title="Miscibility")
plt.tight_layout()
plt.savefig("PCA_plot.png", dpi=300)
plt.close()

# ---- Step 8: Correlation heatmap ----
corr = pd.DataFrame(X_scaled, columns=X.columns).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("Feature_Correlation_Heatmap.png", dpi=300)
plt.close()

# ---- Step 9: Bayesian logistic regression (PyMC) ----
selected_features = [
    'delta_MolWt',
    'ratio_LogP',
    'delta_TPSA',
    'delta_nF',
    'ionic_st', 'pH', 'temp_K'
]

X_sel = df_features[selected_features].values
X_sel_scaled = StandardScaler().fit_transform(X_sel)
n_features = X_sel_scaled.shape[1]

with pm.Model() as model_bin:
    beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    logits = pm.math.dot(X_sel_scaled, beta) + intercept
    p = pm.Deterministic("p", pm.math.sigmoid(logits))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_bin.values)
    trace_bin = pm.sample(2000, tune=1000, chains=4, cores=2, target_accept=0.95,
                          return_inferencedata=True, random_seed=42)

# ---- Step 10: Model summary and coefficient plot ----
summary = az.summary(trace_bin, var_names=['beta'])
print(summary)

# Coefficient plot
import numpy as np
means = summary['mean'].values
lower = summary['hdi_3%'].values
upper = summary['hdi_97%'].values
x = np.arange(len(selected_features))

plt.figure(figsize=(7, 4))
plt.bar(x, means, yerr=[means - lower, upper - means], capsize=5, alpha=0.7, color='skyblue')
plt.xticks(x, selected_features, rotation=30, ha='right')
plt.ylabel('Coefficient (mean)')
plt.title('Feature Importance (Posterior mean ± 95% credible interval)')
plt.tight_layout()
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.savefig("Bayesian_coefficients.png", dpi=300)
plt.close()

print("Analysis finished. Outputs:")
print("  - PCA_plot.png")
print("  - Feature_Correlation_Heatmap.png")
print("  - Bayesian_coefficients.png")
print("  - Console: Bayesian model summary")

if SMILES_F_error_log:
    print("Warning: Failed to parse the following F SMILES:")
    print(SMILES_F_error_log)
if SMILES_DPPC_error_log:
    print("Warning: Failed to parse the following DPPC SMILES:")
    print(SMILES_DPPC_error_log)
