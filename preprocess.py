import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path

BASE = Path(__file__).resolve().parent

merged = pd.read_csv(BASE / 'merged_patients.csv', index_col=0)

# Collapse sub-stages (e.g., IIA, IIB) into broad integers 0-3
stage_mapping = {
    'Stage I': 0, 'Stage IA': 0, 'Stage IB': 0,
    'Stage II': 1, 'Stage IIA': 1, 'Stage IIB': 1,
    'Stage III': 2, 'Stage IIIA': 2, 'Stage IIIB': 2, 'Stage IIIC': 2,
    'Stage IV': 3,
}
merged['stage_int'] = merged['ajcc_pathologic_stage'].map(stage_mapping)
merged_clean = merged.dropna(subset=['stage_int']).copy()

y = merged_clean['stage_int'].values.astype(int)
X = merged_clean.drop(columns=['ajcc_pathologic_stage', 'stage_int'])

# Slice by known column counts from the original merging process
rna_cols = X.columns[0:60660].tolist()
mirna_cols = X.columns[60660:].tolist()

# Log transform to compress dynamic range of expression data
X_rna_log = np.log2(X[rna_cols].astype(float).values + 1)
X_mirna_log = np.log2(X[mirna_cols].astype(float).values + 1)

# Select top features based on ANOVA F-value
selector_rna = SelectKBest(score_func=f_classif, k=5000)
X_rna_scaled = StandardScaler().fit_transform(selector_rna.fit_transform(X_rna_log, y))

selector_mirna = SelectKBest(score_func=f_classif, k=500)
X_mirna_scaled = StandardScaler().fit_transform(selector_mirna.fit_transform(X_mirna_log, y))

X_combined = np.hstack([X_rna_scaled, X_mirna_scaled])
np.save(BASE / 'X_combined.npy', X_combined)
np.save(BASE / 'y_labels.npy', y)

# Save labels with patient IDs for the clinical preparation step
pd.DataFrame({'patient_id': merged_clean.index, 'stage': y}).to_csv(BASE / 'patient_labels.csv', index=False)