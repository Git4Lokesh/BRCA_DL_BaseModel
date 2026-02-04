import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE = Path(__file__).resolve().parent

X_rna_scaled = np.load(BASE / 'X_rna_scaled.npy')
X_mirna_scaled = np.load(BASE / 'X_mirna_scaled.npy')
y = np.load(BASE / 'y_labels.npy')

patient_df = pd.read_csv(BASE / 'patient_labels.csv')
clinical_raw = pd.read_csv(BASE / 'BRCA_complete_clinical.csv').set_index('submitter_id')
clinical_subset = clinical_raw.loc[patient_df['patient_id'].values].copy()

# Filter for standard clinical baseline features
clinical_cols = ['age_at_diagnosis', 'gender', 'race', 'ethnicity']
X_clinical = clinical_subset[clinical_cols].copy()

X_clinical['age_at_diagnosis'] = X_clinical['age_at_diagnosis'].fillna(X_clinical['age_at_diagnosis'].median())
for col in ['gender', 'race', 'ethnicity']:
    X_clinical[col] = X_clinical[col].fillna('Unknown')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age_at_diagnosis']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['gender', 'race', 'ethnicity'])
    ]
)

X_clin_transformed = preprocessor.fit_transform(X_clinical)
X_combined = np.hstack([X_rna_scaled, X_mirna_scaled, X_clin_transformed])

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, stratify=y, random_state=42
)

# Calculate class weights: total_samples / (n_classes * count_per_class)
class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
class_weights_normalized = class_weights / class_weights.min()

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Save for training script
with open(BASE / 'data_loaders.pkl', 'wb') as f:
    pickle.dump({
        'train_loader': train_loader,
        'test_loader': test_loader,
        'class_weights': class_weights_normalized
    }, f)