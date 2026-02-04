import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from pathlib import Path
from model import CrossOmicsAttentionNet

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and align clinical features
X_omics = np.load(Path('X_combined.npy'))
y = np.load(Path('y_labels.npy'))
patient_ids = pd.read_csv(Path('patient_labels.csv'))['patient_id'].values

clin_df = pd.read_csv(Path('BRCA_complete_clinical.csv'), index_col='submitter_id').reindex(patient_ids)
X_clin_raw = clin_df[['age_at_diagnosis', 'gender', 'race', 'ethnicity']].copy()
X_clin_raw['age_at_diagnosis'] = X_clin_raw['age_at_diagnosis'].fillna(X_clin_raw['age_at_diagnosis'].median())
X_clin_encoded = pd.get_dummies(X_clin_raw)

# Normalize clinical data and force 12-dim padding/slicing for model compatibility
X_clin = StandardScaler().fit_transform(X_clin_encoded)
if X_clin.shape[1] < 12:
    X_clin = np.hstack([X_clin, np.zeros((X_clin.shape[0], 12 - X_clin.shape[1]))])
else:
    X_clin = X_clin[:, :12]

X_final = np.hstack([X_clin, X_omics])
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y)

# Training setup with boosted weights for the rare Stage IV class
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights[3] *= 2.0 

model = CrossOmicsAttentionNet(rna_dim=5000, mirna_dim=500).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

best_acc = 0.0
for epoch in range(100):
    model.train()
    for X_batch, y_batch in DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=32, shuffle=True):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_b, y_b in DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=32):
            logits = model(X_b.to(DEVICE))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(y_b.numpy())
    
    acc = balanced_accuracy_score(targets, preds)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1}: New Best Acc {acc:.4f}")

# Final Evaluation
model.load_state_dict(torch.load('best_model.pth'))
print(classification_report(y_test, preds, target_names=['Stage I', 'Stage II', 'Stage III', 'Stage IV']))