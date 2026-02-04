import re
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

def extract_patient_id(sample_id: str) -> str:
    if pd.isna(sample_id):
        return sample_id
    s = str(sample_id)
    # Extracts the TCGA-XX-XXXX format
    m = re.search(r'(TCGA-[A-Za-z0-9]+-[A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    # Fallback to first 12 chars if regex fails
    if len(s) >= 12 and s.startswith('TCGA'):
        return s[:12]
    return s

clinical_path = BASE / 'BRCA_complete_clinical.csv'
clinical = pd.read_csv(clinical_path, dtype=str)
clinical['patient_id'] = clinical['submitter_id'].astype(str)
clinical = clinical.set_index('patient_id')

mrna_path = BASE / 'BRCA_RNA_raw_fpkm.csv'
mrna = pd.read_csv(mrna_path, dtype=str)
gene_col = mrna.columns[0]
mrna = mrna.set_index(gene_col)
mrna = mrna.apply(pd.to_numeric, errors='coerce')
mrna_t = mrna.T.reset_index()
mrna_t = mrna_t.rename(columns={'index': 'sample_id'})
mrna_t['patient_id'] = mrna_t['sample_id'].apply(extract_patient_id)
mrna_t = mrna_t.set_index('patient_id')

if 'sample_id' in mrna_t.columns:
    mrna_t = mrna_t.drop(columns=['sample_id'])

# Average samples if one patient has multiple vials/entries
if not mrna_t.index.is_unique:
    mrna_t = mrna_t.groupby(level=0).mean()

# miRNA filename search
mirna_fnames = ['TCGA_BRCA_miRNA_by_sample_RPM.csv', 'TCGA_BRCA_miRNA.csv']
mirna_path = None
for fn in mirna_fnames:
    p = BASE / fn
    if p.exists():
        mirna_path = p
        break
if mirna_path is None:
    raise FileNotFoundError('miRNA file not found.')

mirna = pd.read_csv(mirna_path, dtype=str, index_col=0)
mirna = mirna.apply(pd.to_numeric, errors='coerce')
mirna_t = mirna.T.reset_index()
mirna_t = mirna_t.rename(columns={'index': 'sample_id'})
mirna_t['patient_id'] = mirna_t['sample_id'].apply(extract_patient_id)
mirna_t = mirna_t.set_index('patient_id')

if 'sample_id' in mirna_t.columns:
    mirna_t = mirna_t.drop(columns=['sample_id'])

if not mirna_t.index.is_unique:
    mirna_t = mirna_t.groupby(level=0).mean()

# Intersection alignment
clinical_patients = set(clinical.index.dropna())
mrna_patients = set(mrna_t.index.dropna())
mirna_patients = set(mirna_t.index.dropna())
common = sorted(clinical_patients & mrna_patients & mirna_patients)

clinical_sub = clinical.loc[common]
mrna_sub = mrna_t.loc[common]
mirna_sub = mirna_t.loc[common]

# Merging pathologic stage with molecular features
merged = pd.concat([clinical_sub[['ajcc_pathologic_stage']].copy(), mrna_sub, mirna_sub], axis=1)
merged.to_csv(BASE / 'merged_patients.csv')