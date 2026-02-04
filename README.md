# 🧬 BRCA Multi-Omics Stage Predictor

A high-performance **Cross-Modal Attention Network** designed to predict breast cancer (BRCA) pathologic stages. This project integrates three distinct data modalities: **mRNA (Transcriptomics)**, **miRNA (Epigenomics)**, and **Clinical Metadata**.

---

## 🛠️ The Data Engineering Stack

| Tool | Purpose |
| :--- | :--- |
| **Docker** | Ensures the environment (Python, PyTorch, PySpark) is identical across all machines. |
| **Apache Spark** | Handles the heavy lifting of transposing and joining high-dimensional genomic datasets. |
| **Airflow** | Orchestrates the sequence from raw TCGA downloads to final model evaluation. |

---

## 📂 Repository Structure

* `data_processing.py`: Extracts TCGA patient IDs and aligns clinical/molecular files via inner joins.
* `preprocess.py`: Performs $Log_2$ transformation and **SelectKBest** feature selection (Top 5000 mRNA, 500 miRNA).
* `model.py`: The core **CrossOmicsAttentionNet** architecture using PyTorch Multihead Attention.
* `train.py`: Training script with **Boosted Class Weights** to handle imbalanced cancer stages.
* `prepare_pytorch_data.py`: Handles final tensor conversion and stratified train/test splitting.

---

## 🧠 Model Architecture: Cross-Modal Attention

The model utilizes **Clinical Metadata** as a **Query** to attend to **Genomic Features** (Key/Value). This allows the network to weigh specific gene expressions differently based on the patient's demographic profile.

### Key Mathematical Components:
* **Feature Scaling:** $Z = \frac{x - \mu}{\sigma}$ (StandardScaler)
* **Dimensionality Reduction:** ANOVA F-value based selection ($k=5000$).
* **Loss Function:** Weighted Cross-Entropy to prioritize rare, high-severity cancer stages:
    $$Loss = -\sum_{c=1}^{C} w_c \cdot y_c \log(\hat{y}_c)$$

---

## 🚀 Getting Started

### 1. Environment Setup (Docker)
To avoid dependency issues with PySpark and PyTorch, build and run the container:
```bash
docker build -t brca-omics-pipeline .
docker run -it brca-omics-pipeline
