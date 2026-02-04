# Cross-Omics Attention Network for BRCA Staging

A Multi-Modal Deep Learning framework designed to predict **AJCC Pathological Stages (I–IV)** for Breast Invasive Carcinoma (BRCA) by integrating clinical metadata with high-dimensional transcriptomic and microRNA data.



## 🌟 Project Highlights
* **Multi-Modal Fusion:** Dynamically integrates three distinct data streams: Clinical Metadata, mRNA Expression (5,000 features), and miRNA Expression (500 features).
* **Transformer-Based Attention:** Implements a **Cross-Attention mechanism** where clinical features act as the "Query" to extract relevant signals from genomic "Keys/Values."
* **Rare Class Detection:** Utilizes **Strategic Boosted Weighting** to achieve detection in rare Stage IV cases—a population often "ignored" by standard models due to extreme imbalance.
* **Robust Performance:** Achieves a **Weighted AUC of 0.75** and **65% Accuracy** on a complex 4-class biological classification task.

---

## 🏗️ Architecture
The core model, `CrossOmicsAttentionNet`, utilizes separate encoding branches for each modality. These embeddings are fused through a multi-head attention layer, allowing the model to learn interactions between a patient's clinical profile and their molecular signature rather than simply concatenating raw features.



---

## 📈 Evolution of the Model (Experimental Log)

The project navigated through five major iterations to overcome the "Curse of Dimensionality" and severe class imbalance.

| Iteration | Strategy | Accuracy | AUC | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Baseline | 67% | N/A | **Overfit:** Memorized majority class; 0% recall on Stage IV. |
| 2 | Regularization | 36% | N/A | **Underfit:** Constraints (Dropout 0.5) were too aggressive. |
| 3 | Auto-Weights | 60% | 0.69 | **Stable:** Mathematically fair, but still missed rare signals. |
| 4 | SMOTE | 35% | 0.62 | **Failed:** Synthetic noise in 5k dimensions destroyed majority recall. |
| **5** | **Boosted Weights** | **65%** | **0.75** | **CHAMPION:** Optimal balance of overall accuracy and minority safety. |

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.9+
* PyTorch
* Scikit-Learn
* Imbalanced-Learn (`pip install imbalanced-learn`)
* Pandas / NumPy
* Matplotlib

### Execution Order
To reproduce the final results, run the scripts in the following order:

1.  **`data_processing.py`**: Aligns clinical and genomic datasets by unique patient barcodes.
2.  **`preprocess.py`**: Conducts Log2 transformation, scaling, and `SelectKBest` (F-classif) feature selection (5,000 genes).
3.  **`train.py`**: Executes the training loop with **Boosted Weighting** and early stopping.

---

## 📊 Final Results Summary
The champion model demonstrates strong predictive power across the entire disease progression spectrum:

* **Stage I (Early):** 53% Recall
* **Stage II (Mid):** 71% Recall
* **Stage III (Advanced):** 63% Recall
* **Stage IV (Critical):** 25% Recall (Successfully identifies high-risk outliers)



---

## 🚀 Future Roadmap
* **External Validation:** Testing the model on the **METABRIC** dataset to evaluate cross-platform generalizability (Microarray vs. RNA-Seq).
* **Model Interpretability:** Integrating **SHAP (SHapley Additive exPlanations)** to identify specific gene sets driving Stage IV predictions.
* **Late Fusion Ensemble:** Exploring expert mixtures to handle technological differences between disparate genomic datasets.
