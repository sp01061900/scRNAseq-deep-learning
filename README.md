# scRNA-seq Deep Learning Comparison


# scRNA-seq Deep Learning Comparison

This project benchmarks **deep learning and classical methods** for single-cell RNA sequencing (scRNA-seq) analysis.  
We compare Autoencoder (AE), Variational Autoencoder (VAE), Transformer, and PCA embeddings on the PBMC 3K dataset.

---

## 📂 Repository Structure
scRNAseq-deep-learning/
│── data/                  # Raw and preprocessed datasets
│── src/                   # Training scripts for each model
│   ├── train_ae.py
│   ├── train_vae.py
│   ├── train_transformer.py
│   └── train_pca.py
│── results/               # Latent embeddings, annotated files, plots, metrics.csv
│── notebooks/             # Jupyter notebooks for comparison and visualization
│   └── comparison.ipynb
│── docs/                  # Documentation
│   ├── HOW_TO_RUN.md
│   └── PROJECT_DETAILS.md
│── requirements.txt       # Dependencies
│── README.md              # Project overview


---

## 🚀 Features
- Preprocessing pipeline for scRNA-seq (normalization, log-transform, HVG selection).
- Deep learning models:
  - Autoencoder (AE)
  - Variational Autoencoder (VAE)
  - Transformer encoder
- Classical baseline: PCA
- Downstream clustering with Leiden + UMAP visualization.
- Metrics: Silhouette, Davies–Bouldin, Calinski–Harabasz.
- Reproducible workflow with Snakemake.

---

## 📊 Results
- **AE**: Clear cluster separation, strong silhouette score.  
- **VAE**: Latent collapse (single cluster, metrics NA).  
- **Transformer**: Moderate separation, overlapping clusters.  
- **PCA**: Good baseline separation, competitive silhouette score.  


## 🧾 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Download PBMC 3K dataset into data/.

3. Run Prepocessing
python src/preprocessing.py --input data/pbmc3k.h5ad --output data/preprocessed.h5ad

4. Train Models
python src/train_ae.py --data data/preprocessed.h5ad --out results/latent_ae.h5ad
python src/train_vae.py --data data/preprocessed.h5ad --out results/latent_vae.h5ad
python src/train_transformer.py --data data/preprocessed.h5ad --out results/latent_transformer.h5ad
python src/train_pca.py --data data/preprocessed.h5ad --out results/latent_pca.h5ad

5. Visualize and compute metrics:
python src/visualize.py --data results/latent_ae.h5ad
python src/visualize.py --data results/latent_vae.h5ad
python src/visualize.py --data results/latent_transformer.h5ad
python src/visualize.py --data results/latent_pca.h5ad

6. Explore results in notebook:
jupyter notebook notebooks/comparison.ipynb

7. Learnings

AE and PCA embeddings preserve biological heterogeneity best.

VAE requires tuning (KL weight, latent dimension) to avoid collapse.

Transformer captures global structure but needs more training for fine detail.

Classical PCA remains a strong baseline for scRNA-seq clustering.
