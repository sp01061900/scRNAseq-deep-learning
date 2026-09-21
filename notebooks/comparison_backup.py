import os
print("metrics.csv exists:", os.path.exists("results/metrics.csv"))
print("annotated files:", [f for f in os.listdir("results") if f.startswith("annotated_")])


import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("results/metrics.csv", header=None,
                 names=["method", "ARI", "NMI", "Silhouette"])

print(df)

# Plot silhouette scores
plt.figure(figsize=(6,4))
plt.bar(df["method"], pd.to_numeric(df["Silhouette"], errors="coerce"))
plt.ylabel("Silhouette Score")
plt.title("Comparison of Methods")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import scanpy as sc

methods = ["ae", "vae", "transformer", "pca"]
scores = []

for m in methods:
    adata = sc.read_h5ad(f"results/annotated_{m}.h5ad")
    latent_key = [k for k in adata.obsm.keys() if m in k][0]
    X = adata.obsm[latent_key]
    labels = adata.obs["leiden"]

    # Compute metrics only if >1 cluster
    if len(set(labels)) > 1:
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    else:
        db, ch = "NA", "NA"

    scores.append([m, db, ch])

df_scores = pd.DataFrame(scores, columns=["method", "Davies-Bouldin", "Calinski-Harabasz"])
print(df_scores)

