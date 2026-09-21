import scanpy as sc
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import csv

def choose_latent_key(adata):
    for key in ["X_latent_transformer", "X_latent_vae", "X_latent_ae", "X_latent"]:
        if key in adata.obsm.keys():
            return key
    if "X_pca" in adata.obsm.keys():
        return "X_pca"
    raise ValueError("No latent representation found in adata.obsm")

def compute_metrics(adata, latent_key, method):
    ari, nmi, sil = "NA", "NA", "NA"

    # Always try silhouette, but only if >1 cluster
    labels = adata.obs["leiden"]
    if len(set(labels)) > 1:
        try:
            sil = silhouette_score(adata.obsm[latent_key], labels)
        except Exception as e:
            print(f"Silhouette failed: {e}")

    # Compute ARI/NMI only if ground truth labels exist
    if "true_labels" in adata.obs:
        ari = adjusted_rand_score(adata.obs["true_labels"], labels)
        nmi = normalized_mutual_info_score(adata.obs["true_labels"], labels)

    print(f"Metrics for {method}: ARI={ari}, NMI={nmi}, Silhouette={sil}")

    # Append metrics to CSV
    with open("results/metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([method, ari, nmi, sil])

def visualize(input_path, output_dir="results/"):
    adata = sc.read_h5ad(input_path)
    latent_key = choose_latent_key(adata)
    method = latent_key.replace("X_latent_", "").replace("X_", "")
    print(f"Using latent representation: {latent_key}")

    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.umap(adata)

    # Save plot
    ax = sc.pl.umap(adata, color=["leiden"], show=False, title=f"UMAP - {method}")
    fig = ax.figure
    fig.savefig(f"{output_dir}/umap_{method}.png", dpi=150)
    plt.close(fig)

    # Save annotated dataset
    adata.write(f"{output_dir}/annotated_{method}.h5ad")

    # Compute metrics
    compute_metrics(adata, latent_key, method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to latent h5ad file")
    args = parser.parse_args()
    visualize(args.data)

