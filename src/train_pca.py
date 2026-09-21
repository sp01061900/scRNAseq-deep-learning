import argparse
import scanpy as sc

def main(args):
    adata = sc.read_h5ad(args.data)
    sc.pp.pca(adata, n_comps=50)
    adata.obsm["X_pca"] = adata.obsm["X_pca"]
    adata.write(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    main(args)

