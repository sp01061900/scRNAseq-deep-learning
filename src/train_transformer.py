import argparse
import scanpy as sc
import torch
import torch.nn as nn

# Simple Transformer Encoder for scRNA-seq
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Add sequence dimension (seq_len=1)
        x = self.embedding(x).unsqueeze(1)
        z = self.transformer(x)
        return z.squeeze(1)

def main(args):
    adata = sc.read_h5ad(args.data)
    X = torch.tensor(adata.X.toarray(), dtype=torch.float32)

    model = TransformerEncoder(X.shape[1], latent_dim=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop (identity reconstruction)
    for epoch in range(20):
        optimizer.zero_grad()
        z = model(X)
        # Decode back to input space
        x_hat = torch.matmul(z, model.embedding.weight)  # simple linear decode
        loss = loss_fn(x_hat, X)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss={loss.item():.4f}")

    # Save latent representation
    with torch.no_grad():
        latent = model(X)
    adata.obsm["X_latent_transformer"] = latent.numpy()
    adata.write(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    main(args)

