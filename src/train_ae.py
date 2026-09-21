import argparse
import scanpy as sc
import torch
import torch.nn as nn

# Simple Autoencoder
class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def main(args):
    adata = sc.read_h5ad(args.data)
    X = torch.tensor(adata.X.toarray(), dtype=torch.float32)

    model = AE(X.shape[1], latent_dim=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(20):
        optimizer.zero_grad()
        x_hat, z = model(X)
        loss = loss_fn(x_hat, X)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss={loss.item():.4f}")

    # Save latent representation
    with torch.no_grad():
        _, latent = model(X)
    adata.obsm["X_latent_ae"] = latent.numpy()
    adata.write(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    main(args)

