import argparse
import scanpy as sc
import torch
import torch.nn as nn

# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        # Encoder: outputs mean and log variance
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    recon_loss = nn.MSELoss()(x_hat, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def main(args):
    adata = sc.read_h5ad(args.data)
    X = torch.tensor(adata.X.toarray(), dtype=torch.float32)

    model = VAE(X.shape[1], latent_dim=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(20):
        optimizer.zero_grad()
        x_hat, z, mu, logvar = model(X)
        loss = vae_loss(x_hat, X, mu, logvar)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss={loss.item():.4f}")

    # Save latent representation
    with torch.no_grad():
        _, latent, _, _ = model(X)
    adata.obsm["X_latent_vae"] = latent.numpy()
    adata.write(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    main(args)

