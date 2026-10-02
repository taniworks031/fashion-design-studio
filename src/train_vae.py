import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import FashionMNISTIdxDataset
from networks import VAE
from torch.utils.data import DataLoader


# Loss function combining Reconstruction Loss (BCE/MSE) and KL Divergence
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
  # Binary Cross Entropy reconstruction loss
  BCE = nn.functional.binary_cross_entropy(
      recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction="sum"
  )
  # KL Divergence term: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return BCE + beta * KLD


def train_vae():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Robust path resolution to locate the 'data' folder from anywhere
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "..", "data")

  # Load data using custom parser
  train_dataset = FashionMNISTIdxDataset(data_dir, train=True)
  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

  # Initialize VAE with a 20-dimensional latent space
  latent_dim = 20
  model = VAE(latent_dim=latent_dim).to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 50
  beta = 1.0  # You can experiment with beta=0.5 or 4.0 later for beta-VAE

  print("Starting Variational Autoencoder (VAE) training...")
  for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, _ in train_loader:
      images = images.to(device)

      optimizer.zero_grad()
      recon_images, mu, logvar = model(images)
      loss = vae_loss_function(recon_images, images, mu, logvar, beta=beta)

      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss:"
        f" {train_loss / len(train_loader.dataset):.4f}"
    )

  # Save VAE weights using robust path resolution
  models_dir = os.path.join(current_dir, "..", "models")
  os.makedirs(models_dir, exist_ok=True)
  torch.save(model.state_dict(), os.path.join(models_dir, "vae.pt"))
  print("VAE saved successfully to models/vae.pt!")


if __name__ == "__main__":
  train_vae()