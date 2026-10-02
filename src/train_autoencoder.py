import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import FashionMNISTIdxDataset
from networks import ConvAutoencoder
from torch.utils.data import DataLoader


def train_denoising_autoencoder():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Robust path resolution to locate the 'data' folder from anywhere
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "..", "data")

  # Load data using custom parser
  train_dataset = FashionMNISTIdxDataset(data_dir, train=True)
  test_dataset = FashionMNISTIdxDataset(data_dir, train=False)

  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

  # Initialize model, loss, and optimizer
  model = ConvAutoencoder().to(device)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 50
  print("Starting Denoising Autoencoder training...")
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
      images = images.to(device)

      # Add random Gaussian noise to inputs for Denoising AE component
      noise = 0.4 * torch.randn_like(images)
      noisy_images = torch.clamp(images + noise, 0.0, 1.0)

      optimizer.zero_grad()
      outputs = model(noisy_images)  # Output tries to reconstruct CLEAN images
      loss = criterion(outputs, images)  # Compare output with clean original
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss:"
        f" {running_loss/len(train_loader):.4f}"
    )

  # Calculate reconstruction error threshold on test set for anomaly detection ("not clothing")
  model.eval()
  mse_losses = []
  with torch.no_grad():
    for images, _ in test_loader:
      images = images.to(device)
      outputs = model(images)
      # Compute per-sample MSE
      batch_losses = torch.mean((outputs - images) ** 2, dim=[1, 2, 3])
      mse_losses.extend(batch_losses.cpu().numpy())

  # Set threshold at Mean + 2 * Standard Deviation of test reconstruction errors
  threshold = np.mean(mse_losses) + 2 * np.std(mse_losses)
  print(f"Computed Anomaly Detection MSE Threshold: {threshold:.5f}")

  # Save trained weights using robust path resolution
  models_dir = os.path.join(current_dir, "..", "models")
  os.makedirs(models_dir, exist_ok=True)
  torch.save(
      {"state_dict": model.state_dict(), "threshold": threshold},
      os.path.join(models_dir, "denoising_autoencoder.pt"),
  )
  print(
      "Denoising Autoencoder saved successfully to"
      " models/denoising_autoencoder.pt!"
  )


if __name__ == "__main__":
  train_denoising_autoencoder()