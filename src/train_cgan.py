import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import FashionMNISTIdxDataset
from networks import ConditionalDiscriminator, ConditionalGenerator
from torch.utils.data import DataLoader


def train_cgan():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Load data using robust path resolution
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "..", "data")

  train_dataset = FashionMNISTIdxDataset(data_dir, train=True)
  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

  # Hyperparameters
  latent_dim = 100
  num_classes = 10
  lr = 0.0002
  b1, b2 = 0.5, 0.999
  epochs = 50  # 40 epochs for diverse non-collapsed generation

  # Initialize Generator and Discriminator
  generator = ConditionalGenerator(latent_dim, num_classes).to(device)
  discriminator = ConditionalDiscriminator(num_classes).to(device)

  # Optimizers and Loss
  optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
  optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
  adversarial_loss = nn.BCELoss()

  print("Starting Conditional GAN (cGAN) training...")
  for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
      batch_size = imgs.size(0)
      imgs, labels = imgs.to(device), labels.to(device)

      # Ground truths for BCE loss (with label smoothing for stability)
      valid = torch.ones(batch_size, 1, device=device) * 0.9
      fake = torch.zeros(batch_size, 1, device=device) + 0.1

      # ---------------------
      #   Train Discriminator
      # ---------------------
      optimizer_D.zero_grad()

      # Loss for real images
      validity_real = discriminator(imgs, labels)
      d_real_loss = adversarial_loss(validity_real, valid)

      # Sample noise and labels for generator input
      z = torch.randn(batch_size, latent_dim, device=device)
      gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)

      # Generate fake images
      gen_imgs = generator(z, gen_labels)

      # Loss for fake images
      validity_fake = discriminator(gen_imgs.detach(), gen_labels)
      d_fake_loss = adversarial_loss(validity_fake, fake)

      # Total Discriminator loss
      d_loss = (d_real_loss + d_fake_loss) / 2
      d_loss.backward()
      optimizer_D.step()

      # -----------------
      #   Train Generator
      # -----------------
      optimizer_G.zero_grad()

      # Train generator to fool the discriminator
      validity_gen = discriminator(gen_imgs, gen_labels)
      g_loss = adversarial_loss(validity_gen, valid)

      g_loss.backward()
      optimizer_G.step()

    print(
        f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss:"
        f" {g_loss.item():.4f}"
    )

  # Save cGAN generator weights
  models_dir = os.path.join(current_dir, "..", "models")
  os.makedirs(models_dir, exist_ok=True)
  torch.save(
      generator.state_dict(), os.path.join(models_dir, "cgan_generator.pt")
  )
  print(
      "Conditional GAN Generator saved successfully to models/cgan_generator.pt!"
  )


if __name__ == "__main__":
  train_cgan()