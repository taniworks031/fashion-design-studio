import os
import torch
from dataset import FashionMNISTIdxDataset
from networks import (
    ConditionalGenerator,
    ConvAutoencoder,
    FashionCNNClassifier,
    VAE,
)
from torch.utils.data import DataLoader


def evaluate_cgan_and_autoencoder():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"\n--- Evaluating Models on {device} ---")

  current_dir = os.path.dirname(os.path.abspath(__file__))
  models_dir = os.path.join(current_dir, "..", "models")

  # 1. Load trained classifier
  classifier = FashionCNNClassifier().to(device)
  classifier.load_state_dict(
      torch.load(
          os.path.join(models_dir, "classifier.pt"), map_location=device, weights_only=True
      )
  )
  classifier.eval()

  # 2. Load trained cGAN generator
  latent_dim = 100
  num_classes = 10
  cgan_gen = ConditionalGenerator(latent_dim, num_classes).to(device)
  cgan_gen.load_state_dict(
      torch.load(
          os.path.join(models_dir, "cgan_generator.pt"), map_location=device, weights_only=True
      )
  )
  cgan_gen.eval()

  # Test cGAN class-conditional accuracy (Requirement: >= 80%)
  print("Testing cGAN generation accuracy against classifier...")
  num_samples = 500
  correct_predictions = 0

  with torch.no_grad():
    for target_class in range(num_classes):
      # Generate noise and target labels for a specific class
      z = torch.randn(num_samples // num_classes, latent_dim, device=device)
      labels = torch.full(
          (num_samples // num_classes,),
          target_class,
          dtype=torch.long,
          device=device,
      )

      gen_imgs = cgan_gen(z, labels)
      # Pass generated images through classifier
      outputs = classifier(gen_imgs)
      _, predicted = torch.max(outputs, 1)

      correct_predictions += (predicted == labels).sum().item()

  cgan_accuracy = (correct_predictions / num_samples) * 100
  print(
      f"cGAN Conditional Generation Accuracy: {cgan_accuracy:.2f}% (Target:"
      " >= 80%)"
  )

  # 3. Test Denoising Autoencoder Anomaly Detection Threshold
  checkpoint = torch.load(
      os.path.join(models_dir, "denoising_autoencoder.pt"),
      map_location=device,
      weights_only=False,
  )
  print(
      "Denoising Autoencoder Anomaly Threshold loaded successfully:"
      f" {checkpoint['threshold']:.5f}"
  )
  print("Evaluation completed successfully!")


if __name__ == "__main__":
  evaluate_cgan_and_autoencoder()