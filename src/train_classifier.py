import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FashionMNISTIdxDataset
from networks import FashionCNNClassifier


def train_classifier():
  # Set device (GPU if available, else CPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Robust path resolution to locate the 'data' folder from anywhere
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "..", "data")

  # Load local datasets using your custom parser
  train_dataset = FashionMNISTIdxDataset(data_dir, train=True)
  test_dataset = FashionMNISTIdxDataset(data_dir, train=False)

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  # Initialize model, loss function, and optimizer
  model = FashionCNNClassifier().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  epochs = 50
  print("Starting CNN Classifier training...")
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss:"
        f" {running_loss/len(train_loader):.4f}"
    )

  # Evaluate on test set
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f"Test Accuracy of CNN Classifier: {accuracy:.2f}%")

  # Save trained weights using robust path resolution
  models_dir = os.path.join(current_dir, "..", "models")
  os.makedirs(models_dir, exist_ok=True)
  torch.save(model.state_dict(), os.path.join(models_dir, "classifier.pt"))
  print("Classifier saved successfully to models/classifier.pt!")


if __name__ == "__main__":
  train_classifier()