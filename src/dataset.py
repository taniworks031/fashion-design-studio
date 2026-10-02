import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FashionMNISTIdxDataset(Dataset):

  def __init__(self, data_dir, train=True):
    img_file = (
        "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    )
    lbl_file = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"

    with open(os.path.join(data_dir, img_file), "rb") as f:
      self.images = (
          np.frombuffer(f.read(), np.uint8, offset=16)
          .reshape(-1, 1, 28, 28)
          .astype(np.float32)
          / 255.0
      )

    with open(os.path.join(data_dir, lbl_file), "rb") as f:
      self.labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(
          np.int64
      )

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])