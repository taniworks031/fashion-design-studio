import torch
import torch.nn as nn


# 1. Standard CNN Classifier (for scoring generated images & accuracy check)
class FashionCNNClassifier(nn.Module):

  def __init__(self):
    super(FashionCNNClassifier, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Output: 32 x 14 x 14
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Output: 64 x 7 x 7
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10),  # 10 clothing classes
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


# 2. Convolutional Autoencoder & Denoising AE
class ConvAutoencoder(nn.Module):

  def __init__(self):
    super(ConvAutoencoder, self).__init__()
    # Encoder
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> 16 x 14 x 14
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32 x 7 x 7
        nn.ReLU(),
    )
    # Decoder
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(
            16, 1, 3, stride=2, padding=1, output_padding=1
        ),  # -> 1 x 28 x 28
        nn.Sigmoid(),
    )

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


# 3. Variational Autoencoder (VAE) with Reparameterization Trick
class VAE(nn.Module):

  def __init__(self, latent_dim=20):
    super(VAE, self).__init__()
    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
    )
    self.fc_mu = nn.Linear(128, latent_dim)
    self.fc_logvar = nn.Linear(128, latent_dim)

    self.decoder_fc = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 28 * 28),
        nn.Sigmoid(),
    )

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x):
    h = self.encoder(x)
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    z = self.reparameterize(mu, logvar)
    recon_x = self.decoder_fc(z).view(-1, 1, 28, 28)
    return recon_x, mu, logvar


# 4. Conditional GAN (cGAN) Generator & Discriminator
class ConditionalGenerator(nn.Module):

  def __init__(self, latent_dim=100, num_classes=10):
    super(ConditionalGenerator, self).__init__()
    self.label_emb = nn.Embedding(num_classes, num_classes)

    self.model = nn.Sequential(
        nn.Linear(latent_dim + num_classes, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 512),
        nn.BatchNorm2d(512) if False else nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, 28 * 28),
        nn.Tanh(),  # Output scaled to [-1, 1]
    )

  def forward(self, noise, labels):
    # Concatenate noise vector and class embedding
    gen_input = torch.cat((noise, self.label_emb(labels)), -1)
    img = self.model(gen_input)
    return img.view(img.size(0), 1, 28, 28)


class ConditionalDiscriminator(nn.Module):

  def __init__(self, num_classes=10):
    super(ConditionalDiscriminator, self).__init__()
    self.label_emb = nn.Embedding(num_classes, num_classes)

    self.model = nn.Sequential(
        nn.Linear(28 * 28 + num_classes, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

  def forward(self, img, labels):
    img_flat = img.view(img.size(0), -1)
    d_input = torch.cat((img_flat, self.label_emb(labels)), -1)
    validity = self.model(d_input)
    return validity