import io
import os
import sys

# Ensure Python can find the 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from networks import ConditionalGenerator, ConvAutoencoder, VAE

# App Config
st.set_page_config(
    page_title="Generative Fashion Design Studio", layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clothing Classes Mapping for Fashion-MNIST
CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


@st.cache_resource
def load_models():
  # Load cGAN Generator
  cgan = ConditionalGenerator(100, 10).to(device)
  cgan.load_state_dict(
      torch.load(
          os.path.join(current_dir, "models", "cgan_generator.pt"),
          map_location=device,
          weights_only=True,
      )
  )
  cgan.eval()

  # Load VAE Model
  vae = VAE(latent_dim=20).to(device)
  vae.load_state_dict(
      torch.load(
          os.path.join(current_dir, "models", "vae.pt"),
          map_location=device,
          weights_only=True,
      )
  )
  vae.eval()

  # Load Denoising Autoencoder & Threshold
  denoise_ckpt = torch.load(
      os.path.join(current_dir, "models", "denoising_autoencoder.pt"),
      map_location=device,
      weights_only=False,
  )
  denoiser = ConvAutoencoder().to(device)
  denoiser.load_state_dict(denoise_ckpt["state_dict"])
  denoiser.eval()
  threshold = denoise_ckpt["threshold"]

  return cgan, vae, denoiser, threshold


cgan_gen, vae_model, denoise_model, anomaly_threshold = load_models()

# Sidebar Navigation
st.sidebar.title("🎨 Fashion Design Studio")
app_mode = st.sidebar.selectbox(
    "Choose Studio Mode",
    [
        "1. cGAN Grid Generator",
        "2. VAE Latent Morphing",
        "3. Denoising & Anomaly Detector",
    ],
)

# ----------------------------------------------------
# TAB 1: cGAN Grid Generator
# ----------------------------------------------------
if app_mode == "1. cGAN Grid Generator":
  st.title("✨ AI Fashion Sketch Generator (Conditional GAN)")
  st.write(
      "Select a clothing category and generate a custom grid of new AI-proposed"
      " clothing designs."
  )

  selected_class_name = st.selectbox(
      "Select Clothing Category", list(CLASSES.values())
  )
  class_id = [
      k for k, v in CLASSES.items() if v == selected_class_name
  ][0]

  grid_size = st.slider(
      "Number of designs to generate", min_value=1, max_value=16, value=8
  )

  if st.button("Generate Designs"):
    with torch.no_grad():
      z = torch.randn(grid_size, 100, device=device)
      labels = torch.tensor(
          [class_id] * grid_size, dtype=torch.long, device=device
      )
      fake_imgs = cgan_gen(z, labels).cpu().numpy()

    # Plot grid
    cols = 4
    rows = (grid_size + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2))
    axes = axes.flatten() if grid_size > 1 else [axes]

    for i in range(grid_size):
      # Clean Rescaling from Tanh range [-1, 1] to [0, 1] without harsh over-saturation
      img = (fake_imgs[i].squeeze() + 1.0) / 2.0
      img = np.clip(img, 0.0, 1.0)

      axes[i].imshow(img, cmap="gray")
      axes[i].axis("off")
      axes[i].set_title(selected_class_name)

    for j in range(grid_size, len(axes)):
      axes[j].axis("off")

    st.pyplot(fig)

# ----------------------------------------------------
# TAB 2: VAE Latent Morphing
# ----------------------------------------------------
elif app_mode == "2. VAE Latent Morphing":
  st.title("🎛️ Latent Space Design Morphing (VAE)")
  st.write(
      "Drag the latent space sliders to smoothly morph clothing features and"
      " explore the VAE manifold."
  )

  col1, col2 = st.columns(2)
  with col1:
    z1_slider = st.slider(
        "Latent Dimension 0", -3.0, 3.0, 0.0, key="z0_slider"
    )
  with col2:
    z2_slider = st.slider(
        "Latent Dimension 1", -3.0, 3.0, 0.0, key="z1_slider"
    )

  with torch.no_grad():
    # Base latent vector of size 20 filled with zeros
    z = torch.zeros(1, 20, device=device)
    z[0, 0] = z1_slider
    z[0, 1] = z2_slider
    decoded_img = vae_model.decoder_fc(z).view(28, 28).cpu().numpy()

  st.subheader("Morphed Clothing Output")
  fig, ax = plt.subplots(figsize=(3, 3))
  ax.imshow(decoded_img, cmap="gray")
  ax.axis("off")
  st.pyplot(fig)

# ----------------------------------------------------
# TAB 3: Denoising & Anomaly Detection
# ----------------------------------------------------
elif app_mode == "3. Denoising & Anomaly Detector":
  st.title("🧼 Sketch Denoising & Clothing Validation")
  st.write(
      "Upload a noisy or hand-drawn sketch. The Denoising Autoencoder will"
      " clean it up, and the anomaly detector will flag if it is valid"
      " clothing."
  )

  uploaded_file = st.file_uploader(
      "Choose a sketch image...", type=["png", "jpg", "jpeg"]
  )

  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))

    col1, col2 = st.columns(2)
    with col1:
      st.subheader("Uploaded Sketch")
      st.image(image, width=150)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
      reconstructed = denoise_model(img_tensor)
      mse = torch.mean((reconstructed - img_tensor) ** 2).item()

    is_clothing = mse <= anomaly_threshold

    recon_np = reconstructed.squeeze().cpu().detach().numpy()
    recon_img = Image.fromarray((recon_np * 255).astype(np.uint8))

    with col2:
      st.subheader("Denoised Output")
      st.image(recon_img, width=150)

    st.markdown("---")
    st.subheader("Validation Report")
    st.metric(label="Reconstruction MSE Error", value=f"{mse:.5f}")
    st.metric(label="Learned Threshold", value=f"{anomaly_threshold:.5f}")

    if is_clothing:
      st.success(
          "✅ **Status: Valid Clothing Item** (Reconstruction error is within"
          " acceptable threshold)."
      )
    else:
      st.error(
          "❌ **Status: Flagged as 'Not Clothing'** (Reconstruction error"
          " exceeds anomaly threshold)."
      )