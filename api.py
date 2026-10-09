import base64
import io
import os
import sys
from typing import List

# Ensure Python can find the 'src' folder regardless of execution path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Now import network architectures from src/
from networks import (
    ConditionalGenerator,
    ConvAutoencoder,
    FashionCNNClassifier,
    VAE,
)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Generative Fashion Design Studio API",
    description="API backend for cGAN design generation, VAE latent space morphing, Denoising & Anomaly Detection, and CNN Classification.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load Models
models_dir = os.path.join(current_dir, "models")

# cGAN
cgan_gen = ConditionalGenerator(100, 10).to(device)
cgan_gen_path = os.path.join(models_dir, "cgan_generator.pt")
if os.path.exists(cgan_gen_path):
  cgan_gen.load_state_dict(
      torch.load(cgan_gen_path, map_location=device, weights_only=True)
  )
cgan_gen.eval()

# VAE
vae_model = VAE(latent_dim=20).to(device)
vae_path = os.path.join(models_dir, "vae.pt")
if os.path.exists(vae_path):
  vae_model.load_state_dict(
      torch.load(vae_path, map_location=device, weights_only=True)
  )
vae_model.eval()

# Denoising AE
denoise_ckpt_path = os.path.join(models_dir, "denoising_autoencoder.pt")
denoise_model = ConvAutoencoder().to(device)
anomaly_threshold = 0.02
if os.path.exists(denoise_ckpt_path):
  denoise_ckpt = torch.load(
      denoise_ckpt_path, map_location=device, weights_only=False
  )
  denoise_model.load_state_dict(denoise_ckpt["state_dict"])
  anomaly_threshold = float(denoise_ckpt["threshold"])
denoise_model.eval()

# CNN Classifier
classifier_model = FashionCNNClassifier().to(device)
classifier_path = os.path.join(models_dir, "classifier.pt")
if os.path.exists(classifier_path):
  classifier_model.load_state_dict(
      torch.load(classifier_path, map_location=device, weights_only=True)
  )
classifier_model.eval()


def fig_to_base64(img_np: np.ndarray) -> str:
  img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
  pil_img = Image.fromarray(img_np, mode="L")
  buffer = io.BytesIO()
  pil_img.save(buffer, format="PNG")
  return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Request Schemas
class GenerateRequest(BaseModel):
  class_id: int = Field(
      ..., ge=0, le=9, description="Clothing category ID (0-9)"
  )
  num_samples: int = Field(
      1, ge=1, le=16, description="Number of images to generate (1-16)"
  )


class MorphRequest(BaseModel):
  latent_vector: List[float] = Field(
      ..., description="20-dimensional latent vector"
  )


@app.get("/")
def read_root():
  return {
      "status": "online",
      "app": "Generative Fashion Design Studio API",
      "device": str(device),
      "anomaly_threshold": anomaly_threshold,
      "classes": CLASSES,
  }


@app.post("/generate")
def generate_sketches(req: GenerateRequest):
  with torch.no_grad():
    z = torch.randn(req.num_samples, 100, device=device)
    labels = torch.tensor(
        [req.class_id] * req.num_samples, dtype=torch.long, device=device
    )
    fake_imgs = cgan_gen(z, labels).cpu().numpy()

  images_b64 = []
  for i in range(req.num_samples):
    img = (fake_imgs[i].squeeze() + 1.0) / 2.0
    images_b64.append(fig_to_base64(img))

  return {
      "class_id": req.class_id,
      "class_name": CLASSES[req.class_id],
      "num_samples": req.num_samples,
      "images_base64": images_b64,
  }


@app.post("/morph")
def morph_latent(req: MorphRequest):
  if len(req.latent_vector) != 20:
    raise HTTPException(
        status_code=400, detail="latent_vector must be of length 20"
    )

  with torch.no_grad():
    z = torch.tensor([req.latent_vector], dtype=torch.float32, device=device)
    decoded = vae_model.decoder_fc(z).view(28, 28).cpu().numpy()

  return {"image_base64": fig_to_base64(decoded)}


@app.post("/denoise")
async def denoise_and_validate(file: UploadFile = File(...)):
  try:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

  transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
  )
  img_tensor = transform(image).unsqueeze(0).to(device)

  with torch.no_grad():
    reconstructed = denoise_model(img_tensor)
    mse = float(torch.mean((reconstructed - img_tensor) ** 2).item())

  is_clothing = mse <= anomaly_threshold
  recon_np = reconstructed.squeeze().cpu().detach().numpy()

  return {
      "mse_error": mse,
      "threshold": anomaly_threshold,
      "is_clothing": is_clothing,
      "denoised_image_base64": fig_to_base64(recon_np),
  }


@app.post("/classify")
async def classify_sketch(file: UploadFile = File(...)):
  try:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

  transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
  )
  img_tensor = transform(image).unsqueeze(0).to(device)

  with torch.no_grad():
    outputs = classifier_model(img_tensor)
    probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    pred_class = int(np.argmax(probs))

  return {
      "predicted_class_id": pred_class,
      "predicted_class_name": CLASSES[pred_class],
      "confidence": float(probs[pred_class]),
      "all_probabilities": {CLASSES[i]: float(probs[i]) for i in range(10)},
  }


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)