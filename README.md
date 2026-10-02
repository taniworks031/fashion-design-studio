# Generative Fashion Design Studio 👗✨

An AI-powered fashion design web application built using PyTorch, FastAPI, and Streamlit, leveraging the Fashion-MNIST dataset via local `.idx-ubyte` raw binary files.

## Features
1. **cGAN Grid Generator**: Generate class-conditional clothing sketches across all 10 categories.
2. **VAE Latent Space Morphing**: Explore and interpolate clothing representations via latent space sliders.
3. **Denoising Autoencoder & Anomaly Detector**: Clean noisy sketches and automatically flag non-clothing anomalies using learned MSE thresholds.
4. **CNN Classifier Evaluation**: Verify generative model quality against a high-accuracy classifier ($\ge 80\%$ target).

## Project Structure
- `src/`: Core PyTorch dataset loaders, network architectures, and training scripts.
- `models/`: Saved model weights (`classifier.pt`, `vae.pt`, `cgan_generator.pt`, `denoising_autoencoder.pt`).
- `data/`: Local raw Fashion-MNIST binary files.
- `api.py`: FastAPI backend server.
- `app.py`: Interactive Streamlit frontend web app.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run FastAPI Backend: `python api.py`
3. Run Streamlit Frontend: `streamlit run app.py`