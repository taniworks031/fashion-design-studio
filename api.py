import io
import os
import sys

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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel