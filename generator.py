# generator.py

import torch
import numpy as np
from model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
noise_dim = 100
generator = Generator(noise_dim=noise_dim).to(device)
generator.load_state_dict(torch.load("dcgan_generator.pth", map_location=device))
generator.eval()

def generate_images(digit: int, count: int = 5):
    labels = torch.tensor([digit] * count).to(device)
    noise = torch.randn(count, noise_dim, 1, 1).to(device)

    with torch.no_grad():
        images = generator(noise, labels).cpu().numpy()

    # Rescale from [-1, 1] to [0, 1] and squeeze
    images = (images + 1) / 2
    images = np.squeeze(images, axis=1)

    return images
