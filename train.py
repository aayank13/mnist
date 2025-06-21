# train.py - Conditional DCGAN Training Script for MNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from model import Generator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
NOISE_DIM = 100
LABEL_DIM = 10
NUM_EPOCHS = 100
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Discriminator Architecture
class Discriminator(nn.Module):
    def __init__(self, label_dim=10):
        super().__init__()
        self.label_embed = nn.Embedding(label_dim, label_dim)
        
        self.net = nn.Sequential(
            # Input: 1 x 28 x 28 + label embedding
            nn.Conv2d(1 + label_dim, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        label_embed = self.label_embed(labels).unsqueeze(2).unsqueeze(3)  # (B, 10, 1, 1)
        label_embed = label_embed.expand(-1, -1, x.size(2), x.size(3))  # (B, 10, H, W)
        x = torch.cat([x, label_embed], dim=1)
        return self.net(x)

# Initialize models
generator = Generator(noise_dim=NOISE_DIM, label_dim=LABEL_DIM).to(device)
discriminator = Discriminator(label_dim=LABEL_DIM).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

# Training function
def train_epoch(epoch):
    generator.train()
    discriminator.train()
    
    total_loss_G = 0
    total_loss_D = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (real_images, real_labels) in enumerate(progress_bar):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        
        # Ground truths
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real images
        real_outputs = discriminator(real_images, real_labels)
        d_real_loss = criterion(real_outputs, valid)
        
        # Fake images
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
        fake_labels = torch.randint(0, LABEL_DIM, (batch_size,)).to(device)
        fake_images = generator(noise, fake_labels)
        fake_outputs = discriminator(fake_images.detach(), fake_labels)
        d_fake_loss = criterion(fake_outputs, fake)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
        fake_labels = torch.randint(0, LABEL_DIM, (batch_size,)).to(device)
        fake_images = generator(noise, fake_labels)
        fake_outputs = discriminator(fake_images, fake_labels)
        
        # Generator loss
        g_loss = criterion(fake_outputs, valid)
        g_loss.backward()
        optimizer_G.step()
        
        # Update progress bar
        total_loss_G += g_loss.item()
        total_loss_D += d_loss.item()
        
        progress_bar.set_postfix({
            'G_Loss': f'{g_loss.item():.4f}',
            'D_Loss': f'{d_loss.item():.4f}'
        })
    
    return total_loss_G / len(train_loader), total_loss_D / len(train_loader)

# Training loop
print("Starting training...")
g_losses = []
d_losses = []

for epoch in range(NUM_EPOCHS):
    g_loss, d_loss = train_epoch(epoch)
    g_losses.append(g_loss)
    d_losses.append(d_loss)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - G_Loss: {g_loss:.4f}, D_Loss: {d_loss:.4f}")
    
    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f"dcgan_generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"dcgan_discriminator_epoch_{epoch+1}.pth")
        
        # Generate sample images
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(16, NOISE_DIM, 1, 1).to(device)
            labels = torch.arange(10).repeat(2).to(device)[:16]
            fake_images = generator(noise, labels).cpu()
            
            # Denormalize
            fake_images = (fake_images + 1) / 2
            
            # Plot
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(fake_images[i].squeeze(), cmap='gray')
                ax.set_title(f'Digit: {labels[i].item()}')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'samples_epoch_{epoch+1}.png')
            plt.close()

# Save final models
torch.save(generator.state_dict(), "dcgan_generator.pth")
torch.save(discriminator.state_dict(), "dcgan_discriminator.pth")

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig('training_losses.png')
plt.show()

print("Training completed!")
print("Final models saved as 'dcgan_generator.pth' and 'dcgan_discriminator.pth'") 