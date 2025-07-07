# train_unet.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import SporeDataset
from unet import UNet

# Configuración
IMAGE_DIR = './dataset/images'
MASK_DIR = './dataset/masks'
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    dataset = SporeDataset(IMAGE_DIR, MASK_DIR, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "unet_sporas.pth")
    print("✅ Modelo guardado como unet_sporas.pth")

if __name__ == "__main__":
    train()
