import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import BATCH_SIZE, DEVICE, EPOCHS, LR, SAVE_DIR
from data.dataset import HistDataDataset
from models.diffusion import create_model


def train():
    dataset = HistDataDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model, scheduler = create_model()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        pbar = tqdm(loader)

        for batch in pbar:
            batch = batch.to(DEVICE)

            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch.shape[0],),
                device=DEVICE
            ).long()

            noisy_images = scheduler.add_noise(batch, noise, timesteps)

            noise_pred = model(noisy_images, timesteps).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch_{epoch}.pt", _use_new_zipfile_serialization=True)


if __name__ == "__main__":
    train()