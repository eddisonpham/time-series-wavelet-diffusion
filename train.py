import torch
from torch.utils.data import DataLoader
import os
import sys

from config import BATCH_SIZE, DEVICE, EPOCHS, LR, SAVE_DIR
from data.dataset import HistDataDataset
from models.diffusion import create_model


def train():
    print(f"[WaveDiff] Starting training", flush=True)
    print(f"[WaveDiff] Device: {DEVICE}", flush=True)
    print(f"[WaveDiff] Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}", flush=True)
    print(f"[WaveDiff] Save dir: {SAVE_DIR}", flush=True)

    dataset = HistDataDataset()
    print(f"[WaveDiff] Dataset size: {len(dataset)} windows", flush=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model, scheduler = create_model()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        steps = 0

        for batch in loader:
            batch = batch.to(DEVICE)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch.shape[0],), device=DEVICE
            ).long()

            noisy_images = scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"[WaveDiff] Epoch {epoch + 1}/{EPOCHS} | avg_loss={avg_loss:.4f}", flush=True)
        sys.stdout.flush()

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pt"),
            _use_new_zipfile_serialization=True
        )
        print(f"[WaveDiff] Checkpoint saved: model_epoch_{epoch}.pt", flush=True)

    print("[WaveDiff] Training complete!", flush=True)


if __name__ == "__main__":
    train()