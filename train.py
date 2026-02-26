import torch
from torch.utils.data import DataLoader
import os

from config import BATCH_SIZE, DEVICE, EPOCHS, LR, SAVE_DIR
from data.dataset import HistDataDataset
from models.diffusion import create_model

CHECKPOINT_EVERY = 5
CHECKPOINT_KEEP_LAST = 3

def cleanup_checkpoints(save_dir, keep_last=CHECKPOINT_KEEP_LAST):
    files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('model_epoch_') and f.endswith('.pt')],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    for f in files[:-keep_last]:
        try:
            os.remove(os.path.join(save_dir, f))
        except Exception:
            pass

def train():
    print(f"[WaveDiff] Starting training\n[WaveDiff] Device: {DEVICE}\n[WaveDiff] Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}\n[WaveDiff] Save dir: {SAVE_DIR}", flush=True)

    dataset = HistDataDataset()
    print(f"[WaveDiff] Dataset size: {len(dataset)} windows", flush=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model, scheduler = create_model()
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        total_loss, steps = 0.0, 0
        for batch in loader:
            batch = batch.to(DEVICE)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=DEVICE).long()
            noisy_images = scheduler.add_noise(batch, noise, timesteps)
            loss = torch.nn.functional.mse_loss(model(noisy_images, timesteps).sample, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); steps += 1

        avg_loss = total_loss / max(1, steps)
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == EPOCHS - 1:
            print(f"[WaveDiff] Epoch {epoch + 1}/{EPOCHS} | avg_loss={avg_loss:.4f}", flush=True)

            checkpoint_name = f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, checkpoint_name), _use_new_zipfile_serialization=True)
            print(f"[WaveDiff] Checkpoint saved: {checkpoint_name}", flush=True)

    cleanup_checkpoints(SAVE_DIR, CHECKPOINT_KEEP_LAST)
    print("[WaveDiff] Training complete!", flush=True)

if __name__ == "__main__":
    train()