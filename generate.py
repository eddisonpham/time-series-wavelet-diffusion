import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import os
import sys

from config import SAVE_DIR, EPOCHS, DEVICE, IMAGE_SIZE, NUM_TIMESTEPS
from models.diffusion import create_model


def generate(output_path: str = None):
    model, scheduler = create_model()

    checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{EPOCHS - 1}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    sample = torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
    scheduler.set_timesteps(NUM_TIMESTEPS)

    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    img = sample.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#05070f")
    ax.set_facecolor("#05070f")
    im = ax.imshow(img, cmap="inferno", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="#f0ede8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#f0ede8")
    ax.set_title("Generated Wavelet Scalogram", color="#f0ede8", fontsize=14, pad=12)
    ax.tick_params(colors="#71717a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#27272a")
    ax.set_xlabel("Time", color="#71717a")
    ax.set_ylabel("Scale", color="#71717a")
    plt.tight_layout()

    # Output path: from env var, argument, or default
    if output_path is None:
        output_path = os.environ.get("WAVEDIFF_OUTPUT_PATH", "generated_scalogram.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved scalogram to: {output_path}")
    return output_path


if __name__ == "__main__":
    out = os.environ.get("WAVEDIFF_OUTPUT_PATH", "generated_scalogram.png")
    generate(output_path=out)