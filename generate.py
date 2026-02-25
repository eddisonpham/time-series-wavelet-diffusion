import torch
import matplotlib.pyplot as plt
from config import *
from models.diffusion import create_model


def generate():
    model, scheduler = create_model()
    model.load_state_dict(torch.load(f"{SAVE_DIR}/model_epoch_{EPOCHS-1}.pt"))
    model.to(DEVICE)
    model.eval()

    sample = torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

    scheduler.set_timesteps(NUM_TIMESTEPS)

    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    img = sample.squeeze().cpu().numpy()

    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.title("Generated Wavelet Scalogram")
    plt.show()


if __name__ == "__main__":
    generate()