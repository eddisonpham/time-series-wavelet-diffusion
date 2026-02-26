import torch
import os

# Wavelet
WAVELET = os.environ.get("WAVEDIFF_WAVELET", "morl")
SCALES = int(os.environ.get("WAVEDIFF_SCALES", 128))

# Image
IMAGE_SIZE = int(os.environ.get("WAVEDIFF_IMAGE_SIZE", 128))

# Training
BATCH_SIZE = int(os.environ.get("WAVEDIFF_BATCH_SIZE", 16))
EPOCHS = int(os.environ.get("WAVEDIFF_EPOCHS", 10))
LR = float(os.environ.get("WAVEDIFF_LR", 1e-4))
NUM_TIMESTEPS = int(os.environ.get("WAVEDIFF_NUM_TIMESTEPS", 1000))
SAVE_DIR = os.environ.get("WAVEDIFF_SAVE_DIR", "./checkpoints")

# Data
CSV_PATH = os.environ.get("WAVEDIFF_CSV_PATH", "data/index_time_series.csv")
COLUMN = os.environ.get("WAVEDIFF_COLUMN", "Close")
WINDOW_SIZE = int(os.environ.get("WAVEDIFF_WINDOW_SIZE", 256))
STRIDE = int(os.environ.get("WAVEDIFF_STRIDE", 64))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"