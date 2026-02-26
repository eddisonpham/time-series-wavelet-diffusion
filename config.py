import torch

# Wavelet
WAVELET = "morl"
SCALES = 128

# Image
IMAGE_SIZE = 128

# Training
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_TIMESTEPS = 1000
SAVE_DIR = "./checkpoints"

# Data
CSV_PATH = "data/index_time_series.csv"
COLUMN = "Close"
WINDOW_SIZE = 256
STRIDE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

