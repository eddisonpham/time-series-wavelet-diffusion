import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from config import CSV_PATH, COLUMN, WINDOW_SIZE, STRIDE
from transforms.wavelet import timeseries_to_scalogram


class HistDataDataset(Dataset):
    def __init__(self):
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        if COLUMN not in df.columns:
            raise ValueError(f"Column '{COLUMN}' not found in CSV")
        series = df[COLUMN].values.astype(np.float32)

        for i in range(0, len(series) - WINDOW_SIZE, STRIDE):
            window = series[i:i + WINDOW_SIZE]
            window = (window - window.mean()) / (window.std() + 1e-8)
            scalogram = timeseries_to_scalogram(window)
            self.windows.append(scalogram)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        signal = self.windows[idx]
        img = timeseries_to_scalogram(signal)
        return img