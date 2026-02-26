# Time-Series Diffusion via Wavelet Scalograms

A PyTorch/HuggingFace-based implementation of diffusion models for generating synthetic financial time-series in the wavelet scalogram domain.

---

## 📈 Approach

This project trains a **Denoising Diffusion Probabilistic Model (DDPM)** to take windows of 1D time-series data, transform them into 2D wavelet scalograms, and learn to generate new examples in this space.

**Pipeline:**
1. **1D Time-Series** →  
2. **Continuous Wavelet Transform (CWT)** →  
3. **Scalogram (S×T)** →  
4. **Resize to 1×H×W tensor** →  
5. **UNet2DModel + DDPM Scheduler** →  
6. **Generated Synthetic Scalogram**

- **Model:**  
  - `UNet2DModel` (from Hugging Face Diffusers)  
  - `DDPMScheduler`  
  - MSE noise prediction objective

---

## 🗂️ Project Structure

```
ts_diffusion_wavelet/
├── requirements.txt   # Dependencies
├── config.py          # Hyperparameter & path config
├── train.py           # Training script
├── generate.py        # Synthetic data generation
│
├── data/
│   └── dataset.py     # Time-series to sliding window dataset
│
├── transforms/
│   └── wavelet.py     # CWT & scalogram transforms
│
├── models/
│   └── diffusion.py   # Model and diffusion scheduler construction
│
└── utils/
    └── inverse.py     # (Optional) Inverse CWT utility
```

---

## 1️⃣ Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

- Python 3.9+
- CUDA GPU recommended for training

---

## 2️⃣ Data Preparation

1. **Download historical data**:  
   [https://www.histdata.com/](https://www.histdata.com/)

2. **Export as CSV**.
3. **Place CSV file at**:  
   `data/eurusd.csv`
4. **Ensure it contains a `"Close"` column**.  
   If not, update `COLUMN` in `config.py`.

---

## 3️⃣ Configure Training

Key configuration (`config.py`):

```python
CSV_PATH = "data/eurusd.csv"
COLUMN = "Close"
WINDOW_SIZE = 256
STRIDE = 64
WAVELET = "morl"
SCALES = 128
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
```

**Key Parameters:**

- `WINDOW_SIZE`: Length of sliding time-series window
- `STRIDE`: Step for window overlap
- `SCALES`: Number of CWT scales
- `IMAGE_SIZE`: Output scalogram size (pixels)

---

## 4️⃣ Training

Run training:

```bash
python train.py
```

This will:
- Load & normalize data from CSV
- Form overlapping windows
- Convert each window to a scalogram
- Train a DDPM model
- Save checkpoints to: `./checkpoints/model_epoch_X.pt`

---

## 5️⃣ Generation

After training, generate a synthetic scalogram:

```bash
python generate.py
```

- Loads the latest model checkpoint
- Samples from noise and denoises with DDPM
- Displays the resulting scalogram

---

## 6️⃣ What Does the Model Learn?

- Approximates the probability distribution `p(scalogram)`
- **Does *not* directly learn `p(time_series)`**
- Captures:
  - Multi-scale temporal patterns
  - Frequency structure
  - Cross-scale dependencies (through attention blocks)

---

## 7️⃣ (Optional) Inverse Mapping to Time-Series

To attempt reconstruction back to 1D time-series:
- Apply inverse CWT (**see:** `utils/inverse.py`)
- Note: Accurate inversion requires handling wavelet phase, which may not be perfectly possible for real-valued scalograms.

_This project currently outputs scalograms only._

---

## 8️⃣ Extensions & Ideas

- Try `UNet1DModel` for direct 1D diffusion modeling
- Add conditioning (e.g., regimes or labels)
- Swap DDPM for DDIM or DPM-Solver for faster sampling
- SDE-based diffusion models
- Train on log-returns instead of raw prices
- Add evaluation metrics (ACF, PSD, Hurst exponent, etc.)

---

## 9️⃣ Hardware Notes

- **GPU strongly recommended.**
- For CPU training:
  - Consider reducing `IMAGE_SIZE`, `BATCH_SIZE`, or diffusion `NUM_TIMESTEPS` for faster runtime.

---

## 📝 Summary

This repository provides:

- Wavelet-based time-frequency representation of financial data
- 2D UNet diffusion architecture with attention
- End-to-end synthesis of new (realistic) time-series patterns in scalogram space

Leverages generative image models for advanced time-series structure modeling.
  
---
