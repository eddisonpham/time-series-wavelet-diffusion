# Time-Series Diffusion via Wavelet Scalograms

This project trains a DDPM (Denoising Diffusion Probabilistic Model) on time-series data by:

1. Converting 1D financial time-series into 2D wavelet scalograms
2. Training a 2D diffusion UNet on those scalogram images
3. Generating new synthetic scalograms from noise

The model is implemented using HuggingFace diffusers.

------------------------------------------------------------
ARCHITECTURE OVERVIEW
------------------------------------------------------------

Time Series (1D)
        ↓
Continuous Wavelet Transform (CWT)
        ↓
Scalogram (S × T)
        ↓
Resize → 1×H×W tensor
        ↓
UNet2D + DDPM Scheduler
        ↓
Generated Scalogram

Model:
- UNet2DModel
- DDPMScheduler
- MSE noise prediction objective

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------

ts_diffusion_wavelet/

├── requirements.txt
├── config.py
├── train.py
├── generate.py
│
├── data/
│   └── dataset.py
│
├── transforms/
│   └── wavelet.py
│
├── models/
│   └── diffusion.py
│
└── utils/
    └── inverse.py

------------------------------------------------------------
1. INSTALLATION
------------------------------------------------------------

Create a virtual environment and install dependencies:

pip install -r requirements.txt

Requirements:
- Python 3.9+
- CUDA GPU recommended

------------------------------------------------------------
2. PREPARE DATA (HISTDATA)
------------------------------------------------------------

1. Download historical data from:
   https://www.histdata.com/

2. Export as CSV.

3. Place the CSV at:

data/eurusd.csv

4. Ensure it contains a "Close" column.
   If not, modify COLUMN in config.py.

------------------------------------------------------------
3. CONFIGURE TRAINING
------------------------------------------------------------

Edit config.py:

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

Important parameters:

- WINDOW_SIZE: length of time-series segment
- STRIDE: sliding window overlap
- SCALES: wavelet resolution
- IMAGE_SIZE: scalogram resize dimension

------------------------------------------------------------
4. TRAINING
------------------------------------------------------------

Run:

python train.py

This will:

- Load CSV
- Create sliding windows
- Convert each window to a scalogram
- Train a DDPM model
- Save checkpoints to:

./checkpoints/model_epoch_X.pt

------------------------------------------------------------
5. GENERATE SYNTHETIC SCALOGRAMS
------------------------------------------------------------

After training:

python generate.py

This will:

- Load the last checkpoint
- Sample from pure Gaussian noise
- Iteratively denoise
- Display the generated scalogram

------------------------------------------------------------
6. WHAT THE MODEL LEARNS
------------------------------------------------------------

The diffusion model learns:

p(scalogram)

It does NOT directly learn:

p(time_series)

The learned distribution captures:

- Multi-scale temporal patterns
- Frequency structure
- Cross-scale dependencies (via attention layers)

------------------------------------------------------------
7. CONVERTING BACK TO TIME-SERIES (OPTIONAL)
------------------------------------------------------------

To reconstruct time-series from generated scalograms:

- Use inverse CWT
- Requires careful handling of phase information
- May not be perfectly invertible unless complex coefficients are preserved

This project currently generates scalograms only.

------------------------------------------------------------
8. EXTENDING THE PROJECT
------------------------------------------------------------

Possible improvements:

- Use UNet1DModel for direct 1D diffusion
- Add conditional diffusion (e.g., regime conditioning)
- Replace DDPM with DDIM or DPM-Solver
- Add SDE-based diffusion
- Train on log returns instead of raw prices
- Add evaluation metrics (ACF, PSD, Hurst exponent)

------------------------------------------------------------
9. HARDWARE NOTES
------------------------------------------------------------

GPU strongly recommended.

If running on CPU:
- Reduce IMAGE_SIZE
- Reduce BATCH_SIZE
- Reduce NUM_TIMESTEPS

------------------------------------------------------------
SUMMARY
------------------------------------------------------------

This repository implements:

- Wavelet-based time-frequency representation
- 2D attention UNet diffusion model
- End-to-end synthetic time-series structure generation

It leverages image diffusion architectures for financial time-series modeling.