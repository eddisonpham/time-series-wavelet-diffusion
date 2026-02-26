# WaveDiff — Wavelet Diffusion Studio

A full-stack platform for generating synthetic financial time-series via **Denoising Diffusion Probabilistic Models (DDPM)** operating in the wavelet scalogram domain. Includes a React UI, FastAPI backend, and SQLite run database.

---

## 📈 Approach

Trains a DDPM to learn the distribution of 2D wavelet scalograms derived from 1D financial time-series windows.

**Pipeline:**
1. **Synthetic Data** — Merton Jump Diffusion price simulation
2. **Sliding Windows** — overlapping 1D time-series chunks
3. **CWT** — Continuous Wavelet Transform → 2D scalogram (S×T)
4. **Resize** → `1×H×W` tensor
5. **UNet2DModel + DDPMScheduler** — MSE noise prediction
6. **Generated Scalogram** — PNG output, stored and viewable in the UI

---

## 🗂️ Project Structure

```
ts_diffusion_wavelet/
├── start.sh               # One-command startup (backend + frontend)
├── requirements.txt       # Python dependencies
├── config.py              # Hyperparams — reads from env vars (UI-overridable)
├── train.py               # Training script with live log streaming
├── generate.py            # Scalogram generation — saves PNG (no plt.show)
├── data_downloader.py     # Merton Jump Diffusion data generator
│
├── app/                   # FastAPI backend
│   ├── main.py            # API routes: /data, /train, /generate, /image
│   ├── database.py        # SQLite engine + session context manager
│   └── models.py          # SQLAlchemy tables: DataRun, TrainRun, GenerationRun
│
├── frontend/              # React + Vite UI
│   ├── vite.config.js     # Proxies /api → localhost:8000
│   └── src/
│       ├── App.jsx        # Sidebar layout + tab routing
│       ├── globals.css    # Design system (dark, amber accent)
│       └── components/
│           ├── DataPanel.jsx     # MJD param sliders, data generation
│           ├── TrainPanel.jsx    # Hyperparams, live log terminal
│           ├── GeneratePanel.jsx # Run generation, scalogram viewer + gallery
│           └── HistoryPanel.jsx  # All runs, image gallery, stats
│
├── data/
│   └── dataset.py         # Sliding window dataset + CWT transform
├── models/
│   └── diffusion.py       # UNet2DModel + DDPMScheduler construction
├── transforms/
│   └── wavelet.py         # CWT & scalogram utilities
└── utils/
    └── inverse.py         # (Optional) inverse CWT
```

---

## 1️⃣ Installation

**Requirements:** Python 3.9+, Node.js 18+. CUDA GPU strongly recommended.

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

---

## 2️⃣ Data Generation

Synthetic OHLCV data is generated via **Merton Jump Diffusion** — no external data source required.

```bash
python data_downloader.py --days 30 --mu 0.05 --sigma 0.2 --lam 0.1
```

**Parameters:**

| Flag | Description | Default |
|------|-------------|---------|
| `--days` | Simulation length (minutes = days × 1440) | `30` |
| `--mu` | Drift | `0.05` |
| `--sigma` | Volatility | `0.2` |
| `--lam` | Jump intensity | `0.1` |
| `--jump_mean` | Mean jump size | `-0.02` |
| `--jump_std` | Jump size std | `0.1` |
| `--year`, `--month` | Simulation start date | `2022`, `1` |

Output: `data/index_time_series.csv`

---

## 3️⃣ Configuration

All parameters in `config.py` are **overridable via environment variables**, which is how the UI controls them per-run:

```python
WAVELET      = os.environ.get("WAVEDIFF_WAVELET", "morl")
IMAGE_SIZE   = int(os.environ.get("WAVEDIFF_IMAGE_SIZE", 128))
BATCH_SIZE   = int(os.environ.get("WAVEDIFF_BATCH_SIZE", 16))
EPOCHS       = int(os.environ.get("WAVEDIFF_EPOCHS", 10))
LR           = float(os.environ.get("WAVEDIFF_LR", 1e-4))
NUM_TIMESTEPS = int(os.environ.get("WAVEDIFF_NUM_TIMESTEPS", 1000))
WINDOW_SIZE  = int(os.environ.get("WAVEDIFF_WINDOW_SIZE", 256))
STRIDE       = int(os.environ.get("WAVEDIFF_STRIDE", 64))
SAVE_DIR     = os.environ.get("WAVEDIFF_SAVE_DIR", "./checkpoints")
```

---

## 4️⃣ Running the Platform

```bash
chmod +x start.sh
./start.sh
```

This starts:
- **FastAPI backend** → `http://localhost:8000` (auto-reload)
- **Vite frontend** → `http://localhost:5173`
- **API docs** → `http://localhost:8000/docs`

Or run manually:
```bash
# Terminal 1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
cd frontend && npm run dev
```

---

## 5️⃣ UI Workflow

The UI exposes the full pipeline across four panels:

**Data** → configure MJD params with sliders, generate CSV, view run history.

**Train** → set all hyperparameters, launch training, watch a live log terminal streaming directly from the subprocess. Each run is saved in SQLite with its own checkpoint directory.

**Generate** → select any successful training run, click generate, view the resulting scalogram. Supports click-to-expand lightbox and a thumbnail gallery of all past generations.

**History** → tabbed view of all data runs, training runs, and generated images with summary stats.

---

## 6️⃣ Database

Uses **SQLite** (zero-config) via SQLAlchemy. DB file: `wavediff.db` (auto-created on startup).

Three tables:

| Table | Contents |
|-------|----------|
| `data_runs` | MJD params, row count, status |
| `train_runs` | All hyperparams, epoch count, checkpoint path, status |
| `generation_runs` | Associated train run, PNG path, status |

---

## 7️⃣ CLI Usage (without UI)

```bash
# Generate data
python data_downloader.py

# Train (uses config.py defaults)
python train.py

# Generate scalogram → saves to generated_scalogram.png
python generate.py

# Or specify output path via env
WAVEDIFF_OUTPUT_PATH=./out.png python generate.py
```

---

## 8️⃣ What the Model Learns

- Approximates `p(scalogram)` — not `p(time_series)` directly
- Captures multi-scale temporal patterns, frequency structure, and cross-scale dependencies via UNet attention blocks
- Inverse CWT to 1D is possible but imperfect — see `utils/inverse.py`

---

## 9️⃣ Extensions

- `UNet1DModel` for direct 1D diffusion
- Conditioning on market regimes or volatility labels
- Swap DDPMScheduler for DDIM / DPM-Solver for faster sampling
- Evaluation metrics: ACF, PSD, Hurst exponent
- Train on log-returns instead of raw prices
- SDE-based diffusion (score matching)

---

## 🖥️ Hardware

GPU strongly recommended. For CPU, reduce `IMAGE_SIZE` (e.g. 64), `BATCH_SIZE` (4–8), and `NUM_TIMESTEPS` (200–500) in the UI or via env vars.