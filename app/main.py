from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
import os
import json
import time
import asyncio
from datetime import datetime

from app.database import init_db, get_session
from app.models import DataRun, TrainRun, GenerationRun

app = FastAPI(title="WaveDiff API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.on_event("startup")
def startup():
    init_db()
    os.makedirs(os.path.join(BASE_DIR, "generated"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class DataParams(BaseModel):
    days: int = 30
    mu: float = 0.05
    sigma: float = 0.2
    lam: float = 0.1
    jump_mean: float = -0.02
    jump_std: float = 0.1
    year: int = 2022
    month: int = 1

class TrainParams(BaseModel):
    window_size: int = 256
    stride: int = 64
    wavelet: str = "morl"
    scales: int = 128
    image_size: int = 128
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-4
    num_timesteps: int = 1000

class GenerateParams(BaseModel):
    train_run_id: int

# ─── Data Generation ──────────────────────────────────────────────────────────

@app.post("/api/data")
def generate_data(params: DataParams):
    with get_session() as session:
        run = DataRun(
            created_at=datetime.utcnow(),
            params=json.dumps(params.dict()),
            csv_path="data/index_time_series.csv",
            rows=0,
            status="running"
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    cmd = [
        sys.executable, os.path.join(BASE_DIR, "data_downloader.py"),
        "--year", str(params.year),
        "--month", str(params.month),
        "--days", str(params.days),
        "--mu", str(params.mu),
        "--sigma", str(params.sigma),
        "--lam", str(params.lam),
        "--jump_mean", str(params.jump_mean),
        "--jump_std", str(params.jump_std),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)

    rows = 0
    for line in result.stdout.splitlines():
        if "Rows generated:" in line:
            rows = int(line.split(":")[1].strip())

    status = "success" if result.returncode == 0 else "failed"
    with get_session() as session:
        run = session.get(DataRun, run_id)
        run.rows = rows
        run.status = status
        run.log = result.stdout + result.stderr
        session.commit()

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    return {"id": run_id, "rows": rows, "status": status}

@app.get("/api/data")
def list_data_runs():
    with get_session() as session:
        runs = session.query(DataRun).order_by(DataRun.created_at.desc()).all()
        return [{"id": r.id, "created_at": r.created_at, "params": json.loads(r.params),
                 "rows": r.rows, "status": r.status} for r in runs]

# ─── Training ─────────────────────────────────────────────────────────────────

training_jobs: dict[int, dict] = {}

def run_training(run_id: int, params: dict):
    log_path = os.path.join(BASE_DIR, "logs", f"train_{run_id}.log")
    env = os.environ.copy()
    env.update({
        "WAVEDIFF_WINDOW_SIZE": str(params["window_size"]),
        "WAVEDIFF_STRIDE": str(params["stride"]),
        "WAVEDIFF_WAVELET": params["wavelet"],
        "WAVEDIFF_SCALES": str(params["scales"]),
        "WAVEDIFF_IMAGE_SIZE": str(params["image_size"]),
        "WAVEDIFF_BATCH_SIZE": str(params["batch_size"]),
        "WAVEDIFF_EPOCHS": str(params["epochs"]),
        "WAVEDIFF_LR": str(params["lr"]),
        "WAVEDIFF_NUM_TIMESTEPS": str(params["num_timesteps"]),
        "WAVEDIFF_SAVE_DIR": os.path.join(BASE_DIR, "checkpoints", f"run_{run_id}"),
    })

    training_jobs[run_id] = {"status": "running", "log_path": log_path}

    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "train.py")],
            stdout=f, stderr=subprocess.STDOUT,
            env=env, cwd=BASE_DIR
        )
        training_jobs[run_id]["pid"] = proc.pid
        proc.wait()
        status = "success" if proc.returncode == 0 else "failed"

    training_jobs[run_id]["status"] = status
    with get_session() as session:
        run = session.get(TrainRun, run_id)
        run.status = status
        session.commit()

@app.post("/api/train")
def start_training(params: TrainParams, background_tasks: BackgroundTasks):
    save_dir = os.path.join(BASE_DIR, "checkpoints", f"run_{{id}}")
    with get_session() as session:
        run = TrainRun(
            created_at=datetime.utcnow(),
            params=json.dumps(params.dict()),
            epochs=params.epochs,
            status="pending",
            log_path=""
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id
        log_path = os.path.join(BASE_DIR, "logs", f"train_{run_id}.log")
        run.log_path = log_path
        run.save_dir = os.path.join(BASE_DIR, "checkpoints", f"run_{run_id}")
        session.commit()

    background_tasks.add_task(run_training, run_id, params.dict())
    return {"id": run_id, "status": "pending"}

@app.get("/api/train/{run_id}/logs")
async def stream_logs(run_id: int):
    with get_session() as session:
        run = session.get(TrainRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        log_path = run.log_path

    async def event_stream():
        pos = 0
        while True:
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    f.seek(pos)
                    chunk = f.read()
                    if chunk:
                        pos += len(chunk.encode())
                        for line in chunk.splitlines():
                            yield f"data: {json.dumps({'line': line})}\n\n"

            job = training_jobs.get(run_id)
            if job and job["status"] in ("success", "failed"):
                yield f"data: {json.dumps({'done': True, 'status': job['status']})}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/train")
def list_train_runs():
    with get_session() as session:
        runs = session.query(TrainRun).order_by(TrainRun.created_at.desc()).all()
        result = []
        for r in runs:
            job = training_jobs.get(r.id)
            status = job["status"] if job else r.status
            result.append({
                "id": r.id,
                "created_at": r.created_at,
                "params": json.loads(r.params),
                "epochs": r.epochs,
                "status": status,
                "save_dir": r.save_dir
            })
        return result

@app.get("/api/train/{run_id}")
def get_train_run(run_id: int):
    with get_session() as session:
        run = session.get(TrainRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Not found")
        job = training_jobs.get(run_id)
        status = job["status"] if job else run.status
        return {
            "id": run.id,
            "created_at": run.created_at,
            "params": json.loads(run.params),
            "epochs": run.epochs,
            "status": status,
            "save_dir": run.save_dir
        }

# ─── Generation ───────────────────────────────────────────────────────────────

@app.post("/api/generate")
def generate_scalogram(params: GenerateParams):
    with get_session() as session:
        train_run = session.get(TrainRun, params.train_run_id)
        if not train_run:
            raise HTTPException(status_code=404, detail="Training run not found")
        if train_run.status != "success":
            raise HTTPException(status_code=400, detail=f"Training run status: {train_run.status}")
        train_params = json.loads(train_run.params)
        save_dir = train_run.save_dir

    out_filename = f"gen_{params.train_run_id}_{int(time.time())}.png"
    out_path = os.path.join(BASE_DIR, "generated", out_filename)

    env = os.environ.copy()
    env.update({
        "WAVEDIFF_IMAGE_SIZE": str(train_params.get("image_size", 128)),
        "WAVEDIFF_NUM_TIMESTEPS": str(train_params.get("num_timesteps", 1000)),
        "WAVEDIFF_SAVE_DIR": save_dir,
        "WAVEDIFF_EPOCHS": str(train_params.get("epochs", 10)),
        "WAVEDIFF_OUTPUT_PATH": out_path,
    })

    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "generate.py")],
        capture_output=True, text=True, env=env, cwd=BASE_DIR
    )

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    with get_session() as session:
        gen = GenerationRun(
            created_at=datetime.utcnow(),
            train_run_id=params.train_run_id,
            image_path=out_path,
            image_filename=out_filename,
            status="success"
        )
        session.add(gen)
        session.commit()
        session.refresh(gen)
        gen_id = gen.id

    return {"id": gen_id, "image_url": f"/api/image/{out_filename}"}

@app.get("/api/generate")
def list_generations():
    with get_session() as session:
        gens = session.query(GenerationRun).order_by(GenerationRun.created_at.desc()).all()
        return [{
            "id": g.id,
            "created_at": g.created_at,
            "train_run_id": g.train_run_id,
            "image_url": f"/api/image/{g.image_filename}",
            "status": g.status
        } for g in gens]

@app.get("/api/image/{filename}")
def get_image(filename: str):
    path = os.path.join(BASE_DIR, "generated", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/png")

@app.delete("/api/train/{run_id}")
def delete_train_run(run_id: int):
    with get_session() as session:
        run = session.get(TrainRun, run_id)
        if run:
            session.delete(run)
            session.commit()
    return {"ok": True}