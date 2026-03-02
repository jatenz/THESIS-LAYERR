from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import subprocess

app = FastAPI(title="Viewer Loyalty API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_and_run(
    file: UploadFile = File(...),
    freq: str = Form("W"),
    train_ratio: float = Form(0.7),
    alpha: float = Form(1.0),
    enable_churn: bool = Form(True),
    churn_k: int = Form(3),
    do_hmm: bool = Form(False),
    do_leiden: bool = Form(False),
):
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(UPLOAD_DIR, f"{run_id}_{file.filename}")
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cmd = [
        "python",
        os.path.join(BASE_DIR, "run.py"),
        "--csv", csv_path,
        "--out", run_dir,
        "--freq", freq,
        "--train_ratio", str(train_ratio),
        "--alpha", str(alpha),
    ]

    if enable_churn:
        cmd += ["--enable_churn", "--churn_k", str(churn_k)]

    if do_hmm:
        cmd += ["--do_hmm"]

    if do_leiden:
        cmd += ["--do_leiden"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return JSONResponse(
            status_code=400,
            content={"error": result.stderr}
        )

    return {
        "status": "completed",
        "run_id": run_id,
        "output": result.stdout[-1000:]
    }