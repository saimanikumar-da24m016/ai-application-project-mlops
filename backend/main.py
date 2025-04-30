# backend/main.py

import os, io, time, sqlite3
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import numpy as np

from model_utils import MODEL, IMG_SIZE

# ── setup feedback DB path ───────────────────────────────────────────
DATA_DIR = os.getenv("FEEDBACK_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "feedback.db")

# ── FastAPI + CORS + Prometheus ─────────────────────────────────────
app = FastAPI(title="Standalone ALL-Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app, include_in_schema=False)

# ── SQLite setup ────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS feedback(
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        REAL,
  correct   TEXT,
  predicted TEXT,
  img       BLOB
)
""")
conn.commit()

# ── helper ───────────────────────────────────────────────────────────
def _predict(img: Image.Image):
    arr   = np.asarray(img.resize(IMG_SIZE[:2]).convert("RGB"), dtype="float32")[None] / 255.0
    probs = MODEL.predict(arr)[0].tolist()
    return probs, int(np.argmax(probs))

# ── endpoints ───────────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img  = Image.open(io.BytesIO(data))
    probs, idx = _predict(img)
    return {"probabilities": probs, "predicted_class": idx}

@app.post("/feedback")
async def feedback(
    correct_label:   str = Form(...),
    predicted_label: str = Form(...),
    file:            UploadFile = File(...),
):
    blob = await file.read()
    conn.execute(
        "INSERT INTO feedback(ts,correct,predicted,img) VALUES(?,?,?,?)",
        (time.time(), correct_label, predicted_label, blob),
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    return JSONResponse({"stored": True, "feedback_count": count})

@app.get("/feedback/count")
def feedback_count():
    count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    return {"feedback_count": count}

@app.get("/ping")
def ping():
    return {"status": "ok"}
