from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import numpy as np, io, time, sqlite3, os
from model_utils import MODEL, IMG_SIZE

app = FastAPI(title="Standalone ALL-Classifier")
DB = "feedback.db"
import os

DB_DIR = os.getenv("FEEDBACK_DIR", "/app/data")
os.makedirs(DB_DIR, exist_ok=True)
DB = os.path.join(DB_DIR, "feedback.db")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation (v6+)
Instrumentator().instrument(app).expose(app, include_in_schema=False)

# SQLite feedback DB
DB = "feedback.db"
conn = sqlite3.connect(DB, check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS feedback(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts REAL,
                  correct TEXT,
                  predicted TEXT,
                  img BLOB)""")

def _predict(img: Image.Image):
    arr = (
        np.asarray(img.resize(IMG_SIZE[:2]).convert("RGB"), dtype="float32")[None]
        / 255.0
    )
    probs = MODEL.predict(arr)[0].tolist()
    return probs, int(np.argmax(probs))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    probs, pred_idx = _predict(img)
    return {"probabilities": probs, "predicted_class": pred_idx}

@app.post("/feedback")
async def feedback(
    correct_label: str = Form(...),
    predicted_label: str = Form(...),
    file: UploadFile = File(...),
):
    blob = await file.read()
    conn.execute(
        "INSERT INTO feedback(ts,correct,predicted,img) VALUES(?,?,?,?)",
        (time.time(), correct_label, predicted_label, blob),
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    return JSONResponse({"stored": True, "feedback_count": count})

@app.get("/ping")
def ping():
    return {"status": "ok"}
