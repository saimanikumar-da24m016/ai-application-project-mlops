import io, os, sqlite3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import psutil
from apscheduler.schedulers.background import BackgroundScheduler
from drift import detect_drift
from feedback import process_feedback_loop
from PIL import Image
import uuid
from datetime import datetime

app = FastAPI()

# ----- Prometheus Metrics -----
REQUEST_COUNT = Counter("all_requests_total", "Total inference requests")
REQUEST_LATENCY = Histogram("all_request_latency_seconds", "Inference latency")
DRIFT_GAUGE = Gauge("all_data_drift", "Drift flag: 0=no,1=yes")
CPU_GAUGE    = Gauge("system_cpu_percent", "CPU utilization percent")
MEMORY_GAUGE = Gauge("system_memory_percent", "Memory utilization percent")
DISK_IO_GAUGE    = Gauge("disk_io_bytes", "Disk I/O bytes")
NETWORK_IO_GAUGE = Gauge("network_io_bytes", "Network I/O bytes")
FD_GAUGE         = Gauge("open_file_descriptors", "Open file handles")

# ----- Load model -----
MODEL_URI = os.getenv("MODEL_URI", "models:/ALL_Model/Production")
model = mlflow.pyfunc.load_model(MODEL_URI)

# ----- Initialize feedback & drift DBs -----
for db in ("feedback.db", "drift.db"):
    conn = sqlite3.connect(db); c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS metrics (
        ts TIMESTAMP, key TEXT, value REAL
      )
    """)
    c.execute("""
      CREATE TABLE IF NOT EXISTS feedback (
        ts TIMESTAMP, image_id TEXT, correct_label TEXT
      )
    """) if db=="feedback.db" else None
    conn.commit(); conn.close()

# ----- Background Scheduler -----
sched = BackgroundScheduler()
sched.add_job(lambda: detect_drift("drift.db"), 'interval', hours=24)
sched.add_job(lambda: process_feedback_loop("feedback.db"), 'interval', hours=1)
sched.start()

# ----- Helpers -----
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224,224))
    # TODO: your exact preprocessing
    arr = ...  
    return arr

def postprocess(preds):
    # preds → label, confidence
    label = ...  
    conf  = ...  
    return label, conf

def record_input_metric(key, value):
    conn = sqlite3.connect("drift.db")
    conn.execute("INSERT INTO metrics VALUES (?,?,?)", (datetime.utcnow(), key, value))
    conn.commit(); conn.close()

# ----- Endpoints -----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    timer = REQUEST_LATENCY.time()
    try:
        data = await file.read()
        arr = preprocess(data)
        preds = model.predict(arr)
        label, conf = postprocess(preds)
        image_id = str(uuid.uuid4())
        # record drift feature—e.g. mean pixel intensity
        record_input_metric("mean_pixel", float(arr.mean()))
        return JSONResponse({
          "image_id": image_id,
          "label": label,
          "confidence": conf,
          # if you generate a segmented image on disk:
          "segmented_image_url": f"/segmented/{image_id}.jpg"
        })
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        timer.observe_duration()

@app.post("/feedback")
async def feedback(payload: dict):
    ts = datetime.utcnow()
    conn = sqlite3.connect("feedback.db")
    conn.execute(
      "INSERT INTO feedback VALUES (?,?,?)",
      (ts, payload["image_id"], payload["correct_label"])
    )
    conn.commit(); conn.close()
    return {"status":"ok"}

@app.get("/drift-status")
def drift_status():
    conn = sqlite3.connect("drift.db")
    cur = conn.cursor()
    cur.execute("SELECT value FROM metrics WHERE key='drift_flag' ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    flag = bool(row[0]) if row else False
    return {"drift": flag}

@app.get("/metrics")
def metrics():
    # update system metrics
    CPU_GAUGE.set(psutil.cpu_percent())
    MEMORY_GAUGE.set(psutil.virtual_memory().percent)
    disk = psutil.disk_io_counters(); DISK_IO_GAUGE.set(disk.read_bytes + disk.write_bytes)
    net = psutil.net_io_counters(); NETWORK_IO_GAUGE.set(net.bytes_sent + net.bytes_recv)
    FD_GAUGE.set(psutil.Process().num_fds())
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)
