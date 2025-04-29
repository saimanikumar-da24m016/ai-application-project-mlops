import sqlite3
import numpy as np
from datetime import datetime, timedelta

# load your baseline distribution from file or DB
BASELINE_MEAN = 123.4  
THRESH = 10.0  # absolute difference threshold

def detect_drift(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cutoff = datetime.utcnow() - timedelta(hours=24)
    cur.execute(
      "SELECT value FROM metrics WHERE key='mean_pixel' AND ts > ?", (cutoff,)
    )
    vals = [r[0] for r in cur.fetchall()]
    drifted = False
    if vals:
        recent_mean = np.mean(vals)
        if abs(recent_mean - BASELINE_MEAN) > THRESH:
            drifted = True
    # store drift flag
    conn.execute(
      "INSERT INTO metrics VALUES (?,?,?)",
      (datetime.utcnow(), "drift_flag", float(drifted))
    )
    conn.commit(); conn.close()
