import sqlite3, subprocess
from datetime import datetime, timedelta

FEEDBACK_THRESH = 20  # e.g. retrain after 20 corrections

def process_feedback_loop(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cutoff = datetime.utcnow() - timedelta(hours=1)
    cur.execute(
      "SELECT COUNT(*) FROM feedback WHERE ts > ?", (cutoff,)
    )
    count = cur.fetchone()[0]
    if count >= FEEDBACK_THRESH:
        # trigger an MLflow retraining job
        subprocess.Popen([
          "mlflow", "run", ".", "-P", f"feedback_db={db_path}"
        ])
        # clear old feedback
        conn.execute("DELETE FROM feedback WHERE ts <= ?", (cutoff,))
        conn.commit()
    conn.close()
