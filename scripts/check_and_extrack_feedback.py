# scripts/check_and_extract_feedback.py

import os, io, sqlite3, csv
from PIL import Image

FEEDBACK_DB = os.getenv("FEEDBACK_DB", "/opt/data/feedback.db")
RAW_DIR     = os.getenv("RAW_DIR", "/opt/data/raw/Original")
FEED_DIR    = os.path.join(RAW_DIR, "feedback")  # new folder for feedback images
MANIFEST_CSV= os.path.join(RAW_DIR, "feedback_manifest.csv")
THRESH      = int(os.getenv("FEEDBACK_THRESH", "100"))

def check_and_extract_feedback(**context):
    os.makedirs(FEED_DIR, exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    (count,) = cur.fetchone()

    # push flag to XCom
    context['ti'].xcom_push(key='should_retrain', value=(count >= THRESH))

    if count >= THRESH:
        # Extract every row as image + label into FEED_DIR
        cur.execute("SELECT id,correct,img FROM feedback")
        rows = cur.fetchall()
        with open(MANIFEST_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path","label"])
            for rid, correct, blob in rows:
                img_path = os.path.join(FEED_DIR, f"fb_{rid}.jpg")
                with open(img_path, "wb") as imgf:
                    imgf.write(blob)
                writer.writerow([img_path, correct])
    conn.close()
    return count
