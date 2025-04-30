import requests
import numpy as np
from PIL import Image

# 1) pick any image from your dataset
img_path = "data/raw/Original/Benign/WBC-Benign-001.jpg"
img      = Image.open(img_path).convert("RGB").resize((224, 224))
arr      = np.asarray(img, dtype="float32") / 255.0        # shape (224,224,3)
payload  = {"inputs": [arr.tolist()]}                      # batch of 1

resp = requests.post("http://localhost:1234/invocations", json=payload, timeout=60)
resp.raise_for_status()
print("Prediction:", resp.json())
