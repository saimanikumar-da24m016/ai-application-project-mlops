# scripts/test_classifier.py

import os
import mlflow
import pandas as pd
import tensorflow as tf
import argparse
from scripts.train_classifier import parse_csv

def test_classifier(data_dir: str, run_id: str, batch_size: int = 32):
    """
    Loads the model from the given run_id, runs evaluate() on test.csv,
    and logs or prints the results.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.keras.load_model(model_uri)

    test_ds, _ = parse_csv(f"{data_dir}/test.csv")
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    loss, acc = model.evaluate(test_ds)
    print(f"🧪 Test loss={loss:.4f}, accuracy={acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test image classifier from MLflow")
    parser.add_argument("data_dir", help="dir containing test.csv")
    parser.add_argument("run_id", help="MLflow run ID where model is logged")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    test_classifier(args.data_dir, args.run_id, args.batch_size)
