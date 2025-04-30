# scripts/register_model.py

import os
import mlflow
import argparse

def register_model(run_id: str, model_name: str = "ALL_Classifier"):
    """
    Takes a run_id, points to runs:/{run_id}/model, and registers it.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    print(f"✅ Registered model '{mv.name}' version {mv.version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Register MLflow model for serving")
    parser.add_argument("run_id", help="MLflow run ID where model is logged")
    parser.add_argument("--model_name", default="ALL_Classifier")
    args = parser.parse_args()
    register_model(args.run_id, args.model_name)
