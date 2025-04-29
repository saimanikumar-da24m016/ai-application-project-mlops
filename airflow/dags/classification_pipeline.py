from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

# import your preprocessing and manifest generation and training
from scripts.preprocess import resize_images
from scripts.prepare_data import make_manifest
from scripts.train_classifier import train_classifier

def default_args():
    return {
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2025, 1, 1),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

with DAG(
    dag_id="all_classification_pipeline",
    default_args=default_args(),
    description="Preprocess raw images, build manifests, train classifier",
    schedule_interval="@daily",
    catchup=False,
    tags=["ALL", "classification"],
) as dag:

    start = DummyOperator(task_id="start")

    # 1️⃣ Preprocessing: resize raw images → data/processed
    preprocess = PythonOperator(
        task_id="preprocess_images",
        python_callable=resize_images,
        op_kwargs={
            "input_dir": "/opt/data/raw/Original",
            "output_dir": "/opt/data/processed",
            "size": (224, 224),
        },
    )

    # 2️⃣ Prepare data: generate train/val/test CSVs
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=make_manifest,
        op_kwargs={
            "raw_dir": "/opt/data/processed",
            "out_dir": "/opt/data/processed_manifest",
            "val_size": 0.1,
            "test_size": 0.1,
            "seed": 42,
        },
    )

    # 3️⃣ Train classifier: logs to MLflow and saves under models/classifier
    train = PythonOperator(
        task_id="train_classifier",
        python_callable=train_classifier,
        op_kwargs={
            "data_dir": "/opt/data/processed_manifest",
            "model_output": "/opt/models/classifier",
            "epochs": 3,
            "batch_size": 8,
        },
    )


    end = DummyOperator(task_id="end")

    start >> preprocess >> prepare >> train >> end
