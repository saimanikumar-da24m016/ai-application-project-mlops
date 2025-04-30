# dags/all_classification_pipeline.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

from scripts.preprocess       import resize_images
from scripts.prepare_data     import make_manifest
from scripts.train_classifier import train_classifier
from scripts.test_classifier  import test_classifier
from scripts.register_model   import register_model

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
    description="Preprocess → manifest → train → test → register for serving",
    schedule_interval="@daily",
    catchup=False,
    tags=["ALL", "classification"],
) as dag:

    start = DummyOperator(task_id="start")

    preprocess = PythonOperator(
        task_id="preprocess_images",
        python_callable=resize_images,
        op_kwargs={
            "input_dir": "/opt/data/raw/Original",
            "output_dir": "/opt/data/processed",
            "size": (224, 224),
        },
    )

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

    train = PythonOperator(
        task_id="train_classifier",
        python_callable=train_classifier,
        op_kwargs={
            "data_dir": "/opt/data/processed_manifest",
            "model_output": "/opt/models/classifier",
            "epochs": 5,
            "batch_size": 8,
        },
        # the returned run_id is pushed to XCom automatically
    )

    test = PythonOperator(
        task_id="test_classifier",
        python_callable=test_classifier,
        op_kwargs={
            "data_dir": "/opt/data/processed_manifest",
            "run_id": "{{ ti.xcom_pull(task_ids='train_classifier') }}",
            "batch_size": 8,
        },
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        op_kwargs={
            "run_id": "{{ ti.xcom_pull(task_ids='train_classifier') }}",
            "model_name": "ALL_Classifier",
        },
    )

    end = DummyOperator(task_id="end")

    start >> preprocess >> prepare >> train >> test >> register >> end
