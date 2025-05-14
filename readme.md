# 🧠 Acute Lymphoblastic Leukemia (ALL) Classifier – MLOps Project

**Repository:** [https://github.com/saimanikumar-da24m016/ai-application-project-mlops](https://github.com/saimanikumar-da24m016/ai-application-project-mlops)

**Team:** Sai Mani Kumar Devathi (DA24M016), Sathwik Pentela (DA24M017)

This project implements a complete MLOps pipeline for classifying Acute Lymphoblastic Leukemia (ALL) from white blood cell microscopy images. It integrates FastAPI, Streamlit, Prometheus, Grafana, Airflow, and MLflow to cover data ingestion, training, monitoring, and production deployment.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Quick Start](#quick-start)
6. [Component Details](#component-details)
7. [Monitoring & Logging](#monitoring--logging)
8. [Retraining Pipeline](#retraining-pipeline)
9. [License](#license)

---

## Features

* Streamlit frontend for image upload, prediction display, and feedback collection
* Airflow orchestration for ETL, training, and feedback-based retraining
* FastAPI backend with Prometheus metrics for inference endpoints
* Grafana dashboards for real-time system and ML performance
* MLflow for experiment tracking, model registry, and serving
* Docker Compose setups for development and combined backend/frontend stacks

---

## Tech Stack

| Layer              | Tool/Framework      |
| ------------------ | ------------------- |
| Frontend           | Streamlit           |
| API & Inference    | FastAPI             |
| Orchestration      | Apache Airflow      |
| Tracking & Serving | MLflow              |
| Monitoring         | Prometheus, Grafana |
| Containers         | Docker, Compose     |
| Data Storage       | Local filesystem    |

---

## Project Structure

```
.
├── airflow
│   ├── dags             # Airflow DAGs for ETL, training, retraining
│   ├── logs             # Task logs
│   └── mlruns           # Shared MLflow tracking
├── backend              # FastAPI app and inference logic
├── data
│   ├── raw              # Raw microscopy images
│   ├── processed        # Preprocessed data
│   └── processed_manifest
├── frontend_streamlit   # Streamlit app for user interaction
├── mlflow-server        # MLflow tracking server & registry
├── mlflow-model-server  # MLflow model serving
├── mlruns               # Local MLflow experiment data
├── models
│   └── classifier       # Saved model artifacts
├── monitoring
│   ├── prometheus       # Scrape configs
│   └── grafana          # Dashboard provisioning
├── scripts              # Utility scripts (data prep, evaluation)
└── docker-compose.*.yml # Dev and prod compose files
```

---

## Prerequisites

* Docker & Docker Compose (version 2+)
* Python 3.8+ (for local script runs)
* (Optional) Airflow installed if running outside Docker

---

## Quick Start

1. Clone the repo:

   ```bash
   git clone https://github.com/saimanikumar-da24m016/ai-application-project-mlops.git
   cd ai-application-project-mlops
   ```
2. Start development environment:

   ```bash
   docker-compose -f docker-compose.dev.yml up --build -d
   ```
3. Start backend & frontend:

   ```bash
   docker-compose -f docker-compose.backend-frontend.yml up --build -d
   ```
4. Access services:

   * Streamlit UI: [http://localhost:8501](http://localhost:8501)
   * FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   * MLflow UI: [http://localhost:5000](http://localhost:5000)
   * Prometheus: [http://localhost:9090](http://localhost:9090)
   * Grafana: [http://localhost:3000](http://localhost:3000)

---

## Component Details

* **Airflow**: DAGs automate data ingestion, training, validation, and feedback retraining. Config in `airflow/dags/`.
* **FastAPI**: Provides `/predict` and `/feedback` endpoints; instrumented for Prometheus metrics.
* **Streamlit**: User interface for uploading images, viewing predictions, and sending feedback.
* **MLflow**: Tracks experiments, registers models, and serves the production model.
* **Monitoring**: Prometheus scrapes metrics, Grafana visualizes dashboards.

---

## Retraining Pipeline

User feedback is periodically ingested via Airflow DAGs to enrich training data. New models are evaluated and promoted in MLflow based on performance.

