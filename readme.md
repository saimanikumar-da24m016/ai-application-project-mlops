

# 🧠 Acute Lymphoblastic Leukemia (ALL) Classifier – MLOps Project

This project is an end-to-end MLOps pipeline that provides an interactive web-based classifier for Acute Lymphoblastic Leukemia (ALL) detection from WBC microscopy images. It integrates **FastAPI**, **Streamlit**, **Prometheus**, **Grafana**, **Airflow**, and **MLflow** for a full production-grade ML lifecycle.
```
docker-compose -f docker-compose.dev.yml up --build -d

docker-compose -f docker-compose.backend-frontend.yml up --build -d
```
---

## 🚀 Features

- 🖼️ Streamlit-based frontend for predictions and user feedback  
- 🔁 Feedback loop with Airflow for model retraining  
- ⚙️ FastAPI backend with Prometheus instrumentation  
- 📊 Real-time system + API monitoring with Grafana  
- 🔄 MLflow for experiment tracking & model serving  
- 🐳 Docker + Docker Compose setup for reproducibility  

---

## 🧩 Project Structure

