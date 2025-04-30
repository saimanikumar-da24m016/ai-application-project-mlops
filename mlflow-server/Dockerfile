# mlflow-model-server/Dockerfile
FROM python:3.9-slim-bullseye

# --- system libraries TensorFlow needs ------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps ----------------------------------------------------------
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
        mlflow \
        numpy

# --------------------------------------------------------------------------
WORKDIR /app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENTRYPOINT ["mlflow", "models", "serve"]
CMD ["--model-uri=models:/ALL_Classifier/latest", "--no-conda", "--host=0.0.0.0", "--port=1234"]
