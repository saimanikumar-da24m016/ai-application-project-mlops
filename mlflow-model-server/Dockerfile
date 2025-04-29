FROM python:3.9-slim-bullseye

# Install MLflow CLI
RUN pip install --no-cache-dir mlflow

# Set working directory
WORKDIR /app

# Set entrypoint to mlflow CLI
ENTRYPOINT ["mlflow", "models", "serve"]

# Set default arguments passed to ENTRYPOINT
CMD ["--model-uri=runs:/ALL_Classifier/latest/model", "--host=0.0.0.0", "--port=1234"]
