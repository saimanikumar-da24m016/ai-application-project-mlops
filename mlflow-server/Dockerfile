FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlruns/mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]