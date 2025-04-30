FROM python:3.9-slim-bullseye

# Core serving stack
RUN pip install --no-cache-dir mlflow \
    && pip install --no-cache-dir tensorflow-cpu==2.15.0 pillow numpy

WORKDIR /app

# point to the tracking server inside the docker network
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# serve the "latest" version in the registry
ENTRYPOINT ["mlflow", "models", "serve"]
CMD ["--model-uri=models:/ALL_Classifier/latest", "--no-conda", "--host=0.0.0.0", "--port=1234"]
