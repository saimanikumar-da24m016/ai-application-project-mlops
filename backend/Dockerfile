FROM python:3.9-slim-bullseye

# system libs Pillow / numpy use
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# -------- requirements file lives INSIDE backend folder ----------
COPY backend/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt
# -----------------------------------------------------------------

WORKDIR /app
COPY backend /app

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
