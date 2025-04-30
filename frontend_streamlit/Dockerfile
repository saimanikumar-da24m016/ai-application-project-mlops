FROM python:3.9-slim-bullseye

# copy dependency list and install
COPY requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# app code
WORKDIR /app
COPY . /app

EXPOSE 8501
ENV BACKEND_URL=http://backend:8000/predict
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
