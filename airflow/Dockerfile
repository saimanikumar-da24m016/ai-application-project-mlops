FROM apache/airflow:2.10.5-python3.9

USER root
# Copy and install all Python dependencies at image build time
COPY requirements-airflow.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Switch back to airflow user
USER airflow