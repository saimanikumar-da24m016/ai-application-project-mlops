docker-compose -f docker-compose.dev.yml up --build -d





```
project-root/
├── airflow/                          # Airflow DAGs & setup
│   ├── Dockerfile
│   ├── requirements.txt
│   └── dags/
│       ├── data_ingestion_dag.py
│       └── model_training_dag.py
│
├── backend/                          # FastAPI + MLflow + Prometheus + Drift/Feedback
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── drift.py
│   └── feedback.py
│
├── frontend/                         # React UI
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── src/
│       ├── App.js
│       ├── api.js
│       └── components/
│           ├── Upload.js
│           ├── Result.js
│           ├── DriftAlert.js
│           └── Feedback.js
│
├── monitoring/                       # Prometheus + Grafana provisioning
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           └── datasources/
│               └── datasource.yml
│
├── scripts/                          # Reusable Python steps for Airflow/MLflow
│   ├── segment.py
│   └── train.py
│
├── data/                             # (mounted or versioned via DVC)
│   ├── raw/
│   └── processed/
│
├── models/                           # output of MLflow runs (can be a local mount)
│
├── logs/                             # any log files / Airflow, API, scheduler logs
│
├── docs/                             # Documentation deliverables
│   ├── architecture.png
│   ├── HLD.md
│   ├── LLD.md
│   ├── test_plan.xlsx
│   └── user_manual.pdf
│
├── MLproject                         # MLflow project entry points
├── conda.yaml                        # environment for MLflow projects
├── docker-compose.yml                # brings up frontend, backend, airflow, prometheus, grafana
└── README.md                         # overview + how to run everything




```