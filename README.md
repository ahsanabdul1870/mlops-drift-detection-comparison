
# MLOps Drift Detection Comparison (Adult Income)

End-to-end MLOps project for:

- baseline model training and tracking with MLflow
- synthetic drift generation (light, medium, heavy)
- detector benchmarking (KS, PSI, JS)
- FastAPI model serving
- Prometheus metrics collection
- Grafana observability dashboards
- GitHub Actions CI/CD with drift-triggered retraining

## 1. What This Project Does

This project uses the UCI Adult Income dataset and compares drift detectors under controlled drift levels.

You can:

1. Train a baseline Random Forest model on clean data.
2. Generate drifted datasets (10%, 30%, 60%).
3. Run KS/PSI/JS drift experiments and log all results to MLflow.
4. Serve model predictions via FastAPI at /predict.
5. Export operational metrics via /metrics for Prometheus.
6. Visualize drift, accuracy degradation, detector performance, and alerts in Grafana.
7. Use CI/CD to run tests, build image, run drift checks, and auto-trigger retraining.

## 2. Project Structure

```text
.
├── data/
│   ├── adult.csv
│   ├── data_clean.csv
│   ├── data_drift_light.csv
│   ├── data_drift_medium.csv
│   └── data_drift_heavy.csv
├── src/
│   ├── api.py                 # FastAPI service (/predict, /metrics, /health)
│   ├── ci_drift_guard.py      # CI drift check + outputs for GitHub Actions
│   ├── drift_detector.py      # KS, PSI, JS detectors + MLflow logging
│   ├── drift_simulator.py     # Generates clean/light/medium/heavy datasets
│   ├── monitor.py             # Runtime monitoring state helper
│   ├── retrain.py             # Drift-triggered retraining entrypoint
│   └── train.py               # Baseline and reusable training pipeline
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   ├── dashboards/
│   └── provisioning/
├── mlflow/
│   ├── mlruns/
│   ├── monitoring/
│   └── serving_model/
├── tests/
├── .github/workflows/
│   └── ci_cd.yml
├── requirements.txt
└── README.md
```

## 3. Prerequisites

Local:

- Python 3.11+
- pip
- virtual environment support
- Docker + Docker Compose

Optional but useful:

- curl
- jq

## 4. Quick Start (Local Python)

From project root:

```bash
cd /home/ahsan/Videos/mlops/project

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Generate datasets:

```bash
python src/drift_simulator.py
```

Train baseline model and log benchmark:

```bash
python src/train.py
```

Run detector experiments across all drift levels:

```bash
python src/drift_detector.py
```

Run tests:

```bash
pytest tests -q
```

## 5. Start the API (Local)

The API serves the current model from mlflow/serving_model.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- GET http://localhost:8000/health
- POST http://localhost:8000/predict
- GET http://localhost:8000/metrics

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
	-H "Content-Type: application/json" \
	-d '[
		{
			"age": 39,
			"workclass": "Private",
			"fnlwgt": 77516,
			"education": "Bachelors",
			"educational-num": 13,
			"marital-status": "Never-married",
			"occupation": "Adm-clerical",
			"relationship": "Not-in-family",
			"race": "White",
			"gender": "Male",
			"capital-gain": 2174,
			"capital-loss": 0,
			"hours-per-week": 40,
			"native-country": "United-States"
		}
	]'
```

## 6. Run the Whole Stack (Docker + Prometheus + Grafana)

This starts API, Prometheus, and Grafana together.

```bash
cd /home/ahsan/Videos/mlops/project
docker compose -f docker/docker-compose.yml up -d --build
```

Services:

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
	- username: admin
	- password: admin

Stop stack:

```bash
docker compose -f docker/docker-compose.yml down
```

## 7. MLflow Tracking

Runs are logged in mlflow/mlruns.

Start MLflow UI locally:

```bash
mlflow ui --backend-store-uri mlflow/mlruns --host 0.0.0.0 --port 5000
```

Open: http://localhost:5000

Key experiments:

- adult_income_baseline
- adult_income_drift_detection
- adult_income_ci_drift_guard
- adult_income_retraining

## 8. Grafana Dashboard Panels

Pre-provisioned dashboard includes:

1. Drift score over time.
2. Accuracy degradation curve.
3. Detector comparison panel.
4. Alert timeline.

Prometheus scrape interval is set to 15 seconds in [prometheus/prometheus.yml](prometheus/prometheus.yml).

## 9. CI/CD Flow (GitHub Actions)

Workflow file: [.github/workflows/ci_cd.yml](.github/workflows/ci_cd.yml)

On push/PR:

1. Run unit tests.
2. Build Docker image.
3. Run drift guard checks.
4. If drift is detected:
	 - trigger retraining,
	 - log retraining run to MLflow,
	 - redeploy API container,
	 - log retraining trigger timing.

## 10. Recommended Run Order

If you are running everything from scratch:

1. python src/drift_simulator.py
2. python src/train.py
3. python src/drift_detector.py
4. uvicorn src.api:app --host 0.0.0.0 --port 8000
5. docker compose -f docker/docker-compose.yml up -d --build
6. mlflow ui --backend-store-uri mlflow/mlruns --host 0.0.0.0 --port 5000

## 11. Notes

- If /predict fails, retrain first to ensure mlflow/serving_model exists.
- Monitor state is persisted at mlflow/monitoring/state.json.
- CI drift report is written to mlflow/monitoring/ci_drift_report.json.

