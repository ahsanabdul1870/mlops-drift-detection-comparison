# MLOps Drift Detection Comparison Report

## Abstract

This project implements an end-to-end MLOps workflow for detecting data drift, comparing detector behavior, and monitoring system health for the Adult Income classification task. The pipeline generates clean and synthetic drifted datasets, trains and serves a Random Forest model, benchmarks three drift detectors (KS, PSI, and Jensen-Shannon), and exposes runtime metrics through FastAPI, Prometheus, Grafana, and MLflow. The main goal is to understand how detector sensitivity changes across drift levels and to show how drift detection can be operationalized into a monitoring and retraining workflow.

## 1. Introduction And Motivation

Data drift is one of the most common causes of model performance degradation in production machine learning systems. A model that performs well during development may become unreliable when the incoming data distribution changes over time. This project addresses that problem by building a practical drift monitoring pipeline around the Adult Income dataset.

The motivation for the project is threefold. First, we want to detect drift early before it affects model quality. Second, we want to compare different drift detectors under controlled conditions so that their relative strengths and weaknesses are visible. Third, we want to connect drift detection to retraining and observability tools so the entire workflow can be monitored and reproduced.

## 2. System Architecture And Workflow

The system is organized as a simple but complete MLOps loop. The workflow is:

1. Generate clean and drifted datasets.
2. Train a baseline model on the clean data.
3. Run drift detectors on clean, light, medium, and heavy drift datasets.
4. Persist monitoring state for downstream services.
5. Serve predictions through a FastAPI application.
6. Export metrics through the `/metrics` endpoint for Prometheus scraping.
7. Visualize drift and monitoring metrics in Grafana.
8. Trigger retraining when drift is detected in CI/CD.

The main code modules that implement this flow are:

- `src/drift_simulator.py` for generating synthetic drift datasets.
- `src/train.py` for model training and MLflow logging.
- `src/drift_detector.py` for KS, PSI, and JS drift experiments.
- `src/ci_drift_guard.py` for CI-based drift checks and retraining decisions.
- `src/retrain.py` for drift-triggered retraining.
- `src/monitor.py` for shared runtime state storage.
- `src/api.py` for serving the model and exposing Prometheus metrics.

### Architecture Summary

| Component | Purpose |
| --- | --- |
| Dataset generation | Create clean and drifted versions of Adult Income data |
| Model training | Train and evaluate a Random Forest classifier |
| Drift detection | Measure drift with KS, PSI, and JS methods |
| Monitoring state | Store accuracy, drift scores, alert counts, and timing data |
| API service | Serve predictions and expose metrics |
| Prometheus | Scrape application metrics from the API |
| Grafana | Visualize drift, accuracy, alerts, and detector comparison |
| MLflow | Track experiments, parameters, and metrics |
| CI/CD | Run drift checks and trigger retraining when necessary |

## 3. Dataset Preparation And Drift Simulation

The project uses the Adult Income dataset as the base source. The drift simulator creates one clean dataset and three drifted variants:

- `data_clean.csv`
- `data_drift_light.csv`
- `data_drift_medium.csv`
- `data_drift_heavy.csv`

The drift generator applies both numeric and categorical changes. Numeric features are shifted by a controlled offset and noise, while categorical features are probabilistically changed to alternate categories. This design gives the project a controlled way to test detector behavior at different levels of distribution shift.

The drift levels are intentionally simple and interpretable:

- light drift: small fraction of rows modified
- medium drift: moderate fraction of rows modified
- heavy drift: large fraction of rows modified

This setup allows direct comparison of detector sensitivity under increasing shift intensity.

## 4. Model Training Setup

The baseline model is implemented in `src/train.py`. The training pipeline uses a `RandomForestClassifier` with preprocessing applied through a scikit-learn pipeline.

### Training Pipeline

- Numeric features use median imputation.
- Categorical features use most-frequent imputation followed by one-hot encoding.
- The data is split into train and test sets using a stratified split.
- The model is trained on the clean dataset and evaluated on a held-out test split.

### Logged Metrics

The training script logs the following metrics to MLflow:

- accuracy
- precision
- recall
- F1 score

The trained pipeline is also saved to `mlflow/serving_model`, which is later loaded by the FastAPI service.

## 5. Drift Detection Methods And Threshold Logic

The project compares three classical drift detectors in `src/drift_detector.py`.

### KS Detector

The Kolmogorov-Smirnov detector compares numeric feature distributions and flags drift when any feature has a p-value below `0.05`. The detector score is the mean KS statistic across common numeric columns.

### PSI Detector

The Population Stability Index detector measures how much a feature distribution changes relative to the reference data. In this project, drift is detected when the average PSI score exceeds `0.1`.

### JS Detector

The Jensen-Shannon detector computes distribution distance on histogram-based probabilities. Drift is detected when the average JS distance exceeds `0.1`.

### Threshold Logic Used In The Project

| Detector | Score Meaning | Drift Rule |
| --- | --- | --- |
| KS | Mean KS statistic over numeric features | Any feature p-value < 0.05 |
| PSI | Mean PSI across numeric features | Average PSI > 0.1 |
| JS | Mean Jensen-Shannon distance | Average JS distance > 0.1 |

The CI version in `src/ci_drift_guard.py` runs the same detector logic against a selected dataset and records whether retraining should be triggered.

## 6. Experimental Results

The project produced real detector outputs by running `python src/drift_detector.py`. The results below summarize the observed scores and detection behavior.

### Detector Results Across Drift Levels

| Detector | Clean | Light | Medium | Heavy | Detection Pattern |
| --- | ---: | ---: | ---: | ---: | --- |
| KS | 0.0000 | 0.0401 | 0.1128 | 0.2922 | Detects light, medium, and heavy drift |
| PSI | 0.0000 | 0.0005 | 0.0330 | 0.3215 | Detects heavy drift only |
| JS | 0.0000 | 0.0182 | 0.1300 | 0.2737 | Detects medium and heavy drift |

### Interpretation Of Results

- KS is the most sensitive detector in this setup because it reacts even at light drift.
- PSI is the most conservative detector because it only exceeds its threshold at heavy drift.
- JS sits between KS and PSI and begins flagging at medium drift.

### Suggested Graphs For The Report

The following visualizations are already supported by the project dashboard and should be included in the final report if possible:

- drift score over time
- accuracy degradation curve
- detector comparison panel
- alert timeline

These plots are already visible in Grafana once the API, Prometheus, and detector pipeline are running.

## 7. Monitoring Stack Setup: FastAPI, Prometheus, Grafana, MLflow

The project includes a complete monitoring stack.

### FastAPI

The API in `src/api.py` exposes:

- `GET /health` for service health checks
- `POST /predict` for inference
- `GET /metrics` for Prometheus scraping

The `/metrics` endpoint exports request counters, prediction latency, drift scores, accuracy, alert count, detector timing, and retraining trigger timing.

### Prometheus

Prometheus is configured in `prometheus/prometheus.yml` to scrape the API at `api:8000/metrics` every 15 seconds.

### Grafana

Grafana is configured to use Prometheus as the default datasource and loads dashboards automatically from the provisioning files. The dashboard shows the drift score trend, model accuracy, detector comparison, and alert activity.

### MLflow

MLflow is used to log:

- training runs
- detector experiments
- CI drift guard results
- retraining runs

This gives the project experiment tracking and reproducibility across the full pipeline.

## 8. Discussion: Most Sensitive And Most Conservative Detector

Based on the observed results, KS is the most sensitive detector and PSI is the most conservative.

### Why KS Is Most Sensitive

KS reacts to feature-wise distribution differences and can detect smaller changes in the sample distributions. In the experiment, it flagged drift at the light level, which shows that it is suitable for early warning.

### Why PSI Is Most Conservative

PSI aggregates the feature shift in a more coarse way and uses a threshold that requires stronger movement before signaling drift. In this project, PSI only exceeded the threshold under heavy drift, which makes it less sensitive but potentially less noisy.

### Why JS Is In The Middle

JS distance captures distribution divergence using histogram probabilities. It is more responsive than PSI in this setup but less aggressive than KS, so it offers a middle ground between early detection and stability.

## 9. Limitations And Future Work

This project is a strong demonstration of drift monitoring, but it has several limitations.

### Limitations

- Only one dataset domain is used.
- Drift is synthetic rather than fully observed from real production traffic.
- Thresholds are heuristic and manually chosen.
- Only three drift detectors are compared.
- The system does not yet perform adaptive threshold tuning.
- The retraining logic is triggered by drift detection but is not a full online learning system.

### Future Work

- Evaluate the pipeline on additional datasets.
- Add real production data streams and delayed labels.
- Compare more detectors and ensemble strategies.
- Learn thresholds dynamically from historical behavior.
- Add automatic model version rollback when retraining underperforms.
- Expand dashboards with detector-specific precision and recall over time.

## 10. Conclusion

This project demonstrates a complete MLOps pipeline for drift detection, monitoring, and retraining. The results show that detector choice strongly affects sensitivity: KS is the earliest detector, PSI is the most conservative, and JS is balanced between them. By combining synthetic drift generation, model training, detector benchmarking, FastAPI serving, Prometheus metrics, Grafana visualization, and MLflow tracking, the project provides a reproducible framework for studying model robustness under distribution shift.

The main takeaway is that drift detection should not be treated as a single metric or tool. It is better understood as a monitored workflow that connects data validation, detector behavior, operational telemetry, and retraining decisions into one system.

## Appendix: Key Commands Used In The Project

```bash
python src/drift_simulator.py
python src/train.py
python src/drift_detector.py
python src/ci_drift_guard.py
python src/retrain.py
uvicorn src.api:app --host 0.0.0.0 --port 8000
docker compose -f docker/docker-compose.yml up -d --build
```
