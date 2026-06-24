# AI Banking Risk Intelligence Platform

A production-style **credit-risk prediction system** for retail banking, built to
showcase the full **ML engineering lifecycle**: data → model → experiment
tracking → API → containerisation, with a clear roadmap to Kubernetes,
monitoring, drift detection, and automated retraining.

> **Status:** MVP. The model, MLflow tracking, FastAPI service, and Docker setup
> are fully working today. The "Future Roadmap" section describes the
> production-hardening stages.

---

## Business Problem

Banks must decide whether to approve loans **quickly, consistently, and
defensibly**. Manual or rules-only underwriting is slow, inconsistent between
officers, and hard to audit. A poorly calibrated process leads to two costly
mistakes:

- **Approving bad borrowers** → loan defaults and direct financial loss.
- **Rejecting good borrowers** → lost revenue and poor customer experience.

This platform predicts the probability that an applicant is a **good** or **bad**
credit risk, returning a calibrated risk score and a simple risk band
(Low / Medium / High) that underwriters and downstream systems can act on.

## Why This Matters in Banking

- **Risk-adjusted decisions:** consistent, data-driven scoring across all branches.
- **Speed:** sub-second predictions enable instant pre-approvals.
- **Auditability & governance:** every model is versioned and every experiment is
  tracked (MLflow), which is essential for regulatory compliance (e.g. model risk
  management / SR 11-7, GDPR explainability expectations).
- **Operational safety:** drift detection and monitoring (roadmap) catch model
  degradation before it impacts the loan book.

---

## Architecture (text diagram)

```
                ┌──────────────────────────────────────────────────────────┐
                │                     TRAINING PIPELINE                      │
                │                                                            │
   data/  ──►   │  data_preprocessing.py ─► train_model.py ─► model.pkl      │
   *.csv        │            │                    │                          │
                │            └────────► MLflow ◄───┘  (params/metrics/model)  │
                └──────────────────────────────────────────────────────────┘
                                          │
                                          ▼  (saved artifact: credit_risk_model.pkl)
                ┌──────────────────────────────────────────────────────────┐
                │                     SERVING LAYER                          │
   Client  ──►  │   FastAPI (api/main.py)                                    │
   (JSON)       │     GET  /health        GET /model-info     POST /predict  │
                │              │                                             │
                │              └────────► loads model bundle (joblib)        │
                └──────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                          Docker / docker-compose (port 8000)
```

---

## Tech Stack

| Layer                | Technology                                  |
| -------------------- | ------------------------------------------- |
| Language             | Python 3.11                                 |
| Data / ML            | pandas, NumPy, scikit-learn, XGBoost        |
| Experiment tracking  | MLflow                                      |
| Model serialization  | joblib                                      |
| API                  | FastAPI + Uvicorn + Pydantic                |
| Testing              | pytest + httpx (FastAPI TestClient)         |
| Packaging            | Docker, docker-compose                      |

---

## Current MVP Features

- ✅ **Credit-risk model** (XGBoost, auto-fallback to RandomForest) with a
  realistic synthetic German-credit-style dataset generated automatically if no
  CSV is present.
- ✅ **MLflow experiment tracking** — logs parameters, metrics
  (accuracy, precision, recall, F1, ROC AUC), and the model artifact.
- ✅ **FastAPI service** with `/health`, `/model-info`, and `/predict`.
- ✅ **Automatic input validation** via Pydantic (categories + value ranges).
- ✅ **Dockerised** for one-command startup.
- ✅ **Tests** for the API contracts.

---

## Project Structure

```
ai-banking-risk-platform/
├── data/
│   └── german_credit_data.csv      # auto-generated if missing
├── src/
│   ├── data_preprocessing.py       # load, clean, encode, split
│   ├── train_model.py              # train + MLflow tracking + save model
│   ├── evaluate_model.py           # classification report + saved summary
│   └── utils.py                    # paths, schema, synthetic data generator
├── api/
│   ├── main.py                     # FastAPI app + endpoints
│   └── schemas.py                  # Pydantic request/response models
├── models/
│   └── credit_risk_model.pkl       # trained model bundle (created by training)
├── mlruns/                         # MLflow tracking data
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run Locally

### 1. Set up the environment

```bash
cd ~/Projects/ai-banking-risk-platform
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

This generates the dataset (if missing), trains the model, logs to MLflow, and
saves `models/credit_risk_model.pkl`.

```bash
python -m src.train_model
```

### 3. (Optional) Evaluate the model

```bash
python -m src.evaluate_model
```

### 4. (Optional) Explore experiments in the MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

### 6. Run the tests

```bash
pytest -q
```

---

## Example API Request

**Request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "job": 2,
    "credit_amount": 4500,
    "duration": 24,
    "housing": "own",
    "saving_accounts": "moderate",
    "checking_account": "moderate",
    "purpose": "car"
  }'
```

**Response**

```json
{
  "prediction": "good",
  "risk_score": 0.18,
  "risk_level": "Low"
}
```

---

## Run with Docker

```bash
# Build the image and start the service (model is trained during the build)
docker compose up --build

# API now available at http://localhost:8000
# Health check:
curl http://localhost:8000/health
```

---

## Future Roadmap

The MVP is intentionally focused. The following stages turn it into a fully
production-grade, governed AI system:

1. **Kubernetes deployment** — containerised API behind a Deployment + Service
   with liveness/readiness probes, horizontal pod autoscaling, and resource
   limits.
2. **Helm charts** — templated, environment-specific (dev/staging/prod) releases
   for repeatable, version-controlled deployments.
3. **GitHub Actions CI/CD** — automated lint, test, build, and image push on
   every commit; gated deploys to staging/production.
4. **Prometheus + Grafana monitoring** — request latency, throughput, error
   rates, and prediction distribution dashboards with alerting.
5. **Evidently AI drift detection** — scheduled data-drift and target-drift
   reports comparing live traffic to the training distribution.
6. **MLflow Model Registry** — promote models through Staging → Production
   stages; the API loads the current Production model instead of a local file.
7. **Automated retraining** — triggered by drift alerts or on a schedule
   (Airflow / cron), with automatic evaluation gates before promotion.
8. **Model governance & audit logs** — full lineage (data version, code version,
   model version), prediction logging, and explainability (SHAP) for regulatory
   compliance and audits.

---

## License

This is a portfolio/demonstration project. Use freely as a reference.
