# Fraud Detection for E-Commerce and Bank Transactions

End-to-end fraud detection project for the 10 Academy Week 8&9 challenge. The project covers data cleaning, geolocation enrichment, feature engineering, model training, explainability, API serving, Docker packaging, and a Dash dashboard.

## Project Layout

```text
Data/                         Raw datasets supplied with the challenge
app/
  serve_model.py              Flask REST API for real-time predictions
  dashboard.py                Flask + Dash dashboard for fraud insights
src/fraud_detection/
  config.py                   Shared paths and constants
  data.py                     Data loading, cleaning, geolocation merge
  features.py                 Feature engineering and preprocessing helpers
  train.py                    Model training and experiment logging
  explain.py                  SHAP and LIME explainability utilities
scripts/
  run_eda.py                  Generates reports/eda_summary.md
reports/                      Generated analysis outputs
models/                       Trained model artifacts
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For the core training/API/dashboard workflow without heavier explainability and MLflow extras:

```bash
pip install -r requirements-core.txt
```

## Run EDA

```bash
python scripts/run_eda.py
```

The script writes `reports/eda_summary.md` with data quality checks, fraud rates, and high-level fraud patterns.

## Train Models

```bash
python -m src.fraud_detection.train --dataset ecommerce
python -m src.fraud_detection.train --dataset creditcard
```

By default, the trainer compares Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and MLP models. It saves the best model by average precision to `models/`.

To enable MLflow logging:

```bash
python -m src.fraud_detection.train --dataset ecommerce --use-mlflow
```

## Explain Models

```bash
python -m src.fraud_detection.explain --dataset ecommerce --model models/ecommerce_best_model.joblib
python -m src.fraud_detection.explain --dataset creditcard --model models/creditcard_best_model.joblib
```

Outputs are written under `reports/explainability/`.

## Serve API

```bash
python app/serve_model.py --dataset ecommerce --model models/ecommerce_best_model.joblib
```

Prediction endpoint:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"purchase_value\": 42, \"source\": \"SEO\", \"browser\": \"Chrome\", \"sex\": \"M\", \"age\": 31, \"country\": \"United States\", \"signup_time\": \"2015-01-01 10:00:00\", \"purchase_time\": \"2015-01-02 11:30:00\", \"user_id\": 1, \"device_id\": \"abc\", \"ip_address\": 12345}"
```

## Dashboard

A live deployment is available at [fraud-detection-dashboard-xatf.onrender.com](https://fraud-detection-dashboard-xatf.onrender.com/).

To run it locally:

```bash
python app/dashboard.py
```

Open `http://127.0.0.1:8050/`.

## Docker

```bash
docker build -t fraud-detection-dashboard .
docker run -p 8050:8050 fraud-detection-dashboard
```
