from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.features import add_ecommerce_features


def create_app(model_path: Path, dataset: str) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    artifact = joblib.load(model_path)
    pipe = artifact["pipeline"]
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "dataset": dataset, "model": artifact.get("model_name")})

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True, silent=False)
        records = payload if isinstance(payload, list) else [payload]
        frame = pd.DataFrame(records)
        if dataset == "ecommerce":
            frame = add_ecommerce_features(frame)
        probabilities = pipe.predict_proba(frame)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        response = [
            {"fraud_probability": float(prob), "prediction": int(pred)}
            for prob, pred in zip(probabilities, predictions)
        ]
        app.logger.info("prediction_count=%s fraud_predictions=%s", len(response), int(predictions.sum()))
        return jsonify(response if isinstance(payload, list) else response[0])

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ecommerce", "creditcard"], default="ecommerce")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "ecommerce_best_model.joblib")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app = create_app(args.model, args.dataset)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

