from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CREDITCARD_FULL_ZIP_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .data import load_creditcard, load_ecommerce_enriched
from .portable import export_pipeline
from .features import (
    ECOMMERCE_CATEGORICAL,
    creditcard_feature_target,
    ecommerce_feature_target,
)


def build_preprocessor(X: pd.DataFrame, categorical: list[str]) -> ColumnTransformer:
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=14, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, random_state=RANDOM_STATE),
    }


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if name == "ecommerce":
        X, y = ecommerce_feature_target(load_ecommerce_enriched())
        return X, y, [c for c in ECOMMERCE_CATEGORICAL if c in X.columns]
    if name == "creditcard":
        # Prefer the full 284,807-row dataset (the same data seeded into Postgres).
        if CREDITCARD_FULL_ZIP_PATH.exists():
            card = pd.read_csv(CREDITCARD_FULL_ZIP_PATH).drop_duplicates().reset_index(drop=True)
        else:
            card = load_creditcard()
        X, y = creditcard_feature_target(card)
        return X, y, []
    raise ValueError(f"Unknown dataset: {name}")


def evaluate_pipeline(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }


def train(
    dataset: str,
    use_mlflow: bool = False,
    model_names: list[str] | None = None,
    sample_size: int | None = None,
) -> Path:
    X, y, categorical = load_dataset(dataset)
    if sample_size and sample_size < len(X):
        sampled_indices = []
        for label, group_indices in y.groupby(y).groups.items():
            class_target = max(2, int(round(sample_size * len(group_indices) / len(X))))
            class_target = min(len(group_indices), class_target)
            sampled = pd.Index(group_indices).to_series().sample(class_target, random_state=RANDOM_STATE)
            sampled_indices.extend(sampled.tolist())
        X = X.loc[sampled_indices].reset_index(drop=True)
        y = y.loc[sampled_indices].astype(int).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    preprocessor = build_preprocessor(X, categorical)
    results = {}
    best_name = None
    best_score = -1.0
    best_pipe = None

    mlflow = None
    if use_mlflow:
        import mlflow as mlflow_module

        mlflow = mlflow_module
        mlflow.set_experiment(f"fraud_detection_{dataset}")

    candidates = candidate_models()
    if model_names:
        candidates = {name: candidates[name] for name in model_names}

    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        if mlflow:
            with mlflow.start_run(run_name=name):
                pipe.fit(X_train, y_train)
                metrics = evaluate_pipeline(pipe, X_test, y_test)
                mlflow.log_params({"dataset": dataset, "model": name})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                mlflow.sklearn.log_model(pipe, name)
        else:
            pipe.fit(X_train, y_train)
            metrics = evaluate_pipeline(pipe, X_test, y_test)
        results[name] = metrics
        if metrics["average_precision"] > best_score:
            best_name = name
            best_score = metrics["average_precision"]
            best_pipe = pipe

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{dataset}_best_model.joblib"
    joblib.dump({"dataset": dataset, "model_name": best_name, "pipeline": best_pipe, "features": X.columns.tolist()}, model_path)
    (REPORTS_DIR / f"{dataset}_model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {best_name} model to {model_path}")
    try:
        json_path, _ = export_pipeline(
            best_pipe,
            MODELS_DIR / f"{dataset}_model_portable",
            extra={"dataset": dataset, "model_name": best_name, "features": X.columns.tolist()},
        )
        print(f"Saved NumPy-only export to {json_path.with_suffix('')}.{{json,npz}}")
    except ValueError as exc:
        print(f"Skipped NumPy-only export: {exc}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ecommerce", "creditcard"], required=True)
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(candidate_models().keys()))
    parser.add_argument("--sample-size", type=int)
    args = parser.parse_args()
    train(args.dataset, use_mlflow=args.use_mlflow, model_names=args.models, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
