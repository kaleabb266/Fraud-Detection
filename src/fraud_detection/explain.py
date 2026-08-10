from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .config import REPORTS_DIR
from .data import load_creditcard, load_ecommerce_enriched
from .features import creditcard_feature_target, ecommerce_feature_target


def load_features(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    if dataset == "ecommerce":
        return ecommerce_feature_target(load_ecommerce_enriched())
    if dataset == "creditcard":
        return creditcard_feature_target(load_creditcard())
    raise ValueError(f"Unknown dataset: {dataset}")


def explain(dataset: str, model_path: Path, sample_size: int = 300) -> None:
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    artifact = joblib.load(model_path)
    pipe = artifact["pipeline"]
    X, _ = load_features(dataset)
    sample = X.sample(min(sample_size, len(X)), random_state=42)
    transformed = pipe.named_steps["preprocess"].transform(sample)
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    model = pipe.named_steps["model"]

    out_dir = REPORTS_DIR / "explainability" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model.predict_proba, transformed)
    shap_values = explainer(transformed)
    shap.summary_plot(shap_values[:, :, 1], transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=160)
    plt.close()

    lime_explainer = LimeTabularExplainer(
        transformed.toarray() if hasattr(transformed, "toarray") else transformed,
        feature_names=list(feature_names),
        class_names=["legit", "fraud"],
        mode="classification",
    )
    row = transformed[0].toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
    lime_exp = lime_explainer.explain_instance(row, model.predict_proba, num_features=12)
    lime_exp.save_to_file(out_dir / "lime_example.html")
    print(f"Saved explainability outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ecommerce", "creditcard"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    args = parser.parse_args()
    explain(args.dataset, args.model, args.sample_size)


if __name__ == "__main__":
    main()

