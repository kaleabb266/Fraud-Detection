"""Score trained pipelines with NumPy only, without importing scikit-learn.

The dashboard runs in 256 MiB, and importing scikit-learn alone costs about
70 MB, enough to get the container killed. ``export_pipeline`` turns a fitted
``preprocess -> model`` pipeline from train.py into a JSON file (preprocessing
parameters) plus an .npz file (tree arrays), and ``PortableModel`` reproduces
``predict_proba`` from them.

Supported: ColumnTransformer of [SimpleImputer + StandardScaler] numeric and
[SimpleImputer + OneHotEncoder(handle_unknown="ignore")] categorical steps,
followed by RandomForestClassifier, DecisionTreeClassifier,
GradientBoostingClassifier (binary) or LogisticRegression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_VERSION = 1


def _paths(prefix: Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    return prefix.with_suffix(".json"), prefix.with_suffix(".npz")


# --------------------------------------------------------------------------- #
# Export (needs scikit-learn; run at training time)
# --------------------------------------------------------------------------- #

def _export_preprocessor(ct) -> dict:
    blocks = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" or trans == "drop" or len(cols) == 0:
            continue
        steps = dict(trans.named_steps)
        imputer = steps["imputer"]
        if "scaler" in steps:
            scaler = steps["scaler"]
            blocks.append(
                {
                    "kind": "numeric",
                    "columns": list(cols),
                    "fill": imputer.statistics_.astype(float).tolist(),
                    "mean": scaler.mean_.astype(float).tolist(),
                    "scale": scaler.scale_.astype(float).tolist(),
                }
            )
        elif "onehot" in steps:
            onehot = steps["onehot"]
            if onehot.handle_unknown != "ignore" or onehot.drop is not None:
                raise ValueError("Only OneHotEncoder(handle_unknown='ignore', drop=None) is supported")
            blocks.append(
                {
                    "kind": "onehot",
                    "columns": list(cols),
                    "fill": [str(v) for v in imputer.statistics_],
                    "categories": [[str(v) for v in cats] for cats in onehot.categories_],
                }
            )
        else:
            raise ValueError(f"Unsupported transformer block: {name}")
    return {"blocks": blocks}


def _tree_arrays(tree) -> dict[str, np.ndarray]:
    t = tree.tree_
    return {
        "left": t.children_left.astype(np.int32),
        "right": t.children_right.astype(np.int32),
        "feature": t.feature.astype(np.int32),
        "threshold": t.threshold.astype(np.float64),
        "value": t.value[:, 0, :].astype(np.float64),  # (n_nodes, n_outputs/classes)
    }


def _pack_trees(trees: list) -> dict[str, np.ndarray]:
    parts = [_tree_arrays(t) for t in trees]
    offsets = np.cumsum([0] + [len(p["left"]) for p in parts]).astype(np.int64)
    return {
        "offsets": offsets,
        "left": np.concatenate([p["left"] for p in parts]),
        "right": np.concatenate([p["right"] for p in parts]),
        "feature": np.concatenate([p["feature"] for p in parts]),
        "threshold": np.concatenate([p["threshold"] for p in parts]),
        "value": np.concatenate([p["value"] for p in parts]),
    }


def export_pipeline(pipe, prefix: Path, extra: dict | None = None) -> tuple[Path, Path]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    pre, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    meta: dict = {"format": FORMAT_VERSION, "preprocess": _export_preprocessor(pre), **(extra or {})}
    if list(model.classes_) != [0, 1]:
        raise ValueError("Only binary 0/1 classifiers are supported")

    if isinstance(model, RandomForestClassifier):
        meta["model"] = {"kind": "forest"}
        arrays = _pack_trees(list(model.estimators_))
    elif isinstance(model, DecisionTreeClassifier):
        meta["model"] = {"kind": "forest"}
        arrays = _pack_trees([model])
    elif isinstance(model, GradientBoostingClassifier):
        dummy = np.zeros((1, model.n_features_in_))
        meta["model"] = {
            "kind": "boosting",
            "init": float(model._raw_predict_init(dummy)[0, 0]),
            "learning_rate": float(model.learning_rate),
        }
        arrays = _pack_trees(list(model.estimators_[:, 0]))
    elif isinstance(model, LogisticRegression):
        meta["model"] = {
            "kind": "linear",
            "coef": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0]),
        }
        arrays = {}
    else:
        raise ValueError(f"Unsupported model type: {type(model).__name__}")

    json_path, npz_path = _paths(prefix)
    json_path.write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


# --------------------------------------------------------------------------- #
# Scoring (NumPy + pandas only)
# --------------------------------------------------------------------------- #

class PortableModel:
    def __init__(self, meta: dict, arrays: dict[str, np.ndarray]):
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"Unsupported portable model format: {meta.get('format')}")
        self.meta = meta
        self.model_name = meta.get("model_name", "trained pipeline")
        self.arrays = arrays

    @classmethod
    def load(cls, prefix: Path) -> "PortableModel":
        json_path, npz_path = _paths(prefix)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        return cls(meta, arrays)

    @staticmethod
    def exists(prefix: Path) -> bool:
        return all(p.exists() for p in _paths(prefix))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for block in self.meta["preprocess"]["blocks"]:
            cols = block["columns"]
            if block["kind"] == "numeric":
                x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                fill = np.asarray(block["fill"])
                x = np.where(np.isnan(x), fill, x)
                parts.append((x - np.asarray(block["mean"])) / np.asarray(block["scale"]))
            else:
                n = len(df)
                for col, fill, cats in zip(cols, block["fill"], block["categories"]):
                    values = df[col].astype(object).where(df[col].notna(), fill).astype(str).to_numpy()
                    index = {c: i for i, c in enumerate(cats)}
                    onehot = np.zeros((n, len(cats)))
                    for row, v in enumerate(values):
                        j = index.get(v)
                        if j is not None:
                            onehot[row, j] = 1.0
                    parts.append(onehot)
        return np.hstack(parts)

    def _leaf_values(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf value row for every (sample, tree): (n, n_trees, k)."""
        a = self.arrays
        offsets, left, right = a["offsets"], a["left"], a["right"]
        feature, threshold, value = a["feature"], a["threshold"], a["value"]
        # sklearn trees compare float32-cast inputs against float64 thresholds.
        X32 = X.astype(np.float32).astype(np.float64)
        n_trees = len(offsets) - 1
        out = np.empty((X.shape[0], n_trees, value.shape[1]))
        for i in range(X.shape[0]):
            row = X32[i]
            for t in range(n_trees):
                base = offsets[t]
                node = 0
                while left[base + node] != -1:
                    if row[feature[base + node]] <= threshold[base + node]:
                        node = left[base + node]
                    else:
                        node = right[base + node]
                out[i, t] = value[base + node]
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transform(df)
        spec = self.meta["model"]
        if spec["kind"] == "forest":
            leaves = self._leaf_values(X)
            leaves = leaves / leaves.sum(axis=2, keepdims=True)
            p1 = leaves[:, :, 1].mean(axis=1)
        elif spec["kind"] == "boosting":
            raw = spec["init"] + spec["learning_rate"] * self._leaf_values(X)[:, :, 0].sum(axis=1)
            p1 = 1.0 / (1.0 + np.exp(-raw))
        elif spec["kind"] == "linear":
            raw = X @ np.asarray(spec["coef"]) + spec["intercept"]
            p1 = 1.0 / (1.0 + np.exp(-raw))
        else:
            raise ValueError(f"Unknown model kind: {spec['kind']}")
        return np.column_stack([1.0 - p1, p1])
