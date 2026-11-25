"""Score trained pipelines with NumPy only, without importing scikit-learn.

The dashboard runs in 256 MiB, and importing scikit-learn alone costs about
70 MB, enough to get the container killed. ``export_pipeline`` turns a fitted
``preprocess -> model`` pipeline from train.py into a JSON file (preprocessing
parameters) plus an .npz file (tree arrays), and ``PortableModel`` reproduces
``predict_proba`` from them.

Supported: ColumnTransformer of [SimpleImputer + StandardScaler] numeric and
[SimpleImputer + OneHotEncoder(handle_unknown="ignore")] categorical steps,
followed by RandomForestClassifier, DecisionTreeClassifier,
GradientBoostingClassifier (binary) or LogisticRegression. Anything else
raises ValueError from ``export_pipeline``, so it cannot export silently and
then score wrongly.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_VERSION = 2  # 2: json/npz pair is tied together by a shared export_id


def _paths(prefix: Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    return prefix.with_suffix(".json"), prefix.with_suffix(".npz")


def remove_export(prefix: Path) -> None:
    """Delete an export so a stale one can't outlive the model it came from."""
    for path in _paths(prefix):
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Export (needs scikit-learn; run at training time)
# --------------------------------------------------------------------------- #

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_imputer(imputer, n_cols: int, block: str) -> None:
    _require(type(imputer).__name__ == "SimpleImputer", f"{block}: imputer must be a SimpleImputer")
    _require(
        isinstance(imputer.missing_values, float) and np.isnan(imputer.missing_values),
        f"{block}: SimpleImputer(missing_values=np.nan) is required",
    )
    _require(not imputer.add_indicator, f"{block}: SimpleImputer(add_indicator=True) is not supported")
    # Columns that were all-NaN at fit time are dropped (keep_empty_features=False),
    # which would shift every later feature.
    _require(len(imputer.statistics_) == n_cols, f"{block}: imputer dropped empty columns")


def _export_preprocessor(ct) -> dict:
    from sklearn.pipeline import Pipeline

    _require(type(ct).__name__ == "ColumnTransformer", "preprocess step must be a ColumnTransformer")
    blocks = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        _require(isinstance(trans, Pipeline), f"{name}: transformer must be a Pipeline, got {trans!r}")
        _require(not isinstance(cols, slice) and all(isinstance(c, str) for c in cols),
                 f"{name}: columns must be selected by name")
        if len(cols) == 0:
            continue
        steps = dict(trans.named_steps)
        if set(steps) == {"imputer", "scaler"}:
            imputer, scaler = steps["imputer"], steps["scaler"]
            _check_imputer(imputer, len(cols), name)
            _require(type(scaler).__name__ == "StandardScaler", f"{name}: scaler must be a StandardScaler")
            _require(scaler.with_mean and scaler.with_std, f"{name}: StandardScaler must center and scale")
            _require(not np.isnan(imputer.statistics_.astype(float)).any(), f"{name}: imputer has NaN fill values")
            blocks.append(
                {
                    "kind": "numeric",
                    "columns": list(cols),
                    "fill": imputer.statistics_.astype(float).tolist(),
                    "mean": scaler.mean_.astype(float).tolist(),
                    "scale": scaler.scale_.astype(float).tolist(),
                }
            )
        elif set(steps) == {"imputer", "onehot"}:
            imputer, onehot = steps["imputer"], steps["onehot"]
            _check_imputer(imputer, len(cols), name)
            _require(type(onehot).__name__ == "OneHotEncoder", f"{name}: encoder must be a OneHotEncoder")
            _require(
                onehot.handle_unknown == "ignore"
                and onehot.drop is None
                and onehot.min_frequency is None
                and onehot.max_categories is None,
                f"{name}: only OneHotEncoder(handle_unknown='ignore') without drop/min_frequency/"
                "max_categories is supported",
            )
            blocks.append(
                {
                    "kind": "onehot",
                    "columns": list(cols),
                    "fill": [str(v) for v in imputer.statistics_],
                    "categories": [[str(v) for v in cats] for cats in onehot.categories_],
                }
            )
        else:
            raise ValueError(f"{name}: unsupported steps {sorted(steps)}")
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

    steps = getattr(pipe, "named_steps", {})
    _require(set(steps) == {"preprocess", "model"}, "pipeline must be exactly preprocess -> model")
    pre, model = steps["preprocess"], steps["model"]
    meta: dict = {"format": FORMAT_VERSION, "preprocess": _export_preprocessor(pre), **(extra or {})}
    _require(list(getattr(model, "classes_", [])) == [0, 1], "only binary 0/1 classifiers are supported")

    if isinstance(model, RandomForestClassifier):
        meta["model"] = {"kind": "forest"}
        arrays = _pack_trees(list(model.estimators_))
    elif isinstance(model, DecisionTreeClassifier):
        meta["model"] = {"kind": "forest"}
        arrays = _pack_trees([model])
    elif isinstance(model, GradientBoostingClassifier):
        _require(model.init is None, "GradientBoostingClassifier(init=...) is not supported")
        _require(model.loss == "log_loss", "only GradientBoostingClassifier(loss='log_loss') is supported")
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

    # Write both files to temp names, then swap them in. A shared export_id lets
    # load() reject a json/npz pair from two different exports.
    export_id = uuid.uuid4().hex
    meta["export_id"] = export_id
    arrays = {**arrays, "export_id": np.array(export_id)}
    json_path, npz_path = _paths(prefix)
    json_tmp, npz_tmp = Path(f"{json_path}.tmp"), Path(f"{npz_path}.tmp")
    try:
        json_tmp.write_text(json.dumps(meta), encoding="utf-8")
        with open(npz_tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(npz_tmp, npz_path)
        os.replace(json_tmp, json_path)
    finally:
        json_tmp.unlink(missing_ok=True)
        npz_tmp.unlink(missing_ok=True)
    return json_path, npz_path


# --------------------------------------------------------------------------- #
# Scoring (NumPy + pandas only)
# --------------------------------------------------------------------------- #

def _category_key(value, fill: str) -> str | None:
    """Mirror SimpleImputer(missing_values=np.nan) + OneHotEncoder(handle_unknown="ignore").

    sklearn treats only NaN (and pandas NA, which it can't handle anyway) as
    missing, so those get the fill value. A Python ``None`` in an object column
    is not "missing" to SimpleImputer: it reaches the encoder as an unknown
    category and encodes as all zeros, so it returns None here.
    """
    if value is None:
        return None
    if value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return fill
    return str(value)


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
        npz_id = str(arrays.pop("export_id")) if "export_id" in arrays else None
        if meta.get("export_id") is None or npz_id != meta.get("export_id"):
            raise ValueError(f"{json_path.name} and {npz_path.name} come from different exports")
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
                    index = {c: i for i, c in enumerate(cats)}
                    onehot = np.zeros((n, len(cats)))
                    for row, v in enumerate(df[col].astype(object).to_numpy()):
                        key = _category_key(v, fill)
                        j = None if key is None else index.get(key)
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
