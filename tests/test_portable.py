import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.fraud_detection.portable import PortableModel, export_pipeline, remove_export

NUMERIC = ["amount", "age"]
CATEGORICAL = ["browser", "sex"]


def make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "amount": rng.gamma(2.0, 30.0, n),
            "age": rng.integers(18, 70, n).astype(float),
            "browser": rng.choice(["Chrome", "Safari", "FireFox", "IE"], n),
            "sex": rng.choice(["M", "F"], n),
        }
    )
    logit = 0.03 * (df["amount"] - 60) + (df["browser"] == "IE") * 1.5 - (df["sex"] == "F") * 0.7
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, y


def make_pipeline(model, scaler=None, onehot=None):
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", scaler or StandardScaler())]), NUMERIC),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", onehot or OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", model)])


MODELS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=15, max_depth=6, random_state=0),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=0),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=30, random_state=0),
    "logistic_regression": lambda: LogisticRegression(max_iter=500),
}


def edge_frames(df):
    """Inputs where a NumPy re-implementation can drift from sklearn."""
    base = df.head(5).reset_index(drop=True)

    def one_row(**changes):
        row = base.head(1).copy().astype(object)
        for col, value in changes.items():
            row.at[0, col] = value
        return row

    multi = base.copy().astype(object)
    multi.at[0, "sex"] = None
    multi.at[1, "browser"] = np.nan
    multi.at[2, "browser"] = "Opera"  # unseen category
    multi.at[3, "age"] = np.nan
    return {
        "plain": base,
        "one row, sex=None": one_row(sex=None),
        "one row, sex=NaN": one_row(sex=np.nan),
        "one row, all categories None": one_row(sex=None, browser=None),
        "one row, unseen browser": one_row(browser="Opera"),
        "one row, age=NaN": one_row(age=np.nan),
        "one row, age=None": one_row(age=None),
        "multi row, mixed blanks": multi,
    }


@pytest.mark.parametrize("name", sorted(MODELS))
def test_portable_matches_sklearn(tmp_path, name):
    df, y = make_data()
    pipe = make_pipeline(MODELS[name]()).fit(df, y)
    export_pipeline(pipe, tmp_path / name, extra={"model_name": name})
    portable = PortableModel.load(tmp_path / name)
    assert portable.model_name == name

    for label, frame in edge_frames(df).items():
        expected = pipe.predict_proba(frame)[:, 1]
        got = portable.predict_proba(frame)[:, 1]
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12, err_msg=f"{name}: {label}")


@pytest.mark.parametrize(
    "pipe_factory",
    [
        lambda: make_pipeline(LogisticRegression(), scaler=StandardScaler(with_mean=False)),
        lambda: make_pipeline(LogisticRegression(), scaler=StandardScaler(with_std=False)),
        lambda: make_pipeline(LogisticRegression(), onehot=OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                                          min_frequency=5)),
        lambda: make_pipeline(LogisticRegression(), onehot=OneHotEncoder(handle_unknown="ignore", max_categories=2)),
        lambda: make_pipeline(LogisticRegression(), onehot=OneHotEncoder(handle_unknown="error")),
        lambda: make_pipeline(MLPClassifier(max_iter=50, random_state=0)),
    ],
    ids=["no-mean", "no-std", "min_frequency", "max_categories", "handle_unknown=error", "mlp"],
)
def test_unsupported_pipelines_are_rejected(tmp_path, pipe_factory):
    df, y = make_data()
    pipe = pipe_factory().fit(df, y)
    with pytest.raises(ValueError):
        export_pipeline(pipe, tmp_path / "model")
    assert not (tmp_path / "model.json").exists()
    assert not (tmp_path / "model.npz").exists()


def test_mismatched_json_and_npz_are_rejected(tmp_path):
    df, y = make_data()
    forest = make_pipeline(MODELS["random_forest"]()).fit(df, y)
    boosting = make_pipeline(MODELS["gradient_boosting"]()).fit(df, y)
    export_pipeline(forest, tmp_path / "a")
    export_pipeline(boosting, tmp_path / "b")
    (tmp_path / "a.json").write_text((tmp_path / "b.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="different exports"):
        PortableModel.load(tmp_path / "a")


def test_old_format_without_export_id_is_rejected(tmp_path):
    df, y = make_data()
    export_pipeline(make_pipeline(MODELS["decision_tree"]()).fit(df, y), tmp_path / "m")
    meta = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    meta.pop("export_id")
    meta["format"] = 2
    (tmp_path / "m.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        PortableModel.load(tmp_path / "m")


def test_remove_export(tmp_path):
    df, y = make_data()
    export_pipeline(make_pipeline(MODELS["decision_tree"]()).fit(df, y), tmp_path / "m")
    assert PortableModel.exists(tmp_path / "m")
    remove_export(tmp_path / "m")
    assert not PortableModel.exists(tmp_path / "m")
    remove_export(tmp_path / "m")  # no error when already gone
