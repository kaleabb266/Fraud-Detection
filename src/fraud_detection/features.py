from __future__ import annotations

import numpy as np
import pandas as pd


ECOMMERCE_CATEGORICAL = ["source", "browser", "sex", "country"]
ECOMMERCE_DROP = [
    "class",
    "signup_time",
    "purchase_time",
    "device_id",
    "user_id",
    "ip_address",
    "ip_int",
    "lower_bound_ip_address",
    "upper_bound_ip_address",
]


def add_ecommerce_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signup_time"] = pd.to_datetime(out["signup_time"], errors="coerce")
    out["purchase_time"] = pd.to_datetime(out["purchase_time"], errors="coerce")
    elapsed = (out["purchase_time"] - out["signup_time"]).dt.total_seconds()
    out["seconds_since_signup"] = elapsed.clip(lower=0).fillna(0)
    out["hour_of_day"] = out["purchase_time"].dt.hour.fillna(-1).astype(int)
    out["day_of_week"] = out["purchase_time"].dt.dayofweek.fillna(-1).astype(int)
    out["is_instant_purchase"] = (out["seconds_since_signup"] <= 1).astype(int)
    out["user_transaction_count"] = out.groupby("user_id")["user_id"].transform("size")
    out["device_transaction_count"] = out.groupby("device_id")["device_id"].transform("size")
    out["device_user_count"] = out.groupby("device_id")["user_id"].transform("nunique")
    out["ip_transaction_count"] = out.groupby("ip_address")["ip_address"].transform("size")
    out["purchase_value_log"] = np.log1p(out["purchase_value"])
    return out


def ecommerce_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    featured = add_ecommerce_features(df)
    y = featured["class"].astype(int)
    X = featured.drop(columns=[c for c in ECOMMERCE_DROP if c in featured.columns], errors="ignore")
    return X, y


def creditcard_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    X = X.assign(amount_log=np.log1p(X["Amount"]), hour_of_day=(X["Time"] // 3600) % 24)
    return X, y

