from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CREDITCARD_PATH, FRAUD_DATA_PATH, IP_COUNTRY_PATH


def load_ecommerce() -> pd.DataFrame:
    df = pd.read_csv(FRAUD_DATA_PATH)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
    return df.drop_duplicates().reset_index(drop=True)


def load_ip_country() -> pd.DataFrame:
    return pd.read_csv(IP_COUNTRY_PATH).drop_duplicates().sort_values("lower_bound_ip_address")


def load_creditcard() -> pd.DataFrame:
    return pd.read_csv(CREDITCARD_PATH).drop_duplicates().reset_index(drop=True)


def add_country(ecommerce: pd.DataFrame, ip_country: pd.DataFrame) -> pd.DataFrame:
    """Map each numeric IP address to its country range using an asof join."""
    tx = ecommerce.copy()
    ranges = ip_country.copy()
    tx["_row_id"] = np.arange(len(tx))
    tx["ip_int"] = np.floor(tx["ip_address"]).astype("int64")
    ranges["lower_bound_ip_address"] = ranges["lower_bound_ip_address"].astype("int64")
    ranges["upper_bound_ip_address"] = ranges["upper_bound_ip_address"].astype("int64")

    merged = pd.merge_asof(
        tx.sort_values("ip_int"),
        ranges.sort_values("lower_bound_ip_address"),
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )
    in_range = merged["ip_int"].le(merged["upper_bound_ip_address"])
    merged.loc[~in_range, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")
    return merged.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)


def load_ecommerce_enriched() -> pd.DataFrame:
    return add_country(load_ecommerce(), load_ip_country())
