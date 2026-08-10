import pandas as pd

from src.fraud_detection.features import add_ecommerce_features


def test_add_ecommerce_features_creates_time_and_velocity_fields():
    df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "signup_time": ["2015-01-01 00:00:00", "2015-01-01 00:00:00"],
            "purchase_time": ["2015-01-01 00:00:01", "2015-01-02 03:00:00"],
            "purchase_value": [10, 20],
            "device_id": ["a", "a"],
            "ip_address": [10.0, 10.0],
        }
    )
    out = add_ecommerce_features(df)
    assert out.loc[0, "is_instant_purchase"] == 1
    assert out.loc[1, "hour_of_day"] == 3
    assert out["user_transaction_count"].tolist() == [2, 2]
    assert out["device_user_count"].tolist() == [1, 1]

