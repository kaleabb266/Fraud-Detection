from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.config import REPORTS_DIR
from fraud_detection.data import load_creditcard, load_ecommerce_enriched
from fraud_detection.features import add_ecommerce_features


def section(title: str) -> str:
    return f"\n## {title}\n\n"


def table_text(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string() + "\n```"


def target_summary(df: pd.DataFrame, target: str) -> str:
    counts = df[target].value_counts().sort_index()
    fraud_rate = df[target].mean()
    return (
        f"- Rows: {len(df):,}\n"
        f"- Columns: {df.shape[1]:,}\n"
        f"- Duplicate rows: {df.duplicated().sum():,}\n"
        f"- Missing values: {int(df.isna().sum().sum()):,}\n"
        f"- Legitimate transactions: {counts.get(0, 0):,}\n"
        f"- Fraud transactions: {counts.get(1, 0):,}\n"
        f"- Fraud rate: {fraud_rate:.3%}\n"
    )


def ecommerce_patterns(df: pd.DataFrame) -> str:
    featured = add_ecommerce_features(df)
    browser = featured.groupby("browser")["class"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
    source = featured.groupby("source")["class"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
    country = featured.groupby("country")["class"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False).head(10)
    instant_rate = featured.groupby("is_instant_purchase")["class"].mean()
    return (
        "### Browser Fraud Pattern\n\n"
        + table_text(browser)
        + "\n\n### Source Fraud Pattern\n\n"
        + table_text(source)
        + "\n\n### Top Countries by Fraud Count\n\n"
        + table_text(country)
        + "\n\n### Instant Purchase Fraud Rate\n\n"
        + table_text(instant_rate.to_frame("fraud_rate"))
        + "\n"
    )


def creditcard_patterns(df: pd.DataFrame) -> str:
    amount = df.groupby("Class")["Amount"].describe()
    time_bins = pd.cut(df["Time"] / 3600, bins=24)
    by_time = df.assign(hour_bin=time_bins).groupby("hour_bin", observed=False)["Class"].agg(["count", "sum", "mean"])
    return (
        "### Amount Distribution by Class\n\n"
        + table_text(amount)
        + "\n\n### Fraud by Time Bin\n\n"
        + table_text(by_time)
        + "\n"
    )


def main() -> None:
    ecommerce = load_ecommerce_enriched()
    creditcard = load_creditcard()

    REPORTS_DIR.mkdir(exist_ok=True)
    report = "# Fraud Detection EDA Summary\n"
    report += section("E-Commerce Data Quality") + target_summary(ecommerce, "class")
    report += section("E-Commerce Fraud Patterns") + ecommerce_patterns(ecommerce)
    report += section("Credit Card Data Quality") + target_summary(creditcard, "Class")
    report += section("Credit Card Fraud Patterns") + creditcard_patterns(creditcard)

    output = REPORTS_DIR / "eda_summary.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
