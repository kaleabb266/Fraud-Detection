from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from flask import Flask, jsonify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection import data as csv_data
from fraud_detection import db
from fraud_detection.features import add_ecommerce_features
from fraud_detection.portable import PortableModel


REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
PALETTE = {
    "ink": "#0f1c2e",
    "muted": "#5b6b7c",
    "line": "#d5dde8",
    "surface": "#ffffff",
    "bg": "#eef3f8",
    "navy": "#0b3a5b",
    "teal": "#0f766e",
    "blue": "#1d4ed8",
    "red": "#c2410c",
    "amber": "#b45309",
    "green": "#15803d",
    "soft": "#f7fafc",
}

server = Flask(__name__)


DATA_SOURCE = "postgres" if db.database_url() else "csv"


def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if DATA_SOURCE == "postgres":
        ecommerce = db.load_ecommerce_enriched()
        # The dashboard only charts time, amount and label; skip V1-V28.
        creditcard = db.load_creditcard(columns=["Time", "Amount", "Class"])
    else:
        ecommerce = csv_data.load_ecommerce_enriched()
        creditcard = csv_data.load_creditcard()
    ecommerce["purchase_time"] = pd.to_datetime(ecommerce["purchase_time"], errors="coerce")
    ecommerce["signup_time"] = pd.to_datetime(ecommerce["signup_time"], errors="coerce")
    ecommerce["purchase_date"] = ecommerce["purchase_time"].dt.date
    ecommerce["signup_to_purchase_hours"] = (
        (ecommerce["purchase_time"] - ecommerce["signup_time"]).dt.total_seconds() / 3600
    ).clip(lower=0)
    ecommerce["is_instant"] = (ecommerce["signup_to_purchase_hours"] <= (1 / 3600)).astype(int)

    creditcard["hour"] = ((creditcard["Time"] // 3600) % 24).astype(int)
    creditcard["day"] = (creditcard["Time"] // 86400).astype(int) + 1
    return ecommerce, creditcard


def load_metrics(dataset: str) -> dict:
    path = REPORTS_DIR / f"{dataset}_model_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def best_model(metrics: dict) -> tuple[str, dict]:
    if not metrics:
        return "Not trained", {}
    name = max(metrics, key=lambda item: metrics[item].get("average_precision", 0))
    return name, metrics[name]


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def format_num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def pretty_model(name: str) -> str:
    return name.replace("_", " ").title()


def base_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["soft"],
        font={"family": "IBM Plex Sans, sans-serif", "color": PALETTE["ink"], "size": 12},
        title={"font": {"size": 15, "color": PALETTE["ink"], "family": "Sora, sans-serif"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 42, "r": 28, "t": 58, "b": 44},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, linecolor=PALETTE["line"], zeroline=False, title_font={"size": 12})
    fig.update_yaxes(gridcolor="#e4ebf2", linecolor=PALETTE["line"], zeroline=False, title_font={"size": 12})
    return fig


def kpi(title: str, value: str, detail: str = "", tone: str = "") -> html.Section:
    classes = f"kpi {tone}".strip()
    return html.Section([html.P(title), html.H2(value), html.Span(detail)], className=classes)


def metric_tile(label: str, value: str) -> html.Div:
    return html.Div([html.Span(label), html.Strong(value)], className="metric-tile")


def section_label(title: str, subtitle: str = "") -> html.Div:
    children = [html.H2(title)]
    if subtitle:
        children.append(html.P(subtitle))
    return html.Div(children, className="section-label")


def empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data for selected filters",
        showarrow=False,
        font={"size": 14, "color": PALETTE["muted"]},
    )
    fig.update_layout(title=title)
    return base_figure(fig)


def filter_ecom(df: pd.DataFrame, start_date, end_date, countries, browsers, sources, sexes) -> pd.DataFrame:
    filtered = df
    if start_date:
        filtered = filtered[filtered["purchase_date"] >= pd.to_datetime(start_date).date()]
    if end_date:
        filtered = filtered[filtered["purchase_date"] <= pd.to_datetime(end_date).date()]
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if browsers:
        filtered = filtered[filtered["browser"].isin(browsers)]
    if sources:
        filtered = filtered[filtered["source"].isin(sources)]
    if sexes:
        filtered = filtered[filtered["sex"].isin(sexes)]
    return filtered


def filter_card(df: pd.DataFrame, hours, amount_range) -> pd.DataFrame:
    filtered = df
    if hours:
        filtered = filtered[filtered["hour"].isin(hours)]
    if amount_range and len(amount_range) == 2:
        lo, hi = amount_range
        filtered = filtered[(filtered["Amount"] >= lo) & (filtered["Amount"] <= hi)]
    return filtered


@server.get("/api/summary")
def summary():
    return jsonify(
        {
            "data_source": DATA_SOURCE,
            "ecommerce": {
                "transactions": int(len(ecom_df)),
                "fraud_cases": int(ecom_df["class"].sum()),
                "fraud_rate": float(ecom_df["class"].mean()),
            },
            "creditcard": {
                "transactions": int(len(card_df)),
                "fraud_cases": int(card_df["Class"].sum()),
                "fraud_rate": float(card_df["Class"].mean()),
            },
        }
    )


ecom_df, card_df = load_dashboard_data()
ecom_metrics = load_metrics("ecommerce")
card_metrics = load_metrics("creditcard")
ecom_best_name, ecom_best = best_model(ecom_metrics)
card_best_name, card_best = best_model(card_metrics)

country_options = [{"label": c, "value": c} for c in sorted(ecom_df["country"].dropna().unique())]
browser_options = [{"label": c, "value": c} for c in sorted(ecom_df["browser"].dropna().unique())]
source_options = [{"label": c, "value": c} for c in sorted(ecom_df["source"].dropna().unique())]
sex_options = [{"label": c, "value": c} for c in sorted(ecom_df["sex"].dropna().unique())]
hour_options = [{"label": f"{h:02d}:00", "value": h} for h in range(24)]
card_amount_max = float(min(card_df["Amount"].quantile(0.99), 5000))

app = Dash(__name__, server=server, title="Adey Fraud Intelligence", suppress_callback_exceptions=True)

app.layout = html.Main(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.P("Adey Innovations · Risk Operations", className="brand-kicker"),
                        html.H1("Fraud Intelligence"),
                        html.Span("Live monitoring across e-commerce and bank-card rails"),
                    ],
                    className="title-block",
                ),
                html.Div(
                    [
                        metric_tile("E-commerce champion", pretty_model(ecom_best_name)),
                        metric_tile("Credit-card champion", pretty_model(card_best_name)),
                        metric_tile("E-com AP", format_pct(ecom_best.get("average_precision"))),
                        metric_tile("Card AP", format_pct(card_best.get("average_precision"))),
                    ],
                    className="header-metrics",
                ),
            ],
            className="app-header",
        ),
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            className="main-tabs",
            children=[
                dcc.Tab(label="Overview", value="overview", className="main-tab", selected_className="main-tab--selected"),
                dcc.Tab(label="E-commerce", value="ecommerce", className="main-tab", selected_className="main-tab--selected"),
                dcc.Tab(label="Credit Card", value="creditcard", className="main-tab", selected_className="main-tab--selected"),
                dcc.Tab(label="Model Performance", value="models", className="main-tab", selected_className="main-tab--selected"),
                dcc.Tab(label="Prediction Demo", value="predict", className="main-tab", selected_className="main-tab--selected"),
            ],
        ),
        html.Div(id="tab-content", className="tab-content"),
    ],
    className="app-shell",
)


def overview_tab() -> html.Div:
    ecom_fraud_value = float(ecom_df.loc[ecom_df["class"] == 1, "purchase_value"].sum())
    card_fraud_value = float(card_df.loc[card_df["Class"] == 1, "Amount"].sum())
    return html.Div(
        [
            section_label("Portfolio pulse", "Both datasets at a glance — volume, loss, and model readiness."),
            html.Div(
                [
                    html.Section(
                        [
                            html.P("E-commerce rail"),
                            html.Div(
                                [
                                    kpi("Transactions", format_num(len(ecom_df)), "Enriched with IP → country"),
                                    kpi("Fraud cases", format_num(int(ecom_df["class"].sum())), f"${ecom_fraud_value:,.0f} labeled loss", "alert"),
                                    kpi("Fraud rate", format_pct(float(ecom_df["class"].mean())), "Baseline portfolio risk"),
                                    kpi("Countries", format_num(ecom_df["country"].nunique()), "Geographic coverage"),
                                ],
                                className="summary-grid",
                            ),
                        ],
                        className="rail-panel",
                    ),
                    html.Section(
                        [
                            html.P("Credit-card rail"),
                            html.Div(
                                [
                                    kpi("Transactions", format_num(len(card_df)), "Duplicates removed"),
                                    kpi("Fraud cases", format_num(int(card_df["Class"].sum())), f"${card_fraud_value:,.0f} labeled loss", "alert"),
                                    kpi("Fraud rate", format_pct(float(card_df["Class"].mean())), "Severe class imbalance"),
                                    kpi("Median fraud $", f"${card_df.loc[card_df['Class'] == 1, 'Amount'].median():,.2f}", "vs legit median"),
                                ],
                                className="summary-grid",
                            ),
                        ],
                        className="rail-panel",
                    ),
                ],
                className="rail-grid",
            ),
            html.Div(
                [
                    dcc.Graph(figure=overview_ecom_trend()),
                    dcc.Graph(figure=overview_card_hour()),
                    dcc.Graph(figure=model_metric_chart(ecom_metrics, "E-commerce · Average Precision by Model", focus="average_precision")),
                    dcc.Graph(figure=model_metric_chart(card_metrics, "Credit Card · Average Precision by Model", focus="average_precision")),
                ],
                className="chart-grid",
            ),
        ]
    )


def ecommerce_tab() -> html.Div:
    return html.Div(
        [
            section_label("E-commerce monitoring", "Filter by time, geography, acquisition source, and client fingerprint."),
            html.Section(
                [
                    html.Div(
                        [
                            html.Label("Purchase date range"),
                            dcc.DatePickerRange(
                                id="ecom-date",
                                min_date_allowed=ecom_df["purchase_date"].min(),
                                max_date_allowed=ecom_df["purchase_date"].max(),
                                start_date=ecom_df["purchase_date"].min(),
                                end_date=ecom_df["purchase_date"].max(),
                                display_format="YYYY-MM-DD",
                            ),
                        ],
                        className="filter-field",
                    ),
                    html.Div(
                        [html.Label("Country"), dcc.Dropdown(id="ecom-country", options=country_options, multi=True, placeholder="All countries")],
                        className="filter-field",
                    ),
                    html.Div(
                        [html.Label("Browser"), dcc.Dropdown(id="ecom-browser", options=browser_options, multi=True, placeholder="All browsers")],
                        className="filter-field",
                    ),
                    html.Div(
                        [html.Label("Source"), dcc.Dropdown(id="ecom-source", options=source_options, multi=True, placeholder="All sources")],
                        className="filter-field",
                    ),
                    html.Div(
                        [html.Label("Sex"), dcc.Dropdown(id="ecom-sex", options=sex_options, multi=True, placeholder="All")],
                        className="filter-field",
                    ),
                ],
                className="filter-band",
            ),
            html.Div(id="ecom-kpis", className="summary-grid"),
            html.Div(
                [
                    dcc.Graph(id="ecom-trend"),
                    dcc.Graph(id="ecom-country-chart"),
                    dcc.Graph(id="ecom-browser-chart"),
                    dcc.Graph(id="ecom-source-chart"),
                    dcc.Graph(id="ecom-device-chart"),
                    dcc.Graph(id="ecom-velocity-chart"),
                ],
                className="chart-grid",
            ),
        ]
    )


def creditcard_tab() -> html.Div:
    return html.Div(
        [
            section_label("Credit-card monitoring", "Hour-of-day and amount filters on the anonymized bank-card stream."),
            html.Section(
                [
                    html.Div(
                        [
                            html.Label("Hour of day"),
                            dcc.Dropdown(id="card-hour", options=hour_options, multi=True, placeholder="All hours"),
                        ],
                        className="filter-field",
                    ),
                    html.Div(
                        [
                            html.Label("Amount range (USD)"),
                            dcc.RangeSlider(
                                id="card-amount",
                                min=0,
                                max=card_amount_max,
                                step=10,
                                value=[0, card_amount_max],
                                marks={0: "$0", int(card_amount_max): f"${card_amount_max:,.0f}"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        className="filter-field wide",
                    ),
                ],
                className="filter-band card-filters",
            ),
            html.Div(id="card-kpis", className="summary-grid"),
            html.Div(
                [
                    dcc.Graph(id="card-hour-chart"),
                    dcc.Graph(id="card-day-chart"),
                    dcc.Graph(id="card-amount-chart"),
                    dcc.Graph(id="card-class-chart"),
                ],
                className="chart-grid",
            ),
        ]
    )


def models_tab() -> html.Div:
    return html.Div(
        [
            section_label("Model performance", "Hold-out metrics from training runs — ranked by average precision for imbalance."),
            html.Div(
                [
                    dcc.Graph(figure=model_metric_chart(ecom_metrics, "E-commerce Model Comparison")),
                    dcc.Graph(figure=model_metric_chart(card_metrics, "Credit-Card Model Comparison")),
                ],
                className="chart-grid",
            ),
            html.Section(
                [
                    html.H2("Champion model snapshot"),
                    html.Div(
                        [
                            best_model_panel("E-commerce", ecom_best_name, ecom_best),
                            best_model_panel("Credit Card", card_best_name, card_best),
                        ],
                        className="model-panels",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    dcc.Graph(figure=recall_precision_scatter(ecom_metrics, "E-commerce · Fraud Precision vs Recall")),
                    dcc.Graph(figure=recall_precision_scatter(card_metrics, "Credit Card · Fraud Precision vs Recall")),
                ],
                className="chart-grid",
            ),
        ]
    )


PORTABLE_ECOM_MODEL = MODELS_DIR / "ecommerce_model_portable"
JOBLIB_ECOM_MODEL = MODELS_DIR / "ecommerce_best_model.joblib"


def ecommerce_model_available() -> bool:
    return PortableModel.exists(PORTABLE_ECOM_MODEL) or JOBLIB_ECOM_MODEL.exists()


@lru_cache(maxsize=1)
def load_ecommerce_model() -> tuple[object, str]:
    """Load the champion once. Prefer the NumPy-only export: unpickling the
    scikit-learn pipeline imports scikit-learn (~70 MB), too much for 256 MiB."""
    if PortableModel.exists(PORTABLE_ECOM_MODEL):
        model = PortableModel.load(PORTABLE_ECOM_MODEL)
        return model, model.model_name
    artifact = joblib.load(JOBLIB_ECOM_MODEL)
    return artifact["pipeline"], artifact.get("model_name", "trained pipeline")


def prediction_tab() -> html.Div:
    model_ready = ecommerce_model_available()
    status = "Model loaded" if model_ready else "Train ecommerce model to enable live scoring"
    return html.Div(
        [
            section_label("Prediction demo", "Score a synthetic e-commerce transaction with the champion pipeline."),
            html.Div(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H2("Transaction inputs"),
                                    html.Span(status, className="status-pill" + ("" if model_ready else " warn")),
                                ],
                                className="panel-head",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [html.Label("Purchase value (USD)"), dcc.Input(id="pred-value", type="number", value=42, min=0)],
                                        className="filter-field",
                                    ),
                                    html.Div(
                                        [html.Label("Age"), dcc.Input(id="pred-age", type="number", value=31, min=18, max=100)],
                                        className="filter-field",
                                    ),
                                    html.Div(
                                        [html.Label("Source"), dcc.Dropdown(id="pred-source", options=source_options, value="SEO")],
                                        className="filter-field",
                                    ),
                                    html.Div(
                                        [html.Label("Browser"), dcc.Dropdown(id="pred-browser", options=browser_options, value="Chrome")],
                                        className="filter-field",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Sex"),
                                            dcc.Dropdown(
                                                id="pred-sex",
                                                options=[{"label": "M", "value": "M"}, {"label": "F", "value": "F"}],
                                                value="M",
                                            ),
                                        ],
                                        className="filter-field",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Country"),
                                            dcc.Dropdown(id="pred-country", options=country_options, value="United States"),
                                        ],
                                        className="filter-field",
                                    ),
                                    html.Div(
                                        [html.Label("Hours since signup"), dcc.Input(id="pred-hours", type="number", value=24, min=0, step=0.1)],
                                        className="filter-field",
                                    ),
                                    html.Button("Score transaction", id="score-button", n_clicks=0, className="score-btn"),
                                ],
                                className="prediction-grid",
                            ),
                        ],
                        className="panel prediction-panel",
                    ),
                    html.Section(id="prediction-output", className="prediction-output"),
                ],
                className="prediction-layout",
            ),
        ]
    )


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    if tab == "ecommerce":
        return ecommerce_tab()
    if tab == "creditcard":
        return creditcard_tab()
    if tab == "models":
        return models_tab()
    if tab == "predict":
        return prediction_tab()
    return overview_tab()


@app.callback(
    Output("ecom-kpis", "children"),
    Output("ecom-trend", "figure"),
    Output("ecom-country-chart", "figure"),
    Output("ecom-browser-chart", "figure"),
    Output("ecom-source-chart", "figure"),
    Output("ecom-device-chart", "figure"),
    Output("ecom-velocity-chart", "figure"),
    Input("ecom-date", "start_date"),
    Input("ecom-date", "end_date"),
    Input("ecom-country", "value"),
    Input("ecom-browser", "value"),
    Input("ecom-source", "value"),
    Input("ecom-sex", "value"),
)
def update_ecommerce(start_date, end_date, countries, browsers, sources, sexes):
    filtered = filter_ecom(ecom_df, start_date, end_date, countries, browsers, sources, sexes)
    fraud = filtered[filtered["class"] == 1]
    loss = float(fraud["purchase_value"].sum()) if not fraud.empty else 0.0
    top_country = fraud["country"].value_counts().index[0] if not fraud.empty else "n/a"
    avg_velocity = filtered["signup_to_purchase_hours"].median() if not filtered.empty else None
    instant_rate = float(filtered["is_instant"].mean()) if not filtered.empty else None

    kpis = [
        kpi("Transactions", format_num(len(filtered)), "Selected e-commerce volume"),
        kpi("Fraud cases", format_num(int(filtered["class"].sum())) if not filtered.empty else "0", f"${loss:,.0f} labeled fraud value", "alert"),
        kpi("Fraud rate", format_pct(float(filtered["class"].mean())) if not filtered.empty else "n/a", "Share of selected transactions"),
        kpi("Top fraud country", top_country, f"Instant purchase rate: {format_pct(instant_rate)}" if instant_rate is not None else ""),
        kpi(
            "Median velocity",
            f"{avg_velocity:,.1f} hrs" if avg_velocity is not None else "n/a",
            "Signup → purchase",
        ),
        kpi("Unique devices", format_num(filtered["device_id"].nunique()) if not filtered.empty else "0", "Device fingerprint coverage"),
    ]

    if filtered.empty:
        empty = empty_fig
        return kpis, empty("Detected fraud over time"), empty("Fraud cases by country"), empty("Browser risk"), empty("Source risk"), empty("Top fraud-linked devices"), empty("Purchase velocity")

    trend = filtered.groupby("purchase_date", as_index=False).agg(fraud_cases=("class", "sum"), volume=("class", "size"))
    trend_fig = go.Figure()
    trend_fig.add_bar(x=trend["purchase_date"], y=trend["volume"], name="Volume", marker_color="#94a3b8", opacity=0.45)
    trend_fig.add_scatter(
        x=trend["purchase_date"],
        y=trend["fraud_cases"],
        name="Fraud cases",
        mode="lines+markers",
        line={"color": PALETTE["red"], "width": 2.5},
        marker={"size": 5},
        yaxis="y2",
    )
    trend_fig.update_layout(
        title="Detected fraud & volume over time",
        yaxis={"title": "Transactions"},
        yaxis2={"title": "Fraud cases", "overlaying": "y", "side": "right", "showgrid": False},
    )
    trend_fig = base_figure(trend_fig)

    country = fraud.groupby("country", as_index=False).size().sort_values("size", ascending=False).head(12)
    country_fig = (
        empty_fig("Fraud cases by country")
        if country.empty
        else base_figure(
            px.bar(
                country,
                x="country",
                y="size",
                title="Fraud cases by country",
                labels={"size": "Fraud cases", "country": "Country"},
                color_discrete_sequence=[PALETTE["navy"]],
            )
        )
    )
    country_fig.update_layout(xaxis_tickangle=-28)

    browser = (
        filtered.groupby("browser", as_index=False)
        .agg(transactions=("class", "size"), fraud_rate=("class", "mean"), fraud_cases=("class", "sum"))
        .sort_values("fraud_rate", ascending=False)
    )
    browser_fig = base_figure(
        px.scatter(
            browser,
            x="transactions",
            y="fraud_rate",
            size="fraud_cases",
            color="browser",
            title="Browser risk profile",
            labels={"fraud_rate": "Fraud rate", "transactions": "Transactions"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    )
    browser_fig.update_yaxes(tickformat=".1%")

    source = (
        filtered.groupby("source", as_index=False)
        .agg(transactions=("class", "size"), fraud_cases=("class", "sum"), fraud_rate=("class", "mean"))
        .sort_values("fraud_rate", ascending=False)
    )
    source_fig = base_figure(
        px.bar(
            source,
            x="source",
            y="fraud_rate",
            text="fraud_cases",
            title="Fraud rate by acquisition source",
            labels={"fraud_rate": "Fraud rate", "source": "Source"},
            color="source",
            color_discrete_sequence=[PALETTE["teal"], PALETTE["blue"], PALETTE["amber"]],
        )
    )
    source_fig.update_traces(texttemplate="%{text:,} fraud", textposition="outside")
    source_fig.update_yaxes(tickformat=".1%", range=[0, max(0.15, float(source["fraud_rate"].max()) * 1.25)])
    source_fig.update_layout(showlegend=False)

    device = fraud.groupby("device_id", as_index=False).size().sort_values("size", ascending=False).head(12)
    device_fig = (
        empty_fig("Top fraud-linked devices")
        if device.empty
        else base_figure(
            px.bar(
                device,
                x="size",
                y="device_id",
                orientation="h",
                title="Top fraud-linked devices",
                labels={"size": "Fraud cases", "device_id": "Device ID"},
                color_discrete_sequence=[PALETTE["teal"]],
            )
        )
    )
    device_fig.update_layout(yaxis={"categoryorder": "total ascending"})

    velocity = filtered[["signup_to_purchase_hours", "class"]].copy()
    velocity["velocity_bucket"] = pd.cut(
        velocity["signup_to_purchase_hours"],
        bins=[-0.01, 0.001, 1, 24, 168, 720, float("inf")],
        labels=["Instant", "<1h", "1–24h", "1–7d", "7–30d", "30d+"],
    )
    vel = velocity.groupby("velocity_bucket", observed=False).agg(fraud_rate=("class", "mean"), volume=("class", "size")).reset_index()
    velocity_fig = base_figure(
        px.bar(
            vel,
            x="velocity_bucket",
            y="fraud_rate",
            title="Fraud rate by signup→purchase velocity",
            labels={"fraud_rate": "Fraud rate", "velocity_bucket": "Time since signup"},
            color_discrete_sequence=[PALETTE["red"]],
        )
    )
    velocity_fig.update_yaxes(tickformat=".0%")
    return kpis, trend_fig, country_fig, browser_fig, source_fig, device_fig, velocity_fig


@app.callback(
    Output("card-kpis", "children"),
    Output("card-hour-chart", "figure"),
    Output("card-day-chart", "figure"),
    Output("card-amount-chart", "figure"),
    Output("card-class-chart", "figure"),
    Input("card-hour", "value"),
    Input("card-amount", "value"),
)
def update_creditcard(hours, amount_range):
    filtered = filter_card(card_df, hours, amount_range)
    fraud = filtered[filtered["Class"] == 1]
    loss = float(fraud["Amount"].sum()) if not fraud.empty else 0.0
    kpis = [
        kpi("Transactions", format_num(len(filtered)), "After filters"),
        kpi("Fraud cases", format_num(int(filtered["Class"].sum())) if not filtered.empty else "0", f"${loss:,.0f} labeled fraud", "alert"),
        kpi("Fraud rate", format_pct(float(filtered["Class"].mean())) if not filtered.empty else "n/a", "Selected window"),
        kpi(
            "Median amount",
            f"${filtered['Amount'].median():,.2f}" if not filtered.empty else "n/a",
            f"Fraud median ${fraud['Amount'].median():,.2f}" if not fraud.empty else "",
        ),
    ]
    if filtered.empty:
        empty = empty_fig
        return kpis, empty("Fraud by hour"), empty("Fraud by day"), empty("Amount distribution"), empty("Class comparison")

    hourly = filtered.groupby("hour", as_index=False)["Class"].agg(fraud_cases="sum", transactions="size")
    hourly["fraud_rate"] = hourly["fraud_cases"] / hourly["transactions"]
    hour_fig = go.Figure()
    hour_fig.add_bar(x=hourly["hour"], y=hourly["fraud_cases"], name="Fraud cases", marker_color=PALETTE["red"])
    hour_fig.add_scatter(
        x=hourly["hour"],
        y=hourly["fraud_rate"],
        name="Fraud rate",
        mode="lines+markers",
        line={"color": PALETTE["blue"], "width": 2},
        yaxis="y2",
    )
    hour_fig.update_layout(
        title="Credit-card fraud by hour of day",
        xaxis={"title": "Hour", "dtick": 2},
        yaxis={"title": "Fraud cases"},
        yaxis2={"title": "Fraud rate", "overlaying": "y", "side": "right", "tickformat": ".2%", "showgrid": False},
    )
    hour_fig = base_figure(hour_fig)

    daily = filtered.groupby("day", as_index=False)["Class"].agg(fraud_cases="sum", volume="size")
    day_fig = go.Figure()
    day_fig.add_scatter(x=daily["day"], y=daily["volume"], name="Volume", mode="lines", line={"color": "#94a3b8", "width": 2})
    day_fig.add_scatter(
        x=daily["day"],
        y=daily["fraud_cases"],
        name="Fraud cases",
        mode="lines+markers",
        line={"color": PALETTE["red"], "width": 2},
        yaxis="y2",
    )
    day_fig.update_layout(
        title="Transaction volume & fraud by day index",
        xaxis={"title": "Day since start"},
        yaxis={"title": "Transactions"},
        yaxis2={"title": "Fraud cases", "overlaying": "y", "side": "right", "showgrid": False},
    )
    day_fig = base_figure(day_fig)

    sample = filtered.sample(min(25000, len(filtered)), random_state=42)
    amount_fig = base_figure(
        px.histogram(
            sample,
            x="Amount",
            color=sample["Class"].map({0: "Legitimate", 1: "Fraud"}),
            nbins=60,
            title="Transaction amount distribution",
            labels={"Amount": "Amount (USD)", "color": "Class"},
            color_discrete_map={"Legitimate": PALETTE["blue"], "Fraud": PALETTE["red"]},
            barmode="overlay",
            opacity=0.75,
        )
    )
    amount_fig.update_xaxes(range=[0, sample["Amount"].quantile(0.995)])

    summary = (
        filtered.groupby("Class", as_index=False)["Amount"]
        .agg(count="count", median="median", mean="mean")
        .assign(Class=lambda d: d["Class"].map({0: "Legitimate", 1: "Fraud"}))
    )
    class_fig = base_figure(
        px.bar(
            summary,
            x="Class",
            y="median",
            text="count",
            title="Median amount by class",
            labels={"median": "Median amount (USD)", "Class": "Class"},
            color="Class",
            color_discrete_map={"Legitimate": PALETTE["teal"], "Fraud": PALETTE["red"]},
        )
    )
    class_fig.update_traces(texttemplate="%{text:,} tx", textposition="outside")
    class_fig.update_layout(showlegend=False)
    return kpis, hour_fig, day_fig, amount_fig, class_fig


@app.callback(
    Output("prediction-output", "children"),
    Input("score-button", "n_clicks"),
    State("pred-value", "value"),
    State("pred-age", "value"),
    State("pred-source", "value"),
    State("pred-browser", "value"),
    State("pred-sex", "value"),
    State("pred-country", "value"),
    State("pred-hours", "value"),
)
def score_transaction(n_clicks, purchase_value, age, source, browser, sex, country, hours):
    if not n_clicks:
        return [
            html.Div(
                [
                    html.H2("Ready to score"),
                    html.P("Enter transaction details and run the champion e-commerce model."),
                    dcc.Graph(figure=risk_gauge(0.0, idle=True), config={"displayModeBar": False}),
                ]
            )
        ]

    if not ecommerce_model_available():
        return [
            html.Div(
                [
                    html.H2("Model unavailable", className="risk-high"),
                    html.P("Train the e-commerce model before scoring transactions."),
                    html.Code("python -m fraud_detection.train --dataset ecommerce"),
                ]
            )
        ]

    pipe, model_name = load_ecommerce_model()
    purchase_time = pd.Timestamp("2015-07-01 12:00:00")
    signup_time = purchase_time - pd.Timedelta(hours=float(hours or 0))
    row = pd.DataFrame(
        [
            {
                "user_id": 999999,
                "signup_time": signup_time,
                "purchase_time": purchase_time,
                "purchase_value": float(purchase_value or 0),
                "device_id": "demo-device",
                "source": source,
                "browser": browser,
                "sex": sex,
                "age": int(age or 0),
                "ip_address": 732758368.0,
                "country": country,
            }
        ]
    )
    scored = add_ecommerce_features(row)
    probability = float(pipe.predict_proba(scored)[0, 1])
    if probability >= 0.7:
        verdict, tone = "High risk — review / block", "risk-high"
    elif probability >= 0.4:
        verdict, tone = "Elevated risk — step-up auth", "risk-mid"
    else:
        verdict, tone = "Lower risk — allow with monitoring", "risk-low"

    return [
        html.Div(
            [
                html.H2(verdict, className=tone),
                html.P(f"Fraud probability: {probability:.2%}"),
                html.Span(f"Model: {pretty_model(model_name)}"),
                dcc.Graph(figure=risk_gauge(probability), config={"displayModeBar": False}),
            ]
        )
    ]


def overview_ecom_trend() -> go.Figure:
    daily = ecom_df.groupby("purchase_date", as_index=False)["class"].agg(fraud_cases="sum", volume="size")
    fig = go.Figure()
    fig.add_scatter(
        x=daily["purchase_date"],
        y=daily["fraud_cases"],
        fill="tozeroy",
        name="Fraud cases",
        line={"color": PALETTE["red"], "width": 2},
        fillcolor="rgba(194, 65, 12, 0.12)",
    )
    fig.update_layout(title="E-commerce fraud trend", xaxis_title="Purchase date", yaxis_title="Fraud cases")
    return base_figure(fig)


def overview_card_hour() -> go.Figure:
    hourly = card_df.groupby("hour", as_index=False)["Class"].sum()
    fig = base_figure(
        px.bar(
            hourly,
            x="hour",
            y="Class",
            title="Credit-card fraud by hour",
            labels={"hour": "Hour of day", "Class": "Fraud cases"},
            color_discrete_sequence=[PALETTE["navy"]],
        )
    )
    fig.update_layout(showlegend=False)
    return fig


def risk_gauge(probability: float, idle: bool = False) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=0 if idle else probability * 100,
            number={"suffix": "%" if not idle else "", "font": {"size": 28, "family": "IBM Plex Mono, monospace"}},
            title={"text": "Fraud score" if not idle else "Awaiting score", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": PALETTE["navy"]},
                "bgcolor": PALETTE["soft"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 70], "color": "#ffedd5"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": PALETTE["red"], "width": 3}, "thickness": 0.75, "value": 50},
            },
        )
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 10}, height=260, paper_bgcolor=PALETTE["surface"])
    return fig


def model_metric_chart(metrics: dict, title: str, focus: str | None = None) -> go.Figure:
    if not metrics:
        return empty_fig(title)
    if focus:
        rows = [
            {"model": pretty_model(name), "score": values.get(focus, 0)}
            for name, values in metrics.items()
        ]
        fig = px.bar(
            pd.DataFrame(rows).sort_values("score", ascending=False),
            x="model",
            y="score",
            title=title,
            labels={"score": "Average precision", "model": "Model"},
            color_discrete_sequence=[PALETTE["teal"]],
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(showlegend=False)
        return base_figure(fig)

    rows = []
    for name, values in metrics.items():
        report = values.get("classification_report", {})
        fraud = report.get("1", {})
        rows.append(
            {
                "model": pretty_model(name),
                "ROC-AUC": values.get("roc_auc"),
                "Avg Precision": values.get("average_precision"),
                "Fraud Recall": fraud.get("recall"),
                "Fraud Precision": fraud.get("precision"),
            }
        )
    long = pd.DataFrame(rows).melt(id_vars="model", var_name="metric", value_name="score")
    fig = px.bar(
        long,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title=title,
        color_discrete_sequence=[PALETTE["navy"], PALETTE["teal"], PALETTE["amber"], PALETTE["blue"]],
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return base_figure(fig)


def recall_precision_scatter(metrics: dict, title: str) -> go.Figure:
    if not metrics:
        return empty_fig(title)
    rows = []
    for name, values in metrics.items():
        fraud = values.get("classification_report", {}).get("1", {})
        rows.append(
            {
                "model": pretty_model(name),
                "precision": fraud.get("precision"),
                "recall": fraud.get("recall"),
                "average_precision": values.get("average_precision"),
            }
        )
    df = pd.DataFrame(rows)
    fig = base_figure(
        px.scatter(
            df,
            x="recall",
            y="precision",
            size="average_precision",
            color="model",
            text="model",
            title=title,
            labels={"recall": "Fraud recall", "precision": "Fraud precision"},
        )
    )
    fig.update_traces(textposition="top center")
    fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return fig


def best_model_panel(dataset: str, name: str, values: dict) -> html.Div:
    report = values.get("classification_report", {}) if values else {}
    fraud = report.get("1", {})
    return html.Div(
        [
            html.H3(dataset),
            metric_tile("Champion", pretty_model(name)),
            metric_tile("Average precision", format_pct(values.get("average_precision")) if values else "n/a"),
            metric_tile("ROC-AUC", format_pct(values.get("roc_auc")) if values else "n/a"),
            metric_tile("Fraud recall", format_pct(fraud.get("recall")) if fraud else "n/a"),
            metric_tile("Fraud precision", format_pct(fraud.get("precision")) if fraud else "n/a"),
            metric_tile("Accuracy", format_pct(report.get("accuracy")) if report else "n/a"),
        ],
        className="model-panel",
    )


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #eef3f8;
                --bg-accent: #e3edf5;
                --surface: #ffffff;
                --ink: #0f1c2e;
                --muted: #5b6b7c;
                --line: #d5dde8;
                --navy: #0b3a5b;
                --teal: #0f766e;
                --blue: #1d4ed8;
                --red: #c2410c;
                --amber: #b45309;
                --soft: #f7fafc;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: "IBM Plex Sans", sans-serif;
                color: var(--ink);
                background:
                    radial-gradient(1200px 500px at 8% -10%, #d9e8f4 0%, transparent 55%),
                    radial-gradient(900px 420px at 100% 0%, #d7efe8 0%, transparent 50%),
                    linear-gradient(180deg, #f5f8fb 0%, var(--bg) 40%, var(--bg-accent) 100%);
                min-height: 100vh;
            }
            .app-shell { max-width: 1360px; margin: 0 auto; padding: 28px 24px 48px; }
            .app-header {
                display: grid;
                grid-template-columns: 1.2fr 1.4fr;
                gap: 20px;
                align-items: end;
                margin-bottom: 20px;
                padding-bottom: 18px;
                border-bottom: 1px solid var(--line);
            }
            .brand-kicker {
                margin: 0 0 8px;
                color: var(--teal);
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }
            .title-block h1 {
                margin: 0;
                font-family: Sora, sans-serif;
                font-size: clamp(28px, 4vw, 40px);
                font-weight: 800;
                letter-spacing: -0.03em;
                color: var(--navy);
            }
            .title-block span { display: block; margin-top: 8px; color: var(--muted); font-size: 15px; }
            .header-metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
            .metric-tile {
                background: rgba(255,255,255,0.72);
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 12px 14px;
                backdrop-filter: blur(6px);
            }
            .metric-tile span { display: block; color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
            .metric-tile strong { display: block; margin-top: 5px; font-size: 15px; font-family: "IBM Plex Mono", monospace; }
            .main-tabs { border-bottom: 1px solid var(--line) !important; }
            .main-tab {
                font-family: "IBM Plex Sans", sans-serif !important;
                font-weight: 600 !important;
                color: var(--muted) !important;
                border: 0 !important;
                background: transparent !important;
                padding: 12px 16px !important;
            }
            .main-tab--selected {
                color: var(--navy) !important;
                border-bottom: 3px solid var(--teal) !important;
            }
            .tab-content { padding-top: 22px; }
            .section-label { margin-bottom: 14px; }
            .section-label h2 {
                margin: 0;
                font-family: Sora, sans-serif;
                font-size: 20px;
                letter-spacing: -0.02em;
                color: var(--navy);
            }
            .section-label p { margin: 6px 0 0; color: var(--muted); font-size: 14px; }
            .rail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }
            .rail-panel {
                background: rgba(255,255,255,0.78);
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 16px;
            }
            .rail-panel > p {
                margin: 0 0 12px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--teal);
            }
            .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 0 0 16px; }
            .kpi, .panel, .prediction-output, .filter-band {
                background: rgba(255,255,255,0.88);
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 16px;
            }
            .kpi p { margin: 0; color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
            .kpi h2 {
                margin: 8px 0 4px;
                font-family: "IBM Plex Mono", monospace;
                font-size: 24px;
                letter-spacing: -0.02em;
                color: var(--ink);
            }
            .kpi span { color: var(--muted); font-size: 12px; }
            .kpi.alert h2 { color: var(--red); }
            .filter-band {
                display: grid;
                grid-template-columns: 1.15fr repeat(4, 1fr);
                gap: 14px;
                align-items: end;
                margin-bottom: 16px;
            }
            .filter-band.card-filters { grid-template-columns: 1fr 2fr; }
            .filter-field label {
                display: block;
                margin-bottom: 6px;
                color: #334155;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.02em;
            }
            .filter-field input, .filter-field .Select-control {
                width: 100%;
                min-height: 40px;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 8px 10px;
                font-family: "IBM Plex Sans", sans-serif;
                background: #fff;
            }
            .filter-field.wide { grid-column: span 1; }
            .chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
            .dash-graph {
                background: rgba(255,255,255,0.92);
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 6px;
            }
            .panel h2, .panel-head h2 {
                margin: 0;
                font-family: Sora, sans-serif;
                font-size: 18px;
                color: var(--navy);
            }
            .panel-head {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                margin-bottom: 16px;
            }
            .status-pill {
                display: inline-flex;
                align-items: center;
                padding: 6px 10px;
                border-radius: 999px;
                background: #dcfce7;
                color: #166534;
                font-size: 12px;
                font-weight: 700;
            }
            .status-pill.warn { background: #ffedd5; color: #9a3412; }
            .model-panels { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
            .model-panel { display: grid; gap: 10px; align-content: start; }
            .model-panel h3 { margin: 0; font-family: Sora, sans-serif; font-size: 16px; color: var(--navy); }
            .prediction-layout { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; align-items: start; }
            .prediction-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; align-items: end; }
            .score-btn {
                height: 42px;
                border: 0;
                border-radius: 8px;
                background: linear-gradient(135deg, var(--navy), var(--teal));
                color: white;
                font-family: "IBM Plex Sans", sans-serif;
                font-weight: 700;
                cursor: pointer;
                grid-column: 1 / -1;
            }
            .score-btn:hover { filter: brightness(1.05); }
            .prediction-output h2 { margin: 0 0 8px; font-family: Sora, sans-serif; }
            .prediction-output p { margin: 0 0 6px; font-size: 18px; font-weight: 700; font-family: "IBM Plex Mono", monospace; }
            .prediction-output span, .prediction-output code { color: var(--muted); font-size: 13px; }
            .risk-high { color: var(--red); }
            .risk-mid { color: var(--amber); }
            .risk-low { color: var(--teal); }
            .DateRangePickerInput { border-radius: 8px !important; border-color: var(--line) !important; }
            @media (max-width: 1100px) {
                .app-header, .rail-grid, .prediction-layout, .model-panels, .chart-grid { grid-template-columns: 1fr; }
                .filter-band, .filter-band.card-filters, .prediction-grid, .header-metrics { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""


if __name__ == "__main__":
    import os

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    app.run(debug=os.environ.get("DASH_DEBUG", "false").lower() == "true", host=host, port=port)
