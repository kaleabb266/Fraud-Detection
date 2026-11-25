"""PostgreSQL storage for the fraud datasets.

The dashboard reads from Postgres whenever ``DATABASE_URL`` is set and falls
back to the CSV files in ``Data/`` otherwise. ``seed()`` loads the raw CSVs
into the database once and builds the cleaned tables the dashboard reads.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

import pandas as pd

from .config import CREDITCARD_FULL_ZIP_PATH, FRAUD_DATA_PATH, IP_COUNTRY_PATH

# Bump when the schema or seeding logic changes so existing databases reseed.
SEED_VERSION = "1"

CARD_FEATURES = [f"V{i}" for i in range(1, 29)]
CARD_COLUMNS = ["Time", *CARD_FEATURES, "Amount", "Class"]
ECOM_COLUMNS = [
    "user_id",
    "signup_time",
    "purchase_time",
    "purchase_value",
    "device_id",
    "source",
    "browser",
    "sex",
    "age",
    "ip_address",
    "class",
]


def database_url() -> str | None:
    """Return a truthy marker when a database is configured (URL or PG* vars)."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    if os.environ.get("PGHOST") and os.environ.get("PGUSER"):
        return "postgresql://"  # configured through libpq PG* variables
    return None


def _parse_url(url: str) -> dict:
    """Parse postgresql://user:password@host:port/dbname, tolerating a raw
    (not percent-encoded) password that contains ':' or '@'."""
    from urllib.parse import unquote

    rest = url.split("://", 1)[1] if "://" in url else url
    rest, _, query = rest.partition("?")
    userinfo, _, hostpart = rest.rpartition("@")  # host never contains '@'
    hostport, _, dbname = hostpart.partition("/")
    host, _, port = hostport.partition(":")
    user, _, password = userinfo.partition(":")  # usernames never contain ':'
    params = {
        "host": host or None,
        "port": port or None,
        "dbname": unquote(dbname) or None,
        "user": unquote(user) or None,
        "password": password or None,
    }
    for pair in filter(None, query.split("&")):
        key, _, value = pair.partition("=")
        params[key] = unquote(value)
    return {k: v for k, v in params.items() if v is not None}


def _connection_kwargs(url: str | None = None) -> dict:
    # Prefer the discrete PG* variables (no URL escaping issues), then the URL.
    if url is None and os.environ.get("PGHOST") and os.environ.get("PGPASSWORD"):
        return {
            "host": os.environ["PGHOST"],
            "port": os.environ.get("PGPORT", "5432"),
            "dbname": os.environ.get("PGDATABASE"),
            "user": os.environ.get("PGUSER"),
            "password": os.environ["PGPASSWORD"],
        }
    url = url or os.environ.get("DATABASE_URL", "")
    return _parse_url(url) if url else {}


@contextmanager
def connect(url: str | None = None):
    import psycopg

    conn = psycopg.connect(connect_timeout=10, **_connection_kwargs(url))
    try:
        yield conn
    finally:
        conn.close()


def wait_for_database(url: str | None = None, timeout: float = 300) -> None:
    import psycopg

    deadline = time.monotonic() + timeout
    while True:
        try:
            with connect(url) as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError as exc:
            if time.monotonic() > deadline:
                raise
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            print(f"[db] waiting for database: {reason}", flush=True)
            time.sleep(5)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #

def _open_fraud_data() -> BinaryIO:
    return open(FRAUD_DATA_PATH, "rb")


def _open_ip_country() -> BinaryIO:
    return open(IP_COUNTRY_PATH, "rb")


@contextmanager
def _open_creditcard() -> Iterator[BinaryIO]:
    with zipfile.ZipFile(CREDITCARD_FULL_ZIP_PATH) as archive:
        with archive.open("creditcard.csv") as member:
            yield member


def _copy_in(cur, table: str, columns: list[str], stream: BinaryIO) -> None:
    cols = ", ".join(_q(c) for c in columns)
    with cur.copy(f"COPY {_q(table)} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER true)") as copy:
        while chunk := stream.read(1 << 20):
            copy.write(chunk)


def _step_fraud_data(cur) -> None:
    cur.execute("DROP TABLE IF EXISTS fraud_data CASCADE")
    cur.execute(
        """
        CREATE TABLE fraud_data (
            row_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id        bigint,
            signup_time    timestamp,
            purchase_time  timestamp,
            purchase_value double precision,
            device_id      text,
            source         text,
            browser        text,
            sex            text,
            age            integer,
            ip_address     double precision,
            "class"        smallint
        )
        """
    )
    with _open_fraud_data() as f:
        _copy_in(cur, "fraud_data", ECOM_COLUMNS, f)


def _step_ip_country(cur) -> None:
    cur.execute("DROP TABLE IF EXISTS ip_address_to_country CASCADE")
    cur.execute(
        """
        CREATE TABLE ip_address_to_country (
            lower_bound_ip_address double precision,
            upper_bound_ip_address double precision,
            country                text
        )
        """
    )
    with _open_ip_country() as f:
        _copy_in(
            cur,
            "ip_address_to_country",
            ["lower_bound_ip_address", "upper_bound_ip_address", "country"],
            f,
        )
    cur.execute(
        "CREATE INDEX ip_address_to_country_lower_idx "
        "ON ip_address_to_country ((floor(lower_bound_ip_address)::bigint))"
    )


def _step_creditcard(cur) -> None:
    cur.execute("DROP TABLE IF EXISTS creditcard CASCADE")
    feature_cols = ",\n".join(f"{_q(c)} double precision" for c in CARD_FEATURES)
    cur.execute(
        f"""
        CREATE TABLE creditcard (
            row_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            "Time"   double precision,
            {feature_cols},
            "Amount" double precision,
            "Class"  smallint
        )
        """
    )
    with _open_creditcard() as f:
        _copy_in(cur, "creditcard", CARD_COLUMNS, f)


def _step_ecommerce_enriched(cur) -> None:
    """Deduplicate transactions and attach a country, mirroring data.add_country."""
    cols = ", ".join(_q(c) for c in ECOM_COLUMNS)
    t_cols = ", ".join(f"t.{_q(c)}" for c in ECOM_COLUMNS)
    cur.execute("DROP TABLE IF EXISTS ecommerce_enriched CASCADE")
    cur.execute(
        f"""
        CREATE TABLE ecommerce_enriched AS
        WITH unique_tx AS (
            SELECT min(row_id) AS row_id, {cols}
            FROM fraud_data
            GROUP BY {cols}
        )
        SELECT t.row_id, {t_cols},
               CASE
                   WHEN r.upper_bound_ip_address IS NOT NULL
                        AND floor(t.ip_address)::bigint <= floor(r.upper_bound_ip_address)::bigint
                   THEN coalesce(r.country, 'Unknown')
                   ELSE 'Unknown'
               END AS country
        FROM unique_tx t
        LEFT JOIN LATERAL (
            SELECT upper_bound_ip_address, country
            FROM ip_address_to_country
            WHERE floor(lower_bound_ip_address)::bigint <= floor(t.ip_address)::bigint
            ORDER BY floor(lower_bound_ip_address)::bigint DESC
            LIMIT 1
        ) r ON true
        """
    )
    cur.execute("ALTER TABLE ecommerce_enriched ADD PRIMARY KEY (row_id)")


def _step_creditcard_clean(cur) -> None:
    """Drop exact duplicate rows, mirroring data.load_creditcard."""
    cols = ", ".join(_q(c) for c in CARD_COLUMNS)
    cur.execute("DROP TABLE IF EXISTS creditcard_clean CASCADE")
    cur.execute(
        f"""
        CREATE TABLE creditcard_clean AS
        SELECT min(row_id) AS row_id, {cols}
        FROM creditcard
        GROUP BY {cols}
        """
    )
    cur.execute("ALTER TABLE creditcard_clean ADD PRIMARY KEY (row_id)")


# Ordered: once a step runs, every later step reruns too, since they derive
# from earlier tables.
SEED_STEPS: list[tuple[str, str, Callable]] = [
    ("fraud_data", "fraud_data", _step_fraud_data),
    ("ip_address_to_country", "ip_address_to_country", _step_ip_country),
    ("creditcard", "creditcard", _step_creditcard),
    ("ecommerce_enriched", "ecommerce_enriched", _step_ecommerce_enriched),
    ("creditcard_clean", "creditcard_clean", _step_creditcard_clean),
]


def seed(url: str | None = None, force: bool = False) -> None:
    """Load the CSV datasets into Postgres. Each step commits on its own, so a
    restart part-way through resumes instead of starting over."""
    with connect(url) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seed_state (
                step     text PRIMARY KEY,
                version  text NOT NULL,
                row_count bigint NOT NULL,
                done_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.commit()
        done = {
            step
            for step, version in conn.execute("SELECT step, version FROM seed_state").fetchall()
            if version == SEED_VERSION
        }

        rerun = force
        for step, table, run in SEED_STEPS:
            if not rerun and step in done:
                print(f"[seed] {step}: up to date", flush=True)
                continue
            rerun = True
            started = time.monotonic()
            print(f"[seed] {step}: loading...", flush=True)
            with conn.cursor() as cur:
                run(cur)
                cur.execute(f"ANALYZE {_q(table)}")
                rows = cur.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO seed_state (step, version, row_count, done_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (step) DO UPDATE
                    SET version = EXCLUDED.version, row_count = EXCLUDED.row_count, done_at = now()
                    """,
                    (step, SEED_VERSION, rows),
                )
            conn.commit()
            print(f"[seed] {step}: {rows:,} rows in {time.monotonic() - started:.1f}s", flush=True)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def _query_df(conn, sql: str, **read_csv_kwargs) -> pd.DataFrame:
    """Stream a query out with COPY and parse it like the source CSV files."""
    buffer = io.BytesIO()
    with conn.cursor() as cur:
        with cur.copy(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER true)") as copy:
            for chunk in copy:
                buffer.write(chunk)
    buffer.seek(0)
    return pd.read_csv(buffer, **read_csv_kwargs)


def load_ecommerce_enriched(url: str | None = None) -> pd.DataFrame:
    cols = ", ".join(_q(c) for c in [*ECOM_COLUMNS, "country"])
    with connect(url) as conn:
        df = _query_df(conn, f"SELECT {cols} FROM ecommerce_enriched ORDER BY row_id")
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
    df["country"] = df["country"].fillna("Unknown")
    return df


def load_creditcard(url: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    cols = ", ".join(_q(c) for c in (columns or CARD_COLUMNS))
    with connect(url) as conn:
        return _query_df(conn, f"SELECT {cols} FROM creditcard_clean ORDER BY row_id")


def seed_summary(url: str | None = None) -> list[dict]:
    with connect(url) as conn:
        rows = conn.execute(
            "SELECT step, version, row_count, done_at FROM seed_state ORDER BY done_at"
        ).fetchall()
    return [
        {"table": step, "version": version, "rows": count, "loaded_at": done_at.isoformat()}
        for step, version, count, done_at in rows
    ]
