"""Load the fraud datasets from Data/ into the Postgres database at DATABASE_URL.

Safe to run on every start: steps that already finished are skipped.

    python scripts/seed_database.py           # seed if needed
    python scripts/seed_database.py --force   # rebuild every table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection import db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Reload every table")
    args = parser.parse_args()

    if not db.database_url():
        print("[seed] DATABASE_URL not set; skipping (dashboard will read the CSV files).", flush=True)
        return 0

    db.wait_for_database()
    db.seed(force=args.force)
    print("[seed] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
