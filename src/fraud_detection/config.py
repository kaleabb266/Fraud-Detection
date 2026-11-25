from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

FRAUD_DATA_PATH = DATA_DIR / "Fraud_Data.csv"
IP_COUNTRY_PATH = DATA_DIR / "IpAddress_to_Country.csv"
CREDITCARD_PATH = DATA_DIR / "creditcard1.csv"
# Full 284,807-row credit card dataset, used to seed Postgres.
CREDITCARD_FULL_ZIP_PATH = DATA_DIR / "creditcard.csv.zip"

RANDOM_STATE = 42
TEST_SIZE = 0.2

