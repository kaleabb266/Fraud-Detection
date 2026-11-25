FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8050

WORKDIR /app

COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY Data ./Data

EXPOSE 8050

# Seed Postgres on first start (no-op once loaded, or without DATABASE_URL),
# then serve the dashboard.
CMD ["sh", "-c", "python scripts/seed_database.py && exec python app/dashboard.py"]
