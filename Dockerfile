FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY models ./models

EXPOSE 5000

CMD ["python", "app/serve_model.py", "--dataset", "ecommerce", "--model", "models/ecommerce_best_model.joblib"]

