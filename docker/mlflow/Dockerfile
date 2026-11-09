# MLflow 3.x tracking server with GenAI extras.
#
# The stock ghcr.io/mlflow/mlflow image lacks the [genai] extras (and the
# DB/S3 drivers) needed for the GenAI tracing / evaluation / prompt
# endpoints, so extend it here. The server version must match the client
# library (mlflow 3.18.0 in pyproject.toml).
#
# After upgrading from a 2.x server, run once (backup first):
#   docker compose run --rm mlflow mlflow db upgrade <backend-store-uri>
# (the backend-store-uri is the same postgresql://... URL used in
# docker-compose.yml).
FROM ghcr.io/mlflow/mlflow:v3.18.0

RUN pip install --no-cache-dir "mlflow[genai]==3.18.0" "psycopg[binary]" boto3
