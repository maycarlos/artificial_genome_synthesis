FROM python:3.10-slim-bullseye

RUN pip install --no-cache-dir mlflow==2.3.1

WORKDIR /mlflow/

ENV ARTIFACT_ROOT /mlflow/artifacts
ENV BACKEND_URI sqlite:////mlflow/mlflow.db

EXPOSE 5000

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000
