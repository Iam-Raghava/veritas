# Veritas AML Add/Search service
FROM python:3.12-slim

WORKDIR /srv

# veritas package first (cached unless veritas/ changes)
COPY veritas/ ./veritas/
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# service
COPY aml/ ./aml/
RUN pip install --no-cache-dir -r aml/requirements.txt

ENV VERITAS_DB=/data/veritas.db
VOLUME /data
EXPOSE 8000

CMD ["uvicorn", "aml.app:app", "--host", "0.0.0.0", "--port", "8000"]
