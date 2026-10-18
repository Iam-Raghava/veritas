# Veritas Product — one-command deployment.
#   docker build -t veritas .
#   docker run -p 8000:8000 veritas
# Then open http://localhost:8000
FROM python:3.11-slim

WORKDIR /app

# Copy the library and product.
COPY veritas/ ./veritas/
COPY product/ ./product/
COPY pyproject.toml ./

# Install the package (no heavy deps for the core).
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir .

EXPOSE 8000

CMD ["python3", "product/api/server.py", "--port", "8000", "--host", "0.0.0.0"]
