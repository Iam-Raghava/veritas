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

# Dense embedding model (multilingual, ~130MB). Downloaded at build
# time from the ONNX community mirror; not committed to git.
ENV VERITAS_DENSE_DIR=/srv/dense-model
RUN mkdir -p /srv/dense-model && \
    curl -sL "https://huggingface.co/Xenova/jina-embeddings-v2-small-en/resolve/main/onnx/model.onnx" \
      -o /srv/dense-model/model.onnx && \
    curl -sL "https://huggingface.co/Xenova/jina-embeddings-v2-small-en/resolve/main/tokenizer.json" \
      -o /srv/dense-model/tokenizer.json && \
    ls -la /srv/dense-model/

# Cross-encoder reranker (bge-reranker-base quantized, ~280MB).
# Retrieve-then-rerank: hybrid gets candidates, reranker scores precisely.
ENV VERITAS_RERANK_DIR=/srv/rerank-model
RUN mkdir -p /srv/rerank-model && \
    curl -sL "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx/model_quantized.onnx" \
      -o /srv/rerank-model/model.onnx && \
    curl -sL "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/tokenizer.json" \
      -o /srv/rerank-model/tokenizer.json && \
    ls -la /srv/rerank-model/ | head -5

ENV VERITAS_DB=/data/veritas.db
VOLUME /data
EXPOSE 8000

CMD ["uvicorn", "aml.app:app", "--host", "0.0.0.0", "--port", "8000"]
