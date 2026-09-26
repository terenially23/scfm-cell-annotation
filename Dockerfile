FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install --no-cache-dir -r requirements.txt

# Vendored geneformer is not on PyPI; it's checked into vendor/ already patched
# for CPU-only inference (see vendor/geneformer/emb_extractor.py).
COPY vendor/geneformer /usr/local/lib/python3.11/site-packages/geneformer

COPY src ./src
COPY models ./models
COPY data/geneformer_full/geneformer_embs.csv ./data/geneformer_full/geneformer_embs.csv
COPY data/geneformer_full/reference_labels.json ./data/geneformer_full/reference_labels.json

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
