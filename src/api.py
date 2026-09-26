"""FastAPI service that annotates a single cell from a gene-symbol -> raw UMI
count dict, using Geneformer's rank-value tokenization scheme reimplemented
directly (no loom/dataset round-trip needed for one cell at a time), followed
by a KNN vote against reference embeddings extracted offline.

Geneformer represents a cell as a rank-ordered sequence of gene tokens, where
each gene's raw count is first scaled to a fixed total (target_sum) and then
divided by that gene's median expression across a large reference corpus
(gene_median_dict) -- so a gene expressed above its "typical" level ranks
higher regardless of the cell's overall depth. This mirrors exactly what
geneformer.tokenizer.TranscriptomeTokenizer.tokenize_anndata does internally.
"""

import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoModel

MODEL_DIR = Path("models/Geneformer-V1-10M")
DICTS_DIR = Path("models/gene_dictionaries_30m")
REFERENCE_DIR = Path("data/geneformer_full")
MODEL_INPUT_SIZE = 2048
TARGET_SUM = 10_000

state: dict = {}


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def tokenize_cell(gene_counts: dict[str, float]) -> list[int]:
    """Reimplements TranscriptomeTokenizer's rank-value encoding for one cell."""
    symbol_to_ensembl = state["symbol_to_ensembl"]
    gene_median = state["gene_median"]
    token_dict = state["token_dict"]

    total_counts = sum(gene_counts.values())
    if total_counts <= 0:
        return []

    scored: list[tuple[float, int]] = []
    for symbol, count in gene_counts.items():
        if count <= 0:
            continue
        ensembl_id = symbol_to_ensembl.get(symbol)
        if ensembl_id is None or ensembl_id not in gene_median or ensembl_id not in token_dict:
            continue
        normalized = (count / total_counts * TARGET_SUM) / gene_median[ensembl_id]
        scored.append((normalized, token_dict[ensembl_id]))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [token for _, token in scored[:MODEL_INPUT_SIZE]]


def embed_cell(token_ids: list[int]) -> np.ndarray:
    model = state["model"]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    return last_hidden.mean(dim=1).squeeze(0).numpy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["symbol_to_ensembl"] = _load_pickle(DICTS_DIR / "gene_name_id_dict_gc30M.pkl")
    state["gene_median"] = _load_pickle(DICTS_DIR / "gene_median_dictionary_gc30M.pkl")
    state["token_dict"] = _load_pickle(DICTS_DIR / "token_dictionary_gc30M.pkl")
    state["model"] = AutoModel.from_pretrained(MODEL_DIR).eval()

    embs_path = REFERENCE_DIR / "geneformer_embs.csv"
    labels_path = REFERENCE_DIR / "reference_labels.json"
    if embs_path.exists() and labels_path.exists():
        embs = pd.read_csv(embs_path)
        embedding_cols = [c for c in embs.columns if c.isdigit()]
        X = embs[embedding_cols].to_numpy()
        with open(labels_path) as f:
            labels = np.array(json.load(f))
        classifier = KNeighborsClassifier(n_neighbors=15, metric="cosine")
        classifier.fit(X, labels)
        state["classifier"] = classifier
    else:
        state["classifier"] = None

    yield
    state.clear()


app = FastAPI(title="Single-Cell Annotation API", lifespan=lifespan)


class CellRequest(BaseModel):
    gene_counts: dict[str, float]


class AnnotationResponse(BaseModel):
    predicted_cell_type: str
    confidence: float
    n_genes_used: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "reference_loaded": state.get("classifier") is not None}


@app.post("/annotate", response_model=AnnotationResponse)
def annotate(request: CellRequest) -> AnnotationResponse:
    if state.get("classifier") is None:
        raise HTTPException(503, "Reference embeddings not loaded; run src/embed_geneformer.py first")

    token_ids = tokenize_cell(request.gene_counts)
    if not token_ids:
        raise HTTPException(422, "No submitted genes matched the model's gene vocabulary")

    embedding = embed_cell(token_ids)
    classifier: KNeighborsClassifier = state["classifier"]
    probs = classifier.predict_proba([embedding])[0]
    top_idx = int(np.argmax(probs))

    return AnnotationResponse(
        predicted_cell_type=classifier.classes_[top_idx],
        confidence=float(probs[top_idx]),
        n_genes_used=len(token_ids),
    )
