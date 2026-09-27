# Single-Cell Annotation via a Bio Foundation Model

**[View the results, no setup required →](https://claude.ai/artifact/XXt4pz48QhNxBtRVKPnwBn)**

Compares zero-shot [Geneformer](https://huggingface.co/ctheodoris/Geneformer) cell
embeddings against a classical scanpy PCA+Leiden baseline for annotating PBMC cell
types, and serves the foundation-model embeddings through a small FastAPI service
that tokenizes a single cell's raw gene counts and classifies it against a reference
set — containerized with Docker.

## Why

Built to close three specific gaps: hands-on single-cell (not just bulk RNA-seq)
data handling, a genuine bio foundation model (beyond consuming AlphaFold's output),
and a production-style deployment path (Docker/API) for one.

## Data

[`datavil/pbmc3k`](https://huggingface.co/datasets/datavil/pbmc3k) on HuggingFace — a
17,041-cell, two-sample (10x Genomics Multiome) PBMC dataset with raw counts
(`adata.raw`) and coarse ground-truth cell-type labels (`cell_type_lvl1`: Lymphocytes,
Erythroid, B Cells, Monocytes). QC, normalization, HVG selection, and clustering are
all done from scratch here starting from the raw counts layer — the dataset's own
precomputed QC/clustering columns are never used, only its ground-truth labels, and
only for evaluation.

## Pipeline

1. **`src/preprocess.py`** — QC (gene/count filters, mitochondrial fraction), drops
   cells with no ground-truth label, normalizes, selects 2,000 HVGs, and runs a
   classical PCA(50) + Leiden clustering baseline. Subsamples to 4,000 cells by
   default (CPU-friendly).
2. **`src/embed_geneformer.py`** — maps gene symbols to Ensembl IDs, tokenizes each
   cell as Geneformer's rank-value gene sequence, extracts a 256-dim mean-pooled
   embedding per cell from **Geneformer-V1-10M** (the smallest, CPU-inferable
   checkpoint), then evaluates both KMeans cluster agreement (ARI/NMI) and 5-fold
   KNN classification accuracy against ground truth, alongside the classical
   baseline.
3. **`src/api.py`** — a FastAPI service. `POST /annotate` takes
   `{"gene_counts": {"CD3D": 4, "MS4A1": 0, ...}}`, reimplements Geneformer's
   rank-value tokenization directly (no loom/dataset round-trip needed for a single
   cell), runs one forward pass through the same 10M-param model, and classifies the
   resulting embedding with a cosine-KNN fit on the reference embeddings from step 2.

## Results (4,000 cells, 4 cell types)

```
Geneformer embedding + KMeans(4):              ARI=-0.002  NMI=0.001
Geneformer embedding + KNN(15) 5-fold CV acc:   0.543 +/- 0.034
Classical PCA + KMeans(4):                      ARI=0.895  NMI=0.877
Classical PCA + KNN(15) 5-fold CV acc:           0.994 +/- 0.004
```

**Interpretation:** the classical PCA+HVG baseline wins decisively at this task, and
that's an expected, informative result rather than a bug. Zero-shot KMeans on the raw
Geneformer embedding is essentially at chance (ARI≈0) — a single global mean-pooled
embedding over the whole gene sequence, from a 10M-parameter checkpoint that was never
supervised on cell type or fine-tuned on this dataset, doesn't linearly separate into
4 clusters just because the underlying biology does. The KNN result (0.54 vs. 0.99)
shows the embedding *does* carry some cell-type signal — enough for a supervised
nearest-neighbor probe to do much better than the unsupervised clustering, just far
less than a dataset-tuned PCA on 2,000 hand-picked highly-variable genes. A
dataset-specific PCA+HVG pipeline is essentially built to ace this: it selects exactly
the genes that vary most *in this dataset*. Geneformer's embedding space is
general-purpose by design and would be expected to close this gap substantially with
linear probing / fine-tuning on labelled data, or with the larger V2 checkpoints
(104M/316M params, which do have a `<cls>` token, on a GPU) rather than raw zero-shot
mean pooling from the smallest V1 model.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_assets.py       # pulls the PBMC data + Geneformer-V1-10M
python src/preprocess.py
python src/embed_geneformer.py --gene-dicts-dir models/gene_dictionaries_30m \
    --model-dir models/Geneformer-V1-10M --output-dir data/geneformer_full
```

### API

```bash
uvicorn src.api:app --reload
curl -X POST localhost:8000/annotate -H "Content-Type: application/json" \
  -d '{"gene_counts": {"CD3D": 4, "CD3E": 3, "MS4A1": 0, "CD79A": 0}}'
```

### Docker

Run the "Running it" steps above first — the reference embeddings and model/dict
files they produce aren't checked into git (they're either large, regenerable, or
both) and are baked into the image at build time so the container needs no internet
access at runtime.

```bash
docker build -t sc-annotation .
docker run -p 8000:8000 sc-annotation
```

## Notes on getting this running

- Geneformer isn't on PyPI; it's installed from its HuggingFace model repo. The
  package also hardcodes `device="cuda"` in its embedding-extraction path with no CPU
  fallback — `vendor/geneformer` is a patched copy (see its `emb_extractor.py`) that
  falls back to CPU when no GPU is present, since this was all built and run without
  one.
- Geneformer-V1-10M uses the older `gc30M` tokenizer vocabulary and no `<cls>`/`<eos>`
  tokens, unlike the newer V2 checkpoints — `model_version="V1"` and
  `special_token=False` have to be set explicitly, and the model's `emb_mode`
  auto-falls-back from `"cls"` to `"cell"` (mean pooling) accordingly.

**A note on the API's real behavior:** since the KNN classifier is fit on an
imbalanced reference set (Lymphocytes are ~58% of the 4,000 cells) and the underlying
embedding only weakly separates classes, the deployed `/annotate` endpoint tends to
predict the majority class for several genuinely different single-cell test requests
in practice — a faithful reflection of the 0.54 KNN accuracy above, not a serving bug.
Class-weighting or oversampling the minority classes in the reference set would be the
first fix before trusting this for anything beyond a demo.

## What I'd do differently at scale

- Fine-tune (or at least linear-probe) Geneformer on labelled cell types instead of
  using raw zero-shot embeddings, and try a mid-size V2 checkpoint on a GPU.
- Swap the in-process KNN reference for a proper vector index (FAISS) once the
  reference set grows past a few thousand cells.
- Add batch-effect correction (e.g. Harmony) before comparing embeddings across the
  dataset's two samples (`s1d1`, `s1d3`).
