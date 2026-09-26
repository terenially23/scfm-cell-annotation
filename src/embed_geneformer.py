"""Extract Geneformer cell embeddings for the processed AnnData and compare against
the classical PCA+Leiden baseline using the held-out cell_type_lvl1 labels.

Geneformer tokenizes each cell as a rank-ordered list of Ensembl gene IDs (by expression,
normalized against the gene's median expression in a large reference corpus), so raw
counts (not log-normalized values) are required as input, and gene symbols must be
mapped to Ensembl IDs using the package's own dictionary.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def build_ensembl_symbol_map(gene_dict_path: str) -> dict[str, str]:
    with open(gene_dict_path, "rb") as f:
        symbol_to_ensembl: dict[str, str] = pickle.load(f)
    return symbol_to_ensembl


def prepare_tokenizer_input(adata: sc.AnnData, symbol_to_ensembl: dict[str, str]) -> sc.AnnData:
    adata = adata.copy()
    adata.X = adata.layers["counts"].copy()

    ensembl_ids = [symbol_to_ensembl.get(g, None) for g in adata.var_names]
    keep = [i for i, e in enumerate(ensembl_ids) if e is not None]
    print(f"mapped {len(keep)}/{adata.n_vars} genes to Ensembl IDs")

    adata = adata[:, keep].copy()
    adata.var["ensembl_id"] = [ensembl_ids[i] for i in keep]
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()
    return adata


def extract_embeddings(
    adata: sc.AnnData, model_dir: str, gene_dicts_dir: str, output_dir: str
) -> tuple[np.ndarray, np.ndarray]:
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer, EmbExtractor

    tmp_h5ad = Path(output_dir) / "for_tokenizer.h5ad"
    tmp_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write(tmp_h5ad)

    loom_dir = Path(output_dir) / "loom_input"
    loom_dir.mkdir(parents=True, exist_ok=True)
    adata.write_loom(loom_dir / "cells.loom")

    tokenizer = TranscriptomeTokenizer(
        {"cell_type_lvl1": "cell_type_lvl1"},
        nproc=4,
        model_version="V1",
        model_input_size=2048,
        special_token=False,
        gene_median_file=f"{gene_dicts_dir}/gene_median_dictionary_gc30M.pkl",
        token_dictionary_file=f"{gene_dicts_dir}/token_dictionary_gc30M.pkl",
        gene_mapping_file=f"{gene_dicts_dir}/ensembl_mapping_dict_gc30M.pkl",
    )
    tokenized_dir = Path(output_dir) / "tokenized"
    tokenizer.tokenize_data(str(loom_dir), str(tokenized_dir), "cells", file_format="loom")

    extractor = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cls",
        model_version="V1",
        token_dictionary_file=f"{gene_dicts_dir}/token_dictionary_gc30M.pkl",
        max_ncells=adata.n_obs,
        emb_layer=-1,
        forward_batch_size=8,
        nproc=4,
    )
    embs = extractor.extract_embs(
        model_dir,
        str(tokenized_dir / "cells.dataset"),
        output_dir,
        "geneformer_embs",
    )

    tokenized_dataset = load_from_disk(str(tokenized_dir / "cells.dataset"))
    labels = np.array(tokenized_dataset["cell_type_lvl1"])
    return embs, labels


def evaluate(embedding: np.ndarray, labels_true: np.ndarray, pca_reference: sc.AnnData, n_clusters: int) -> None:
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(embedding)
    ari = adjusted_rand_score(labels_true, km.labels_)
    nmi = normalized_mutual_info_score(labels_true, km.labels_)
    print(f"Geneformer embedding + KMeans({n_clusters}): ARI={ari:.3f} NMI={nmi:.3f}")

    knn_acc = cross_val_score(
        KNeighborsClassifier(n_neighbors=15, metric="cosine"), embedding, labels_true, cv=5
    )
    print(f"Geneformer embedding + KNN(15) 5-fold CV accuracy: {knn_acc.mean():.3f} +/- {knn_acc.std():.3f}")

    # Classical baseline is aligned to the *original* adata order, not the (possibly
    # reordered/filtered) tokenizer output, so re-align by matching label composition size.
    if pca_reference.n_obs == len(labels_true):
        pca_km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(
            pca_reference.obsm["X_pca_classical"]
        )
        ari_pca = adjusted_rand_score(pca_reference.obs["cell_type_lvl1"].values, pca_km.labels_)
        nmi_pca = normalized_mutual_info_score(pca_reference.obs["cell_type_lvl1"].values, pca_km.labels_)
        print(f"Classical PCA + KMeans({n_clusters}): ARI={ari_pca:.3f} NMI={nmi_pca:.3f}")

        knn_acc_pca = cross_val_score(
            KNeighborsClassifier(n_neighbors=15),
            pca_reference.obsm["X_pca_classical"],
            pca_reference.obs["cell_type_lvl1"].values,
            cv=5,
        )
        print(f"Classical PCA + KNN(15) 5-fold CV accuracy: {knn_acc_pca.mean():.3f} +/- {knn_acc_pca.std():.3f}")
    else:
        print(
            f"Skipping classical baseline comparison: {pca_reference.n_obs} cells in original "
            f"data vs {len(labels_true)} cells retained by the tokenizer."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed.h5ad")
    parser.add_argument("--gene-dicts-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="data/geneformer_run")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.input)
    symbol_to_ensembl = build_ensembl_symbol_map(f"{args.gene_dicts_dir}/gene_name_id_dict_gc30M.pkl")
    prepped = prepare_tokenizer_input(adata, symbol_to_ensembl)

    embs, labels = extract_embeddings(prepped, args.model_dir, args.gene_dicts_dir, args.output_dir)
    embedding = embs.values if hasattr(embs, "values") else np.asarray(embs)

    n_clusters = adata.obs["cell_type_lvl1"].nunique()
    evaluate(embedding, labels, adata, n_clusters)

    np.save(Path(args.output_dir) / "embeddings.npy", embedding)
    with open(Path(args.output_dir) / "reference_labels.json", "w") as f:
        json.dump(labels.tolist(), f)


if __name__ == "__main__":
    main()
