"""QC and preprocessing of raw PBMC counts, independent of the dataset's precomputed layers.

Starts from adata.raw (unfiltered counts) so the QC/normalization steps are our own,
not the dataset publisher's. Ground-truth cell_type_lvl1 labels are kept aside purely
for evaluation later, never used during preprocessing or clustering.
"""

import argparse

import numpy as np
import scanpy as sc


def load_raw_with_labels(path: str) -> sc.AnnData:
    full = sc.read_h5ad(path)
    raw = full.raw.to_adata()
    raw.obs["cell_type_lvl1"] = full.obs["cell_type_lvl1"].values
    raw.obs["sample"] = full.obs["sample"].values
    return raw


def qc_and_filter(adata: sc.AnnData) -> sc.AnnData:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)

    adata = adata[adata.obs["n_genes_by_counts"].between(200, 6000)].copy()
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    return adata


def normalize_and_select_hvg(adata: sc.AnnData, n_top_genes: int = 2000) -> sc.AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return adata


def classical_pipeline(adata: sc.AnnData) -> sc.AnnData:
    hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(hvg, n_neighbors=15, n_pcs=50)
    sc.tl.leiden(hvg, key_added="leiden_classical", resolution=1.0, flavor="igraph", n_iterations=2)
    adata.obs["leiden_classical"] = hvg.obs["leiden_classical"].values
    adata.obsm["X_pca_classical"] = hvg.obsm["X_pca"]
    return adata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pbmc3k_pped.h5ad")
    parser.add_argument("--output", default="data/processed.h5ad")
    parser.add_argument("--subsample", type=int, default=4000, help="cells to keep, 0 = keep all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    adata = load_raw_with_labels(args.input)
    print(f"loaded raw: {adata.shape}")

    adata = qc_and_filter(adata)
    print(f"after QC: {adata.shape}")

    adata = adata[adata.obs["cell_type_lvl1"].notna()].copy()
    print(f"after dropping unlabeled cells: {adata.shape}")

    if args.subsample and adata.n_obs > args.subsample:
        idx = np.random.choice(adata.n_obs, args.subsample, replace=False)
        adata = adata[idx].copy()
        print(f"subsampled to: {adata.shape}")

    adata = normalize_and_select_hvg(adata)
    adata = classical_pipeline(adata)

    adata.write(args.output)
    print(f"wrote {args.output}")
    print(adata.obs["cell_type_lvl1"].value_counts())


if __name__ == "__main__":
    main()
