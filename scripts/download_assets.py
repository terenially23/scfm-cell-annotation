"""Downloads everything the pipeline needs that isn't checked into the repo:
the PBMC dataset mirror, and Geneformer's smallest (10M param, CPU-friendly)
model plus its matching gc30M tokenizer dictionaries.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    hf_hub_download(
        repo_id="datavil/pbmc3k",
        filename="pbmc3k_pped.h5ad",
        repo_type="dataset",
        local_dir=str(data_dir),
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    snapshot_dir = Path(
        snapshot_download(
            repo_id="ctheodoris/Geneformer",
            allow_patterns=["Geneformer-V1-10M/*", "geneformer/gene_dictionaries_30m/*"],
        )
    )

    import shutil

    shutil.copytree(snapshot_dir / "Geneformer-V1-10M", models_dir / "Geneformer-V1-10M", dirs_exist_ok=True)
    shutil.copytree(
        snapshot_dir / "geneformer" / "gene_dictionaries_30m",
        models_dir / "gene_dictionaries_30m",
        dirs_exist_ok=True,
    )
    # The vendored (CPU-patched) geneformer package expects these dictionaries
    # bundled alongside its own source at this path when model_version="V1".
    shutil.copytree(
        snapshot_dir / "geneformer" / "gene_dictionaries_30m",
        ROOT / "vendor" / "geneformer" / "gene_dictionaries_30m",
        dirs_exist_ok=True,
    )
    print("Downloaded PBMC dataset and Geneformer-V1-10M into data/, models/, and vendor/geneformer/")


if __name__ == "__main__":
    main()
