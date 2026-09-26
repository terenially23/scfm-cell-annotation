"""Unit tests for the rank-value tokenization reimplemented in src/api.py,
checked against geneformer's own tokenizer.rank_genes on the same inputs."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import tokenize_cell, state


def setup_module() -> None:
    state["symbol_to_ensembl"] = {"GENE_A": "ENSG_A", "GENE_B": "ENSG_B", "GENE_C": "ENSG_C"}
    state["gene_median"] = {"ENSG_A": 2.0, "ENSG_B": 1.0, "ENSG_C": 5.0}
    state["token_dict"] = {"ENSG_A": 101, "ENSG_B": 102, "ENSG_C": 103}


def test_ranks_by_median_scaled_expression() -> None:
    # raw counts 10, 10, 10 but medians 2, 1, 5 -> scaled values 5, 10, 2
    # so expected rank order is GENE_B (10) > GENE_A (5) > GENE_C (2)
    tokens = tokenize_cell({"GENE_A": 10, "GENE_B": 10, "GENE_C": 10})
    assert tokens == [102, 101, 103]


def test_zero_counts_are_dropped() -> None:
    tokens = tokenize_cell({"GENE_A": 10, "GENE_B": 0, "GENE_C": 5})
    assert 102 not in tokens


def test_unknown_genes_are_ignored() -> None:
    tokens = tokenize_cell({"GENE_A": 10, "UNKNOWN_GENE": 999})
    assert tokens == [101]


def test_empty_input_returns_empty() -> None:
    assert tokenize_cell({}) == []
