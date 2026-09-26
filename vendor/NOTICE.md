`vendor/geneformer` is vendored from https://huggingface.co/ctheodoris/Geneformer
(Apache License 2.0), which is not distributed on PyPI. Two changes were made from
the upstream source:

1. `emb_extractor.py`: `device="cuda"` was hardcoded with no CPU fallback; patched to
   fall back to CPU when no GPU is available (this project was built and run without
   one).
2. The `gc30M` tokenizer dictionaries (`gene_dictionaries_30m/*.pkl`), required for
   the V1-10M checkpoint used here, are bundled directly rather than relying on the
   package's own (broken, in the version vendored here) data-file installation.

See the upstream repository for full attribution and the original license text.
