# Paper reproduction layer

This directory holds clean, protocol-facing utilities and a status manifest.
The hundreds of historical versioned scripts were deliberately not copied:
many referred to personal mount points, pilot-scale grids, V5 checkpoints, or
hand-written intermediate CSVs.

The JSON files in `../configs/benchmarks/` are the authoritative paper-scale
protocols. Exact runs additionally require immutable case manifests and the
artifact hashes listed in `paper-manifest.json`.

Metric rule: paper overall RFI precision/recall/F1 must be computed from summed
`TP/FP/FN/TN` pixel counts (micro aggregation). A mean of per-patch F1 values is
a different metric and must be labeled `macro_patch_f1`.

`evaluate_patch_masks.py` can evaluate a published pair of `.npy` arrays and an
optional metadata JSONL manifest using the paper checkpoint. It does not create
the missing synthetic candidates.
