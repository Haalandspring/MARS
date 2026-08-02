# Reproducibility Status

The public structure is ready for a reproducibility release, but the audited
development snapshot did not include enough artifacts to reproduce the paper's
numbers. This page is deliberately explicit about the boundary between runnable
source and unavailable evidence.

## Current matrix

| Component | Status | Evidence / action needed |
| --- | --- | --- |
| Paper architecture and identity | Ready | One config; 270,769-parameter assertion plus experiment-ID/training-fingerprint tests that separate same-topology loss ablations. |
| Loss and online augmentation | Ready with correction | Paper loss is implemented; the three-family cardinality bug is fixed. Retraining is required to know its numerical effect. |
| Training CLI | Ready | Requires the unpublished 4,800/1,200 arrays. |
| PyTorch checkpoint/history | Missing | Publish epoch-49 `best_f1.pt`, `history.csv`, embedded config, URL, license, and SHA-256. |
| ONNX/TensorRT export | Ready, unverified locally | Requires CUDA/TensorRT and the missing checkpoint. Engines get strict provenance/verification sidecars and should be rebuilt per target system. |
| Filterbank mitigation | Source ready, integration unverified locally | Requires `sigpyproc`, CUDA hardware for paper performance, and a model artifact. Only 8-bit input is currently safe. |
| Training data provenance | Missing | Publish observation/source IDs, extraction and split scripts, array hashes, and data terms. |
| Synthetic RFI benchmark | Blocked | Missing SPECTRALib dependency pin/vendor, candidates, seeds, source patches, raw counts, and results. |
| FRB protection benchmark | Blocked | Missing clean/mixed candidate manifests, support arrays, seeds, checkpoints, and results. |
| Full-filterbank/PRESTO benchmark | Blocked | Missing deterministic 1,000-case manifest, filterbanks, cleaned outputs, and candidate tables. |
| Real GMRT benchmark | Blocked | Missing observation metadata/data and protocol-consistent reruns with one final model and no HYS. |
| Decoder ablations | Blocked | Missing four trained checkpoints, raw pooled counts, and generated summaries. |
| Runtime comparison | Blocked | Missing common input manifest, raw timings, RFDL code/results, and combined table producer. |
| RFDL baseline | External/missing | Record repository commit, environment, preprocessing, checkpoint hashes, and generated metrics. |

## Source-level workflow

1. Install the package and relevant extras.

   ```bash
   python -m pip install -e ".[filterbank,test]"
   ```

2. Put training arrays under `data/training_noise/` following
   [`data/README.md`](../data/README.md), then run:

   ```bash
   mars-train --config configs/train/paper.json
   ```

3. Confirm the log prints exactly 270,769 trainable parameters and that the
   selected artifact contains its full config, `mars-paper-2026-07-31`
   experiment ID, and matching training fingerprint.

4. Export and verify ONNX/TensorRT on the target GPU:

   ```bash
   mars-export-tensorrt \
     --checkpoint artifacts/checkpoints/mars-paper/best_f1.pt \
     --batch-size 64 --precision fp16 --io-dtype fp16
   ```

5. Run a cleaned filterbank with a fully recorded config. Keep hysteresis and
   `zdot` explicit in the command or archived config.

6. Run tests:

   ```bash
   python -m pytest
   ```

## Minimum artifact release

For each binary/data artifact, publish:

- stable download URL and license/access terms;
- SHA-256, byte size, shape/dtype (where applicable);
- producer command and Git commit;
- parent/input hashes;
- experiment ID, artifact role, and canonical training fingerprint;
- random seed or immutable per-case seed manifest;
- software/hardware metadata where results depend on CUDA, TensorRT, Filtool,
  PRESTO, or CPU thread placement.

Required named artifacts include:

```text
best_f1.pt
history.csv
train_patches.npy / train_masks.npy
val_patches.npy / val_masks.npy
synthetic_rfi_patches.jsonl
frb_protection.jsonl
full_filterbank_presto.jsonl
decoder_ablation_counts.csv
real_gmrt_candidates.csv
runtime_raw_repetitions.csv
```

The manifest schema in [`reproduction/paper-manifest.json`](../reproduction/paper-manifest.json)
tracks which paper outputs are still blocked.
