# Local data

Datasets and generated observations are intentionally not stored in Git. The
repository ignores everything under `data/` except this file, as well as all
`*.fil`, `*.npy`, `*.npz`, and `*.jsonl` files anywhere in the workspace.

For inference, an input can be any supported 8-bit SIGPROC filterbank. It does
not need to be copied into this directory:

```bash
mars-mitigate \
  --config configs/pipeline/mitigation.json \
  --input-fil /path/to/observation.fil \
  --output-fil outputs/observation_mars.fil
```

For training, provide frequency-time patch arrays and binary RFI masks through
the paths expected by the training configuration. Patches are single-channel
arrays in the `tanh(z / 6)` domain; masks contain binary targets. Keep all such
arrays local or distribute them through a dedicated dataset archive with its
own license, provenance, checksums, and versioning.

Benchmark inputs, real observations, generated candidates, and result tables
belong in ignored local directories such as `data/`, `outputs/`, or `results/`.
