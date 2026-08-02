# Data contract

Large filterbanks and NumPy arrays are intentionally excluded from Git. The
training loader expects four arrays by default:

```text
data/training_noise/train_patches.npy  # (4800, 512, 512)
data/training_noise/train_masks.npy    # (4800, 512, 512)
data/training_noise/val_patches.npy    # (1200, 512, 512)
data/training_noise/val_masks.npy      # (1200, 512, 512)
```

Patches are single-channel frequency-time arrays in the `tanh(z / 6)` domain;
masks are binary RFI targets. Paper-profile training rejects any other example
counts. The audited development repository did not include the observation
provenance, split-generation script, or hashes for these arrays.

Paper benchmarks additionally require immutable JSONL manifests named in
`configs/benchmarks/`. Those manifests and the two real GMRT observations were
not available in the source snapshot. Do not claim exact paper reproduction
until the missing provenance, inputs, and checksums have been published.
