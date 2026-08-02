# Limitations

- **No final model artifact:** source code cannot validate the paper's reported
  metrics without the selected checkpoint and data.
- **GPU-resident, not out-of-core:** the pipeline tiles inference by segment but
  currently reads the complete filterbank and keeps full-size working arrays in
  memory. Long observations may exceed GPU memory.
- **8-bit input only:** output is uint8 and the input SIGPROC header is retained.
  The code rejects other input bit depths to avoid a corrupt header/data pair.
- **Whole-channel external preflags only:** `preflag_channels` supports upstream
  channel flags; an arbitrary channel-time preflag mask is not yet an input.
- **TensorRT portability:** engines are tied to TensorRT/CUDA/GPU details. Build
  locally and retain the generated, numerically verified metadata sidecar; the
  paper profile rejects missing or unverified sidecars.
- **Optional `zdot` ambiguity:** the paper does not unambiguously identify which
  reported full-filterbank results enabled it. Runs must state the switch.
- **No uncertainty calibration:** the paper reports deterministic fixed-seed
  aggregates rather than confidence intervals or run-to-run variance.
- **External tools:** PRESTO, SIGPROC, PulsarX/Filtool, SPECTRALib, and RFDL have
  their own installation, version, and licensing requirements.
