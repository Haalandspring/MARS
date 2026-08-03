# Runtime limitations

This page describes limitations of the reusable public mitigation system.
Datasets, model binaries, and benchmark outputs are distributed separately from
the source repository.

- **At least 512 channels:** the runtime intentionally rejects filterbanks with
  `C < 512`. The minimum cannot be lowered through configuration.
- **8-bit filterbanks only:** MARS writes uint8 samples while preserving the
  validated SIGPROC header. Other input bit depths fail explicitly rather than
  producing a header/data mismatch.
- **GPU-resident processing:** inference is tiled into segments and patches,
  but the current pipeline reads the complete observation and keeps full-size
  working arrays on the GPU. Long observations can exceed device memory.
- **No in-place cleaning:** input and output paths must be different. The
  source observation is never overwritten by `mars-mitigate`.
- **Whole-channel external preflags:** `preflag_channels` accepts upstream
  channel flags; an arbitrary channel-time external mask is not yet supported.
- **Optional processing changes science data:** hysteresis and `zdot` are
  disabled in the main mitigation profile. Runs that enable them must preserve
  the resolved configuration with the output.
- **Stack-local PyTorch determinism:** deterministic science profiles disable
  cuDNN autotuning and require deterministic algorithms, but byte identity is
  promised only for the same input, checkpoint, configuration, GPU, and
  PyTorch/CUDA/cuDNN stack.
- **TensorRT portability:** engines depend on TensorRT, CUDA, GPU architecture,
  precision, and build configuration. Build on the deployment machine and keep
  the verified metadata sidecar. PyTorch deterministic flags do not govern
  kernels inside the TensorRT engine.
- **No uncertainty estimate in a single mask:** the runtime produces a
  thresholded segmentation mask, not calibrated per-pixel uncertainty or an
  ensemble confidence interval.

The package fails explicitly at these boundaries instead of silently changing
the input contract or falling back to a different model/backend.
