# Model artifacts

Model checkpoints, ONNX graphs, TensorRT engines, and training histories are
generated or separately distributed artifacts. They are intentionally ignored
by Git and are not part of the public source repository.

The default mitigation profile expects this local layout:

```text
artifacts/
  checkpoints/
    mars-paper-historical/
      best_f1.pt
  onnx/
  tensorrt/
```

Place the released checkpoint at the path configured by
`configs/pipeline/mitigation.json`, or pass another compatible checkpoint
explicitly to `mars-mitigate`. A checkpoint must match the configured MARS
architecture and artifact identity; the runtime rejects incompatible metadata
instead of silently loading a different model.

TensorRT engines are tied to the GPU architecture and TensorRT/CUDA software
stack. Build and verify an engine on the deployment system:

```bash
mars-export-tensorrt \
  --checkpoint artifacts/checkpoints/mars-paper-historical/best_f1.pt \
  --batch-size 64 \
  --precision fp16 \
  --io-dtype fp16
```

The exporter writes a sidecar containing the checkpoint and engine hashes,
model identity, build environment, and numerical verification result. Use an
engine in the mitigation pipeline only when that verification passed.

Public model releases should provide a stable download location, SHA-256 hash,
license/access terms, and the compatible MARS version. Local copies remain
ignored even after they are placed under `artifacts/`.
