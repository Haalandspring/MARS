# Model artifacts

The paper checkpoint, ONNX graph, TensorRT engine, and training history were not
present in the audited development repository, so this source release does not
pretend to ship them.

Expected local layout:

```text
artifacts/
  checkpoints/
    mars-paper/
      best_f1.pt
      history.csv
      config.json
  onnx/
  tensorrt/
```

`best_f1.pt` must embed the full resolved paper configuration, experiment ID
`mars-paper-2026-07-31`, matching training fingerprint, and 270,769 trainable
parameters. TensorRT engines are hardware- and version-bound; build one locally
with `mars-export-tensorrt` rather than treating an engine as a portable model
file. The paper pipeline rejects engines without a matching, successfully
verified sidecar.

Before a reproducibility release, add checkpoint/history download URLs and
SHA-256 hashes to this file. See `docs/reproducibility.md`.
