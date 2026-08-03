# Observation search workflow

This folder contains the code used on real observations. The normal sequence is:

1. mitigate RFI in an input filterbank;
2. run the PRESTO periodicity search; and
3. match the acceleration-search output to a target when validating a known source.

The scripts are thin entry points over the reusable implementation in
`src/mars_rfi/`; there is only one copy of the model and mitigation pipeline.

Install once from the repository root before using the scripts:

```bash
python -m pip install -e ".[mitigation]"
```

## 1. Mitigate RFI

```bash
python search/mitigate_rfi.py \
  --config configs/pipeline/mitigation.json \
  --input-fil observation.fil \
  --output-fil observation_mars.fil \
  --tensorrt-engine artifacts/tensorrt/mars-fp16.engine
```

For production work, use an engine whose sidecar records
`verification.status=passed`. Diagnostic engines whose verification failed are
not loaded by the production pipeline.

## 2. Run PRESTO

```bash
python search/run_presto.py \
  --fil observation_mars.fil \
  --dm 73.81 \
  --zmax 0 \
  --numharm 8 \
  --no-container \
  --out-dir search_outputs/J0139
```

Use `--sif <image.sif>` instead of `--no-container` when PRESTO is provided by
an Apptainer/Singularity image. Commands are printed before execution.

## 3. Match a known target

```bash
python search/match_candidates.py \
  --accel-output search_outputs/J0139/<ACCEL_FILE> \
  --target-period-ms <PERIOD_MS> \
  --output search_outputs/J0139/target-match.json
```

The equivalent installed commands are `mars-mitigate`,
`mars-search-presto`, and `mars-search-match`.
