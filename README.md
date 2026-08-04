# MARS

MARS (**M**orphology-**A**ware **R**FI **S**egmentation) removes radio-frequency
interference from 8-bit SIGPROC filterbank observations. The public interface is
one script and one configuration file:

```bash
python mars.py -f input.fil -o output.fil
```

The script predicts an RFI mask, replaces contaminated samples, applies the
configured zero-DM projection and baseline removal, rescales the data, and
writes a cleaned filterbank while preserving the SIGPROC header contract.

## Setup

The validated environment is Linux x86-64, Python 3.13, and an NVIDIA GPU with
a driver capable of CUDA 13. Create a clean environment, then install the exact
tested MARS, ONNX, and TensorRT dependencies:

```bash
conda create -n mars python=3.13 -y
conda activate mars
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

`requirements.txt` installs both execution paths: PyTorch FP16 inference and
ONNX/TensorRT FP16 compilation. Direct and transitive package versions are
locked to the clean environment used for the end-to-end checks. The system
NVIDIA driver is the only prerequisite that pip cannot install.

The production checkpoint is included in the repository at the location already
selected by `config.json`:

```text
artifacts/checkpoints/mars-paper-historical/best_f1.pt
```

Its SHA-256 is
`3acf3997bd83a8836e512974a2093bce319ede7f47ce3868757df545c4bd69a4`.
After installing the dependencies, run MARS directly from the repository:

```bash
python mars.py \
  -f /path/to/observation.fil \
  -o /path/to/observation_mars.fil
```

Only `-f` and `-o` are command-line options. Every processing option is kept in
[`config.json`](config.json).

If `-o` ends in `.fil`, it is used as the output filename. Otherwise it is
treated as an output directory:

```bash
python mars.py -f observation.fil -o cleaned/
```

This writes `cleaned/observation_mars.fil`.

## Configuration

Edit `config.json` before running. Important options include:

| Setting | Purpose |
| --- | --- |
| `checkpoint` | PyTorch checkpoint used when `tensorrt_path` is `null`. |
| `tensorrt_path` | Verified TensorRT engine; set to `null` for PyTorch. |
| `batch_size` | Patch inference batch size. |
| `threshold` | Binary RFI-mask threshold. |
| `raw_compute_dtype` | Pre/post-processing precision; default `float16`. |
| `hys_enabled` | Enable or disable mask hysteresis. |
| `science_zdot` | Enable the zero-DM projection. |
| `science_zdot_stage` | Apply `zdot` before or after replacement. |
| `replacement_fill_mode` | Replacement strategy for masked samples. |
| `final_baseline_enabled` | Enable final baseline removal. |
| `baseline_width` | Baseline width in seconds. |
| `rescale_mode` | Output rescaling method. |
| `write_mask_files` | Optionally write diagnostic mask filterbanks. |

The supplied production configuration uses FP16 computation, disables
hysteresis, enables `zdot` after RFI replacement, applies baseline removal, and
uses Filtool-style block rescaling.

### TensorRT FP16

TensorRT cannot run the `.pt` checkpoint directly. **Before enabling TensorRT,
first compile and verify an engine on the machine that will run MARS.** The
included checkpoint is used automatically.

After installing `requirements.txt`, compile on the deployment GPU:

```bash
python compile_tensorrt.py
```

`compile_tensorrt.py` reads `checkpoint`, `batch_size`, `patch_size`, and the
`tensorrt_build` section from `config.json`. It performs this mandatory sequence:

```text
.pt checkpoint -> ONNX export -> FP16 TensorRT build -> numerical verification
               -> .engine + .engine.json verification sidecar
```

Do not configure the engine until compilation reports that verification passed.
Then set:

```json
"tensorrt_path": "artifacts/tensorrt/mars-fp16.engine"
```

Run the normal interface; no additional TensorRT command-line option is needed:

```bash
python mars.py -f input.fil -o output.fil
```

The engine and its `.engine.json` sidecar must remain together. MARS rejects a
missing, mismatched, failed, or unverified engine instead of silently falling
back to PyTorch. Recompile when the GPU architecture, CUDA/TensorRT versions,
checkpoint, model shape, patch size, or batch size changes. To return to the
PyTorch FP16 path, set `tensorrt_path` back to `null`.

## Processing flow

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70}}}%%
flowchart TB
    IN["8-bit SIGPROC filterbank<br/>frequency channels × time samples"]

    subgraph LOAD["1 · Input validation and loading"]
        direction TB
        HDR["Read and preserve SIGPROC header"]
        CHECK["Validate 8-bit input<br/>and at least 512 channels"]
        GPU["Load observation to GPU<br/>raw storage: FP32 by default"]
        HDR --> CHECK --> GPU
    end

    subgraph MASK["2 · RFI-mask branch"]
        direction TB
        SEG["Split into approximately 2 s segments"]
        NORM["Robust per-channel normalization<br/>FP16 computation"]
        PATCH["Arrange normalized data as<br/>512 × 512 model patches"]
        BACKEND{"Inference backend selected<br/>by tensorrt_path"}
        PT["PyTorch checkpoint<br/>FP16 autocast"]
        TRT["Precompiled and verified<br/>TensorRT FP16 engine"]
        LOGIT["MARS logits"]
        THRESH["Threshold at 0.5<br/>hysteresis disabled"]
        RECON["Reconstruct the full-observation<br/>binary RFI mask"]
        SEG --> NORM --> PATCH --> BACKEND
        BACKEND -->|"null"| PT --> LOGIT
        BACKEND -->|"engine path"| TRT --> LOGIT
        LOGIT --> THRESH --> RECON
    end

    subgraph SCIENCE["3 · Science-data mitigation branch"]
        direction TB
        RAW["Keep the original observation<br/>separate from normalized NN input"]
        REPLACE["Replace mask-selected samples<br/>production mode: zero fill"]
        ZDOT["Zero-DM projection<br/>zdot after replacement"]
        BASE["Final time-domain baseline removal<br/>1 s window"]
        SCALE["Filtool-style block rescaling<br/>mean 128, standard deviation 6"]
        WRITE["Write uint8 samples with<br/>the preserved SIGPROC header"]
        RAW --> REPLACE --> ZDOT --> BASE --> SCALE --> WRITE
    end

    OUT["Cleaned SIGPROC filterbank"]
    IN --> HDR
    GPU --> SEG
    GPU --> RAW
    RECON -->|"mask controls replacement"| REPLACE
    WRITE --> OUT
```

## Model flow

The production model is the paper `TRTShapeUNet512` configuration: four encoder
widths `[8, 16, 32, 64]`, morphology-aware context, additive skip connections,
and horizontal/vertical refinement at every decoder scale. It has 270,769
trainable parameters.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 60, "rankSpacing": 72}}}%%
flowchart TB
    X["Normalized patch<br/>B × 1 × 512 × 512"]

    subgraph ENCODER["Encoder · Conv 3×3 + BatchNorm + ReLU"]
        direction TB
        S0["Stem / skip S0<br/>B × 8 × 512 × 512"]
        S1["Stride-2 down / skip S1<br/>B × 16 × 256 × 256"]
        S2["Stride-2 down / skip S2<br/>B × 32 × 128 × 128"]
        S3["Stride-2 down<br/>B × 64 × 64 × 64"]
        S0 --> S1 --> S2 --> S3
    end

    subgraph SHAPE["Morphology-aware bottleneck · 64 channels"]
        direction TB
        FORK{"Three parallel RFI-shape views"}
        LOCAL["Compact/local branch<br/>3 × 3 convolution"]
        HOR["Horizontal branch<br/>1 × 9 convolution"]
        VER["Vertical branch<br/>9 × 1 convolution"]
        CONCAT["Concatenate: 192 channels"]
        FUSE["1 × 1 fusion: 192 → 64<br/>plus residual input"]
        FORK --> LOCAL --> CONCAT
        FORK --> HOR --> CONCAT
        FORK --> VER --> CONCAT
        CONCAT --> FUSE
    end

    subgraph DECODER["Full-resolution additive decoder"]
        direction TB
        U2["Nearest upsample + 1×1 projection<br/>add skip S2 + 3×3 refine<br/>B × 32 × 128 × 128"]
        U2H["Horizontal morphology refine<br/>1 × 31 then 3 × 3 + residual"]
        U2V["Vertical morphology refine<br/>31 × 1 then 3 × 3 + residual"]
        U1["Nearest upsample + 1×1 projection<br/>add skip S1 + 3×3 refine<br/>B × 16 × 256 × 256"]
        U1H["Horizontal morphology refine<br/>1 × 31 then 3 × 3 + residual"]
        U1V["Vertical morphology refine<br/>31 × 1 then 3 × 3 + residual"]
        U0["Nearest upsample + 1×1 projection<br/>add skip S0 + 3×3 refine<br/>B × 8 × 512 × 512"]
        U0H["Horizontal morphology refine<br/>1 × 31 then 3 × 3 + residual"]
        U0V["Vertical morphology refine<br/>31 × 1 then 3 × 3 + residual"]
        U2 --> U2H --> U2V --> U1 --> U1H --> U1V --> U0 --> U0H --> U0V
    end

    HEAD["1 × 1 segmentation head<br/>B × 1 × 512 × 512 logits"]
    MASKOUT["Thresholded binary RFI mask"]

    X --> S0
    S3 --> FORK
    FUSE --> U2
    S2 -.-> U2
    S1 -.-> U1
    S0 -.-> U0
    U0V --> HEAD --> MASKOUT
```

## Supported inputs

- 8-bit SIGPROC filterbanks;
- at least 512 frequency channels (`C < 512` is intentionally unsupported);
- input and output must be different files; and
- the observation must fit the current GPU-resident working set.

The current implementation loads the complete observation into the
GPU-resident working set; it is not an out-of-core streaming implementation.

## Repository layout

```text
mars.py                simple RFI mitigation entry point
compile_tensorrt.py    config-driven FP16 TensorRT compiler and verifier
config.json            all user-selectable mitigation options
requirements.txt       runtime dependencies
src/mars_rfi/          minimal runtime and TensorRT implementation
artifacts/checkpoints/mars-paper-historical/best_f1.pt
                       production checkpoint required by mars.py
```

All other directories—including search, training, tests, article validation,
datasets, additional checkpoints, TensorRT binaries, generated filterbanks, and
benchmark results—are local-only and excluded from the public Git repository.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). MARS is
released under the [MIT License](LICENSE).
