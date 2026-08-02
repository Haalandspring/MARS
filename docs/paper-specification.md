# Paper-Facing Specification

This document records the code-facing specification stated in the MARS paper,
*MARS: A Lightweight Morphology-Aware RFI Segmentation Network for Mask-Guided
Mitigation in Radio Astronomy* (draft dated 31 July 2026). It is intended as a
traceable reference for repository configuration and reproducibility work. It
does not add implementation details that are absent from the paper.

## Scope and terminology

The paper distinguishes between:

- **MARS**, the neural mask predictor; and
- the **GPU RFI mitigation pipeline**, which performs preprocessing, invokes
  MARS, reconstructs and applies the mask, conditions the output, and writes a
  cleaned filterbank.

MARS localises RFI; it does not synthesize a cleaned observation. The learned
mask and the replacement policy are separate parts of the system. (Paper
§2.1 / pp. 3–4; §4 / p. 9)

## Neural architecture

### Inputs and outputs

- Input: one normalized frequency-time patch with shape `1 × H × W`.
- Deployment patch size: `H = W = 512`.
- The frequency axis is `H`; the time axis is `W`.
- Output: one full-resolution RFI logit map with shape `1 × H × W`.
- RFI probabilities are obtained with a sigmoid.
- The reported model has **270,769 trainable parameters**.

(Paper §2.1 / pp. 3–4)

### Encoder-decoder backbone

- Compact U-Net-like encoder-decoder with channel widths
  `(8, 16, 32, 64)`.
- Feature blocks use convolution, batch normalization, and ReLU.
- The encoder has three stride-two downsampling stages.
- The decoder uses nearest-neighbor upsampling.
- Upsampled decoder features are projected by `1 × 1` convolutions and fused
  with the corresponding encoder features by addition, rather than channel
  concatenation.
- A final `1 × 1` convolution produces the single RFI-logit channel.

(Paper §2.1 / p. 3; Table 1 / p. 4)

### Morphology-aware bottleneck

The deepest feature map is processed by three parallel convolutional branches:

- local branch: `3 × 3`;
- horizontal branch: `1 × 9`, targeting narrowband structures extended in
  time; and
- vertical branch: `9 × 1`, targeting broadband or short-duration structures
  extended in frequency.

The branch outputs are concatenated, fused by a `1 × 1` Conv-BN-ReLU block,
and added to the bottleneck input as a residual correction. (Paper §2.1 /
p. 3)

### Decoder refinement

The three decoder upsampling stages, from coarse to fine, are called `up2`,
`up1`, and `up0`; their output resolutions are `H/4 × W/4`, `H/2 × W/2`, and
`H × W` respectively. After each upsampling stage, refinement is applied in
this order:

1. horizontal residual block: `1 × 31` convolution followed by `3 × 3`;
2. vertical residual block: `31 × 1` convolution followed by `3 × 3`.

(Paper §2.1 / p. 3)

## End-to-end data flow

The paper specifies the following processing sequence. (Paper §§2.2–2.5 /
pp. 4–7)

1. Read raw filterbank data as a channel-by-time array.
2. Divide the observation into temporal segments whose duration is close to
   two seconds and whose length is tileable by 512 time samples.
3. Before normalization, screen the raw segment for degenerate, saturated,
   unusable, persistently contaminated, or otherwise pre-flagged channel
   segments. Retain these flags as auxiliary masks.
4. Normalize each channel independently using its median and scaled median
   absolute deviation, with a standard-deviation fallback.
5. Apply bounded `tanh` compression to form the neural-network input.
6. Construct `512 × 512` patches and run batched neural inference.
7. Apply sigmoid and the configured mask threshold.
8. Reconstruct the segment-level neural mask and combine it with the auxiliary
   masks.
9. Optionally promote a channel to a segment-wide flag based on neural-mask
   occupancy.
10. Apply the final mask to a separate mean/std-normalized representation of
    the original data.
11. Optionally apply post-replacement zero-DM matched filtering (`zdot`).
12. Remove the baseline with a running-median profile.
13. Rescale the complete stream to an 8-bit filterbank and force replaced
    pixels to the output mean.
14. Return the final output to the CPU for filterbank writing. The other listed
    processing stages are described as GPU operations.

## Segmentation, normalization, and patch construction

### Segment length

The target duration is `T_seg = 2 s`. With patch size `P = 512` and sampling
time `t_samp`, the number of time patches and actual segment length are:

```text
K = max(1, round(T_seg / (P * t_samp)))
L = K * P
```

(Paper §2.2 / p. 4)

### Network-input normalization

For each channel in a segment, the paper computes the channel median and a
Gaussian-consistent scaled median absolute deviation. If that scale is below a
minimum valid value, the channel standard deviation is used instead. A
safeguarded scale is used to avoid division by zero. The normalized value `z`
is then compressed as:

```text
x = tanh(z / 6)
```

The `tanh` scale is therefore **6**. (Paper §2.2 / pp. 4–5)

### Pre-normalization screening

- A channel segment is degenerate when its sample standard deviation is below
  `1e-6`.
- A channel segment is saturated when more than 50% of its samples exceed a
  robust segment-level bright threshold.
- Persistent contamination is inferred from outlying channel medians,
  channel spreads, bright-sample occupancy, and coherence across neighboring
  frequency channels.
- These flags are recorded before per-channel normalization and retained as
  auxiliary masks.

(Paper §2.2 / p. 5)

### Patch tiling and reconstruction

- Time is tiled with non-overlapping 512-sample blocks.
- Frequency uses complete non-overlapping 512-channel blocks.
- If the number of channels is not a multiple of 512, one additional block is
  taken from the final 512 channels. During reconstruction, only rows not
  already covered by complete blocks are copied from this overlapping block.
- For fewer than 512 channels, blocks from different time intervals are
  stacked to construct a `512 × 512` network input and separated after
  inference.
- A final temporal tail shorter than 512 samples receives no neural prediction;
  it is treated as masked and assigned the neutral replacement value.

(Paper §2.2 / p. 5)

## Inference and replacement defaults

Unless a benchmark explicitly says otherwise, the paper-facing defaults are:

| Setting | Paper value |
| --- | ---: |
| Sigmoid mask threshold | `0.5` |
| Hysteresis in reported main results | disabled |
| Degenerate channel std threshold | `1e-6` |
| Channel neural-mask occupancy promotion threshold | `0.5` |
| Replacement in mean/std-normalized stream | zero |
| Baseline running-median width | `1.0 s` |
| Output type | 8-bit filterbank |
| Output target mean | `128` |
| Output target standard deviation | `6` |
| Final value for masked samples | `128` |

(Paper §§2.2–2.4 / pp. 5–7; §4.5 / p. 13)

The network-input representation and replacement representation are distinct:
the former uses robust median/scaled-MAD normalization plus `tanh`; the latter
uses channel mean/std normalization to follow a Filtool-like equalization
convention. (Paper §2.3 / p. 6)

### Optional hysteresis

The optional completion stage defines a high-confidence seed mask and a lower
confidence support mask. It grows seed pixels through connected support pixels
using anisotropic dilation, with optional row-support gating, temporal closing,
and final dilation. It guards against excessive growth by limiting both total
masked fraction and the fraction added beyond the seed; exceeding either limit
causes a fallback to the original seed mask. Seed pixels are always retained.

Hysteresis is disabled for the reported synthetic RFI, FRB-protection,
full-filterbank, and real-GMRT main results. (Paper §2.4 / pp. 6–7; Appendix C /
p. 28)

### Deployment format

The PyTorch checkpoint is exported to ONNX and compiled into a TensorRT engine
with fixed `512 × 512` spatial dimensions. The production configuration uses
FP16 input, output, and inference, and supports batched patches. (Paper §2.5 /
p. 7)

## Training defaults

| Setting | Paper value |
| --- | ---: |
| Training source patches | `4,800` |
| Validation source patches | `1,200` |
| Patch shape | `512 × 512` |
| Random seed | `1234` |
| Epochs | `50` |
| Batch size | `64` |
| Optimizer | AdamW |
| Initial learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Positive-class BCE weight | `3.0` |
| Dice coefficient | `1.0` |
| Astronomy-preservation coefficient | `1.0` |
| LR plateau patience | `4 epochs` |
| LR reduction factor | `0.5` |
| Minimum learning rate | `1e-6` |
| Training augmentation | enabled |
| Validation augmentation | disabled |
| Validation hysteresis | disabled |

All 50 epochs are completed. The checkpoint with the highest validation F1 is
selected; the paper reports validation F1 `0.9962` at epoch 49 for the selected
checkpoint. (Paper §3 / pp. 7–9; Figure 2 / p. 9; Appendix B / p. 27)

### Loss

The total objective is the sum of:

- pixel-weighted, positive-class-weighted binary cross entropy;
- a weighted Dice loss computed per patch and averaged over the batch; and
- an astronomy-preservation penalty on clean injected-pulse pixels.

The astronomy term averages squared predicted RFI probability on pixels that
belong to the astronomical support mask but not the RFI target. If a batch has
no astronomical support pixels, the same type of penalty is applied over
non-RFI pixels. The paper's no-astronomy-loss ablation sets the astronomy
coefficient to zero while keeping the architecture, data, augmentation, seed,
optimizer, learning rate, batch size, and schedule unchanged. (Paper §3 /
p. 8)

### Online augmentation

Training augmentation is applied online in the normalized input domain and
includes horizontal, vertical, compact, block-like, periodic, and composite
RFI. A fraction of samples remains RFI-negative. Synthetic dispersed pulses
are also injected as non-RFI examples. The stated shared augmentation settings
are:

- sampling times: `128, 256, 512, 1024, 1310 µs`;
- at most three RFI families per patch;
- extra-family probability: `0.35`;
- main FRB pulse-width range: `1–10 ms`;
- low-DM FRB pulse-width range: `1–5 ms`;
- low-DM bright-medium pulse-width range: `3–6 ms`.

The paper also describes low-DM and bright-pulse augmentation branches. The
astronomical support mask excludes injected pulse pixels that are part of the
RFI target. (Paper §3 / pp. 7–8; Appendix B / p. 27)

## Benchmark specifications and reported results

### Synthetic RFI mask benchmark

Protocol:

- clean `512 × 512` backgrounds;
- SPECTRALib 0.0.23 injection in a raw-like amplitude domain;
- six families: persistent narrowband, repeating narrowband, impulsive
  narrowband, repeating broadband, impulsive broadband, and mixed RFI;
- five sampling times: `128, 256, 512, 1024, 1310 µs`;
- 500 fixed-seed examples per family and sampling time;
- `6 × 5 × 500 = 15,000` patches;
- identical candidate patches for MARS and RFDL;
- threshold `0.5`; no hysteresis;
- metrics: pixel precision, recall, and F1.

(Paper §4.1 / pp. 9–10; Appendix E / pp. 29–30)

Reported aggregate results:

| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| MARS | 0.995 | 0.963 | 0.978 |
| RFDL | 0.903 | 0.946 | 0.924 |

MARS family F1 values are 0.999 impulsive broadband, 0.986 impulsive
narrowband, 0.985 persistent narrowband, 0.983 mixed, 0.966 repeating
broadband, and 0.948 repeating narrowband. Its mean F1 across all
family/sampling-time cells is 0.977, versus 0.800 for RFDL. MARS repeating
narrowband F1 decreases from 0.980 at 128 µs to 0.912 at 1310 µs. (Paper §5.1 /
pp. 15–16)

### Astronomical-signal protection benchmark

Protocol:

- two cases: clean signal-only and signal plus mixed RFI;
- five sampling times: `128, 256, 512, 1024, 1310 µs`;
- 13 DMs: `10, 20, 30, 50, 75, 100, 150, 300, 600, 1000, 1500, 2000,
  3000 pc cm^-3`;
- nominal peak S/N: `6, 10, 20`;
- pulse widths: `1.31, 5.24, 15.72 ms`;
- 16 fixed-seed repeats per grid point;
- `5 × 13 × 3 × 3 × 16 = 9,360` patches per case;
- pulse/RFI support overlap is limited to at most 15%;
- MARS, RFDL, and no-astronomy-loss models are compared;
- threshold `0.5`; no hysteresis.

The metric is retained injected fluence. In the mixed-RFI case, pulse pixels
that overlap ground-truth RFI are excluded from the evaluated pulse support.
(Paper §4.2 / pp. 10–11; Appendix E / pp. 29–30)

Reported mean retained fluence:

| Model | Clean signal | Signal + mixed RFI |
| --- | ---: | ---: |
| MARS | 0.976 | 0.964 |
| RFDL | 0.577 | 0.536 |
| MARS without astronomy loss | 0.888 | 0.837 |

At DM `10 pc cm^-3` and S/N 20, MARS retains 0.86 in the clean case and 0.83
with mixed RFI; RFDL retains 0.19 and 0.17. In the clean case, the same point
improves from 0.48 without the astronomy loss to 0.86 with it. (Paper §5.2 /
pp. 16–19)

### Synthetic full-filterbank PRESTO benchmark

Protocol:

- SIGPROC `fake` binary-pulsar filterbanks;
- SPECTRALib contamination;
- 4096 channels, 8-bit samples, 300 s duration;
- first-channel frequency `1550 MHz`;
- channel offset `-0.292968752 MHz`;
- five sampling times, 200 fixed-seed filterbanks per sampling time, for 1,000
  contaminated inputs;
- each contaminated file is supplied unchanged to the GPU pipeline and
  `filtool`;
- PRESTO defaults: `zmax=200`, `numharm=8`, `numdms=1`, `nobary=True`,
  `zerodm=False`, and red-noise removal enabled;
- search flow: `prepsubband`, `realfft`, red-noise removal, `accelsearch`;
- metric: significance of the candidate matched to the injected target period
  or frequency, including harmonic matching.

(Paper §4.3 / p. 12; Appendix D / pp. 28–29; Appendix E / p. 30)

SIGPROC generation ranges:

| Parameter | Paper distribution/range |
| --- | --- |
| `seed` | integer in `[0, 2147483647]` |
| period at 128/256 µs | 1–10 ms |
| period at 512 µs | 1.5–10 ms |
| period at 1024 µs | 2.5–10 ms |
| period at 1310 µs | 3–10 ms |
| width | 5–50% of period |
| `snrpeak` | 0.08–0.10 |
| DM | 50–3000 pc cm^-3 |
| binary period | 1–10 h |
| pulsar/companion mass | 0.5–5 solar masses each |
| orbital phase | 0–1 |
| eccentricity | 0–1 |

Period, width, `snrpeak`, DM, binary period, and masses are sampled
log-uniformly; phase and eccentricity are sampled uniformly. `nifs=1`, start
time MJD 50000, and SIGPROC dispersion and sampling-time smearing remain
enabled. (Paper Appendix D / pp. 28–29)

Reported median GPU-to-`filtool` target-significance ratios:

| Sampling time | Median ratio | Median GPU-minus-`filtool` significance |
| ---: | ---: | ---: |
| 128 µs | 0.946 | -1.12 |
| 256 µs | 0.913 | -1.38 |
| 512 µs | 0.898 | -1.15 |
| 1024 µs | 0.982 | -0.15 |
| 1310 µs | 0.988 | -0.08 |

(Paper §5.3 / p. 19)

### Real GMRT checks

Both cleaned outputs use matched PRESTO settings and red-noise removal.
Because `zmax=0`, these are non-accelerated periodicity searches performed with
`accelsearch`. (Paper §4.4 / pp. 12–13)

| Source | DM (pc cm^-3) | `zmax` | `numharm` | GPU pipeline significance | `filtool` significance |
| --- | ---: | ---: | ---: | ---: | ---: |
| J0139+5814 | 73.81 | 0 | 8 | 13.89 | 13.88 |
| J0659+1414 | 14.05 | 0 | 16 | 17.09 | 16.90 |

(Paper §5.4 / pp. 19–20; Table 7 / p. 20)

### Compute-time benchmark

Protocol:

- GPU: NVIDIA GH200;
- CPU: AMD EPYC 9825;
- input: 4096 channels and approximately 100 s duration;
- one excluded warm-up plus three measured runs;
- report the mean of the three measured runs;
- use the fastest tested mean among `filtool` CPU thread counts;
- MARS pipeline uses TensorRT FP16;
- exclude file I/O, model/engine loading, output writing, and PRESTO;
- GPU pipeline time includes preprocessing, normalization, patch construction,
  inference, reconstruction, replacement, zero-DM filtering, baseline
  correction, and rescaling;
- RFDL time covers mask-prediction preprocessing and neural forward pass only.

(Paper §4.6 / pp. 13–14)

Reported ranges across sampling times:

| System | Compute time |
| --- | ---: |
| GPU mitigation pipeline | 0.34–3.31 s |
| `filtool` | 2.08–22.8 s |
| RFDL mask prediction | 23.3–240.7 s |

The paper reports the GPU pipeline as 6.2–7.0 times faster than the fastest
tested multi-threaded `filtool`, while RFDL mask prediction alone takes
69.3–73.6 times the GPU full-pipeline time. (Paper §5.5 / p. 21)

### Decoder-refinement ablation

Each variant is retrained from scratch under the MARS training protocol and
evaluated on the same fixed benchmark sets without hysteresis. RFI F1 is pooled
over 15,000 patches; mixed-RFI FRB retention and RFI-only recall use 9,360
patches and exclude physical FRB/RFI overlap. (Paper Appendix A / pp. 24–27)

| Variant | RFI F1 | Mixed-RFI FRB retention | RFI-only recall |
| --- | ---: | ---: | ---: |
| MARS | 0.978 | 0.964 | 0.992 |
| No vertical refinement | 0.969 | 0.935 | 0.989 |
| No horizontal refinement | 0.985 | 0.928 | 0.990 |
| No decoder refinement | 0.969 | 0.888 | 0.986 |

At DM `10 pc cm^-3`, S/N 20 in the mixed-RFI benchmark, retained fluence is
0.83, 0.74, 0.68, and 0.53 respectively. (Paper Appendix A / p. 27)

## `filtool` baseline command

The paper gives the following command template. (Paper §4.5 / p. 13)

```bash
filtool -t <NTHREADS> -l 2 --baseline 1.0 --zapthre 4.0 \
  --rfi zdot mask 2 2 kadaneF 2 2 kadaneT 2 2 \
  -o <OUTPUT_ROOT> -f <INPUT.fil>
```

## Parameters required for exact reproduction but not specified by the paper

The following values or artifacts are needed to reproduce the reported runs
exactly, but the paper does not state them. Repository configuration should
make these choices explicit rather than silently presenting them as paper
defaults.

### Model construction

- exact number of convolutions in each ordinary encoder/decoder feature block;
- convolution kernels, padding, bias settings, and downsampling operators not
  otherwise listed above;
- branch output widths in the morphology-aware bottleneck;
- exact BN/ReLU placement, padding, and residual projections in decoder
  refinement blocks;
- weight initialization and batch-normalization hyperparameters.

### Training and loss

- source and provenance of the 4,800/1,200 background patches;
- the split-generation procedure and a manifest or hashes for the split;
- definition and construction of the per-pixel weight map;
- Dice numerical epsilon and exact empty-mask behavior;
- AdamW beta and epsilon values;
- learning-rate scheduler mode, threshold, cooldown, and other unstated
  scheduler options;
- AMP/precision policy, gradient clipping, data-loader worker seeds, and
  deterministic backend settings;
- exact RFI-family sampling probabilities and the probability of an RFI-free
  sample;
- training RFI amplitude, position, width, duration, duty-cycle, repetition,
  and persistence distributions;
- training FRB DM, S/N, placement, profile, and branch probabilities.

### Preprocessing and inference

- minimum valid scaled-MAD value and scale safeguard epsilon;
- robust bright-threshold definition;
- thresholds and neighborhood/coherence rules for persistent channel flags;
- precise stacking/unstacking rule when there are fewer than 512 channels;
- inference batch size and TensorRT builder/workspace/profile settings;
- whether post-replacement `zdot` was enabled for each reported
  full-filterbank result. The paper calls it optional, while the compute-time
  protocol explicitly includes it;
- exact baseline edge handling and block boundaries for output rescaling.

### Optional hysteresis

- support threshold;
- anisotropic neighborhood/kernel;
- number of constrained dilation steps;
- row-support gate threshold;
- temporal-closing and final-dilation kernels/iterations;
- maximum final masked fraction and maximum added-pixel fraction.

### Synthetic benchmarks

- complete fixed-seed manifests or the top-level RNG seeds used to create
  them;
- SPECTRALib injection amplitude, width, duration, persistence, repeat-count,
  duty-cycle, and placement distributions;
- exact pulse-support definition and injected pulse profile;
- mixed-RFI overlap placement/rejection algorithm;
- aggregation convention for main precision, recall, and F1 values, including
  zero-denominator handling;
- PRESTO candidate-match tolerance, harmonic-match procedure, and handling of
  unmatched targets;
- exact software versions or commits for SIGPROC, PRESTO, PulsarX/`filtool`,
  PyTorch, CUDA, ONNX, and TensorRT.

### Real-data and performance reproduction

- GMRT observation identifiers, dates, bands, sampling times, channel counts,
  durations, backend details, and access instructions;
- tested `filtool` thread counts and CPU affinity/NUMA policy;
- benchmark inference batch size, timing API/synchronization procedure, system
  software versions, and power/clock policy;
- per-sampling-time runtime values underlying the reported ranges.

