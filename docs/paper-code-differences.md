# Paper-to-Code Audit

This audit compares the 31 July 2026 MARS paper draft (source PDF SHA-256
`d07533d9da0db49eb645a94435b64374681bc1ed4a44784cb1fe08c23987bdf3`) with
the clean historical `Physics-Informed-Latent-Diffusion` commit
`d94cce6da71a91240ac1a1def973486ca9790013`, inspected on 2 August 2026. It
distinguishes fixes made in this curated repository from claims that still
require new data or reruns.

Severity meanings:

- **Critical**: the paper model/result cannot be identified or reproduced.
- **High**: the implementation can change a reported scientific result.
- **Medium**: reproducibility, portability, or interpretation is materially
  weakened.
- **Low**: naming or documentation drift with limited numerical impact.

## Architecture and training

| Severity | Paper | Historical code | Curated repository / remaining action |
| --- | --- | --- | --- |
| Critical | MARS has channels `(8,16,32,64)`, horizontal and vertical refinement on `up2/up1/up0`, and 270,769 parameters. | `v6/config.py` enabled both switches but set both stage lists to empty. It therefore built the 162,801-parameter no-refinement ablation. | `configs/train/paper.json`, `config.py`, and `build_paper_model()` enable all six refinement blocks and assert 270,769 parameters. |
| Critical | A single selected epoch-49 checkpoint has validation F1 0.9962 and is exported to ONNX/TensorRT. | Training, export, and inference scripts referenced three different checkpoint directories (`no_refine`, `noastro`, and an unsuffixed run); none of the referenced checkpoint, ONNX, engine, history, or embedded configs was present. | Paths are unified under `artifacts/checkpoints/mars-paper/`; checkpoint/engine consumers require the paper experiment ID and canonical training fingerprint. The real files and hashes must still be published, so numerical claims cannot yet be verified. |
| High | The main model combines full decoder refinement, positive BCE weight 3, astronomy loss 1, and the final physical-width augmentation. | No checked-in config had that exact combination: v4 had full refinement but positive weight 2; v5 had positive weight 3 but astronomy loss 0; v6 had the final loss/augmentation but no refinement. | The paper config is now an explicit, reviewed composition of the paper-stated topology and v6 training settings. The missing production checkpoint must prove this was the configuration actually evaluated. |
| High | Paper artifacts must not be confused with topology-identical loss ablations. | The no-astronomy-loss model has the same 270,769 parameters as the paper model, and historical paths/names were the only identity mechanism. CLI loss or data-path overrides could also write to the default paper directory. | Every checkpoint carries an experiment ID, artifact role, and hash of architecture/loss/augmentation/training fields plus the declared split paths/counts. Paper-profile overrides are rejected; explicit ablations have separate identities/directories, and existing output identities cannot be overwritten. Array content hashes still require the unpublished data manifest. |
| High | At most three RFI families may be injected per training patch. | `max_families_per_patch=3`, but the augmentation performed only one Bernoulli draw after the first family, so it could generate at most two. | The implementation now draws each additional family until the configured cap; a regression test makes three families reachable. Results should be retrained/rerun if the paper checkpoint used the old implementation. |
| Medium | The train/validation split is 4,800/1,200 fixed patches. | The loader accepted arbitrary arrays and the repository supplied neither provenance nor hashes. | The expected array contract is documented, but the split builder, observation provenance, deduplication policy, and checksums remain missing. |

The 270,769 count is diagnostic: the base model has 162,801 parameters and the
three horizontal plus three vertical refinement blocks add 107,968.

## Inference and mitigation

| Severity | Paper | Historical code | Curated repository / remaining action |
| --- | --- | --- | --- |
| High | Every main-text benchmark disables hysteresis. | The real-data runner said it was disabled in a comment but set `HYS_ENABLED=True`; J0659 output was even named `_hys`. | Hysteresis is off in all paper defaults. A repository-defined diagnostic profile is available only by explicit config/CLI opt-in; its settings are not claimed by the paper. Both real sources must be rerun before retaining Table 7. |
| High | For fewer than 512 channels, time blocks are stacked into a 512x512 input and separated after inference. | The overlap branch sliced fewer than 512 rows and reshaped them as 512, causing a runtime error. | Consecutive time blocks are now packed into frequency rows, padded, and reversibly unpacked; round-trip tests cover `C=256`, `512`, and `701`. |
| Medium | A final tail shorter than 512 samples is masked and filled neutrally. | An observation shorter than 512 samples reached `torch.cat([])` before tail handling. | Empty patch batches are handled explicitly; the tail path can supply neutral replacement. |
| High | Pre-screened or pre-flagged channel masks are combined with the neural mask. | Global blank arrays were initialized to zero and never populated; only segment-local detectors contributed. | `preflag_channels` now supports explicit upstream whole-channel flags, while local dead/saturated/persistent masks remain active. Pixel-level external flag files are not yet supported. |
| Low | The main mask is `sigmoid(logit) >= 0.5`. | Training used `>=`; PyTorch/TensorRT inference used `>`. | Both inference backends now use `>=`. |
| High | Output is an 8-bit filterbank. | The pipeline always wrote uint8 samples but copied the original header unchanged, corrupting non-8-bit inputs. | Non-8-bit input now fails clearly. A future writer may safely rebuild SIGPROC headers to support conversion from other bit depths. |
| Medium | The implementation is segment based. | The code called itself streaming but read the entire observation and allocated full-size GPU arrays. | Documentation now calls this segment-wise, GPU-resident processing. Out-of-core streaming remains future work. |
| Medium | TensorRT implements the same paper model/checkpoint. | An existing engine bypassed checkpoint/config validation entirely, and a missing engine silently fell back to PyTorch. Export verification only warned when its tolerance was exceeded. | Paper inference requires a matching sidecar and successful numerical verification; verification failure aborts export, and an explicitly requested missing engine is an error. An unsafe opt-out exists only for non-reporting migration/debug work. |

## Experiment protocol and reported results

| Severity | Paper | Historical code | Remaining action |
| --- | --- | --- | --- |
| Critical | Synthetic RFI uses 15,000 fixed-seed SPECTRALib 0.0.23 patches. | The vendor directory was absent, the default config ran only 32 rather than 500 examples per family/cadence, and no immutable candidate set or seeds were supplied. | Publish SPECTRALib installation/pin, the 15,000-case JSONL manifest, source-patch hashes, and raw result counts. `configs/benchmarks/synthetic_rfi_patches.json` locks the intended scale. |
| Critical | FRB clean and mixed sets contain 9,360 cases each. | Old 4-DM/10,080-case configs and V5 commands coexisted with the correct 13-DM configs. | The public benchmark config contains only the paper grid; publish both fixed-seed manifests and input hashes. |
| Critical | Full-filterbank tests contain 1,000 fixed-seed cases. | Generators used unseeded Python `random` and random UUIDs; rerunning cannot recreate the paper data. | Generate and publish one immutable manifest containing the master seed, all sampled parameters, per-component seeds, UUIDs, and checksums. |
| High | Real GMRT results use one final MARS model and no hysteresis. | J0139 used a V5 path, J0659 used a V6 HYS path, and no script generated Table 7 from machine-readable outputs. | Rerun both observations with the same paper checkpoint and protocol, then publish observation metadata and an automatically generated Table 7 CSV. |
| High | Table 8 reports F1 from pooled pixel counts. | Several ablation plots averaged per-patch F1 or weighted already-averaged F1 values; an upstream compact summary had no producer. | Store TP/FP/FN/TN per case and compute micro F1 from summed counts. Macro patch F1, if retained, must be separately named. |
| High | The operational Filtool template includes thread/segment controls. | Quality scripts omitted `-t` and `-l 2`; only speed scripts supplied them. | Use one versioned Filtool command config for both quality and speed paths and rerun the comparison. |
| Medium | PRESTO matching uses a small tolerance and harmonic matching. | The matcher used 1% frequency tolerance and up to harmonic/subharmonic 16 while the search summed 8 harmonics; neither value was stated in the paper. | Freeze and disclose matcher parameters; align the harmonic limit with the search or justify the difference. |
| Medium | The PRESTO runner must find the single-DM product for integer and fractional DMs. | One runner expected the literal CLI label (for example `DM100.dat`), while PRESTO normally formats it as `DM100.00.dat`, causing successful integer-DM runs to be reported missing. | The curated runner discovers prefix-matched `.dat` files, prefers the exact two-decimal PRESTO label, and errors on ambiguity. |
| Critical | Figure 14 compares MARS, fastest Filtool, and RFDL from controlled timing inputs. | The final combined CSV and its producer were missing; the RFDL timing code lived outside the repository, and one merge script used the wrong schema key. | Publish the common input manifest, raw repetition timings, hardware metadata, RFDL runner/version, and a single table/figure producer. |

## Naming and repository scope

- The historical repository name and README still described
  physics-informed latent diffusion, while the paper's final method is a
  morphology-aware U-Net. This repository is named MARS and excludes diffusion
  history from the public code path.
- Historical `v3/v4/v5/v6`, `small_unet`, `light_model`, stale architecture
  images, binary caches, and personal `/mnt/leo/...` paths were not copied.
- The old architecture SVG described `(8,16,24,32)`, only partial refinement,
  and roughly 85.6k parameters. It is not an image of the paper model and is
  deliberately excluded.

## Paper ambiguities that code alone cannot resolve

1. **`zdot` state:** the method calls post-replacement `zdot` optional, the
   runtime includes it, and local runners enable it, but the paper does not say
   unambiguously which full-filterbank results use it. Public defaults keep it
   off and require explicit opt-in.
2. **Data provenance:** the training backgrounds and real GMRT observation IDs,
   dates, backend, cadence, duration, frequency setup, and access route are not
   specified sufficiently for independent retrieval.
3. **Statistical definition:** uncertainty intervals and run-to-run variance are
   absent; most results are single fixed-seed aggregates.
4. **RFDL control:** source patches and morphology augmentation are shared, but
   representation, patch size, preprocessing, objectives, and training stages
   differ. The comparison is controlled, but not an architecture-only ablation.

The paper-facing specification, without code interpretation, is maintained in
[`paper-specification.md`](paper-specification.md).
