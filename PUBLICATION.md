# Manuscript citation and data-editor response

Associated paper:
[MARS: A Lightweight Morphology-Aware RFI Segmentation Network for Mask-Guided Mitigation in Radio Astronomy](https://arxiv.org/abs/2608.05546).

**Archive verified on 2026-09-20:** MARS v0.1.0 is published on Zenodo at
[10.5281/zenodo.22857994](https://doi.org/10.5281/zenodo.22857994).
The source is publicly accessible at <https://github.com/Haalandspring/MARS>.
The archive's 21 public files match release commit
`0a16ead1048fa299f5cdfaa0bdff3116e75a0684`, including the trained checkpoint.
The record's software type, version, MIT license, author names, and ORCIDs
have been checked.

The availability text and BibTeX below are ready to insert into the manuscript.
The response to the editor should be used after those manuscript edits are
complete, with the actual section and line numbers filled in.

## Software availability paragraph

> The MARS pipeline is publicly available under the MIT License at
> https://github.com/Haalandspring/MARS. A frozen copy of the pipeline,
> version v0.1.0, is archived on Zenodo at https://doi.org/10.5281/zenodo.22857994
> (Gong et al., 2026). The archive includes the inference source code,
> configuration, dependency specifications, trained checkpoint, and
> TensorRT compilation and verification code.

This wording describes the deposited pipeline without claiming that the
archive contains every dataset or analysis script. If the release is also
claimed to be the exact version used for the manuscript results, verify that
claim against the analysis records and identify its configuration explicitly.

AASTeX form, for insertion into the manuscript's software-availability text:

```tex
The MARS pipeline is publicly available under the MIT License at
\url{https://github.com/Haalandspring/MARS}.
A frozen copy of the pipeline, version v0.1.0, is archived on Zenodo
\citep{mars_software} at \url{https://doi.org/10.5281/zenodo.22857994}.
The archive includes the inference source code, configuration, dependency
specifications, trained checkpoint, and TensorRT compilation and
verification code.
```

Also add `MARS \citep{mars_software}` to the manuscript's existing
`\software{...}` list, preserving the other software entries.

## Software reference-list entry

The [Zenodo BibTeX export](https://zenodo.org/records/22857994/export/bibtex)
uses an `@software` entry. The following entry preserves its authors, title,
version, year, and DOI while using `@misc` for traditional BibTeX styles.
Insert it into the manuscript's `.bib` file:

```bibtex
@misc{mars_software,
  author       = {Gong, Zhaocheng and White, Jack and Roy, Jayanta and Armour, Wesley},
  title        = {{MARS}: Morphology-Aware {RFI} Segmentation},
  year         = {2026},
  howpublished = {Zenodo},
  note         = {Version v0.1.0, computer software},
  doi          = {10.5281/zenodo.22857994},
  url          = {https://doi.org/10.5281/zenodo.22857994}
}
```

## Response to the data editor

Use the following after the manuscript edits are complete.
Replace `SECTION` and `LINES` with the location in the revised manuscript.

> Thank you for this suggestion. We have added a software-availability
> statement in SECTION (lines LINES) identifying the public MARS repository,
> https://github.com/Haalandspring/MARS. We have also archived a frozen copy
> of the pipeline, version v0.1.0, on Zenodo (https://doi.org/10.5281/zenodo.22857994)
> and cited this software release in the revised manuscript and reference
> list. The Zenodo record lists the authors by their real names—Zhaocheng
> Gong, Jack White, Jayanta Roy, and Wesley Armour—and includes an ORCID
> for each author.

The author names and ORCIDs in `CITATION.cff` were obtained from the
[MARS preprint](https://arxiv.org/html/2608.05546v1). All four names and ORCIDs
were verified against the [published Zenodo record](https://zenodo.org/records/22857994).
