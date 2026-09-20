# Manuscript and data-editor response templates

**Draft: do not submit until the Zenodo release is published and checked.**
The source is publicly accessible at <https://github.com/Haalandspring/MARS>.
The following text anticipates a frozen deposit that has not yet been made
as part of this revision. Complete [RELEASING.md](RELEASING.md) first.

Replace `VERSION`, `YEAR`, and `VERSION_DOI` with the actual archived version,
publication year, and DOI for that specific version. `VERSION_DOI` means a DOI
such as the identifier assigned by Zenodo, without a URL prefix; it is not the
all-versions concept DOI. No DOI in this file is intended to be a real citation.

## Software availability paragraph

> The MARS pipeline is publicly available under the MIT License at
> https://github.com/Haalandspring/MARS. A frozen copy of the pipeline,
> version VERSION, is archived on Zenodo at https://doi.org/VERSION_DOI
> (Gong et al., YEAR). The archive includes the inference source code,
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
A frozen copy of the pipeline, version VERSION, is archived on Zenodo
\citep{mars_software} at \url{https://doi.org/VERSION_DOI}.
The archive includes the inference source code, configuration, dependency
specifications, trained checkpoint, and TensorRT compilation and
verification code.
```

Also add `MARS \citep{mars_software}` to the manuscript's existing
`\software{...}` list, preserving the other software entries.

## Reference-list entry

Prefer the BibTeX export from the published Zenodo record. Check the author
order, version, year, and DOI before inserting it into the manuscript's `.bib`
file. This `@misc` template works with traditional BibTeX styles:

```bibtex
@misc{mars_software,
  author       = {Gong, Zhaocheng and White, Jack and Roy, Jayanta and Armour, Wesley},
  title        = {{MARS}: Morphology-Aware {RFI} Segmentation},
  year         = {YEAR},
  howpublished = {Zenodo},
  note         = {Version VERSION, computer software},
  doi          = {VERSION_DOI},
  url          = {https://doi.org/VERSION_DOI}
}
```

## Response to the data editor

Use the following only after publication and the manuscript edits are complete.
Replace `SECTION` and `LINES` with the location in the revised manuscript.

> Thank you for this suggestion. We have added a software-availability
> statement in SECTION (lines LINES) identifying the public MARS repository,
> https://github.com/Haalandspring/MARS. We have also archived a frozen copy
> of the pipeline, version VERSION, on Zenodo (https://doi.org/VERSION_DOI)
> and cited this software release in the revised manuscript and reference
> list. The Zenodo record lists the authors by their real names—Zhaocheng
> Gong, Jack White, Jayanta Roy, and Wesley Armour—and includes an ORCID
> for each author.

The author names and ORCIDs in `CITATION.cff` were obtained from the
[MARS preprint](https://arxiv.org/html/2608.05546v1). Check that the published
Zenodo record displays all four correctly before sending the response.
