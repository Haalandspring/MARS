# Archiving a citable MARS release

The public development repository is <https://github.com/Haalandspring/MARS>.
A GitHub repository alone does not provide the frozen DOI-bearing copy
requested by the data editor. Follow this procedure to publish that copy.
Version `0.1.0` is the current version in `CITATION.cff` and
`src/mars_rfi/__init__.py`. It was published on 2026-09-20 as
[GitHub release v0.1.0](https://github.com/Haalandspring/MARS/releases/tag/v0.1.0)
and archived at [10.5281/zenodo.22857994](https://doi.org/10.5281/zenodo.22857994).
The fixed release commit is `0a16ead1048fa299f5cdfaa0bdff3116e75a0684`.
Examples below refer to this release; use a new version and tag for subsequent
releases. Citation updates on the development branch do not move the published
tag or change the archived files.

## Prepare the release

1. Select the code revision and configuration appropriate to the revised
   manuscript. Archiving today's pipeline does not establish that every
   historical result used today's defaults. Record the release commit and
   describe any differences relevant to the reported analysis.
2. Keep the version in `CITATION.cff` and `src/mars_rfi/__init__.py` identical.
   Use the corresponding Git tag (`v0.1.0` for version `0.1.0`). Set
   `date-released` in `CITATION.cff` to the actual release date in `YYYY-MM-DD`
   format when publishing, rather than reusing the former draft date.
3. Check the author list below against the agreed software authorship.
   These identities come from the
   [MARS preprint](https://arxiv.org/html/2608.05546v1).

   | Author | ORCID |
   | --- | --- |
   | Zhaocheng Gong | https://orcid.org/0009-0002-6946-7541 |
   | Jack White | https://orcid.org/0000-0003-2690-6858 |
   | Jayanta Roy | https://orcid.org/0000-0002-2892-8025 |
   | Wesley Armour | https://orcid.org/0000-0003-1756-3064 |

4. Validate `CITATION.cff` with
   [cffconvert](https://github.com/citation-file-format/cffconvert), installed
   in a separate tooling environment, using `cffconvert --validate` from the
   repository root. Keep `CITATION.cff` as the single metadata source: Zenodo
   [ignores it if a `.zenodo.json` file is also present](https://help.zenodo.org/docs/github/describe-software/citation-file/).
5. Ensure the committed release includes `mars.py`, `compile_tensorrt.py`,
   `src/mars_rfi/`, `config.json`, `requirements.txt`, `README.md`, `LICENSE`,
   `CITATION.cff`, and the trained checkpoint
   `artifacts/checkpoints/mars-paper-historical/best_f1.pt`. For the currently
   distributed checkpoint, SHA-256 is
   `3acf3997bd83a8836e512974a2093bce319ede7f47ce3868757df545c4bd69a4`.
   Include the documentation changes in the commit before tagging.

The public repository deliberately excludes local datasets, training and
validation experiments, and compiled TensorRT engines. This release provides
the inference pipeline and checkpoint; it is not a deposit of all data and
scripts underlying every manuscript result. Readers can use the PyTorch path
or compile and verify a TensorRT engine on their own GPU as described in the
README. Do not archive the whole local research directory.

## Publish through the GitHub integration

1. Sign in to [Zenodo](https://zenodo.org), connect the GitHub account, and
   [enable `Haalandspring/MARS`](https://help.zenodo.org/docs/github/enable-repository/)
   before publishing the release.
2. Commit and push the reviewed release changes to GitHub. Publish a GitHub
   release for tag `v0.1.0` at that exact commit. Include the version, full
   commit SHA, release scope, and checkpoint checksum in its release notes.
3. Wait for Zenodo to process the release, then open its record. Follow the
   [official archival instructions](https://help.zenodo.org/docs/github/archive-software/github-upload/)
   if processing fails. Merely creating a Git tag does not complete archiving.
4. Inspect the record: resource type **Software**, correct title and version,
   MIT license, all four real author names in order, their ORCIDs, and a
   downloadable archive. Download it and check that the source, configuration,
   requirements, and actual checkpoint are present. Verify the checkpoint hash.
5. Copy the **DOI for this version**, rather than the DOI covering all versions.
   The manuscript must identify the frozen copy. Zenodo explains the distinction
   in its [versioning FAQ](https://zenodo.org/help/versioning).

## Complete the citation and manuscript

After the record is public and its DOI resolves:

- Add a top-level `doi` field to `CITATION.cff` containing the assigned DOI
  (without the `https://doi.org/` prefix). Keep the cited version and release
  date consistent with that record.
- Replace the pending-archive paragraph in `README.md` with the version, DOI
  link, and release link. Update the provisional citation notice as well.
- Complete [PUBLICATION.md](PUBLICATION.md), insert the availability paragraph
  and software citation into the manuscript, and include the software entry
  in its reference list. Use the response template only after these steps.
- Make citation-only updates in a subsequent commit. Preserve the published
  tag and archived files; do not move the tag to include the newly assigned
  DOI. It is normal for the first automatically archived copy to predate the
  addition of its own DOI to the development branch.
- For a later software release, increment the version and remove or replace
  the previous version's DOI before archiving. Do not label changed software
  with the old version DOI.

## Alternative: manual deposit with a reserved DOI

If the DOI must appear inside the first archived copy, create a Zenodo upload
draft and [reserve its DOI](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/)
first. Add that assigned DOI to `CITATION.cff`, finalize the metadata and release
commit, and tag it. From the repository root, export only the committed tag:

```bash
git archive --format=zip --prefix=MARS-0.1.0/ \
  --output=/tmp/MARS-0.1.0.zip v0.1.0
sha256sum /tmp/MARS-0.1.0.zip
```

Follow Zenodo's [manual upload procedure](https://help.zenodo.org/docs/github/archive-software/manual-upload/).
Enter the creators and ORCIDs from `CITATION.cff` in the deposit form; do not
assume metadata inside the ZIP will populate the form. Set the type to
Software, license to MIT, and version to `0.1.0`, and link the GitHub release
and commit. Upload the ZIP to the same draft, publish it, and verify the public
record and download. A reserved DOI alone does not make the code accessible.
Use either this route or automatic GitHub archiving for a given release to
avoid two separate deposits of the same version.

The [AAS software policy](https://journals.aas.org/policy-statement-on-software/)
explains the distinction between citing the paper and citing the archived code.
