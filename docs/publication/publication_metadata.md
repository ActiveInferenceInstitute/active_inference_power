# Publication metadata and v1 release identity

The v1 source is published at GitHub
[`ActiveInferenceInstitute/active_inference_power`](https://github.com/ActiveInferenceInstitute/active_inference_power)
with the immutable tag and release
[`v1.0.0`](https://github.com/ActiveInferenceInstitute/active_inference_power/releases/tag/v1.0.0).
The Zenodo concept DOI is
[10.5281/zenodo.21695160](https://doi.org/10.5281/zenodo.21695160); it identifies
the release family. The version DOI is
[10.5281/zenodo.21695161](https://doi.org/10.5281/zenodo.21695161), and its
immutable record is [Zenodo record 21695161](https://zenodo.org/records/21695161).
The PDF cover uses the concept DOI; the reproducibility section names the
version DOI, record, and exact GitHub release. The abstract remains
repository-only and intentionally contains neither DOI nor Zenodo language.

The identifiers were reserved through a production Zenodo draft and are
verified after publication against the public record. Zenodo's draft response
provides the reserved version DOI and `conceptrecid`; the concept DOI is
constructed only by Zenodo's documented same-prefix concept-record mapping,
never guessed from a tag or placeholder. The identifiers become publicly
registered when the draft is published.

A passing local result bundle remains source-bound release evidence, not a
substitute for the public GitHub and Zenodo checks below. The published release
must show the same immutable PDF, source tag, DOI pair, and cross-links.

## Identifier semantics

Zenodo commonly supplies two related identifiers for a versioned software
release:

- A **concept DOI** identifies the release family and is suitable when a
  citation should follow future versions.
- A **version DOI** identifies the exact deposited archive and must agree with
  the version and archive record being rendered.

The source also retains `publication.zenodo_deposition_id`. It must equal the
numeric suffix of the version DOI, so the editable draft used for the
reserve-first handoff cannot silently be confused with another Zenodo record.

Both values are bound to the Zenodo draft response and later confirmed from the
public record. Do not use a guessed identifier, a `TBD` placeholder, a
prospective DOI URL, or an identifier from another project. The manuscript
only renders a DOI after the corresponding Zenodo deposit has supplied the
version identifier and concept record identity; until then the cover uses the
explicit draft-status field described above.

## Reserve-first publication handoff

Perform the following external handoff in order. A separately provisioned
controlled release environment owns the publication metadata exporter and
renderer; it is not a public repository dependency. Public source users can
reproduce the inputs and inspect the released PDF/HTML; maintainers follow the
controlled runbook in [rendering_pipeline.md](rendering_pipeline.md).

1. Verify the public repository and its write authority, then bind the exact
   canonical URL in publication configuration. The v1 target is
   `ActiveInferenceInstitute/active_inference_power`.
2. Create a production Zenodo draft with `prereserve_doi: true`; retain its
   deposition ID, reserved version DOI, and Zenodo-returned concept record ID.
   Do not publish a second automatic GitHub archive for this same v1 tag.
3. Put the verified DOI pair, version-record URL, repository, tag, and release
   URL in publication configuration, then update metadata sidecars through the
   controlled metadata exporter rather than hand-editing `CITATION.cff`,
   `.zenodo.json`, or `codemeta.json`.

4. Regenerate the complete source-bound candidate: results, quality, strict
   hydration, audits, DOI-bearing PDF/HTML, rendered-alt synchronization,
   renderer provenance, review, and finalization. Inspect the cover, abstract,
   citation block, and all rendered identifier links. The cover DOI must
   replace—not appear alongside—the draft-status field.
5. Push the exact DOI-bearing source to the public repository, create the
   `v1.0.0` tag, and create a GitHub **draft** release with the final PDF and
   release manifest. Update the Zenodo draft's related identifiers with both
   the repository and exact release URL, upload the same PDF and source
   archive, then publish Zenodo. Finally publish the GitHub draft release.
6. Verify the public Zenodo Records API returns the expected concept/version
   DOI pair, title, version, immutable record, files, and GitHub relations;
   verify the GitHub release body, `CITATION.cff`, `codemeta.json`, and PDF all
   point back to the same DOI pair. Preserve those observations as the
   publication handoff receipt.

The local build/render sequence and its fail-closed checks are documented in
[rendering_pipeline.md](rendering_pipeline.md) and
[release_review.md](release_review.md). A new code, manuscript, figure, or
metadata change after hydration invalidates the current render and requires a
fresh pass before external publication.
