# Storey certificate decision and campaign v2

This is the G1 decision record for the expanded conditional-power campaign. It
records immutable evidence receipts; it does not alter the existing finalized release receipt or
relabel either preserved failed evidence bundle.

## Evidence reviewed

The original expanded campaign (`output/campaign/`) completed its simulation,
figure, schema, and provenance stages but remained incomplete because the
same-vector `storey_power_gain` certificate did not meet its predeclared finite
band. The independent precision audit (`output/precision/`) retained the same
qualitative shortfall. Their exact estimates, bands, replication plans, and
receipt identities are historical artifact fields, not maintained prose
constants; inspect the retained JSON receipts when those values matter.

The two other relevant procedures remain separate:

- `storey_qvalue_conservative` is a fixed-lambda null-count upper-bound rule.
  When estimation and rejection use the same p-value vector, it remains an
  assumption-gated diagnostic rather than a general finite-sample FDR theorem.

- `storey_qvalue_split_confirmation` estimates its null-count scale from an
  independent selection stream and rejects only on an independent confirmation
  stream. Its finite result is a conditional calibration diagnostic under its
  declared independent-stream, fixed-tuning, and super-uniform-null contracts;
  it is not an arbitrary-dependence theorem.

## Frozen three-way method review

The retained isolated review receipt at
`output/storey-method-review/storey_method_review.json` fixes the independent
two-groups design, its tuning, replication plan, and random schedule before
execution. It carries the plan, schedule, result, and file digests needed to
identify that historical receipt. Those digest values are deliberately read
from the artifact rather than copied here, because a copied identifier would
silently become stale if a retained receipt or its source closure changes.

The review reports finite procedure-specific FDR estimates and uncertainty
bands for the same-vector plug-in, a fixed-lambda conservative same-vector
rule, and independent split confirmation. The same-vector methods share each
p-value/truth draw; the split decision channel uses a distinct, hash-bound
confirmation stream. These are finite conditional operating characteristics,
not a proof that either same-vector procedure controls FDR under arbitrary
dependence.

To reproduce the frozen review, use a new, previously absent root. The recorded
`output/storey-method-review/` receipt is not a writable destination and the
runner rejects an existing root:

```bash
review_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_storey_review.XXXXXX")"
review_root="$review_parent/receipt"
./.venv/bin/python scripts/run_storey_method_review.py \
  --output-root "$review_root"
```

The command is intentionally non-tunable and rejects an existing root. Its
status is `review_only_not_for_promotion_or_release`; it does not complete a
campaign, modify a certificate policy, or authorize publication.

## Decision

For the new, isolated campaign only, the same-vector
`storey_power_gain` record is predeclared **diagnostic**. It remains fully
serialized with its verdict, finite band, and policy membership. It is neither
deleted nor treated as passing.

The terminal certificate set is:

1. `bh_fdr_control`
2. `bonferroni_fwer`
3. `positive_dependence`
4. `storey_split_confirmation`

The canonical release configuration keeps all five certificates terminal. The
preserved campaign and precision bundles are interpreted using their original,
all-terminal policy and remain fail-closed.

## Frozen v2 lane

The exact v2 source is
[`data/research_lanes/conditional_power_campaign_v2.yaml`](../data/research_lanes/conditional_power_campaign_v2.yaml).
It retains the prior expanded effect, dependence, reliability, action-policy,
certificate, and calibration grids, fixes `lambda=0.5` and confidence `0.99`,
and declares the exact terminal/diagnostic partition. Its output namespace is
the exact canonical path `output/campaign-v2/`. The preserved fail-closed
`output/campaign/` and `output/precision/` roots must never be reused.

The following guarded archive/preflight sequence is the only permitted
source-contract refresh for this exact frozen lane. It first validates the
literal configured namespace and its declared predecessor receipt locations,
then preserves an existing canonical receipt under a clearly named historical
path before regeneration. Do not overwrite the receipt in place. A genuinely
new configuration must declare its own previously absent output namespace.

```bash
campaign_root="$PWD/output/campaign-v2"
archive_root="$PWD/output/campaign-v2-stale-$(date -u +%Y%m%dT%H%M%SZ)"
test -d "$campaign_root"
test ! -e "$archive_root"
test ! -L "$archive_root"
# Read-only preflight: validate the exact frozen namespace and predecessors
# before moving the retained receipt. Run from the project root.
./.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.release_provenance import campaign_lane_predecessor_artifacts

project = Path.cwd().resolve()
records = campaign_lane_predecessor_artifacts(
    project / "data/research_lanes/conditional_power_campaign_v2.yaml",
    project_root=project,
    output_root=project / "output/campaign-v2",
)
print("campaign predecessor preflight passed:", ", ".join(sorted(records)))
PY
mv "$campaign_root" "$archive_root"
./.venv/bin/python scripts/generate_results.py \
  --profile campaign --workers 4 --checkpoint-every 500 \
  --config data/research_lanes/conditional_power_campaign_v2.yaml \
  --output-root output/campaign-v2
./.venv/bin/python scripts/finalize_campaign.py \
  --config data/research_lanes/conditional_power_campaign_v2.yaml \
  --output-root output/campaign-v2
```

The generator, run ledger, manifest, and finalizer bind the policy and exact
configuration digest. A malformed policy, omitted certificate, terminal
failure, or source-witness mismatch must stop finalization. A diagnostic
failure remains visible in the completed v2 campaign rather than being
silently erased. A quality report may record optional reproducibility evidence,
but campaign completion does not require it; neither manuscript hydration,
PDF/HTML rendering, nor `review_release.py` is an input to
`finalize_campaign.py`. Both commands must receive the literal
`output/campaign-v2` argument. The provenance validator resolves that declared
namespace and its declared predecessor receipts together; do not redirect,
copy, or substitute either predecessor file to repair a campaign receipt.

## Retained v2 receipt

The retained v2 manifest at `output/campaign-v2/` is historical campaign
evidence, not a source-current release candidate. Whether that receipt reports
a completed campaign is determined by its own manifest and validation against
the then-current source closure. The receipt itself, rather than this narrative
page, is the authoritative record of its run identifier, configuration and
execution digests, artifact inventory, figure count, checkpoint set, final
file hashes, and hash-bound predecessor certificates from the failed campaign
and precision receipts. Do not copy a digest, a timestamped archive-directory
suffix, or a certificate value into this page: after a source-contract refresh,
the archived receipt and newly generated manifest are the only safe authorities
for their respective identities.

The v2 YAML additionally declares the bounded deterministic source-work plan
used by the campaign artifacts. It changes neither the
simulation grid nor the terminal/diagnostic policy partition, and it is not a
wall-time benchmark, simulated cost, or statistical-evidence claim.

The frozen policy declares four terminal records and retains the same-vector
`storey_power_gain` record as a visible diagnostic. A retained receipt's exact
verdicts, estimates, and finite bounds are authoritative in that receipt; they
must not be copied into prose or rewritten as a pass. Earlier v2 receipts stay
byte-preserved under explicitly named historical paths, but their paths and
hashes are artifact metadata rather than this document's source-current
claims. Completion of the frozen v2 campaign artifact contract, if recorded by
its manifest, does not supersede the preserved fail-closed `output/campaign/`
or `output/precision/` evidence, certify the canonical release, or authorize a
publication or universal adaptive-FDR claim.

## Claim boundary

Campaign v2 can report conditional operating characteristics under its frozen
model, process, testing setting, policy, seeds, and independent split-stream
contract. It cannot claim universal adaptive-FDR control, arbitrary-dependence
validity, a proof for same-vector Storey q-values, or optimality of an
action-selection policy.
