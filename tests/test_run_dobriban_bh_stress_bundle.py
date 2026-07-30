"""Real-file tests for the isolated all-chunk Dobriban/BH stress runner."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
import shutil
from typing import Any, cast

from matplotlib import pyplot as plt
import pytest

from active_inference_power.protocols import (
    DOBRIBAN_BH_BUNDLE_STATUS,
    DOBRIBAN_BH_SMOKE_STATUS,
    DobribanBHProtocol,
    build_dobriban_bh_stress_bundle,
    compact_dobriban_bh_execution_profile,
    dobriban_bh_bundle_digest,
    frozen_dobriban_bh_execution_profile,
    load_dobriban_bh_protocol,
    validate_dobriban_bh_execution_profile,
    validate_dobriban_bh_stress_bundle,
)
from active_inference_power.protocols import dobriban_bh_bundle as bundle_contract
from scripts.artifacts import atomic_write_json, sha256_file
from scripts.audit_figure_accessibility import (
    audit_figure_accessibility,
    audit_rendered_figure_alts,
)
from scripts.run_dobriban_bh_stress_bundle import (
    BUNDLE_FIGURE_FILENAME,
    DASHBOARD_FILENAME,
    DOBRIBAN_BH_BUNDLE_FILENAME,
    FIGURE_REGISTRY_FILENAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    ROOT,
    _parser,
    _validate_staged_output,
    main,
    run_dobriban_bh_stress_bundle,
)


def _compact_real_profile() -> dict[str, object]:
    """Use every declared chunk with real RNGs while keeping tests bounded."""
    protocol = load_dobriban_bh_protocol()
    return compact_dobriban_bh_execution_profile(
        protocol,
        strata=8,
        macro_replications=4,
        target_entries=640,
    )


def _read_real_compact_bundle(
    tmp_path: Path,
) -> tuple[DobribanBHProtocol, dict[str, object], dict[str, Any]]:
    """Produce one finite compact receipt, then return its real decoded payload."""
    target = tmp_path / "real-compact-bundle"
    profile = _compact_real_profile()
    run_dobriban_bh_stress_bundle(target, profile)
    bundle = json.loads((target / DOBRIBAN_BH_BUNDLE_FILENAME).read_text(encoding="utf-8"))
    return load_dobriban_bh_protocol(), profile, bundle


def test_compact_real_all_chunk_bundle_is_atomic_hash_bound_and_accessible(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dobriban-bh-all-chunks"
    receipt = run_dobriban_bh_stress_bundle(target, _compact_real_profile())

    bundle_path = target / DOBRIBAN_BH_BUNDLE_FILENAME
    registry_path = target / "figures" / FIGURE_REGISTRY_FILENAME
    figure_path = target / "figures" / BUNDLE_FIGURE_FILENAME
    dashboard_path = target / DASHBOARD_FILENAME
    index_path = target / INDEX_FILENAME
    manifest_path = target / MANIFEST_FILENAME
    assert {path.name for path in target.iterdir()} == {
        DOBRIBAN_BH_BUNDLE_FILENAME,
        "figures",
        DASHBOARD_FILENAME,
        INDEX_FILENAME,
        MANIFEST_FILENAME,
    }
    assert {path.name for path in (target / "figures").iterdir()} == {
        BUNDLE_FIGURE_FILENAME,
        FIGURE_REGISTRY_FILENAME,
    }
    assert receipt["status"] == DOBRIBAN_BH_SMOKE_STATUS
    assert Path(receipt["output_root"]) == target
    assert Path(receipt["bundle_path"]) == bundle_path
    assert Path(receipt["figure_path"]) == figure_path
    assert Path(receipt["figure_registry_path"]) == registry_path
    assert Path(receipt["dashboard_path"]) == dashboard_path
    assert Path(receipt["index_path"]) == index_path
    assert Path(receipt["manifest_path"]) == manifest_path
    assert receipt["bundle_sha256"] == sha256_file(bundle_path)
    assert receipt["manifest_sha256"] == sha256_file(manifest_path)
    assert not list(tmp_path.glob(".dobriban-bh-all-chunks.dobriban-bh-stress-*"))

    protocol = load_dobriban_bh_protocol()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    validate_dobriban_bh_stress_bundle(bundle, protocol=protocol)
    assert bundle["status"] == DOBRIBAN_BH_SMOKE_STATUS
    assert bundle["evidence_class"] == "stress_test"
    assert bundle["protocol"] == protocol.to_mapping()
    assert bundle["result_sha256"] == receipt["result_sha256"]
    assert bundle["result_sha256"] == dobriban_bh_bundle_digest(bundle)
    assert bundle["execution_profile"] == _compact_real_profile()
    assert [row["prefix"] for row in bundle["chunks"]] == [
        chunk["prefix"] for chunk in protocol.plan["chunks"]
    ]
    assert len(bundle["chunks"]) == 7
    for declared, row in zip(protocol.plan["chunks"], bundle["chunks"], strict=True):
        assert row["declared"] == declared
        result = row["result"]
        assert result["N"] == declared["N"]
        assert result["master_seed"] == declared["master_seed"]
        assert result["seed_spawn_key"] == [0]
        assert result["target_entries"] == 640
        assert result["strata"] == 8
        assert result["macro_replications"] == 4
        assert result["datasets"] == 32
    assert set(bundle["source_hashes"]) == {
        "orchestrator",
        "protocol_adapter",
        "bundle_contract",
        "protocol_plan",
        "artifact_io",
        "figure_accessibility_audit",
        "figure_registry_schema",
        "lockfile",
    }
    assert "Arb certificate" in bundle["claim_boundary"]
    assert "not a release" in bundle["nonclaims"][2]

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    record = registry["figures"][0]
    assert registry["schema_version"] == "2.3"
    assert registry["bundle_result_sha256"] == bundle["result_sha256"]
    assert record["reader_question"]
    assert record["visible_comparison"]
    assert record["takeaway"]
    assert record["visual_encoding"]
    assert record["evaluation_perspective"]
    assert record["alt"] != record["caption"]
    image = plt.imread(figure_path)
    assert image.shape[0] >= 900
    assert image.shape[1] >= 1_500
    figure_audit = audit_figure_accessibility(target)
    assert figure_audit["passed"], figure_audit["findings"]
    alt_audit = audit_rendered_figure_alts(target, rendered_html=index_path)
    assert alt_audit["passed"], alt_audit["findings"]

    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert dashboard["bundle_result_sha256"] == bundle["result_sha256"]
    assert dashboard["figure_registry_sha256"] == sha256_file(registry_path)
    assert [row["prefix"] for row in dashboard["chunks"]] == [
        chunk["prefix"] for chunk in protocol.plan["chunks"]
    ]
    index_html = index_path.read_text(encoding="utf-8")
    assert record["alt"] in index_html
    assert bundle["claim_boundary"] in index_html
    assert "finite stress receipt" in index_html

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_result_sha256"] == bundle["result_sha256"]
    assert manifest["source_hashes"] == bundle["source_hashes"]
    assert set(manifest["artifacts"]) == {
        "stress_bundle",
        "stress_figure",
        "figure_registry",
        "dashboard",
        "index_html",
    }
    for artifact in manifest["artifacts"].values():
        path = target / artifact["path"]
        assert artifact["root"] == "output"
        assert path.is_file()
        assert artifact["sha256"] == sha256_file(path)
        assert artifact["bytes"] == path.stat().st_size


def test_bundle_contract_and_output_validator_fail_closed_on_tampering(
    tmp_path: Path,
) -> None:
    target = tmp_path / "valid-bundle"
    run_dobriban_bh_stress_bundle(target, _compact_real_profile())
    protocol = load_dobriban_bh_protocol()
    bundle = json.loads((target / DOBRIBAN_BH_BUNDLE_FILENAME).read_text(encoding="utf-8"))

    promotable = copy.deepcopy(bundle)
    promotable["status"] = "release_complete"
    promotable["result_sha256"] = dobriban_bh_bundle_digest(promotable)
    with pytest.raises(ValueError, match="non-promotional execution mode"):
        validate_dobriban_bh_stress_bundle(promotable, protocol=protocol)

    missing_chunk = copy.deepcopy(bundle)
    missing_chunk["chunks"].pop()
    missing_chunk["result_sha256"] = dobriban_bh_bundle_digest(missing_chunk)
    with pytest.raises(ValueError, match="all seven declared chunks"):
        validate_dobriban_bh_stress_bundle(missing_chunk, protocol=protocol)

    tampered_root = tmp_path / "tampered-bundle"
    shutil.copytree(target, tampered_root)
    manifest_path = tampered_root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["dashboard"]["sha256"] = "0" * 64
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="dashboard SHA-256 does not match"):
        _validate_staged_output(
            tampered_root,
            protocol=protocol,
            source_hashes=bundle["source_hashes"],
        )


def test_runner_rejects_nonisolated_existing_and_invalid_compact_roots(
    tmp_path: Path,
) -> None:
    profile = _compact_real_profile()
    with pytest.raises(ValueError, match="outside canonical output"):
        run_dobriban_bh_stress_bundle(ROOT / "output", profile)
    with pytest.raises(ValueError, match="outside canonical output"):
        run_dobriban_bh_stress_bundle(ROOT / "output" / "nested-stress", profile)

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_dobriban_bh_stress_bundle(existing, profile)
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"

    invalid = copy.deepcopy(profile)
    invalid["strata"] = 1_000
    with pytest.raises(ValueError, match="fewer strata"):
        run_dobriban_bh_stress_bundle(tmp_path / "invalid-profile", invalid)


def test_cli_requires_an_explicit_root_and_has_no_tuning_flags() -> None:
    with pytest.raises(SystemExit):
        main([])
    help_text = _parser().format_help()
    assert "--output-root" in help_text
    for forbidden in ("--strata", "--macro", "--seed", "--target-entries", "--chunk"):
        assert forbidden not in help_text


def test_execution_profiles_fail_closed_on_noncanonical_tuning() -> None:
    """Exercise both frozen and compact boundaries without running a fake plan."""
    protocol = load_dobriban_bh_protocol()
    compact = _compact_real_profile()
    full = frozen_dobriban_bh_execution_profile(protocol)
    validate_dobriban_bh_execution_profile(compact, protocol=protocol)
    validate_dobriban_bh_execution_profile(full, protocol=protocol)

    def assert_compact_rejected(
        mutate: Callable[[dict[str, object]], None], message: str
    ) -> None:
        candidate = copy.deepcopy(compact)
        mutate(candidate)
        with pytest.raises(ValueError, match=message):
            validate_dobriban_bh_execution_profile(candidate, protocol=protocol)

    missing_key = copy.deepcopy(compact)
    del missing_key["mode"]
    with pytest.raises(ValueError, match="keys differ"):
        validate_dobriban_bh_execution_profile(missing_key, protocol=protocol)
    assert_compact_rejected(
        lambda profile: profile.__setitem__("mode", "user_selected"),
        "mode is not supported",
    )
    assert_compact_rejected(
        lambda profile: profile.__setitem__("strata", True),
        "strata must be an integer",
    )
    assert_compact_rejected(
        lambda profile: profile.__setitem__("target_entries", 0),
        "target_entries must be an integer",
    )

    reordered = copy.deepcopy(compact)
    prefixes = reordered["declared_chunk_prefixes"]
    assert isinstance(prefixes, list)
    prefixes.reverse()
    with pytest.raises(ValueError, match="ordered seven declared chunks"):
        validate_dobriban_bh_execution_profile(reordered, protocol=protocol)

    incomplete_macro = copy.deepcopy(compact)
    macro = incomplete_macro["macro_replications"]
    assert isinstance(macro, dict)
    macro.pop(next(iter(macro)))
    with pytest.raises(ValueError, match="exactly cover declared chunks"):
        validate_dobriban_bh_execution_profile(incomplete_macro, protocol=protocol)

    too_few_macro = copy.deepcopy(compact)
    macro = too_few_macro["macro_replications"]
    assert isinstance(macro, dict)
    macro[str(protocol.plan["chunks"][0]["prefix"])] = 1
    with pytest.raises(ValueError, match="at least 2"):
        validate_dobriban_bh_execution_profile(too_few_macro, protocol=protocol)

    compact_at_public_strata = copy.deepcopy(compact)
    compact_at_public_strata["strata"] = int(protocol.plan["strata"])
    with pytest.raises(ValueError, match="fewer strata"):
        validate_dobriban_bh_execution_profile(compact_at_public_strata, protocol=protocol)

    compact_over_target = copy.deepcopy(compact)
    compact_over_target["target_entries"] = int(protocol.plan["target_entries"]) + 1
    with pytest.raises(ValueError, match="cannot exceed public-plan target_entries"):
        validate_dobriban_bh_execution_profile(compact_over_target, protocol=protocol)

    compact_over_macro = copy.deepcopy(compact)
    macro = compact_over_macro["macro_replications"]
    assert isinstance(macro, dict)
    first_prefix = str(protocol.plan["chunks"][0]["prefix"])
    macro[first_prefix] = int(protocol.plan["chunks"][0]["macro_replications"]) + 1
    with pytest.raises(ValueError, match="cannot exceed declared macro"):
        validate_dobriban_bh_execution_profile(compact_over_macro, protocol=protocol)

    frozen_wrong_strata = copy.deepcopy(full)
    frozen_wrong_strata["strata"] = int(full["strata"]) - 1
    with pytest.raises(ValueError, match="retain protocol strata"):
        validate_dobriban_bh_execution_profile(frozen_wrong_strata, protocol=protocol)

    frozen_wrong_target = copy.deepcopy(full)
    frozen_wrong_target["target_entries"] = int(full["target_entries"]) - 1
    with pytest.raises(ValueError, match="retain protocol target_entries"):
        validate_dobriban_bh_execution_profile(frozen_wrong_target, protocol=protocol)

    frozen_wrong_macro = copy.deepcopy(full)
    macro = frozen_wrong_macro["macro_replications"]
    assert isinstance(macro, dict)
    macro[first_prefix] = 2
    with pytest.raises(ValueError, match="retain every declared macro"):
        validate_dobriban_bh_execution_profile(frozen_wrong_macro, protocol=protocol)

    tampered_protocol = copy.deepcopy(protocol)
    tampered_protocol.plan["chunks"] = {"not": "a declared chunk list"}
    with pytest.raises(ValueError, match="chunks must be a list"):
        bundle_contract._declared_chunks(tampered_protocol)


def test_real_chunk_payload_validation_rejects_each_result_contract_breach(
    tmp_path: Path,
) -> None:
    """Tamper a decoded real compact result instead of using synthetic values."""
    protocol, profile, bundle = _read_real_compact_bundle(tmp_path)
    first_row = cast(dict[str, Any], cast(list[object], bundle["chunks"])[0])
    declared = cast(Mapping[str, Any], first_row["declared"])
    result = cast(dict[str, Any], first_row["result"])
    bundle_contract._validate_result(
        result,
        declared=declared,
        profile=profile,
        protocol=protocol,
    )

    def assert_result_rejected(
        mutate: Callable[[dict[str, Any]], None], message: str
    ) -> None:
        candidate = copy.deepcopy(result)
        mutate(candidate)
        with pytest.raises(ValueError, match=message):
            bundle_contract._validate_result(
                candidate,
                declared=declared,
                profile=profile,
                protocol=protocol,
            )

    assert_result_rejected(
        lambda item: item.__setitem__("schema_version", "1.0"),
        "schema must be '2.0'",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("model_id", "other-model"),
        "model_id does not match",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("protocol_hash", "0" * 64),
        "protocol hash does not match",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("N", int(item["N"]) + 1),
        "N does not match",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("m", 1),
        "m must equal 100",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("strata", int(item["strata"]) + 1),
        "strata does not match",
    )
    assert_result_rejected(
        lambda item: item.__setitem__(
            "macro_replications", int(item["macro_replications"]) + 1
        ),
        "macro replications do not match",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("datasets", int(item["datasets"]) + 1),
        "datasets must equal",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("master_seed", int(item["master_seed"]) + 1),
        "master seed does not match",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("seed_spawn_key", [1]),
        "child stream",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("target_entries", int(item["target_entries"]) + 1),
        "target_entries do not match",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("fdr_hat", 1.1),
        "fdr_hat must be in",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("mcse", 0.0),
        "mcse must be positive",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("ci95_low", 1.1),
        "95% interval must contain",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("one_sided_p_value", 1.1),
        "p-value must be in",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("mean_rejections", -1.0),
        "mean rejections must be non-negative",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("rejection_probability", 1.1),
        "rejection probability must be in",
    )
    assert_result_rejected(
        lambda item: item.__setitem__("evidence_class", "theorem"),
        "evidence class must be stress_test",
    )

    missing_key = copy.deepcopy(result)
    del missing_key["mcse"]
    with pytest.raises(ValueError, match="keys differ"):
        bundle_contract._validate_result(
            missing_key,
            declared=declared,
            profile=profile,
            protocol=protocol,
        )
    with pytest.raises(ValueError, match="must be a mapping"):
        bundle_contract._validate_result(
            [],
            declared=declared,
            profile=profile,
            protocol=protocol,
        )
    with pytest.raises(ValueError, match="finite number"):
        bundle_contract._require_finite_number(True, "case")
    with pytest.raises(ValueError, match="finite number"):
        bundle_contract._require_finite_number(math.inf, "case")
    with pytest.raises(ValueError, match="integer at least"):
        bundle_contract._require_positive_integer(True, "case")
    with pytest.raises(ValueError, match="integer at least"):
        bundle_contract._require_positive_integer(0, "case")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        bundle_contract._require_sha256("A" * 64, "case")
    with pytest.raises(ValueError, match="must be non-empty"):
        bundle_contract._validate_source_hashes({})
    with pytest.raises(ValueError, match="keys must be non-empty"):
        bundle_contract._validate_source_hashes({"": "0" * 64})


def test_real_bundle_top_level_contract_and_builder_fail_closed(
    tmp_path: Path,
) -> None:
    """Use a written compact receipt to test its boundary and self-hash witnesses."""
    protocol, profile, bundle = _read_real_compact_bundle(tmp_path)
    validate_dobriban_bh_stress_bundle(bundle)
    source_hashes = cast(Mapping[str, str], bundle["source_hashes"])
    rows = cast(list[dict[str, Any]], bundle["chunks"])
    results = [cast(Mapping[str, Any], row["result"]) for row in rows]
    rebuilt = build_dobriban_bh_stress_bundle(
        protocol,
        execution_profile=profile,
        chunk_results=results,
        source_hashes=source_hashes,
    )
    assert rebuilt["status"] == DOBRIBAN_BH_SMOKE_STATUS
    with pytest.raises(ValueError, match="every declared chunk"):
        build_dobriban_bh_stress_bundle(
            protocol,
            execution_profile=profile,
            chunk_results=results[:-1],
            source_hashes=source_hashes,
        )
    with pytest.raises(ValueError, match="must be non-empty"):
        build_dobriban_bh_stress_bundle(
            protocol,
            execution_profile=profile,
            chunk_results=results,
            source_hashes={},
        )

    with pytest.raises(ValueError, match="must be a mapping"):
        # Deliberately pass a real non-mapping at runtime to exercise the
        # fail-closed validator branch; the cast keeps that negative test
        # explicit to static analysis.
        validate_dobriban_bh_stress_bundle(cast(Any, []), protocol=protocol)
    missing_key = copy.deepcopy(bundle)
    del missing_key["status"]
    with pytest.raises(ValueError, match="keys differ"):
        validate_dobriban_bh_stress_bundle(missing_key, protocol=protocol)

    for field, value, message in (
        ("schema_version", "2.0", "schema version is unsupported"),
        ("kind", "promotable", "kind is unsupported"),
        ("evidence_class", "theorem", "evidence class must be stress_test"),
        ("status", DOBRIBAN_BH_BUNDLE_STATUS, "status does not match"),
        ("claim_boundary", "unbounded claim", "claim boundary differs"),
        ("nonclaims", [], "claim boundary differs"),
    ):
        candidate = copy.deepcopy(bundle)
        candidate[field] = value
        with pytest.raises(ValueError, match=message):
            validate_dobriban_bh_stress_bundle(candidate, protocol=protocol)

    altered_protocol = copy.deepcopy(bundle)
    protocol_mapping = cast(dict[str, Any], altered_protocol["protocol"])
    protocol_payload = cast(dict[str, Any], protocol_mapping["protocol"])
    model = cast(dict[str, Any], protocol_payload["model"])
    model["alpha"] = 0.02
    with pytest.raises(ValueError, match="protocol mapping does not match"):
        validate_dobriban_bh_stress_bundle(altered_protocol, protocol=protocol)

    empty_sources = copy.deepcopy(bundle)
    empty_sources["source_hashes"] = {}
    with pytest.raises(ValueError, match="must be non-empty"):
        validate_dobriban_bh_stress_bundle(empty_sources, protocol=protocol)
    malformed_sources = copy.deepcopy(bundle)
    malformed_sources["source_hashes"] = {"source": "A" * 64}
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_dobriban_bh_stress_bundle(malformed_sources, protocol=protocol)

    non_list_rows = copy.deepcopy(bundle)
    non_list_rows["chunks"] = {"not": "a list"}
    with pytest.raises(ValueError, match="chunks must be a list"):
        validate_dobriban_bh_stress_bundle(non_list_rows, protocol=protocol)
    missing_row = copy.deepcopy(bundle)
    missing_row["chunks"] = cast(list[dict[str, Any]], missing_row["chunks"])[:-1]
    with pytest.raises(ValueError, match="retain all seven"):
        validate_dobriban_bh_stress_bundle(missing_row, protocol=protocol)
    non_mapping_row = copy.deepcopy(bundle)
    records = cast(list[object], non_mapping_row["chunks"])
    records[0] = []
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_dobriban_bh_stress_bundle(non_mapping_row, protocol=protocol)
    incomplete_row = copy.deepcopy(bundle)
    records = cast(list[object], incomplete_row["chunks"])
    records[0] = {"prefix": "fdr_N50"}
    with pytest.raises(ValueError, match="keys differ"):
        validate_dobriban_bh_stress_bundle(incomplete_row, protocol=protocol)
    mismatched_row = copy.deepcopy(bundle)
    mismatched_records = cast(list[dict[str, Any]], mismatched_row["chunks"])
    mismatched_records[0]["prefix"] = "other"
    with pytest.raises(ValueError, match="row differs"):
        validate_dobriban_bh_stress_bundle(mismatched_row, protocol=protocol)
    stale_digest = copy.deepcopy(bundle)
    stale_digest["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="result_sha256 does not match"):
        validate_dobriban_bh_stress_bundle(stale_digest, protocol=protocol)

    full_mode_receipt = copy.deepcopy(bundle)
    full_mode_receipt["execution_profile"] = frozen_dobriban_bh_execution_profile(protocol)
    full_mode_receipt["status"] = DOBRIBAN_BH_BUNDLE_STATUS
    with pytest.raises(ValueError, match="strata does not match"):
        validate_dobriban_bh_stress_bundle(full_mode_receipt, protocol=protocol)
