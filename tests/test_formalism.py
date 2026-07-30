"""Real-file and symbol-level tests for the formalism-to-code registry."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from active_inference_power.formalism import (
    FormalismEntry,
    FormalismRegistry,
    formalism_registry,
    formalism_rows,
    resolve_code_symbol,
    validate_formalism_registry,
)


ROOT = Path(__file__).resolve().parent.parent


def test_registry_is_deterministic_and_resolves_real_symbols_and_paths() -> None:
    registry = formalism_registry()
    payload = registry.to_mapping()
    assert payload == formalism_registry().to_mapping()
    assert payload["entry_count"] == len(registry.entries)
    assert len(payload["registry_hash"]) == 64
    for entry in registry.entries:
        assert (ROOT / entry.source_paths[0]).is_file()
        assert all((ROOT / path).is_file() for path in entry.source_paths)
        assert all((ROOT / path).is_file() for path in entry.test_paths)
        for symbol in entry.code_symbols:
            module_name, symbol_name = symbol.rsplit(".", 1)
            assert hasattr(importlib.import_module(f"active_inference_power.{module_name}"), symbol_name)
            assert resolve_code_symbol(symbol) is not None
        assert entry.entry_hash in json.dumps(entry.to_mapping(), sort_keys=True) or len(entry.entry_hash) == 64


def test_registry_validation_and_manuscript_rows_are_fail_closed() -> None:
    registry = formalism_registry()
    payload = registry.to_mapping()
    validate_formalism_registry(payload)
    rows = formalism_rows(payload)
    assert rows.count("\n") + 1 == payload["entry_count"]
    assert "Conditional power functional" in rows
    conditional_power = next(
        entry for entry in registry.entries if entry.entry_id == "conditional-power-functional"
    )
    assert "θ_TPR(s, π)" in conditional_power.formal_object
    assert "P(|R ∩ H1| > 0)" in conditional_power.formal_object
    model_process_setting = next(
        entry for entry in registry.entries if entry.entry_id == "model-process-setting-contract"
    )
    assert "s = (M_A, P_N, T); d = (s, π)" in model_process_setting.formal_object
    assert all(line.count("|") == 4 for line in rows.splitlines())
    assert "scenario / PowerScenario (+2 symbols); conditional operating characteristic" in rows
    assert "generative model" in rows
    assert "active inference / run AI test" in rows
    assert "theory / expected false rejections global null" in rows
    assert "expected_false_rejections_global_null" not in rows
    assert "generative_model" not in rows
    with pytest.raises(ValueError, match="schema"):
        validate_formalism_registry({**payload, "schema_version": "1.0"})
    with pytest.raises(ValueError, match="hash"):
        validate_formalism_registry({**payload, "registry_hash": "0" * 64})
    with pytest.raises(ValueError, match="entries"):
        validate_formalism_registry({**payload, "entries": []})
    with pytest.raises(ValueError, match="mapping"):
        validate_formalism_registry([])  # type: ignore[arg-type]


def test_online_fdr_registry_qualifies_alpha_accounting_without_calling_it_evidence() -> None:
    entry = next(
        item
        for item in formalism_registry().entries
        if item.entry_id == "online-fdr-recursions"
    )
    text = " ".join((entry.formal_object, entry.claim, entry.non_claim))
    assert "procedural alpha-wealth W_t^α" in text
    assert "cumulative nominal-level allocation" in text
    assert "remaining budget" in text
    assert "α_t and wealth" not in text


def _materialize_declared_paths(root: Path, *, sources: bool, tests: bool) -> None:
    """Create a real isolated checkout skeleton for registry-path validation."""
    for entry in formalism_registry().entries:
        paths = (
            (entry.source_paths if sources else ())
            + (entry.test_paths if tests else ())
        )
        for relative_path in paths:
            destination = root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.touch()


def test_registry_validation_requires_declared_source_and_test_files(tmp_path: Path) -> None:
    payload = formalism_registry().to_mapping()
    source_only = tmp_path / "source_only"
    _materialize_declared_paths(source_only, sources=False, tests=True)
    with pytest.raises(ValueError, match="formalism source path does not exist"):
        validate_formalism_registry(payload, project_root=source_only)

    test_only = tmp_path / "test_only"
    _materialize_declared_paths(test_only, sources=True, tests=False)
    with pytest.raises(ValueError, match="formalism test path does not exist"):
        validate_formalism_registry(payload, project_root=test_only)

    complete = tmp_path / "complete"
    _materialize_declared_paths(complete, sources=True, tests=True)
    validate_formalism_registry(payload, project_root=complete)


def test_code_symbol_resolution_rejects_unresolvable_references() -> None:
    assert resolve_code_symbol("protocols.DobribanBHResult").__name__ == "DobribanBHResult"
    with pytest.raises(ValueError, match="invalid code symbol"):
        resolve_code_symbol("not-a-symbol")
    with pytest.raises(ValueError, match="cannot be imported"):
        resolve_code_symbol("does_not_exist.Symbol")
    with pytest.raises(ValueError, match="missing attribute"):
        resolve_code_symbol("scenario.DoesNotExist")


def test_entry_and_registry_contracts_reject_invalid_values() -> None:
    base: dict[str, Any] = dict(
        entry_id="entry",
        layer="estimand",
        title="Title",
        formal_object="f",
        code_symbols=("scenario.PowerScenario",),
        source_paths=("src/x.py",),
        test_paths=("tests/x.py",),
        assumptions=("a",),
        evidence_class="formal_conditional",
        claim="claim",
        non_claim="non-claim",
    )
    for key, value in (
        ("entry_id", ""),
        ("title", ""),
        ("formal_object", ""),
        ("layer", "bad"),
        ("evidence_class", "bad"),
        ("claim", ""),
        ("non_claim", ""),
        ("code_symbols", ()),
        ("code_symbols", ("not-a-symbol",)),
        ("source_paths", ("docs/not-src.py",)),
        ("test_paths", ("src/not-tests.py",)),
        ("assumptions", ("",)),
    ):
        with pytest.raises(ValueError):
            FormalismEntry(**{**base, key: value})
    with pytest.raises(ValueError):
        FormalismRegistry(())
    with pytest.raises(ValueError):
        FormalismRegistry(formalism_registry().entries, schema_version="1.0")
