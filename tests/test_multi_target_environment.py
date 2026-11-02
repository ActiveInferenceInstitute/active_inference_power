"""Real multi-target arrival/control/confirmation contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, cast

import pytest

from active_inference_power.multi_target_environment import (
    MultiTargetEnvironmentSpec,
    MultiTargetFiltrationEvent,
    _seed_pairs,
    evaluate_multi_target_control_pair,
    make_paired_static_control,
    run_multi_target_environment,
)


def _spec(**changes: Any) -> MultiTargetEnvironmentSpec:
    values: dict[str, Any] = {
        "target_ids": ("target_a", "target_b", "target_c"),
        "arrival_times": (0, 1, 2),
        "target_truth": (True, False, True),
        "reliability_schedule": (0.68, 0.75, 0.82, 0.9),
        "action_cost_schedule": (1.0, 1.1, 1.2, 1.3),
        "horizon": 4,
    }
    values.update(changes)
    return MultiTargetEnvironmentSpec(**values)


def test_multi_target_spec_is_versioned_and_validates_visible_contract() -> None:
    spec = _spec()
    visible = spec.visible_contract()
    assert visible["contract_version"] == "2.0"
    assert "target_truth" not in visible
    assert "target_truth" in spec.to_mapping()
    assert spec.configuration_hash != _spec(target_truth=(False, False, False)).configuration_hash
    for changes, message in (
        ({"target_ids": ("a", "b")}, "at least three"),
        ({"target_ids": ("a", "a", "c")}, "unique"),
        ({"arrival_times": (0, 1)}, "align"),
        ({"arrival_times": (-1, 1, 2)}, "non-negative"),
        ({"arrival_times": (0, 1, 4)}, "before horizon"),
        ({"reliability_schedule": (0.7,) * 3}, "align"),
        ({"reliability_schedule": (0.5,) * 4}, "reliability"),
        ({"action_cost_schedule": (1.0, 1.0, 1.0, 0.0)}, "cost"),
        ({"confirmation_steps": 0}, "confirmation_steps"),
        ({"alpha": 1.0}, "alpha"),
        ({"policy": "future"}, "policy"),
        ({"precise_reliability_bonus": 0.0}, "bonus"),
        ({"precise_cost_multiplier": 0.0}, "multiplier"),
        ({"contract_version": "1.0"}, "contract_version"),
    ):
        with pytest.raises(ValueError, match=message):
            _spec(**changes)


def test_visible_filtration_excludes_truth_and_independent_confirmation_is_hashed() -> None:
    spec = _spec()
    trace = run_multi_target_environment(spec, seed=11, confirmation_seed=12)
    assert trace == run_multi_target_environment(spec, seed=11, confirmation_seed=12)
    visible = json.dumps(trace.visible_payload(), sort_keys=True)
    assert "selected_target_truth" not in visible
    assert "confirmation" not in visible
    assert trace.visible_trace_hash != trace.confirmation_stream_hash
    assert trace.events[-1].selected_target == trace.selected_target
    with pytest.raises(ValueError, match="differ"):
        run_multi_target_environment(spec, seed=4, confirmation_seed=4)
    with pytest.raises(ValueError, match="stream seeds"):
        run_multi_target_environment(spec, seed=True, confirmation_seed=2)
    with pytest.raises(ValueError, match="selected target"):
        replace(trace, selected_target=2)


def test_multi_target_trace_rejects_corrupted_hashes_streams_and_measurements() -> None:
    """Trace integrity checks apply to every serialized evidence component."""
    trace = run_multi_target_environment(_spec(), seed=19, confirmation_seed=20)
    for changes, message in (
        ({"environment_hash": "not-a-sha256"}, "SHA-256"),
        ({"seed": True}, "stream seeds"),
        ({"confirmation_seed": trace.seed}, "independent"),
        ({"events": ()}, "selected target"),
        ({"confirmation_observations": ()}, "confirmation observations"),
        ({"confirmation_observations": (2,)}, "confirmation observations"),
        ({"confirmation_p_value": 1.1}, "p-value"),
        ({"total_cost": 0.0}, "total cost"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(trace, **changes)

    round_robin = run_multi_target_environment(
        _spec(policy="round_robin"), seed=19, confirmation_seed=20
    )
    assert tuple(event.selected_target for event in round_robin.events) == (0, 1, 2, 0)


def test_filtration_event_and_static_control_fail_closed() -> None:
    event = MultiTargetFiltrationEvent(0, (0, 1), 0, "cheap", (), ())
    assert event.to_mapping()["action"] == "cheap"
    for values in (
        {"time": -1},
        {"available_targets": ()},
        {"selected_target": 2},
        {"action": "oracle"},
        {"visible_observations": (2,)},
        {"visible_target_history": ((-1, ()),)},
    ):
        arguments: dict[str, Any] = {
            "time": 0, "available_targets": (0, 1), "selected_target": 0,
            "action": "cheap", "visible_observations": (), "visible_target_history": (),
        }
        arguments.update(values)
        with pytest.raises(ValueError):
            MultiTargetFiltrationEvent(**arguments)
    dynamic = _spec(policy="round_robin")
    static = make_paired_static_control(dynamic)
    assert static.arrival_times == (0, 0, 0)
    assert len(set(static.reliability_schedule)) == 1
    assert static.precise_cost_multiplier == 1.0


def test_multi_target_pair_is_common_random_number_worker_invariant() -> None:
    dynamic = _spec()
    serial = evaluate_multi_target_control_pair(dynamic, 4, seed=23, workers=1)
    parallel = evaluate_multi_target_control_pair(dynamic, 4, seed=23, workers=3)
    assert serial == parallel
    assert serial["contract_version"] == "2.0"
    filtration = cast(dict[str, Any], serial["formal_filtration_specification"])
    assert "target_truth" in filtration["hidden_from_policy"]
    paired = cast(dict[str, Any], serial["paired_dynamic_minus_static"])
    assert set(paired) == {"power", "fdr", "fwer", "cost", "stopping_time"}
    assert cast(dict[str, Any], paired["power"])["n"] == 4
    representative = cast(dict[str, Any], serial["representative"])
    assert representative["visible_trace_hash"] != representative["confirmation_stream_hash"]


def test_multi_target_pair_rejects_invalid_execution_inputs() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="n_reps"):
        evaluate_multi_target_control_pair(spec, 0)
    with pytest.raises(ValueError, match="workers"):
        evaluate_multi_target_control_pair(spec, 1, workers=0)
    with pytest.raises(ValueError, match="seed"):
        _seed_pairs(True, 1)
    with pytest.raises(ValueError, match="seed"):
        _seed_pairs(-1, 1)
