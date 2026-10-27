"""TEST_DATA integration: small native DB readback and mocked execution only.

No segmentation, feature extraction, tracking solver, or mapper is executed.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integrations.video_masks import region_bridge_experiment as bridge
from integrations.video_masks import region_bridge_inputs as inputs
from integrations.video_masks import region_experiment as regions
from integrations.video_masks import region_geometry as geometry
from integrations.video_masks import region_refinement as refinement
from integrations.video_masks.object_region_policy import POLICY


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_module("object_refinement_cli_test", ROOT / "scripts/run_photo_demo.py")
_legacy_fixture = _load_module(
    "object_refinement_native_fixture", ROOT / "tests/test_video_mask_region_bridge_inputs.py"
).fixture


def _models():
    return [
        dict(label="old", model_id=0, registered_indices=[999], camera_domain_status="CAMERA_DOMAIN_VALID"),
        dict(label="new", model_id=0, registered_indices=[0], camera_domain_status="CAMERA_DOMAIN_VALID"),
        dict(label="new", model_id=1, registered_indices=[1], camera_domain_status="CAMERA_DOMAIN_VALID"),
    ]


def _timeline():
    return [dict(decode_index=index, **({"input_index": index // 6} if index in (0, 6) else {}))
            for index in range(7)]


def test_choose_intermediates_filters_old_before_strict_domain_validation():
    original_models = _models()
    before = deepcopy(original_models)
    added, ledger = inputs.choose_intermediates(_timeline(), original_models, POLICY)
    assert added == [2, 4]
    assert ledger["model_set"] == "new"
    assert ledger["group_by_original_position"] == {0: 0, 1: 1}
    assert original_models == before
    assert inputs.choose_intermediates(_timeline(), original_models) == inputs.select_intermediates(
        _timeline(), original_models
    )


def test_only_old_models_do_not_create_generic_candidates():
    added, ledger = inputs.choose_intermediates(_timeline(), _models()[:1], POLICY)
    assert added == []
    assert "NO_VALID_CAMERA_DOMAIN_MODEL" in ledger["reasons"]
    assert ledger["unregistered_original_positions"] == [0, 1]


@pytest.mark.parametrize("budget,added", [(3, [2]), (4, [2, 4])])
def test_policy_survives_real_db_input_plan_native_record_and_mapper_binding(tmp_path, monkeypatch, budget, added):
    source, baseline = _legacy_fixture(tmp_path)
    previous = bridge.read(source / "PLAN.json")
    previous["source_bindings"] = {}
    bridge.save(source / "PLAN.json", previous)
    assessment = dict(models=_models(), summary={"new": {"largest_valid_component_views": 1}})
    bridge.save(source / "assessment/ASSESSMENT.json", assessment)
    bridge.save(source / "geometry/RESULT.json", {"models": []})
    (source / "geometry/database.db").write_bytes((baseline / "database.db").read_bytes())
    inherited = bridge.read(baseline / "run.json")
    inherited_names = ["images/1.jpg", "images/2.jpg"]
    inherited["effective_options"]["native"]["mapper"] = dict(image_names=inherited_names, min_model_size=2)
    bridge.save(baseline / "run.json", inherited)
    hashes = {str(path): geometry.digest(path) for path in tmp_path.rglob("*") if path.is_file()}

    policy = dict(POLICY, native_max_frames=budget, max_added=budget - 2)
    output = tmp_path / "bridge"
    monkeypatch.setattr(bridge, "_invoke", lambda *a, **k: pytest.fail("native stage launched during preparation"))
    frozen = bridge.prepare(source, output, policy)
    assert frozen["selection_policy"] == policy
    assert bridge.read(output / "RUN_PLAN.json")["selection_policy"] == policy
    assert bridge.read(output / "SUMMARY.json")["status"] == "PREPARED"
    assert bridge.main(["input-worker", "--run", str(output)]) == 0

    plan_path = output / "input/PLAN.json"
    plan = bridge.read(plan_path)
    assert plan["endpoint_expansion_policy"] == policy
    assert plan["endpoint_selection"]["policy"] == policy
    assert plan["added_decode_indices"] == added
    assert plan["endpoint_count"] == budget
    assert plan["prepared_only"] is True and plan["geometry_accepted"] is False
    prepared_sfm = Path(plan["baseline_sfm"])
    native = bridge.read(prepared_sfm / "run.json")
    assert native["preparation"]["policy"] == policy
    assert native["effective_options"] == inherited["effective_options"]
    assert native["status"] == "PREPARED_ONLY" and native["models"] == []
    assert native["geometry_executed"] is False
    assert native["preparation"]["sfm_executed"] is False
    with sqlite3.connect(prepared_sfm / "database.db") as database:
        assert database.execute("SELECT count(*) FROM images").fetchone()[0] == budget
        assert database.execute("SELECT count(*) FROM frames").fetchone()[0] == budget
        assert database.execute("SELECT count(*) FROM matches").fetchone()[0] == 1

    tracking = output / "tracking"
    tracking.mkdir()
    frames = [dict(row, observations=[]) for row in plan["timeline"] if row.get("input_index") is not None]
    bridge.save(tracking / "REGION_TRACKS.json", dict(
        schema="glove-region-tracks-v1", coordinate_convention="opencv_integer_centers", frames=frames,
        source_bindings=[dict(path=str(plan_path), sha256=geometry.digest(plan_path))],
    ))
    bound = geometry.prepare(prepared_sfm, tracking / "REGION_TRACKS.json", max_frames=96)
    assert bound["endpoint_expansion_policy"] == policy
    assert len(bound["frames"]) == budget
    mapper = SimpleNamespace(image_names=list(inherited_names), min_model_size=2)
    receipt = geometry._bind_mapper_image_names(mapper, bound["frames"], 96, bound["endpoint_expansion_policy"])
    assert receipt["policy"] == "versioned_bridge_expansion"
    assert receipt["added"] == [f"images/bridge_decode_{index:06d}.jpg" for index in added]
    assert receipt["removed"] == []
    assert set(mapper.image_names) == {frame["name"] for frame in bound["frames"]}
    assert mapper.min_model_size == 2

    calls = []
    monkeypatch.setattr(geometry, "run", lambda *a, **k: calls.append((a, k)))
    assert bridge.main(["geometry-worker", "--run", str(output)]) == 0
    assert calls == [((str(prepared_sfm), tracking / "REGION_TRACKS.json", output / "geometry"), {"max_frames": 96})]
    assert all(geometry.digest(Path(path)) == digest for path, digest in hashes.items())


@pytest.mark.parametrize("maximum", [64, 96])
def test_geometry_enforces_tightened_resource_budget_at_preparation_boundary(maximum):
    plan = {"endpoint_expansion_policy": dict(POLICY, native_max_frames=3, max_added=1)}
    geometry._validate_frame_limit(3, maximum, plan)
    with pytest.raises(ValueError, match="tightened policy budget"):
        geometry._validate_frame_limit(4, maximum, plan)
    for invalid in (True, 0, 97):
        with pytest.raises(ValueError):
            geometry._validate_frame_limit(3, invalid, plan)


def test_geometry_validates_generic_policy_even_below_legacy_frame_limit():
    plan = {"endpoint_expansion_policy": dict(POLICY, max_added=999)}
    with pytest.raises(ValueError, match="max_added"):
        geometry._validate_frame_limit(2, 64, plan)


@pytest.mark.parametrize("preparer", [bridge.prepare, inputs.prepare])
def test_preparers_refuse_existing_endpoints_exceeding_tightened_budget(tmp_path, preparer):
    source, _ = _legacy_fixture(tmp_path)
    assessment = bridge.read(source / "assessment/ASSESSMENT.json")
    assessment["summary"] = {"new": {"largest_valid_component_views": 1}}
    bridge.save(source / "assessment/ASSESSMENT.json", assessment)
    target = tmp_path / "over_budget"
    with pytest.raises(ValueError, match="existing endpoints exceed"):
        preparer(source, target, dict(POLICY, native_max_frames=1))
    assert not target.exists()


def _prepare_controller(tmp_path, monkeypatch, calls, models):
    source = tmp_path / "source"
    source.mkdir()
    (source / "geometry").mkdir()
    bridge.save(source / "geometry/RESULT.json", {})
    bridge.save(source / "PLAN.json", dict(schema="glove-region-plan-v1", timeline=_timeline()))
    monkeypatch.setattr(refinement, "_code_bindings", lambda: {})

    def reverse_prepare(original, output, geometry_dir):
        calls.append(("reverse_prepare", original, geometry_dir))
        (output / "assessment").mkdir(parents=True)
        bridge.save(output / "assessment/ASSESSMENT.json", {"models": models})
        frozen = {"source_bindings": {str(source / "PLAN.json"): geometry.digest(source / "PLAN.json")}}
        bridge.save(output / "PLAN.json", frozen)
        return frozen

    monkeypatch.setattr(refinement.reverse, "prepare", reverse_prepare)
    output = tmp_path / "refined"
    state = refinement.prepare(source, output)
    assert state["status"] == "PREPARED"
    assert state["policy"] == POLICY
    assert state["automatic_retry"] is False and state["maximum_bridge_passes"] == 1
    assert calls == [("reverse_prepare", source, "geometry")]
    return output


def test_prepared_child_plan_drift_blocks_every_execution(tmp_path, monkeypatch):
    calls = []
    output = _prepare_controller(tmp_path, monkeypatch, calls, _models())
    frozen = bridge.read(output / "REFINEMENT.json")
    assert frozen["prepared_stage_bindings"] == {
        str(output / "reverse/PLAN.json"): geometry.digest(output / "reverse/PLAN.json")
    }
    bridge.save(output / "reverse/PLAN.json", {"source_bindings": {}, "tampered": True})
    monkeypatch.setattr(refinement.reverse, "execute", lambda *a: pytest.fail("reverse launched after child-plan drift"))
    monkeypatch.setattr(refinement.bridge, "execute", lambda *a: pytest.fail("bridge launched after child-plan drift"))
    with pytest.raises(ValueError, match="prelaunch input/code drift"):
        refinement.execute(output)
    assert len(calls) == 1
    assert bridge.read(output / "REFINEMENT.json")["status"] == "PREPARED"


@pytest.mark.parametrize("failure", [None, "reverse", "bridge"])
def test_controller_real_selection_policy_propagation_and_one_attempt(tmp_path, monkeypatch, failure):
    calls = []
    output = _prepare_controller(tmp_path, monkeypatch, calls, _models())

    def reverse_execute(path):
        calls.append(("reverse_execute", path))
        return {"status": "PARTIAL_DIAGNOSTIC" if failure == "reverse" else "EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED"}

    def bridge_prepare(source, target, policy):
        calls.append(("bridge_prepare", source, target, deepcopy(policy)))
        (target / "input").mkdir(parents=True)
        bridge.save(target / "input/PLAN.json", {"added_decode_indices": [2, 4]})

    def bridge_execute(path):
        calls.append(("bridge_execute", path))
        return {"status": "PARTIAL_DIAGNOSTIC" if failure == "bridge" else "BRIDGE_EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED"}

    monkeypatch.setattr(refinement.reverse, "execute", reverse_execute)
    monkeypatch.setattr(refinement.bridge, "prepare", bridge_prepare)
    monkeypatch.setattr(refinement.bridge, "execute", bridge_execute)
    result = refinement.execute(output)
    expected = ["reverse_prepare", "reverse_execute"]
    if failure != "reverse":
        expected += ["bridge_prepare", "bridge_execute"]
        assert calls[2][-1] == POLICY
        assert result["bridge_selection"]["selected_decode_indices"] == [2, 4]
    assert [call[0] for call in calls] == expected
    assert result["status"] == ("PARTIAL_DIAGNOSTIC" if failure else "REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED")
    assert result["geometry_accepted"] is False and result["quality_status"] == "NOT_EVALUATED"
    before = deepcopy(calls)
    with pytest.raises(ValueError, match="unattempted"):
        refinement.execute(output)
    assert calls == before


def test_controller_skips_bridge_without_eligible_frames(tmp_path, monkeypatch):
    models = [dict(label="new", model_id=0, registered_indices=[0, 1], camera_domain_status="CAMERA_DOMAIN_VALID")]
    calls = []
    output = _prepare_controller(tmp_path, monkeypatch, calls, models)
    monkeypatch.setattr(refinement.reverse, "execute", lambda path: {"status": "EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED"})
    monkeypatch.setattr(refinement.bridge, "prepare", lambda *a: pytest.fail("bridge prepared without candidates"))
    monkeypatch.setattr(refinement.bridge, "execute", lambda *a: pytest.fail("bridge executed without candidates"))
    result = refinement.execute(output)
    assert result["status"] == "REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED"
    assert result["stages"]["bridge"] == "NOT_RUN_NO_ELIGIBLE_FRAMES"
    assert result["bridge_selection"]["reasons"] == ["NO_ELIGIBLE_GAPS"]


@pytest.mark.parametrize("command,prepare_only,status,expected", [
    ("refine-object", True, "PREPARED", 0),
    ("refine-object", False, "REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED", 0),
    ("refine-object", False, "PARTIAL_DIAGNOSTIC", 1),
    ("refine-object", False, "PREPARED", 1),
    ("execute-object-refinement", False, "REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED", 0),
    ("execute-object-refinement", False, "DRIFT_REFUSED", 1),
    ("execute-object-refinement", False, "PREPARED", 1),
])
def test_refinement_cli_routes_and_status_exit_codes(tmp_path, monkeypatch, capsys, command, prepare_only, status, expected):
    calls = []
    source, target = tmp_path / "source", tmp_path / "out"
    monkeypatch.setattr(refinement, "prepare", lambda *a: calls.append(("prepare", a)) or {"status": "PREPARED"})
    monkeypatch.setattr(refinement, "execute", lambda *a: calls.append(("execute", a)) or {"status": status})
    argv = [command, "--run", str(source)]
    if command == "refine-object":
        argv += ["--output", str(target), "--source-geometry", "chosen_geometry"]
        if prepare_only:
            argv.append("--prepare-only")
    assert cli.main(argv) == expected
    expected_calls = []
    if command == "refine-object":
        expected_calls.append(("prepare", (source, target, "chosen_geometry")))
    if not prepare_only:
        expected_calls.append(("execute", (target if command == "refine-object" else source,)))
    assert calls == expected_calls
    assert json.loads(capsys.readouterr().out)["status"] == status


@pytest.mark.parametrize("prepare_only,status,expected", [
    (True, "PREPARED", 0),
    (False, "EXPERIMENT_COMPLETE_NOT_ACCEPTED", 0),
    (False, "NATIVE_EXPERIMENTS_COMPLETE_NOT_ACCEPTED", 1),
    (False, "PARTIAL_DIAGNOSTIC", 1),
    (False, "PREPARED", 1),
])
def test_region_cli_routes_and_actual_success_status(tmp_path, monkeypatch, capsys, prepare_only, status, expected):
    paths = [tmp_path / name for name in ("masks", "sfm", "report", "output")]
    calls = []
    monkeypatch.setattr(regions, "prepare", lambda *a: calls.append(("prepare", a)) or {"status": "PREPARED"})
    monkeypatch.setattr(regions, "execute", lambda *a: calls.append(("execute", a)) or {"status": status})
    monkeypatch.setattr(refinement, "run", lambda *a, **k: pytest.fail("refinement ignored --no-refine"))
    argv = ["reconstruct-regions", "--no-refine"]
    for flag, path in zip(("--mask-run", "--sfm-run", "--baseline-report", "--output"), paths):
        argv += [flag, str(path)]
    if prepare_only:
        argv.append("--prepare-only")
    assert cli.main(argv) == expected
    assert calls == [("prepare", tuple(paths))] + ([] if prepare_only else [("execute", (paths[-1],))])
    assert json.loads(capsys.readouterr().out)["status"] == status


@pytest.mark.parametrize("models,refinement_status,expected", [
    ([], None, 0),
    ([{"model_id": 0}], "REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED", 0),
    ([{"model_id": 0}], "PARTIAL_DIAGNOSTIC", 1),
    ([{"model_id": 0}], "PREPARED", 1),
])
def test_region_cli_automatically_refines_only_existing_reconstruction(tmp_path, monkeypatch, capsys, models, refinement_status, expected):
    calls = []
    output = tmp_path / "region"
    (output / "geometry").mkdir(parents=True)
    bridge.save(output / "geometry/RESULT.json", {"models": models})
    monkeypatch.setattr(regions, "prepare", lambda *a: calls.append(("prepare", a)) or {"status": "PREPARED"})
    monkeypatch.setattr(regions, "execute", lambda *a: calls.append(("execute", a)) or {"status": "EXPERIMENT_COMPLETE_NOT_ACCEPTED"})

    def refine(source, target, source_geometry=None):
        calls.append(("refine", source, target, source_geometry))
        return {"status": refinement_status, "geometry_accepted": False}

    monkeypatch.setattr(refinement, "run", refine)
    argv = ["reconstruct-regions", "--mask-run", "m", "--sfm-run", "s", "--baseline-report", "r", "--output", str(output)]
    assert cli.main(argv) == expected
    assert [item[0] for item in calls] == ["prepare", "execute"] + (["refine"] if models else [])
    printed = json.loads(capsys.readouterr().out)
    saved = bridge.read(output / "AUTOMATIC_REFINEMENT.json")
    assert printed["automatic_refinement"] == saved
    assert saved["geometry_accepted"] is False
    if models:
        assert calls[-1] == ("refine", output, output / "refinement", "geometry")
        assert saved["status"] == refinement_status
    else:
        assert saved["status"] == "NOT_APPLICABLE_NO_RECONSTRUCTION"


@pytest.mark.parametrize("prepare_only", [True, False])
def test_region_cli_never_auto_refines_unexecuted_or_failed_source(tmp_path, monkeypatch, prepare_only):
    calls = []
    monkeypatch.setattr(regions, "prepare", lambda *a: calls.append("prepare") or {"status": "PREPARED"})
    monkeypatch.setattr(regions, "execute", lambda *a: calls.append("execute") or {"status": "PARTIAL_DIAGNOSTIC"})
    monkeypatch.setattr(refinement, "run", lambda *a, **k: pytest.fail("refinement started without completed source"))
    argv = ["reconstruct-regions", "--mask-run", "m", "--sfm-run", "s", "--baseline-report", "r", "--output", str(tmp_path / "out")]
    if prepare_only:
        argv.append("--prepare-only")
    assert cli.main(argv) == (0 if prepare_only else 1)
    assert calls == ["prepare"] + ([] if prepare_only else ["execute"])


@pytest.mark.parametrize("command", ["refine-object", "execute-object-refinement", "reconstruct-regions"])
def test_cli_input_errors_return_two_and_do_not_retry(tmp_path, monkeypatch, capsys, command):
    attempts = []

    def reject(*args):
        attempts.append(args)
        raise ValueError("TEST_DATA unbound source")

    if command == "reconstruct-regions":
        monkeypatch.setattr(regions, "prepare", reject)
        monkeypatch.setattr(regions, "execute", lambda *a: pytest.fail("execution after rejected preparation"))
        argv = [command, "--mask-run", "m", "--sfm-run", "s", "--baseline-report", "r", "--output", "o"]
    else:
        monkeypatch.setattr(refinement, "prepare", reject)
        monkeypatch.setattr(refinement, "execute", reject if command == "execute-object-refinement" else
                            lambda *a: pytest.fail("execution after rejected preparation"))
        argv = [command, "--run", str(tmp_path)]
        if command == "refine-object":
            argv += ["--output", str(tmp_path / "output")]
    assert cli.main(argv) == 2
    assert len(attempts) == 1
    assert "INPUT_ERROR: TEST_DATA unbound source" in capsys.readouterr().err
