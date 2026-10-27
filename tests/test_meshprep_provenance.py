"""TEST_DATA producer-edge checks; no PLY geometry, native run, or photo proof.

The surface records are hand-built protocol fixtures, including caller-declared
real labels. Such labels never authenticate native execution or photographs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from integrations.meshprep.provenance import validate_source


CONTRACT = "mvs-surface-run-v1"
PROFILE = "openmvs-2.4.0-cpu-preserve-raw-v1"


def _binding(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _synthetic(root: Path) -> tuple[Path, Path, dict]:
    root.mkdir()
    mesh = root / "TEST_DATA_fixture.ply"
    mesh.write_bytes(b"TEST_DATA protocol mesh bytes; not native geometry\n")
    record = root / "source.json"
    value = {"schema_version": "mesh-prep-source-v1", "data_kind": "synthetic",
             "evidence_scope": "TEST_ONLY", "photo_reconstruction": False,
             "generation": "TEST_DATA hand-built identity fixture",
             "artifacts": [{"role": "fixture_mesh", **_binding(mesh)}]}
    _json(record, value)
    return mesh, record, value


def _surface(root: Path) -> tuple[Path, Path, dict, dict]:
    work = root / "work"
    work.mkdir(parents=True)
    mesh = work / "__mvs_surface.ply"
    raw = work / "__mvs_surface_raw.ply"
    mesh.write_bytes(b"TEST_DATA surface protocol payload; not reconstructed geometry\n")
    raw.write_bytes(b"TEST_DATA pre-cleanup protocol payload; not reconstructed geometry\n")
    plan = {"test_data": True, "contract_version": CONTRACT, "profile": PROFILE,
            "input_origin": "caller_declared_real"}
    value = {"test_data": True, "contract_version": CONTRACT, "profile": PROFILE,
             "status": "SURFACE_GENERATED_NOT_ACCEPTED", "exit_code": 0,
             "input_origin": "caller_declared_real", "input_drift": [],
             "stages": [
                 {"name": "dense", "status": "GEOMETRY_READBACK_PASSED", "returncode": 0},
                 {"name": "mesh", "status": "GEOMETRY_READBACK_PASSED", "returncode": 0,
                  "outputs": {"geometry": [
                      {**_binding(raw), "role": "pre_cleanup_surface", "vertices": 4, "triangles": 4},
                      {**_binding(mesh), "role": "mesh", "vertices": 4, "triangles": 4},
                  ]}},
             ]}
    record = root / "result.json"
    _json(record, value)
    _json(root / "plan.json", plan)
    return mesh, record, value, plan


def _check_result(value: dict, mesh: Path, record: Path, role: str, synthetic: bool, plan: Path | None) -> None:
    assert value["schema_version"] == "mesh-source-link-v1"
    assert value["status"] == "PRODUCER_OUTPUT_BOUND"
    assert value["source_role"] == role and value["synthetic_input"] is synthetic
    assert value["mesh"] == _binding(mesh)
    assert value["source_record"] == _binding(record)
    assert value["plan"] == (None if plan is None else _binding(plan))
    assert value["photo_lineage_status"] == value["native_execution_status"] == "NOT_VERIFIED"


@pytest.mark.parametrize("role", [None, "fixture_mesh"], ids=["default", "explicit"])
def test_typed_synthetic_source_binds_only_test_evidence(tmp_path, role):
    mesh, record, _ = _synthetic(tmp_path / "TEST_DATA")
    result = validate_source(mesh, record, synthetic=True, role=role)
    _check_result(result, mesh, record, "fixture_mesh", True, None)


@pytest.mark.parametrize("role", ["mesh", "pre_cleanup_surface"])
def test_surface_requires_and_preserves_the_explicit_output_role(tmp_path, role):
    mesh, record, value, _ = _surface(tmp_path / "TEST_DATA")
    selected = next(row for row in value["stages"][1]["outputs"]["geometry"] if row["role"] == role)
    mesh = Path(selected["path"])
    result = validate_source(mesh, record, synthetic=False, role=role)
    _check_result(result, mesh, record, role, False, record.parent / "plan.json")


@pytest.mark.parametrize("value", [
    {"status": "PASS", "test_data": True},
    {"data_kind": "synthetic", "photo_reconstruction": False, "files": []},
], ids=["unrelated-success-json", "unversioned-source"])
def test_existing_json_is_not_a_producer_contract(tmp_path, value):
    mesh, record, _ = _synthetic(tmp_path / "TEST_DATA")
    _json(record, value)
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=True)


@pytest.mark.parametrize("fault", ["photo", "scope", "generation", "data-kind", "caller-kind"])
def test_synthetic_source_cannot_claim_another_evidence_kind(tmp_path, fault):
    mesh, record, value = _synthetic(tmp_path / "TEST_DATA")
    if fault == "photo":
        value["photo_reconstruction"] = True
    elif fault == "scope":
        value["evidence_scope"] = "REAL_PHOTO"
    elif fault == "generation":
        value["generation"] = ""
    elif fault == "data-kind":
        value["data_kind"] = "real"
    _json(record, value)
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=fault != "caller-kind")


@pytest.mark.parametrize("fault", ["failed", "bool-exit", "bool-stage-exit", "drift",
                                   "origin", "profile", "caller-kind"])
def test_surface_status_plan_and_caller_kind_must_agree(tmp_path, fault):
    mesh, record, value, plan = _surface(tmp_path / "TEST_DATA")
    if fault == "failed":
        value["status"] = "FAILED"
    elif fault == "bool-exit":
        value["exit_code"] = False
    elif fault == "bool-stage-exit":
        value["stages"][0]["returncode"] = False
    elif fault == "drift":
        value["input_drift"] = ["TEST_DATA changed photo"]
    elif fault == "origin":
        plan["input_origin"] = "caller_declared_synthetic"
    elif fault == "profile":
        plan["profile"] = "TEST_DATA_different_recipe"
    _json(record, value)
    _json(record.parent / "plan.json", plan)
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=fault == "caller-kind", role="mesh")


@pytest.mark.parametrize("role,select_raw", [(None, False), ("dense", False),
                                          ("unknown", False), ("mesh", True)])
def test_surface_role_is_not_inferred_or_substituted(tmp_path, role, select_raw):
    mesh, record, value, _ = _surface(tmp_path / "TEST_DATA")
    if select_raw:
        mesh = Path(value["stages"][1]["outputs"]["geometry"][0]["path"])
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=False, role=role)


@pytest.mark.parametrize("fault", ["hash", "same-bytes-other-path", "duplicate", "escape"])
def test_surface_mesh_binding_is_unique_exact_and_within_work(tmp_path, fault):
    mesh, record, value, _ = _surface(tmp_path / "TEST_DATA")
    geometry = value["stages"][1]["outputs"]["geometry"]
    if fault == "hash":
        geometry[1]["sha256"] = "0" * 64
    elif fault == "same-bytes-other-path":
        other = mesh.parent / "other" / "__mvs_surface.ply"
        other.parent.mkdir()
        other.write_bytes(mesh.read_bytes())
        mesh = other
    elif fault == "duplicate":
        geometry.append(copy.deepcopy(geometry[1]))
    else:
        outside = record.parent / "outside.ply"
        outside.write_bytes(mesh.read_bytes())
        geometry[1].update(_binding(outside))
        geometry[1]["path"] = str(mesh.parent / ".." / "outside.ply")
        mesh = outside
    _json(record, value)
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=False, role="mesh")


@pytest.mark.parametrize("suffix", [',"unused":NaN', ',"unused":1e999', ',"data_kind":"synthetic"'],
                         ids=["nan", "overflow-number", "duplicate-key"])
def test_source_json_rejects_nonfinite_values_and_duplicate_keys(tmp_path, suffix):
    mesh, record, value = _synthetic(tmp_path / "TEST_DATA")
    text = json.dumps(value, allow_nan=False)
    record.write_text(text[:-1] + suffix + "}", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=True)


def test_prepare_rejects_unrelated_source_before_ply_read_or_native_setup(tmp_path, monkeypatch):
    from scripts import prepare_mesh

    mesh, source, _ = _synthetic(tmp_path / "TEST_DATA")
    _json(source, {"test_data": True, "status": "PASS", "unrelated": True})
    output = tmp_path / "prepared"
    output.mkdir()
    args = SimpleNamespace(mesh=mesh, source_run=source, source_role=None, synthetic=True,
                           component=None, target_faces=10000, max_relative_deviation=0.01,
                           max_relative_volume_change=0.02)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid provenance must be rejected before geometry/native setup")

    monkeypatch.setattr(prepare_mesh.ply, "read_ply", forbidden)
    monkeypatch.setattr(prepare_mesh.cad, "configure_runtime", forbidden)
    with pytest.raises((ValueError, prepare_mesh.cad.Rejected)):
        prepare_mesh.prepare(args, output, {}, lambda: None, lambda: None)


@pytest.mark.parametrize("role", [None, 1, ["mesh"]], ids=["null", "number", "list"])
def test_surface_geometry_role_type_is_rejected_with_value_error(tmp_path, role):
    mesh, record, value, _ = _surface(tmp_path / "TEST_DATA")
    value["stages"][1]["outputs"]["geometry"][1]["role"] = role
    _json(record, value)
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=False, role="mesh")


@pytest.mark.parametrize("value", [None, [], 42], ids=["null", "array", "number"])
def test_source_json_root_must_be_an_object(tmp_path, value):
    mesh, record, _ = _synthetic(tmp_path / "TEST_DATA")
    record.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=True)


def test_source_json_byte_limit_is_enforced_before_unbounded_reading(tmp_path, monkeypatch):
    from integrations.meshprep import provenance

    mesh, record, _ = _synthetic(tmp_path / "TEST_DATA")
    monkeypatch.setattr(provenance, "MAX_JSON", 64)
    assert record.stat().st_size > provenance.MAX_JSON
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=True)


@pytest.mark.parametrize("value", [1, 0, "true", None], ids=["one", "zero", "string", "null"])
def test_synthetic_argument_is_a_strict_boolean(tmp_path, value):
    mesh, record, _ = _synthetic(tmp_path / "TEST_DATA")
    with pytest.raises(ValueError):
        validate_source(mesh, record, synthetic=value)
