"""TEST_DATA contract checks only; no model, native CAD, or subprocess execution.

Mesh/CAD payloads are deliberately labelled byte fixtures, not measured results
or evidence that a native geometry conversion succeeded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from integrations.cad_delivery import delivery


@pytest.mark.parametrize("value", [True, 0, 121, float("nan"), float("inf")])
def test_wall_budget_is_finite_and_bounded(value):
    with pytest.raises(ValueError):
        delivery.validate_wall(value)


def test_internal_request_drift_stops_before_launch(tmp_path, monkeypatch):
    request=tmp_path/"request.json"
    request.write_text('{"output":"unused"}',encoding="utf-8")
    monkeypatch.setattr(delivery.subprocess,"Popen",lambda *a,**k:pytest.fail("must not launch"))
    with pytest.raises(ValueError,match="request changed"):
        delivery.coordinate(request,"0"*64)


def test_omitted_delivery_file_not_accepted_as_partial_manifest(tmp_path):
    request=tmp_path/"request.json"
    request.write_text(json.dumps({"schema":"cad-delivery-request-v1","output":str(tmp_path.resolve()),"wall_seconds":120,"reference":None}),encoding="utf-8")
    with pytest.raises(ValueError,match="binding set"):
        delivery.verify_bundle(tmp_path,{"status":"COMPLETED_GEOMETRIC_DELIVERY","exit_code":0,"request_and_outputs":[_binding(request)]})


PREP_FILES = (
    "scripts/prepare_mesh.py",
    "scripts/mesh_to_cad.py",
    "integrations/meshprep/geometry.py",
    "integrations/meshprep/ply_io.py",
    "integrations/meshprep/provenance.py",
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _binding(path: Path) -> dict:
    """Use the production path/bytes/sha256 schema, computed independently."""
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    root = tmp_path / "project"
    implementation = {}
    for name in PREP_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# TEST_DATA implementation: " + name + "\n", encoding="utf-8")
        # prepare_mesh records platform-dependent separators on Windows.
        implementation[name.replace("/", "\\")] = _binding(path)["sha256"]
    monkeypatch.setattr(delivery, "ROOT", root)
    source = root / "source.ply"
    source.write_bytes(b"TEST_DATA upstream mesh\n")
    provenance = root / "source.json"
    _write_json(provenance, {
        "schema_version": "mesh-prep-source-v1", "data_kind": "synthetic",
        "evidence_scope": "TEST_ONLY", "photo_reconstruction": False,
        "generation": "TEST_DATA labelled upstream bytes, not native geometry",
        "artifacts": [{"role": "fixture_mesh", **_binding(source)}],
    })
    run = root / "prepared"
    run.mkdir()
    (run / "selected_before.ply").write_bytes(b"TEST_DATA selected mesh\n")
    (run / "prepared.ply").write_bytes(b"TEST_DATA prepared mesh\n")
    _write_json(run / "selection_map.json", {"test_data": True, "source_face_ids": [0]})
    inputs = {}
    for name, path in (("mesh", source), ("source_record", provenance)):
        bound = _binding(path)
        inputs[name] = {"path": bound["path"], "sha256": bound["sha256"]}
    record = {
        "contract_version": "mesh-prep-v1",
        "status": "PASS_PREPARED_TOPOLOGY",
        "script_exit_code": 0,
        "synthetic_input": True,
        "implementation": implementation,
        "inputs": inputs,
        "source_validation": {
            "schema_version": "mesh-source-link-v1", "status": "PRODUCER_OUTPUT_BOUND",
            "source_schema": "mesh-prep-source-v1",
            "source_role": "fixture_mesh", "synthetic_input": True,
            "mesh": _binding(source), "source_record": _binding(provenance), "plan": None,
            "evidence_scope": "Declared producer output association only; no native execution authentication or upstream photo chain.",
            "photo_lineage_status": "NOT_VERIFIED", "native_execution_status": "NOT_VERIFIED",
        },
        "artifacts": [
            _binding(run / name)
            for name in ("selected_before.ply", "prepared.ply", "selection_map.json")
        ],
    }
    _write_json(run / "mesh_prep.json", record)
    return root, run, record


def _cad_record(tmp_path: Path, *, synthetic: bool = True, scaled: bool = False):
    mesh = tmp_path / "input.ply"
    mesh.write_bytes(b"TEST_DATA CAD input\n")
    cad = tmp_path / "cad"
    cad.mkdir()
    for name in ("model.FCStd", "model.step"):
        (cad / name).write_bytes(("TEST_DATA " + name + "\n").encode("ascii"))
    source = tmp_path / ("scale_ledger.json" if scaled else "source_record.json")
    _write_json(source, {"test_data": True, "note": "source identity byte fixture; not native evidence"})
    source_binding = _binding(source)
    arguments = ["--mesh", str(mesh.resolve()), "--output", str(cad.resolve()),
                 "--source-run", str(source.resolve()), "--scale-status", "reference_scaled" if scaled else "unknown",
                 "--length-unit", "mm" if scaled else "arbitrary"]
    if scaled:
        arguments += ["--scale-reference", str(source.resolve()), "--max-faces", "10000"]
    if synthetic:
        arguments.append("--synthetic")
    record = {
        "test_data": True,
        "status": "PASS_SYNTHETIC_CAD_CONVERSION" if synthetic else "PASS_GEOMETRIC_CAD_CONVERSION",
        "script_exit_code": 0,
        "synthetic_input": synthetic,
        "max_faces": 10000,
        "input": {"mesh": str(mesh.resolve()), "mesh_sha256": _binding(mesh)["sha256"],
                  "source_run": str(source.resolve()), "source_run_sha256": source_binding["sha256"]},
        "argv": (["TEST_DATA_python.exe", str(delivery.ROOT / "scripts/scaled_mesh_to_cad.py"),
                  "--ledger", str(source.resolve()), "--output", str(cad.resolve())] if scaled else
                 ["TEST_DATA_python.exe", str(delivery.ROOT / "scripts/mesh_to_cad.py"), *arguments]),
        "scale": {
            "source_scale_status": "reference_scaled" if scaled else "unknown",
            "declared_length_unit": "mm" if scaled else "arbitrary",
            "scale_reference": {"path": source_binding["path"], "sha256": source_binding["sha256"]} if scaled else None,
        },
        "artifacts": [_binding(cad / name) for name in ("model.FCStd", "model.step")],
    }
    _write_json(cad / "conversion_report.json", record)
    if scaled:
        _write_json(cad / "scale_verification.json", {
            "test_data": True, "status": "PASS_VERIFIED_SCALE_AND_CAD", "exit_code": 0,
            "ledger": source_binding, "data_kind": "synthetic" if synthetic else "real",
            "scale_verified": True, "native_conversion_started": True,
            "converter": _binding(delivery.ROOT / "scripts/mesh_to_cad.py"),
            "converter_arguments": arguments,
        })
    return cad, mesh, record


def _validate_fixture_cad(cad, synthetic, mesh, scaled=False):
    """Expected source identity comes from the fixture layout, not the report."""
    source = cad.parent / ("scale_ledger.json" if scaled else "source_record.json")
    return delivery.validate_cad(cad, synthetic, mesh, scaled,
                                 source_record=source, scale_reference=source if scaled else None)


def test_prepared_preserves_bound_synthetic_evidence(prepared):
    root, run, record = prepared
    result = delivery.prepared_input(run)
    assert result["mesh"] == _binding(run / "prepared.ply")
    assert result["record"] == _binding(run / "mesh_prep.json")
    assert result["artifacts"] == record["artifacts"]
    assert result["upstream"] == [_binding(Path(row["path"])) for row in record["inputs"].values()]
    assert result["synthetic_input"] is True
    assert result["data_kind"] == "synthetic"
    assert result["photo_lineage_status"] == "NOT_VERIFIED"
    assert set(delivery.PREP_CODE) == set(PREP_FILES)
    assert delivery.file_binding(root / PREP_FILES[0]) == _binding(root / PREP_FILES[0])


@pytest.mark.parametrize("fault", ["missing", "changed-role"])
def test_prepared_recomputes_and_requires_its_source_validation(prepared, fault):
    _, run, record = prepared
    if fault == "missing":
        record.pop("source_validation")
    else:
        record["source_validation"]["source_role"] = "mesh"
    _write_json(run / "mesh_prep.json", record)
    with pytest.raises(ValueError):
        delivery.prepared_input(run)


@pytest.mark.parametrize(
    "field,value",
    [("status", "REJECTED"), ("script_exit_code", False)],
    ids=["failed", "bool0"],
)
def test_prepared_rejects_unsuccessful_record(prepared, field, value):
    _, run, record = prepared
    record[field] = value
    _write_json(run / "mesh_prep.json", record)
    with pytest.raises(ValueError, match="not successful"):
        delivery.prepared_input(run)


@pytest.mark.parametrize("value", ["true", 1], ids=["str", "int"])
def test_prepared_requires_boolean_synthetic_flag(prepared, value):
    _, run, record = prepared
    record["synthetic_input"] = value
    _write_json(run / "mesh_prep.json", record)
    with pytest.raises(ValueError, match="must be boolean"):
        delivery.prepared_input(run)


@pytest.mark.parametrize("name", ["mesh", "source_record"], ids=["mesh", "source"])
def test_prepared_rejects_upstream_byte_drift(prepared, name):
    _, run, record = prepared
    Path(record["inputs"][name]["path"]).write_bytes(b"TEST_DATA changed upstream\n")
    with pytest.raises(ValueError, match="upstream input drift"):
        delivery.prepared_input(run)


@pytest.mark.parametrize("name", PREP_FILES, ids=["prep", "cad", "geo", "ply", "provenance"])
def test_prepared_rechecks_each_current_implementation(prepared, name):
    root, run, _ = prepared
    (root / name).write_text("# TEST_DATA changed implementation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation drift"):
        delivery.prepared_input(run)


@pytest.mark.parametrize("name", ["selected_before.ply", "prepared.ply"], ids=["selected", "prepared"])
def test_prepared_rejects_artifact_byte_drift(prepared, name):
    _, run, _ = prepared
    (run / name).write_bytes(b"TEST_DATA changed artifact\n")
    with pytest.raises(ValueError, match="file binding changed|not a bound artifact"):
        delivery.prepared_input(run)


def test_prepared_rejects_duplicate_artifact_in_expected_sized_list(prepared):
    _, run, record = prepared
    # Keep prepared.ply bound and keep the length at three; selection_map is absent.
    record["artifacts"][2] = dict(record["artifacts"][0])
    _write_json(run / "mesh_prep.json", record)
    with pytest.raises(ValueError, match="artifact set differs"):
        delivery.prepared_input(run)


@pytest.mark.parametrize("synthetic,scaled", [(False, False), (True, True)], ids=["real", "scaled"])
def test_cad_accepts_matching_geometry_report_and_bound_files(tmp_path, synthetic, scaled):
    cad, mesh, record = _cad_record(tmp_path, synthetic=synthetic, scaled=scaled)
    assert _validate_fixture_cad(cad, synthetic, mesh, scaled) == record


@pytest.mark.parametrize(
    "field,value",
    [("status", "REJECTED"), ("script_exit_code", False)],
    ids=["failed", "bool0"],
)
def test_cad_requires_successful_native_report(tmp_path, field, value):
    cad, mesh, record = _cad_record(tmp_path)
    record[field] = value
    _write_json(cad / "conversion_report.json", record)
    with pytest.raises(ValueError, match="native CAD report did not pass"):
        _validate_fixture_cad(cad, True, mesh)


def test_cad_rejects_different_source_path_even_with_identical_bytes(tmp_path):
    cad, mesh, record = _cad_record(tmp_path)
    other = tmp_path / "other.ply"
    other.write_bytes(mesh.read_bytes())
    record["input"]["mesh"] = str(other.resolve())
    _write_json(cad / "conversion_report.json", record)
    with pytest.raises(ValueError, match="input does not match"):
        _validate_fixture_cad(cad, True, mesh)


@pytest.mark.parametrize(
    "field,value",
    [("source_scale_status", "unknown"), ("declared_length_unit", "m")],
    ids=["status", "unit"],
)
def test_scaled_cad_rejects_wrong_scale_semantics(tmp_path, field, value):
    cad, mesh, record = _cad_record(tmp_path, scaled=True)
    record["scale"][field] = value
    _write_json(cad / "conversion_report.json", record)
    with pytest.raises(ValueError, match="CAD scale status differs|CAD unit differs"):
        _validate_fixture_cad(cad, True, mesh, scaled=True)


def test_cad_pass_cannot_hide_changed_step_bytes(tmp_path):
    cad, mesh, _ = _cad_record(tmp_path)
    step = cad / "model.step"
    original = step.read_bytes()
    step.write_bytes(b"X" + original[1:])  # Same length, different hash.
    with pytest.raises(ValueError, match="file binding changed"):
        _validate_fixture_cad(cad, True, mesh)


def test_scaled_cad_requires_wrapper_success_as_well_as_geometry(tmp_path):
    cad, mesh, _ = _cad_record(tmp_path, scaled=True)
    wrapper = cad / "scale_verification.json"
    for result in (
        {"status": "REJECTED", "exit_code": 0},
        {"status": "PASS_VERIFIED_SCALE_AND_CAD", "exit_code": False},
    ):
        _write_json(wrapper, result)
        with pytest.raises(ValueError, match="scale wrapper post-validation did not pass"):
            _validate_fixture_cad(cad, True, mesh, scaled=True)
    wrapper.unlink()
    with pytest.raises(FileNotFoundError):
        _validate_fixture_cad(cad, True, mesh, scaled=True)


def test_environment_drops_parent_paths_home_and_tokens(tmp_path, monkeypatch):
    inherited = {
        "PYTHONPATH": "TEST_DATA injected python path",
        "PYTHONHOME": "TEST_DATA injected python home",
        "HOME": "TEST_DATA parent home",
        "USERPROFILE": "TEST_DATA parent profile",
        "OPENAI_API_KEY": "TEST_DATA fake token",
        "HF_TOKEN": "TEST_DATA fake token",
        "PATH": "TEST_DATA injected executable path",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    native = tmp_path / "native"
    temp = tmp_path / "temp"
    env = delivery.environment(native_bin=native, temp=temp)
    assert not (set(inherited) - {"PATH"}) & set(env)
    assert inherited["PATH"] not in env["PATH"]
    assert env["PATH"].split(os.pathsep)[0] == str(native)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["TEMP"] == env["TMP"] == str(temp)
    assert temp.is_dir()


def test_verify_never_reuses_missing_or_failed_delivery_result(tmp_path, monkeypatch):
    run = tmp_path / "delivery"
    run.mkdir()
    # Leftover success-looking artifacts do not authorize a failed-stage resume.
    (run / "viewer.html").write_text("TEST_DATA stale output", encoding="utf-8")
    _write_json(run / "stages.json", {"status": "COMPLETED_GEOMETRIC_DELIVERY", "exit_code": 0})

    def forbidden(*args, **kwargs):
        pytest.fail("An unsuccessful delivery must stop before inspecting prepared inputs")

    monkeypatch.setattr(delivery, "prepared_input", forbidden)
    with pytest.raises(FileNotFoundError):
        delivery.verify_delivery(run)
    for result in (
        {"status": "REJECTED", "exit_code": 2},
        {"status": "COMPLETED_GEOMETRIC_DELIVERY", "exit_code": False},
    ):
        _write_json(run / "RESULT.json", result)
        with pytest.raises(ValueError, match="did not finish successfully"):
            delivery.verify_delivery(run)
    assert (run / "viewer.html").read_text(encoding="utf-8") == "TEST_DATA stale output"
