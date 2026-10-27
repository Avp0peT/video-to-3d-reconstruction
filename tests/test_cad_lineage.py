"""TEST_DATA file-identity counterexamples; no native geometry or measurement.

Success-shaped reports below are deliberately synthetic protocol fixtures.
They assert only the validators' linking behavior, never actual CAD execution.
Ledger-edge tests start after the separate numerical ledger validation boundary.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from integrations.cad_delivery import delivery


def _binding(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")


def _same_bytes_elsewhere(path: Path, name: str) -> Path:
    other = path.with_name(name)
    other.write_bytes(path.read_bytes())
    return other


def _cad_fixture(root: Path, *, scaled: bool = False) -> dict:
    root.mkdir()
    mesh = root / "TEST_DATA_input.ply"
    mesh.write_bytes(b"TEST_DATA non-native mesh payload\n")
    source = root / ("TEST_DATA_scale_ledger.json" if scaled else "TEST_DATA_mesh_prep.json")
    _json(source, {"test_data": True, "purpose": "identity fixture, not a verified geometric record"})
    output = root / "cad"
    output.mkdir()
    for name in ("model.FCStd", "model.step"):
        (output / name).write_bytes(("TEST_DATA protocol bytes, not CAD: " + name).encode("ascii"))
    src = _binding(source)
    converter_arguments = ["--mesh", str(mesh.resolve()), "--output", str(output.resolve()),
                           "--source-run", str(source.resolve()), "--scale-status", "reference_scaled",
                           "--length-unit", "mm", "--scale-reference", str(source.resolve()),
                           "--max-faces", "10000", "--synthetic"]
    report = {
        "test_data": True, "status": "PASS_SYNTHETIC_CAD_CONVERSION", "script_exit_code": 0,
        "synthetic_input": True, "max_faces": 10000,
        "input": {"mesh": str(mesh.resolve()), "mesh_bytes": _binding(mesh)["bytes"],
                  "mesh_sha256": _binding(mesh)["sha256"],
                  "source_run": src["path"], "source_run_sha256": src["sha256"]},
        "scale": {"source_scale_status": "reference_scaled" if scaled else "unknown",
                  "declared_length_unit": "mm" if scaled else "arbitrary",
                  "scale_reference": {"path": src["path"], "sha256": src["sha256"]} if scaled else None},
        "artifacts": [_binding(output / name) for name in ("model.FCStd", "model.step")],
        "argv": ["TEST_DATA_python.exe", str(delivery.ROOT / "scripts/scaled_mesh_to_cad.py"),
                 "--ledger", str(source.resolve()), "--output", str(output.resolve())] if scaled else
                ["TEST_DATA_python.exe", str(delivery.ROOT / "scripts/mesh_to_cad.py"),
                 "--mesh", str(mesh.resolve()), "--source-run", str(source.resolve()),
                 "--output", str(output.resolve()), "--scale-status", "unknown",
                 "--length-unit", "arbitrary", "--synthetic"],
    }
    wrapper = {"test_data": True, "status": "PASS_VERIFIED_SCALE_AND_CAD", "exit_code": 0,
               "ledger": src, "data_kind": "synthetic", "scale_verified": True,
               "native_conversion_started": True,
               "converter": _binding(delivery.ROOT / "scripts/mesh_to_cad.py"),
               "converter_arguments": converter_arguments}
    fixture = {"mesh": mesh, "source": source, "output": output, "scaled": scaled,
               "report": report, "wrapper": wrapper}
    _persist(fixture)
    return fixture


def _persist(fixture: dict) -> None:
    _json(fixture["output"] / "conversion_report.json", fixture["report"])
    if fixture["scaled"]:
        _json(fixture["output"] / "scale_verification.json", fixture["wrapper"])


def _validate(fixture: dict, **overrides):
    kwargs = {"source_record": fixture["source"],
              "scale_reference": fixture["source"] if fixture["scaled"] else None}
    kwargs.update(overrides)
    return delivery.validate_cad(fixture["output"], True, fixture["mesh"], fixture["scaled"], **kwargs)


def test_expected_source_is_required_even_for_a_success_shaped_report(tmp_path):
    fixture = _cad_fixture(tmp_path / "TEST_DATA")
    with pytest.raises(ValueError):
        delivery.validate_cad(fixture["output"], True, fixture["mesh"])


@pytest.mark.parametrize("fault", ["same-bytes-other-path", "wrong-hash", "directory"])
def test_cad_source_record_requires_exact_file_identity(tmp_path, fault):
    fixture = _cad_fixture(tmp_path / "TEST_DATA")
    expected = fixture["source"]
    if fault == "same-bytes-other-path":
        other = _same_bytes_elsewhere(expected, "TEST_DATA_unrelated.json")
        fixture["report"]["input"]["source_run"] = str(other.resolve())
    elif fault == "wrong-hash":
        fixture["report"]["input"]["source_run_sha256"] = "0" * 64
    else:
        expected = fixture["source"].parent
        fixture["report"]["input"].update(source_run=str(expected.resolve()), source_run_sha256=None)
    _persist(fixture)
    with pytest.raises(ValueError):
        _validate(fixture, source_record=expected)


@pytest.mark.parametrize("where", ["report", "argument"])
def test_unscaled_cad_rejects_a_scale_reference(tmp_path, where):
    fixture = _cad_fixture(tmp_path / "TEST_DATA")
    if where == "report":
        bound = _binding(fixture["source"])
        fixture["report"]["scale"]["scale_reference"] = {"path": bound["path"], "sha256": bound["sha256"]}
        _persist(fixture)
    with pytest.raises(ValueError):
        _validate(fixture, scale_reference=fixture["source"] if where == "argument" else None)


@pytest.mark.parametrize("fault", ["same-bytes-other-path", "wrong-hash", "missing"])
def test_scaled_reference_must_be_the_expected_source_ledger(tmp_path, fault):
    fixture = _cad_fixture(tmp_path / "TEST_DATA", scaled=True)
    reference = fixture["report"]["scale"]["scale_reference"]
    if fault == "same-bytes-other-path":
        reference["path"] = str(_same_bytes_elsewhere(fixture["source"], "TEST_DATA_other_reference.json").resolve())
    elif fault == "wrong-hash":
        reference["sha256"] = "0" * 64
    else:
        fixture["report"]["scale"]["scale_reference"] = None
    _persist(fixture)
    with pytest.raises(ValueError):
        _validate(fixture)


@pytest.mark.parametrize("fault", ["ledger", "ledger-bytes", "data-kind", "scale-verified",
                                   "native-started", "converter", "converter-hash"])
def test_scaled_wrapper_binds_its_actual_ledger_and_converter(tmp_path, fault):
    fixture = _cad_fixture(tmp_path / "TEST_DATA", scaled=True)
    wrapper = fixture["wrapper"]
    if fault == "ledger":
        wrapper["ledger"] = _binding(_same_bytes_elsewhere(fixture["source"], "TEST_DATA_other_ledger.json"))
    elif fault == "ledger-bytes":
        wrapper["ledger"]["bytes"] += 1
    elif fault == "data-kind":
        wrapper["data_kind"] = "real"
    elif fault == "scale-verified":
        wrapper["scale_verified"] = 1
    elif fault == "native-started":
        wrapper["native_conversion_started"] = 1
    elif fault == "converter":
        other = fixture["source"].parent / "TEST_DATA_other_converter.py"
        other.write_bytes((delivery.ROOT / "scripts/mesh_to_cad.py").read_bytes())
        wrapper["converter"] = _binding(other)
    else:
        wrapper["converter"]["sha256"] = "0" * 64
    _persist(fixture)
    with pytest.raises(ValueError):
        _validate(fixture)


@pytest.mark.parametrize("fault", ["mesh", "source", "reference", "output", "synthetic", "wrapper-argv"])
def test_scaled_command_edges_cannot_be_replayed_from_another_invocation(tmp_path, fault):
    fixture = _cad_fixture(tmp_path / "TEST_DATA", scaled=True)
    arguments = fixture["wrapper"]["converter_arguments"]
    if fault == "wrapper-argv":
        fixture["report"]["argv"][3] = str(
            _same_bytes_elsewhere(fixture["source"], "TEST_DATA_other_launch_ledger.json").resolve())
    elif fault == "synthetic":
        arguments.remove("--synthetic")
    else:
        option = {"mesh": "--mesh", "source": "--source-run", "reference": "--scale-reference",
                  "output": "--output"}[fault]
        arguments[arguments.index(option) + 1] = str(fixture["source"].parent / "TEST_DATA_foreign_input")
    _persist(fixture)
    with pytest.raises(ValueError):
        _validate(fixture)


@pytest.mark.parametrize("explicit", [False, True], ids=["wrapper-default", "wrapper-explicit"])
def test_real_wrapper_argv_shape_is_distinct_from_converter_arguments(tmp_path, explicit):
    fixture = _cad_fixture(tmp_path / "TEST_DATA", scaled=True)
    if explicit:
        fixture["report"]["max_faces"] = 250
        fixture["report"]["argv"] += ["--max-faces", "250"]
        arguments = fixture["wrapper"]["converter_arguments"]
        arguments[arguments.index("--max-faces") + 1] = "250"
    assert fixture["report"]["argv"][2:] != fixture["wrapper"]["converter_arguments"]
    _persist(fixture)
    assert _validate(fixture) == fixture["report"]


@pytest.mark.parametrize("value", [False, 0, 250], ids=["bool", "zero", "unrecorded-nondefault"])
def test_wrapper_default_does_not_authorize_an_invalid_or_different_face_limit(tmp_path, value):
    fixture = _cad_fixture(tmp_path / "TEST_DATA", scaled=True)
    fixture["report"]["max_faces"] = value
    arguments = fixture["wrapper"]["converter_arguments"]
    arguments[arguments.index("--max-faces") + 1] = str(value)
    _persist(fixture)
    with pytest.raises(ValueError):
        _validate(fixture)


def _scale_link_fixture(root: Path, marker: str = "A") -> tuple[dict, dict]:
    root.mkdir()
    mesh, record, reference = (root / name for name in (
        "TEST_DATA_prepared.ply", "TEST_DATA_mesh_prep.json", "TEST_DATA_reference.json"))
    mesh.write_bytes(("TEST_DATA mesh " + marker).encode("ascii"))
    _json(record, {"test_data": True, "record_identity": marker})
    _json(reference, {"test_data": True, "reference_identity": marker})
    prepared = {"mesh": _binding(mesh), "record": _binding(record), "data_kind": "synthetic",
                "synthetic_input": True}
    request = {"schema": "cad-delivery-request-v1", "prepared": prepared,
               "reference": str(reference.resolve())}
    # Only linking fields are supplied. The separate verify_ledger numerical
    # gate is outside these isolated tests; no numerical PASS is asserted.
    ledger = {"test_data": True, "source_mesh": copy.deepcopy(prepared["mesh"]),
              "source_record": copy.deepcopy(prepared["record"]), "request": _binding(reference),
              "data_kind": "synthetic"}
    return request, ledger


def test_scale_link_accepts_matching_live_file_bindings(tmp_path):
    request, ledger = _scale_link_fixture(tmp_path / "TEST_DATA_A")
    assert delivery.validate_scale_link(request, ledger) is None


@pytest.mark.parametrize("field", ["source_mesh", "source_record", "request", "data_kind"])
def test_independently_valid_but_foreign_scale_edges_are_rejected(tmp_path, field):
    request, ledger = _scale_link_fixture(tmp_path / "TEST_DATA_A")
    _, foreign = _scale_link_fixture(tmp_path / "TEST_DATA_B", "B")
    ledger[field] = "real" if field == "data_kind" else foreign[field]
    with pytest.raises(ValueError):
        delivery.validate_scale_link(request, ledger)


@pytest.mark.parametrize("field", ["source_mesh", "source_record", "request"])
def test_scale_link_rechecks_files_even_when_declared_bindings_match(tmp_path, field):
    request, ledger = _scale_link_fixture(tmp_path / "TEST_DATA_A")
    path = Path(ledger[field]["path"])
    old = path.read_bytes()
    path.write_bytes(b"X" + old[1:])  # Same byte count, stale digest on both sides.
    with pytest.raises(ValueError):
        delivery.validate_scale_link(request, ledger)
