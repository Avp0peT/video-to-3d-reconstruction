"""Synthetic native records test representation only; no reconstruction."""
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from integrations.demo_pipeline import research_code6 as module
from kozlov_lab.stereo.official.indexed_code3d import indexed_body_code, code_equivalence_for_mapping

POINTS = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (2., 3., 5.), (3., 7., 11.)]


def fixture(tmp_path, monkeypatch, coordinates=POINTS):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    # Preserve implementation source bindings without copying product algorithms.
    algorithm = tmp_path / "src/kozlov_lab/stereo/official/indexed_code3d.py"
    algorithm.parent.mkdir(parents=True)
    algorithm.write_bytes((ROOT / "src/kozlov_lab/stereo/official/indexed_code3d.py").read_bytes())
    sfm = tmp_path / "sfm"
    binary = sfm / "models/0/binary"
    binary.mkdir(parents=True)
    records = list(zip(range(1, len(coordinates)+1), coordinates))
    # Deliberately unsorted physical record order, deterministic real ID selection.
    data = struct.pack("<Q", len(records))
    for ident, xyz in reversed(records):
        data += struct.pack("<QdddBBBdQii", ident, *xyz, 1, 2, 3, 0.5, 1, 1, ident-1)
    (binary / "points3D.bin").write_bytes(data)
    desc = sfm / "models/0/descriptions.json"
    desc.write_text(json.dumps({"source": {"num_points3D": len(records), "points3D": {str(i): {"xyz": xyz} for i, xyz in records}}}))
    run = {"contract_version": "sfm-native-object-v1", "status": "SUCCESS", "data_kind": "TEST_ONLY",
           "models": [{"model_id": "0", "usable": True, "roundtrip_verified": True,
                       "binary_path": "models/0/binary", "descriptions_path": "models/0/descriptions.json",
                       "num_points3D": len(records)}]}
    (sfm / "run.json").write_text(json.dumps(run))
    return sfm, tmp_path / "outputs/research"


@pytest.mark.parametrize("contract", ["sfm-native-object-v1", "sfm-native-object-pinhole-v1"])
def test_native_id_order_complete_rational_code(tmp_path, monkeypatch, contract):
    sfm, output = fixture(tmp_path, monkeypatch)
    path=sfm/"run.json"
    doc=json.loads(path.read_text());doc["contract_version"]=contract;path.write_text(json.dumps(doc))
    result = module.export_native_code6(sfm, "0", output)
    assert result["status"] == "SUCCESS" and result["complete"] is True
    assert result["native_point_ids"] == [1, 2, 3, 4, 5, 6]
    assert result["counts"]["quadruples"] == 15
    assert result["controls"]["status"] == "PASS"
    controls = json.loads((output / "CONTROLS.json").read_text())
    assert controls["independent_positive"] is True
    assert controls["independent_negative_no_bijection"] is True
    assert controls["independent_permutations_checked"] == 720
    assert controls["negative_existential_comparison"]["equivalent"] is False
    source = json.loads((output / "SOURCE_POINTS.json").read_text())
    assert source["points"][4]["xyz_hex"][2] == (5.).hex()
    assert "不是整个物体" in (output / "index.html").read_text(encoding="utf-8")


def test_affine_positive_and_coordinate_negative():
    rational = [[Fraction.from_float(v) for v in xyz] for xyz in POINTS]
    left = indexed_body_code(rational, labels=list(range(1, 7)))
    transformed = [[2*x+3*y+1, y+z-4, 3*z+2] for x,y,z in rational]
    right = indexed_body_code(transformed, labels=list(range(1, 7)))
    identity = {i: i for i in range(1, 7)}
    assert code_equivalence_for_mapping(left, right, identity)["equivalent"] is True
    transformed[-1][0] += Fraction(1, 7)
    negative = indexed_body_code(transformed, labels=list(range(1, 7)))
    assert code_equivalence_for_mapping(left, negative, identity)["equivalent"] is False


@pytest.mark.parametrize("kind,expected", [("few", "NOT_APPLICABLE_INSUFFICIENT_POINTS"),
    ("planar", "UNDEFINED_PLANAR"), ("duplicate", "REJECTED"), ("budget", "BUDGET_EXCEEDED"),
    ("drift", "REJECTED"), ("truncated", "REJECTED")])
def test_honest_failures_no_reselection(tmp_path, monkeypatch, kind, expected):
    coordinates = POINTS[:5] if kind == "few" else [(float(i),float(i*i),0.) for i in range(6)] if kind == "planar" else list(POINTS)
    if kind == "duplicate":
        coordinates[-1] = coordinates[0]
    sfm, output = fixture(tmp_path, monkeypatch, coordinates)
    if kind == "drift":
        desc = sfm / "models/0/descriptions.json"
        value = json.loads(desc.read_text())
        value["source"]["points3D"]["1"]["xyz"][0] = 0.1
        desc.write_text(json.dumps(value))
    if kind == "truncated":
        binary = sfm / "models/0/binary/points3D.bin"
        binary.write_bytes(binary.read_bytes()[:-1])
    result = module.export_native_code6(sfm, "0", output, max_entries=1 if kind == "budget" else 10000)
    assert result["status"] == expected
    assert result.get("complete") is not True
    assert (output / "RESULT.json").is_file() and (output / "index.html").is_file()


def test_reject_unbound_model_and_overwrite(tmp_path, monkeypatch):
    sfm, output = fixture(tmp_path, monkeypatch)
    assert module.export_native_code6(sfm, "missing", output)["status"] == "REJECTED"
    with pytest.raises(FileExistsError):
        module.export_native_code6(sfm, "0", output)


def test_partial_source_with_verified_model_is_allowed(tmp_path, monkeypatch):
    sfm, output = fixture(tmp_path, monkeypatch)
    run_path = sfm / "run.json"
    run = json.loads(run_path.read_text())
    run["status"] = "PARTIAL"
    run_path.write_text(json.dumps(run))
    result = module.export_native_code6(sfm, "0", output)
    assert result["status"] == "SUCCESS" and result["source_sfm_status"] == "PARTIAL"


def test_control_budget_unknown_does_not_reselect_or_destroy_code(tmp_path, monkeypatch):
    sfm, output = fixture(tmp_path, monkeypatch)
    result = module.export_native_code6(sfm, "0", output, max_control_permutations=1)
    assert result["status"] == "SUCCESS"
    assert result["native_point_ids"] == [1, 2, 3, 4, 5, 6]
    assert result["controls"]["status"] == "UNKNOWN"
    controls = json.loads((output / "CONTROLS.json").read_text())
    assert controls["independent_negative_no_bijection"] is None
    assert controls["independent_permutations_checked"] == 1


def test_disabled_controls_only_build_representation(tmp_path, monkeypatch):
    sfm, output = fixture(tmp_path, monkeypatch)
    result = module.export_native_code6(sfm, "0", output, controls=False)
    assert result["controls"]["status"] == "NOT_RUN"
    assert not (output / "CONTROLS.json").exists()


def test_independent_oracle_rejects_corrupted_indexed_table():
    from integrations.demo_pipeline.research_controls import independent_table
    payload = indexed_body_code(POINTS, labels=list(range(1, 7))).to_dict()
    independent_table(payload)
    payload["entries"][0]["value"]["numerator_hex"] = "0x777"
    with pytest.raises(ValueError, match="independent complete indexed table differs"):
        independent_table(payload)
