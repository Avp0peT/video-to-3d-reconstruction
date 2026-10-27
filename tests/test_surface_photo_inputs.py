"""TEST_DATA identity-chain checks; no native scene, SfM, or camera evidence.

PNG files are real tiny images. Native BIN/MVSI bytes are explicit placeholders;
both native view extraction and scene reading are isolated with monkeypatch.
The ``real`` metadata below exercises the contract, not an actual-photo claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, PngImagePlugin
import pytest

from integrations.mvs_surface import photo_inputs


MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")


def _binding(path: Path) -> dict:
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8")


def _png(path: Path, index: int, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (3, 2))
    image.putdata([(index * 70 + x * 5, 20 + x, 90 - x) for x in range(6)])
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("TEST_DATA", "canonical" if canonical else "dataset input")
    image.save(path, format="PNG", pnginfo=metadata)


def _flush(case) -> None:
    _json(case.spec_path, case.spec)
    _json(case.manifest_path, case.manifest)
    case.io_meta["manifest_sha256"] = _binding(case.manifest_path)["sha256"]
    _json(case.io / "run.json", case.io_meta)
    _jsonl(case.io / "frames.jsonl", case.io_rows)
    _json(case.sfm / "run.json", case.sfm_meta)
    _jsonl(case.sfm / "frames_map.jsonl", case.map_rows)
    _json(case.bridge / "PLAN.json", case.plan)
    _json(case.bridge / "RESULT.json", case.bridge_result)
    _json(case.bridge / "INPUT_BINDINGS.json", case.input_bindings)


@pytest.fixture
def chain(tmp_path, monkeypatch):
    root = tmp_path / "TEST_DATA"
    root.mkdir()
    case = SimpleNamespace(root=root)
    case.io, case.sfm = root / "io", root / "sfm"
    case.bridge, case.dataset = root / "bridge", root / "dataset"
    case.selected = root / "selection" / "sparse"
    case.source = case.sfm / "models" / "0" / "binary"
    case.image_root = case.bridge
    for directory in (case.io, case.sfm, case.bridge, case.dataset,
                      case.selected, case.source, case.image_root):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (case.selected, case.source):
        for name in MODEL_FILES:
            (directory / name).write_bytes(b"TEST_DATA mocked native BIN: " + name.encode())
    case.scene_path = case.bridge / "scene.mvs"
    case.scene_path.write_bytes(b"TEST_DATA mocked MVSI; not a native scene\n")
    case.manifest_path = root / "dataset_manifest.json"
    case.spec_path = root / "photo_inputs.json"
    case.manifest = {"contract_version": "photo-io-v1", "dataset_id": "TEST_DATA_dataset",
                     "data_kind": "real", "dataset_root": str(case.dataset.resolve()),
                     "frames": []}
    case.io_meta = {"contract_version": "photo-io-v1", "run_id": "TEST_DATA_io",
                    "dataset_id": "TEST_DATA_dataset", "data_kind": "real", "status": "SUCCESS",
                    "input_count": 2, "ok_count": 2, "error_count": 0}
    case.sfm_meta = {"contract_version": "sfm-native-object-v1", "run_id": "TEST_DATA_sfm",
                     "upstream_io_run_id": "TEST_DATA_io", "data_kind": "real", "status": "SUCCESS",
                     "counts": {"input_rows": 2},
                     "models": [{"model_id": "0", "binary_path": "models/0/binary",
                                 "usable": True, "registered_image_ids": [11, 23]}]}
    case.spec = {"schema_version": "surface-photo-input-v1",
                 "bridge_run": str(case.bridge.resolve()), "sfm_run": str(case.sfm.resolve()),
                 "io_run": str(case.io.resolve()), "dataset_manifest": str(case.manifest_path.resolve()),
                 "model_id": "0"}
    case.plan = {"source_images": str(case.sfm.resolve()), "source_sparse": str(case.selected.resolve())}
    case.bridge_result = {"status": "PASS", "input_bindings_unchanged": True}
    case.io_rows, case.map_rows, case.views, case.scene_images = [], [], [], []
    case.raw_paths, case.canonical_paths, case.staged_paths = [], [], []
    for index, image_id in enumerate((11, 23)):
        raw = case.dataset / "raw" / f"{index}.png"
        name = f"images/{index:06d}.png"
        canonical, staged = case.io / name, case.sfm / name
        _png(raw, index)
        _png(canonical, index, canonical=True)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(canonical.read_bytes())
        scene_name = f"undistorted/images/{name}"
        undistorted = case.bridge / scene_name
        undistorted.parent.mkdir(parents=True, exist_ok=True)
        undistorted.write_bytes(canonical.read_bytes())
        case.raw_paths.append(raw)
        case.canonical_paths.append(canonical)
        case.staged_paths.append(staged)
        case.manifest["frames"].append({"image": f"raw/{index}.png"})
        case.io_rows.append({"input_index": index, "status": "OK", "frame_id": f"TEST_DATA_frame_{index}",
                             "image_path": f"raw/{index}.png", "image_sha256": _binding(raw)["sha256"],
                             "image_output": name, "canonical_width": 3, "canonical_height": 2})
        case.map_rows.append({"input_index": index, "status": "REGISTERED",
                              "frame_id": f"TEST_DATA_frame_{index}", "upstream_io_run_id": "TEST_DATA_io",
                              "upstream_image_output": name, "canonical_image_sha256": _binding(canonical)["sha256"],
                              "representative_input_index": index, "staged_image_name": name,
                              "registrations": [{"model_id": "0", "image_id": image_id, "camera_id": 7}]})
        case.views.append({"image_id": image_id, "name": name, "camera_id": 7, "width": 3, "height": 2})
        case.scene_images.append({"index": index, "global_id": image_id,
                                  "name": scene_name, "width": 3, "height": 2})
    case.source_views, case.selected_views = copy.deepcopy(case.views), copy.deepcopy(case.views)
    case.input_bindings = [_binding(case.selected / name) for name in MODEL_FILES]
    case.input_bindings.extend(_binding(path) for path in case.staged_paths)
    case.native_calls = []

    def fake_native_views(path):
        path = Path(path).resolve()
        case.native_calls.append(path)
        if path == case.selected.resolve():
            return copy.deepcopy(case.selected_views)
        if path == case.source.resolve():
            return copy.deepcopy(case.source_views)
        raise ValueError(f"TEST_DATA unexpected native model: {path}")

    def fake_scene(path):
        if Path(path).resolve() != case.scene_path.resolve():
            raise ValueError("TEST_DATA unexpected scene path")
        return {"images": copy.deepcopy(case.scene_images)}

    monkeypatch.setattr(photo_inputs, "_native_views", fake_native_views)
    monkeypatch.setattr(photo_inputs, "read_scene", fake_scene)
    _flush(case)
    return case


def _validate(case):
    return photo_inputs.validate(case.spec_path, case.scene_path, case.image_root)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("contract", ["sfm-native-v1", "sfm-native-profile-v1", "sfm-native-object-v1", "sfm-native-object-video-v1", "sfm-native-object-pinhole-v1"])
def test_two_photo_identities_are_bound_without_native_or_capture_claims(chain, contract):
    chain.sfm_meta["contract_version"] = contract
    _flush(chain)
    before = _snapshot(chain.root)
    result = _validate(chain)
    assert result["status"] == "PHOTO_INPUT_IDENTITIES_BOUND"
    assert result["geometry_lineage"] == result["mask_source_chain"] == result["raw_camera_capture"] == "NOT_VERIFIED"
    assert {Path(p).resolve() for p in chain.native_calls} == {chain.source.resolve(), chain.selected.resolve()}
    assert len(result["frames"]) == 2
    by_name = {row["native_image"]: row for row in result["frames"]}
    for index, view in enumerate(chain.views):
        frame = by_name[view["name"]]
        assert frame["scene_image"] == f"undistorted/images/{view['name']}"
        assert (frame["scene_index"], frame["scene_global_id"]) == (index, view["image_id"])
        assert (frame["image_id"], frame["camera_id"], frame["representative_input_index"]) == (view["image_id"], 7, index)
        assert len(frame["aliases"]) == 1
        alias = frame["aliases"][0]
        assert (alias["input_index"], alias["frame_id"]) == (index, f"TEST_DATA_frame_{index}")
    bindings = {row["path"]: row for row in result["bindings"]}
    # Every on-disk fixture file belongs to the identity chain; the mocked native
    # readers do not make the corresponding BIN files exempt from hashing.
    expected = {str(path.resolve()) for path in chain.root.rglob("*") if path.is_file()}
    assert expected <= set(bindings)
    for path, row in bindings.items():
        assert row == _binding(Path(path))
    assert _snapshot(chain.root) == before
    assert _binding(chain.raw_paths[0])["sha256"] != _binding(chain.canonical_paths[0])["sha256"]


@pytest.mark.parametrize('fault', [None, 'contract', 'identity', 'count'])
def test_video192_ledger_with_two_registered_views(chain, fault):
    """192 real ledger rows do not imply 192 registered or accepted views."""
    c = chain
    c.sfm_meta.update(contract_version='sfm-native-object-video-v1', profile_id='cad-object-video-v1')
    for i in range(2, 192):
        # Unregistered rows remain in the full input ledger; no fake native view.
        c.io_rows.append(dict(c.io_rows[0], input_index=i, frame_id=f'TEST_DATA_frame_{i}'))
        c.map_rows.append(dict(c.map_rows[0], input_index=i, frame_id=f'TEST_DATA_frame_{i}',
                              registrations=[], staged_image_name=None, representative_input_index=None))
        c.manifest['frames'].append(dict(c.manifest['frames'][0]))
    c.io_meta.update(input_count=192, ok_count=192)
    c.sfm_meta['counts']['input_rows'] = 192
    if fault == 'contract': c.sfm_meta['contract_version'] = 'sfm-native-object-video-v999'
    if fault == 'identity': c.map_rows[0]['frame_id'] = 'WRONG'
    if fault == 'count': c.io_meta['input_count'] = 191
    _flush(c)
    if fault:
        with pytest.raises(ValueError): _validate(c)
    else:
        result = _validate(c)
        assert result['status'] == 'PHOTO_INPUT_IDENTITIES_BOUND'
        assert len(result['frames']) == 2
        assert result['geometry_lineage'] == 'NOT_VERIFIED'


INVALID_ASSOCIATIONS = [
    "bridge_failed", "bridge_input_drift", "bridge_wrong_image_root", "binding_missing",
    "binding_duplicate", "binding_hash", "binding_bytes_bool", "model_unknown", "model_unusable",
    "registered_ids", "native_camera", "native_name", "scene_missing_view", "sfm_upstream_run",
    "io_dataset", "io_count_bool", "sfm_count", "row_duplicate_index", "row_frame_id",
    "row_upstream_output", "row_registration_model", "row_registration_image",
    "row_registration_camera", "row_canonical_hash", "row_escape", "manifest_escape",
]


@pytest.mark.parametrize("fault", INVALID_ASSOCIATIONS)
def test_mismatched_or_incomplete_associations_are_rejected(chain, fault):
    if fault == "bridge_failed":
        chain.bridge_result["status"] = "FAIL"
    elif fault == "bridge_input_drift":
        chain.bridge_result["input_bindings_unchanged"] = False
    elif fault == "bridge_wrong_image_root":
        chain.plan["source_images"] = str(chain.io.resolve())
    elif fault == "binding_missing":
        chain.input_bindings.pop()
    elif fault == "binding_duplicate":
        chain.input_bindings[-1] = copy.deepcopy(chain.input_bindings[0])
    elif fault == "binding_hash":
        chain.input_bindings[0]["sha256"] = "0" * 64
    elif fault == "binding_bytes_bool":
        chain.input_bindings[0]["bytes"] = True
    elif fault == "model_unknown":
        chain.spec["model_id"] = "missing-model"
    elif fault == "model_unusable":
        chain.sfm_meta["models"][0]["usable"] = False
    elif fault == "registered_ids":
        chain.sfm_meta["models"][0]["registered_image_ids"] = [11, 99]
    elif fault == "native_camera":
        chain.selected_views[0]["camera_id"] = 8
    elif fault == "native_name":
        chain.selected_views[0]["name"] = "images/wrong.png"
    elif fault == "scene_missing_view":
        chain.scene_images.pop()
    elif fault == "sfm_upstream_run":
        chain.sfm_meta["upstream_io_run_id"] = "other-io"
    elif fault == "io_dataset":
        chain.io_meta["dataset_id"] = "other-dataset"
    elif fault == "io_count_bool":
        chain.io_meta["ok_count"] = True
    elif fault == "sfm_count":
        chain.sfm_meta["counts"]["input_rows"] = 1
    elif fault == "row_duplicate_index":
        chain.map_rows[1]["input_index"] = 0
    elif fault == "row_frame_id":
        chain.map_rows[0]["frame_id"] = "other-frame"
    elif fault == "row_upstream_output":
        chain.map_rows[0]["upstream_image_output"] = chain.io_rows[1]["image_output"]
    elif fault == "row_registration_model":
        chain.map_rows[0]["registrations"][0]["model_id"] = "other-model"
    elif fault == "row_registration_image":
        chain.map_rows[0]["registrations"][0]["image_id"] = 23
    elif fault == "row_registration_camera":
        chain.map_rows[0]["registrations"][0]["camera_id"] = 8
    elif fault == "row_canonical_hash":
        chain.map_rows[0]["canonical_image_sha256"] = "0" * 64
    elif fault == "row_escape":
        chain.io_rows[0]["image_output"] = "../outside.png"
        chain.map_rows[0]["upstream_image_output"] = "../outside.png"
    elif fault == "manifest_escape":
        chain.manifest["frames"][0]["image"] = "../outside.png"
        chain.io_rows[0]["image_path"] = "../outside.png"
    else:
        raise AssertionError(f"unhandled TEST_DATA case: {fault}")
    _flush(chain)
    with pytest.raises(ValueError):
        _validate(chain)


@pytest.mark.parametrize("target", ["raw", "canonical", "staged", "manifest"])
def test_actual_file_drift_is_rejected_despite_success_metadata(chain, target):
    if target == "raw":
        _png(chain.raw_paths[0], 2)
    elif target == "canonical":
        _png(chain.canonical_paths[0], 2, canonical=True)
    elif target == "staged":
        _png(chain.staged_paths[0], 2, canonical=True)
        chain.input_bindings[-2] = _binding(chain.staged_paths[0])
        _json(chain.bridge / "INPUT_BINDINGS.json", chain.input_bindings)
    else:
        chain.manifest_path.write_text(chain.manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _validate(chain)


@pytest.mark.parametrize("fault", ["nonobject", "duplicate_key", "nonfinite"])
def test_spec_requires_strict_object_json(chain, fault):
    # Duplicate/nonfinite forms otherwise contain the complete valid spec, so
    # missing required fields cannot hide a permissive JSON parser.
    valid = json.dumps(chain.spec)
    payload = {"nonobject": "[]",
               "duplicate_key": valid[:-1] + ', "model_id": "0"}',
               "nonfinite": valid[:-1] + ', "TEST_DATA_nonfinite": NaN}'}[fault]
    chain.spec_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        _validate(chain)


@pytest.mark.parametrize("field", ["bridge_run", "sfm_run", "io_run", "dataset_manifest"])
def test_spec_paths_must_be_absolute(chain, field):
    chain.spec[field] = "relative/TEST_DATA"
    _flush(chain)
    with pytest.raises(ValueError):
        _validate(chain)


def test_scene_same_basename_in_wrong_directory_is_rejected(chain):
    expected = chain.bridge / chain.scene_images[0]["name"]
    wrong_name = f"undistorted/other/{chain.views[0]['name']}"
    wrong = chain.bridge / wrong_name
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(expected.read_bytes())
    assert wrong.name == expected.name and _binding(wrong)["sha256"] == _binding(expected)["sha256"]
    chain.scene_images[0]["name"] = wrong_name
    with pytest.raises(ValueError):
        _validate(chain)


def test_image_root_must_be_bridge_not_undistorted_directory(chain):
    wrong_root = chain.bridge / "undistorted"
    assert wrong_root.is_dir()
    with pytest.raises(ValueError):
        photo_inputs.validate(chain.spec_path, chain.scene_path, wrong_root)


def test_scene_index_preserves_scene_array_order_not_native_name_order(chain):
    chain.scene_images.reverse()
    for index, scene_view in enumerate(chain.scene_images):
        scene_view["index"] = index
    result = _validate(chain)
    by_native_name = {row["native_image"]: row for row in result["frames"]}
    assert by_native_name[chain.views[0]["name"]]["scene_index"] == 1
    assert by_native_name[chain.views[1]["name"]]["scene_index"] == 0
    assert by_native_name[chain.views[0]["name"]]["scene_global_id"] == 11
    assert by_native_name[chain.views[1]["name"]]["scene_global_id"] == 23


@pytest.mark.parametrize("fault", ["different_id", "boolean_id"])
def test_scene_global_id_must_be_the_strict_integer_native_image_id(chain, fault):
    if fault == "different_id":
        chain.scene_images[0]["global_id"] = 23
    else:
        # Make all corresponding native/registration IDs numerically equal to
        # True so this rejects the boolean type, not just unequal ID values.
        chain.views[0]["image_id"] = 1
        chain.source_views[0]["image_id"] = 1
        chain.selected_views[0]["image_id"] = 1
        chain.sfm_meta["models"][0]["registered_image_ids"] = [1, 23]
        chain.map_rows[0]["registrations"][0]["image_id"] = 1
        chain.scene_images[0]["global_id"] = True
        _flush(chain)
    with pytest.raises(ValueError):
        _validate(chain)


def _add_alias(chain, *, different_encoding: bool):
    raw = chain.dataset / "raw" / "alias.png"
    canonical = chain.io / "images" / "alias.png"
    _png(raw, 0)
    if different_encoding:
        with Image.open(chain.canonical_paths[0]) as source:
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("TEST_DATA", "same decoded pixels, independent PNG encoding")
            source.save(canonical, format="PNG", pnginfo=metadata)
    else:
        canonical.write_bytes(chain.canonical_paths[0].read_bytes())
    chain.manifest["frames"].append({"image": "raw/alias.png"})
    io_alias = copy.deepcopy(chain.io_rows[0])
    io_alias.update(input_index=2, frame_id="TEST_DATA_alias", image_path="raw/alias.png",
                    image_sha256=_binding(raw)["sha256"], image_output="images/alias.png")
    chain.io_rows.append(io_alias)
    map_alias = copy.deepcopy(chain.map_rows[0])
    map_alias.update(input_index=2, frame_id="TEST_DATA_alias", upstream_image_output="images/alias.png",
                     canonical_image_sha256=_binding(canonical)["sha256"])
    chain.map_rows.append(map_alias)
    chain.io_meta.update(input_count=3, ok_count=3)
    chain.sfm_meta["counts"]["input_rows"] = 3
    _flush(chain)
    return canonical


def test_same_pixels_with_different_png_bytes_cannot_alias_a_native_view(chain):
    canonical = _add_alias(chain, different_encoding=True)
    assert _binding(canonical)["sha256"] != _binding(chain.staged_paths[0])["sha256"]
    with Image.open(canonical) as alias, Image.open(chain.staged_paths[0]) as staged:
        assert (alias.mode, alias.size, alias.tobytes()) == (staged.mode, staged.size, staged.tobytes())
    with pytest.raises(ValueError):
        _validate(chain)


def test_same_canonical_bytes_can_alias_a_native_view(chain):
    canonical = _add_alias(chain, different_encoding=False)
    assert canonical.read_bytes() == chain.staged_paths[0].read_bytes()
    result = _validate(chain)
    assert result["status"] == "PHOTO_INPUT_IDENTITIES_BOUND"
    assert len(result["frames"]) == 2
    frame = next(row for row in result["frames"] if row["representative_input_index"] == 0)
    assert frame["native_image"] == chain.views[0]["name"]
    assert frame["scene_image"] == f"undistorted/images/{chain.views[0]['name']}"
    assert (frame["scene_index"], frame["scene_global_id"]) == (0, 11)
    aliases = {row["input_index"]: row for row in frame["aliases"]}
    assert set(aliases) == {0, 2}
    assert aliases[0]["canonical_image"]["sha256"] == aliases[2]["canonical_image"]["sha256"]
    assert aliases[2]["canonical_image"]["sha256"] == aliases[2]["staged_image"]["sha256"]
