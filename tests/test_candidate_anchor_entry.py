"""Entry helpers with fake native objects; never loads pycolmap/models."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_anchor_entry", ROOT / "scripts/evaluate_candidate_anchor_geometry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixture():
    rotation = np.eye(3)
    pose = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda: rotation), translation=np.zeros(3))
    image = SimpleNamespace(name="000081.png", cam_from_world=lambda: pose)
    camera = SimpleNamespace(model_name="SIMPLE_RADIAL", width=8, height=6, params=np.array([100., 4., 3., 0.]))
    c = dict(image_id=12, input_index=81, params=camera.params.tolist(), R=rotation.tolist(), t=[0., 0., 0.])
    frame = dict(input_index=81, frame_id="f81", sha256="a" * 64, width=8, height=6)
    mapping = dict(database_image_id=12, input_index=81, frame_id="f81", canonical_image_sha256="a" * 64,
                   staged_image_name="000081.png")
    return c, image, camera, frame, mapping


def test_verify_camera_correct_fake_native():
    c, image, camera, frame, mapping = fixture()
    before = deepcopy(c)
    got = entry.verify_camera(c, image, camera, frame, mapping)
    assert got["camera_model"] == "SIMPLE_RADIAL"
    assert got["canonical_image_sha256"] == frame["sha256"]
    assert c == before


@pytest.mark.parametrize("where,key,value", [
    (0, "image_id", 13), (0, "input_index", 82), (4, "input_index", 82),
    (4, "frame_id", "different"), (4, "canonical_image_sha256", "b" * 64),
    (4, "staged_image_name", "wrong.png"), (3, "width", 7), (3, "height", 5),
    (0, "R", [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]),
    (0, "t", [1., 0., 0.]), (0, "params", [101., 4., 3., 0.]),
])
def test_verify_camera_mismatch(where, key, value):
    args = list(fixture())
    args[where][key] = value
    with pytest.raises(ValueError):
        entry.verify_camera(*args)


def test_native_model_and_rotation_rejected():
    args = list(fixture())
    args[2].model_name = "PINHOLE"
    with pytest.raises(ValueError):
        entry.verify_camera(*args)
    args = list(fixture())
    args[0]["R"] = np.zeros((3, 3)).tolist()
    args[1].cam_from_world().rotation.matrix = lambda: np.zeros((3, 3))
    with pytest.raises(ValueError):
        entry.verify_camera(*args)


def test_mask_check_colmap_floor_not_round():
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 1] = 255
    assert entry.mask_check(mask, [1.99, 1.99]) is True
    assert entry.mask_check(mask, [2., 1.99]) is False
    assert entry.mask_check(mask, [-.01, 1.]) is False
    assert entry.mask_check(mask, [4., 1.]) is False
    with pytest.raises(ValueError):
        entry.mask_check(mask, [float("nan"), 1.])


def test_require_bound_needs_explicit_authority(tmp_path):
    p = tmp_path / "source.json"
    p.write_text("{}", encoding="utf-8")
    ref = entry.bind(p)
    assert entry.require_bound(p, [ref]) == ref
    with pytest.raises(ValueError, match="authority binding"):
        entry.require_bound(p, [])
    wrong = dict(ref, sha256="0" * 64)
    with pytest.raises(ValueError, match="authority binding"):
        entry.require_bound(p, [ref, wrong])
    with pytest.raises(ValueError, match="binding drift"):
        entry.require_bound(p, [wrong])
    p.write_text('{"drift":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="binding drift"):
        entry.require_bound(p, [ref])
