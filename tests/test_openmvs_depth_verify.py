"""TEST_DATA: saved-depth outcome checks, never native execution evidence.

The tiny DR encoder below is independent of product parsing/writing helpers.
Every artifact is synthetic and created beneath pytest's tmp_path. No OpenMVS,
native/GPU execution, real image corpus, or aggregate causal claim is involved.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

from integrations.openmvs_depth.verify import (
    DepthVerificationError,
    legacy_nearest_reference,
    verify_depth_pair,
)


_K = ((120.0, 0.25, 1.0), (0.0, 125.0, 0.5), (0.0, 0.0, 1.0))
_R = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
_C = (1.25, -2.5, 3.75)
_LABELS = ((0, 255, 0), (0, 255, 255))
_VALIDITY = ((255, 255, 0), (255, 255, 255))
_MASKED = ((0.0, 2.0, 0.0), (0.0, 4.0, 6.0))
_CONTROL = ((7.0, 2.0, 9.0), (0.0, 4.0, 6.0))


def _dr_bytes(
    depth, *, image_name: bytes = b"image.png", view_ids=(7, 11, 13),
    K=_K, R=_R, C=_C, image_size=None,
) -> bytes:
    """Encode depth-only little-endian DR bytes from explicit test values."""
    height, width = len(depth), len(depth[0])
    assert all(len(row) == width for row in depth)
    values = tuple(value for row in depth for value in row)
    image_size = (width, height) if image_size is None else image_size
    return b"".join((
        struct.pack("<2sBBIIIIff", b"DR", 1, 0, *image_size,
                    width, height, 1.0, 20.0),
        struct.pack("<H", len(image_name)), image_name,
        struct.pack("<I", len(view_ids)),
        struct.pack(f"<{len(view_ids)}I", *view_ids),
        struct.pack("<9d", *(value for row in K for value in row)),
        struct.pack("<9d", *(value for row in R for value in row)),
        struct.pack("<3d", *C),
        struct.pack(f"<{len(values)}f", *values),
    ))


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_labels(path: Path, labels) -> str:
    data = _png_bytes(Image.fromarray(np.array(labels, dtype=np.uint8)))
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _pair(tmp_path: Path, *, labels=_LABELS, validity=_VALIDITY,
          masked=_MASKED, control=_CONTROL) -> dict:
    root = tmp_path / "synthetic-pair"
    root.mkdir()
    for arm in ("masked", "control"):
        (root / arm).mkdir()
    image_size = (len(labels[0]), len(labels))
    image_bytes = _png_bytes(Image.new("RGB", image_size, color=(13, 27, 41)))
    for arm, depth in (("masked", masked), ("control", control)):
        (root / arm / "image.png").write_bytes(image_bytes)
        # Each dmap refers to its own arm's file, even though pixels/bytes match.
        (root / arm / "depth.dmap").write_bytes(_dr_bytes(depth))
    mask = root / "mask.png"
    validity_path = root / "validity.png"
    return {
        "masked_path": root / "masked" / "depth.dmap",
        "control_path": root / "control" / "depth.dmap",
        "input_root": root,
        "masked_image_path": root / "masked" / "image.png",
        "control_image_path": root / "control" / "image.png",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "mask_path": mask,
        "mask_sha256": _write_labels(mask, labels),
        "validity_path": validity_path,
        "validity_sha256": _write_labels(validity_path, validity),
        "reference_id": 7,
        "expected_K": _K,
        "expected_R": _R,
        "expected_C": _C,
        "masked_ignore_label": 0,
        "control_ignore_label": -1,
        "postprocess_dmaps": 0,
    }


def _rewrite_map(pair: dict, arm: str, **changes) -> None:
    depth = changes.pop("depth", _MASKED if arm == "masked" else _CONTROL)
    pair[f"{arm}_path"].write_bytes(_dr_bytes(depth, **changes))


def _assert_evidence_boundary(result: dict) -> None:
    assert result["comparison_scope"] == "one_reference_saved_depth_outcome"
    assert result["native_consumption_status"] == "NOT_VERIFIED"
    assert result["runtime_provenance_status"] == "NOT_VERIFIED"
    assert result["configuration_status"] == "DECLARED_NOT_NATIVE_VERIFIED"
    assert result["mask_alignment_status"] == "CALLER_SUPPLIED"
    assert result["quality_status"] == "NOT_EVALUATED"


def test_giant_broadcast_geometry_is_rejected_before_copy(monkeypatch):
    from integrations.openmvs_depth import verify
    huge = np.broadcast_to(np.array(1.0), (1_000_000_000, 3))

    def forbidden(*args, **kwargs):
        pytest.fail('shape rejection must precede NumPy copy')

    monkeypatch.setattr(verify.np, 'array', forbidden)
    with pytest.raises(DepthVerificationError, match='shape'):
        verify._matrix(huge, (3, 3), 'K')


@pytest.mark.parametrize('K', [
    ((1., 1., 0.), (1., 1., 0.), (0., 0., 1.)),
    ((1e-8, 1., 0.), (1e-8, 1., 0.), (0., 0., 1.)),
])
def test_singular_calibration_cannot_be_supported(tmp_path, K):
    pair = _pair(tmp_path)
    _rewrite_map(pair, 'masked', K=K)
    _rewrite_map(pair, 'control', K=K)
    pair['expected_K'] = K
    with pytest.raises(DepthVerificationError, match='pinhole'):
        verify_depth_pair(**pair)


def test_same_size_supported_requires_nonvacuous_matched_depth_signals(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    before = {path: path.read_bytes() for path in pair["input_root"].rglob("*")
              if path.is_file()}
    result = verify_depth_pair(**pair)

    assert result["outcome_status"] == "SUPPORTED"
    assert result["reasons"] == []
    _assert_evidence_boundary(result)
    assert result["sampling_status"] == "IDENTITY_NO_RESIZE"
    assert result["source_image_size"] == result["actual_depth_size"] == [3, 2]
    assert result["native_reference_id"] == 7
    assert result["view_ids"] == [7, 11, 13]
    assert result["expected_depth_geometry"] == {
        "K": [list(row) for row in _K], "R": [list(row) for row in _R], "C": list(_C),
    }
    assert result["declared_options"] == {
        "masked_ignore_label": 0, "control_ignore_label": -1, "postprocess_dmaps": 0,
    }
    assert result["counts"] == {
        "total_pixels": 6, "ignored_pixels": 3, "known_background_pixels": 2,
        "unknown_pixels": 1, "foreground_pixels": 3, "masked_ignored_nonzero": 0,
        "masked_known_background_nonzero": 0, "masked_unknown_nonzero": 0,
        "masked_foreground_nonzero": 3, "control_known_background_nonzero": 1,
        "control_foreground_nonzero": 3, "known_background_removed": 1,
    }
    assert result["inputs_stable"] is True
    bindings = {item["role"]: item for item in result["inputs"]}
    assert set(bindings) == {"masked_depth", "control_depth", "masked_image",
                             "control_image", "mask", "validity"}
    for binding in bindings.values():
        path = Path(binding["path"])
        assert binding["bytes"] == len(before[path])
        assert binding["sha256"] == hashlib.sha256(before[path]).hexdigest()
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("position,known,unknown", [((0, 0), 1, 0), ((0, 2), 0, 1)])
def test_nonzero_depth_in_any_ignored_pixel_is_violation(
    tmp_path: Path, position: tuple[int, int], known: int, unknown: int,
) -> None:
    masked = [list(row) for row in _MASKED]
    masked[position[0]][position[1]] = 0.25
    result = verify_depth_pair(**_pair(tmp_path, masked=masked))
    assert result["outcome_status"] == "VIOLATION"
    assert result["reasons"] == ["NONZERO_DEPTH_IN_DECLARED_IGNORED_PIXELS"]
    assert result["counts"]["masked_ignored_nonzero"] == 1
    assert result["counts"]["masked_known_background_nonzero"] == known
    assert result["counts"]["masked_unknown_nonzero"] == unknown
    _assert_evidence_boundary(result)


def test_all_zero_depth_pair_is_inconclusive(tmp_path: Path) -> None:
    zeros = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = verify_depth_pair(**_pair(tmp_path, masked=zeros, control=zeros))
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert set(result["reasons"]) == {
        "NO_MASKED_FOREGROUND_DEPTH", "NO_CONTROL_FOREGROUND_DEPTH",
        "NO_CONTROL_BACKGROUND_SIGNAL",
    }
    _assert_evidence_boundary(result)


@pytest.mark.parametrize("arm,reason", [
    ("masked", "NO_MASKED_FOREGROUND_DEPTH"),
    ("control", "NO_CONTROL_FOREGROUND_DEPTH"),
])
def test_each_arm_must_have_foreground_depth(tmp_path: Path, arm: str, reason: str) -> None:
    replacement = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) if arm == "masked" else (
        (7.0, 0.0, 9.0), (0.0, 0.0, 0.0))
    result = verify_depth_pair(**_pair(tmp_path, **{arm: replacement}))
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert result["reasons"] == [reason]
    _assert_evidence_boundary(result)


def test_all_zero_mask_does_not_vacuously_support_exclusion(tmp_path: Path) -> None:
    zeros = ((0, 0, 0), (0, 0, 0))
    result = verify_depth_pair(**_pair(tmp_path, labels=zeros, masked=zeros))
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert set(result["reasons"]) == {
        "NO_MASKED_FOREGROUND_DEPTH", "NO_CONTROL_FOREGROUND_DEPTH",
    }
    _assert_evidence_boundary(result)


def test_all_foreground_mask_has_no_background_evidence(tmp_path: Path) -> None:
    foreground = ((255, 255, 255), (255, 255, 255))
    result = verify_depth_pair(**_pair(tmp_path, labels=foreground, validity=foreground))
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert set(result["reasons"]) == {"NO_KNOWN_BACKGROUND", "NO_CONTROL_BACKGROUND_SIGNAL"}
    assert result["counts"]["ignored_pixels"] == 0
    _assert_evidence_boundary(result)


def test_unknown_pixels_do_not_count_as_known_background(tmp_path: Path) -> None:
    result = verify_depth_pair(**_pair(tmp_path, validity=_LABELS))
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert set(result["reasons"]) == {"NO_KNOWN_BACKGROUND", "NO_CONTROL_BACKGROUND_SIGNAL"}
    assert result["counts"]["unknown_pixels"] == 3
    assert result["counts"]["known_background_pixels"] == 0
    assert result["counts"]["known_background_removed"] == 0
    _assert_evidence_boundary(result)


def test_control_depth_only_in_unknown_ignored_region_is_not_background_signal(tmp_path: Path) -> None:
    control = ((0.0, 2.0, 9.0), (0.0, 4.0, 6.0))
    result = verify_depth_pair(**_pair(tmp_path, control=control))
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert result["reasons"] == ["NO_CONTROL_BACKGROUND_SIGNAL"]
    assert result["counts"]["known_background_pixels"] == 2
    assert result["counts"]["known_background_removed"] == 0
    _assert_evidence_boundary(result)


def test_foreground_marked_geometrically_unknown_is_rejected(tmp_path: Path) -> None:
    validity = ((255, 0, 0), (255, 255, 255))
    with pytest.raises(DepthVerificationError, match="foreground.*unknown"):
        verify_depth_pair(**_pair(tmp_path, validity=validity))


def test_noninteger_legacy_nearest_has_independently_enumerated_grid() -> None:
    labels = np.array(((0, 255, 0, 255, 0), (255, 0, 255, 0, 255),
                       (0, 0, 255, 255, 255)), dtype=np.uint8)
    # 5x3 -> 3x2: source columns [0, 1, 3], rows [0, 1].
    # This expected result is concrete, not computed with the product helper.
    expected = np.array(((0, 255, 255), (255, 0, 0)), dtype=np.uint8)
    np.testing.assert_array_equal(legacy_nearest_reference(labels, (3, 2)), expected)


@pytest.mark.parametrize("ignored_nonzero", [False, True])
def test_resizing_remains_reference_only_even_if_diagnostic_looks_decisive(
    tmp_path: Path, ignored_nonzero: bool,
) -> None:
    labels = ((0, 255, 0, 255, 0), (255, 0, 255, 0, 255), (0, 0, 255, 255, 255))
    validity = ((255, 255, 255, 255, 255), (255, 255, 255, 0, 255),
                (255, 255, 255, 255, 255))
    # Independent projected labels [[0,255,255],[255,0,0]], validity
    # [[255,255,255],[255,255,0]]; no product sampler constructs depth data.
    masked = ((1.0 if ignored_nonzero else 0.0, 2.0, 3.0), (4.0, 0.0, 0.0))
    control = ((5.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    result = verify_depth_pair(**_pair(tmp_path, labels=labels, validity=validity,
                                      masked=masked, control=control))
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert result["reasons"] == ["NATIVE_RESIZE_MAPPING_UNVERIFIED"]
    assert result["sampling_status"] == "REFERENCE_ONLY"
    assert result["source_image_size"] == [5, 3]
    assert result["actual_depth_size"] == [3, 2]
    assert result["counts"]["ignored_pixels"] == 3
    assert result["counts"]["known_background_pixels"] == 2
    assert result["counts"]["unknown_pixels"] == 1
    assert result["counts"]["foreground_pixels"] == 3
    assert result["counts"]["masked_ignored_nonzero"] == int(ignored_nonzero)
    assert result["counts"]["control_known_background_nonzero"] == 2
    _assert_evidence_boundary(result)


@pytest.mark.parametrize("postprocess", [1, 15])
def test_enabled_postprocess_is_outside_frozen_comparison(tmp_path: Path, postprocess: int) -> None:
    pair = _pair(tmp_path)
    pair["postprocess_dmaps"] = postprocess
    result = verify_depth_pair(**pair)
    assert result["outcome_status"] == "INCONCLUSIVE"
    assert result["reasons"] == ["POSTPROCESS_ENABLED_OUTSIDE_FROZEN_COMPARISON"]
    _assert_evidence_boundary(result)


def test_wrong_global_reference_id_is_rejected(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    pair["reference_id"] = 0
    with pytest.raises(DepthVerificationError, match="reference ID"):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("ids", [(8, 11, 13), (7, 11, 14), (7, 13, 11)])
def test_reference_and_ordered_neighbor_misbinding_is_rejected(tmp_path: Path, ids) -> None:
    pair = _pair(tmp_path)
    _rewrite_map(pair, "control", view_ids=ids)
    with pytest.raises(DepthVerificationError, match="identities differ"):
        verify_depth_pair(**pair)


def test_same_artifact_cannot_be_its_own_control(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    pair["control_path"] = pair["masked_path"]
    with pytest.raises(DepthVerificationError, match="distinct depth artifact"):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("case", ["other_arm", "absolute", "outside_root"])
def test_native_image_path_must_bind_own_arm_inside_root(tmp_path: Path, case: str) -> None:
    pair = _pair(tmp_path)
    if case == "other_arm":
        image_name = b"../control/image.png"
    elif case == "absolute":
        image_name = str(pair["masked_image_path"]).encode("utf-8")
    else:
        outside = tmp_path / "outside.png"
        outside.write_bytes(pair["masked_image_path"].read_bytes())
        image_name = b"../../outside.png"
    _rewrite_map(pair, "masked", image_name=image_name)
    with pytest.raises(DepthVerificationError):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("role", ["control_image", "mask", "validity"])
def test_supplied_hash_binds_every_raster_input(tmp_path: Path, role: str) -> None:
    pair = _pair(tmp_path)
    path = pair[f"{role}_path"]
    # A different, decodable same-size PNG still must fail its original digest.
    replacement = Image.new("RGB" if role == "control_image" else "L", (3, 2), color=0)
    path.write_bytes(_png_bytes(replacement))
    with pytest.raises(DepthVerificationError, match="SHA-256 mismatch"):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("role", ["mask", "validity"])
def test_raster_dimensions_must_match_source_image(tmp_path: Path, role: str) -> None:
    pair = _pair(tmp_path)
    pair[f"{role}_sha256"] = _write_labels(pair[f"{role}_path"], ((0, 0), (0, 0)))
    with pytest.raises(DepthVerificationError, match="dimensions disagree"):
        verify_depth_pair(**pair)


def test_depth_pair_dimensions_must_match(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    _rewrite_map(pair, "control", depth=((7.0, 2.0), (9.0, 0.0), (4.0, 6.0)))
    with pytest.raises(DepthVerificationError, match="depth dimensions"):
        verify_depth_pair(**pair)


def test_image_size_header_must_follow_supported_depth_save_convention(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    _rewrite_map(pair, "masked", image_size=(6, 4))
    with pytest.raises(DepthVerificationError, match="header convention"):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("arm,component,value", [
    ("masked", "K", ((121.0, 0.25, 1.0), (0.0, 125.0, 0.5), (0.0, 0.0, 1.0))),
    ("control", "R", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("control", "C", (1.25, -2.5, 4.75)),
])
def test_each_stored_camera_component_is_bound_to_declared_geometry(
    tmp_path: Path, arm: str, component: str, value,
) -> None:
    pair = _pair(tmp_path)
    _rewrite_map(pair, arm, **{component: value})
    with pytest.raises(DepthVerificationError, match=f"native {component} differs"):
        verify_depth_pair(**pair)


def test_two_matching_wrong_cameras_do_not_override_external_expectation(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    for arm in ("masked", "control"):
        _rewrite_map(pair, arm, C=(0.0, 0.0, 0.0))
    with pytest.raises(DepthVerificationError, match="native C differs"):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("key,value", [
    ("expected_K", ((-120.0, 0.0, 1.0), (0.0, 125.0, 0.5), (0.0, 0.0, 1.0))),
    ("expected_R", ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("expected_C", (1.0, 2.0)),
])
def test_invalid_expected_geometry_is_rejected(tmp_path: Path, key: str, value) -> None:
    pair = _pair(tmp_path)
    pair[key] = value
    with pytest.raises(DepthVerificationError):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("key,value", [
    ("masked_ignore_label", 255), ("control_ignore_label", 0),
    ("masked_ignore_label", False), ("reference_id", True), ("postprocess_dmaps", 16),
])
def test_invalid_declared_comparison_contract_is_rejected(tmp_path: Path, key: str, value) -> None:
    pair = _pair(tmp_path)
    pair[key] = value
    with pytest.raises(DepthVerificationError):
        verify_depth_pair(**pair)


@pytest.mark.parametrize("role", ["mask", "validity"])
def test_soft_labels_are_rejected_in_both_label_rasters(tmp_path: Path, role: str) -> None:
    pair = _pair(tmp_path)
    labels = ((0, 127, 0), (0, 255, 255))
    pair[f"{role}_sha256"] = _write_labels(pair[f"{role}_path"], labels)
    with pytest.raises(DepthVerificationError, match="exactly 0/255"):
        verify_depth_pair(**pair)


def test_color_mask_is_not_silently_converted_to_labels(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    data = _png_bytes(Image.new("RGB", (3, 2), color=(0, 0, 0)))
    pair["mask_path"].write_bytes(data)
    pair["mask_sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(DepthVerificationError, match="8-bit grayscale"):
        verify_depth_pair(**pair)
