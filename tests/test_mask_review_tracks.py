"""Hand-counted sidecar fixtures; no SfM/native/model execution."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mask_review_tracks_under_test", ROOT / "integrations/mask_review/tracks.py")
tracks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tracks)


def png(values=None, rgb=None):
    stream = io.BytesIO()
    image = Image.new("RGB" if rgb is not None else "L", (3, 2), rgb or 0)
    if values is not None:
        image.putdata(values)
    image.save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def case(tmp_path):
    # IDs, input indexes, row order and names intentionally have unrelated orders.
    # Shared point labels: 100=[F,B,F], 300=[B,F,B], 901=[F,B].
    definitions = [
        (17, 8, "zebra.png", [255, 0, 255, 0, 255, 0],
         [[1., .49], [.5, .5], [.49, .49], [.5, .5], [2.99, .49]], [300, -1, 100, -1, 901]),
        (904, 2, "apple.png", [255, 0, 0, 255, 0, 0],
         [[1.49, .49], [.99, .49], [2., .49]], [100, 300, 901]),
        (51, 23, "middle.png", [255, 0, 0, 0, 255, 0],
         [[.5, .5], [1.5, .5]], [100, 300]),
    ]
    images, frames, mapping = {}, [], []
    for iid, index, name, pixels, coords, pids in definitions:
        rel = "images/" + name
        path = tmp_path / rel
        path.parent.mkdir(exist_ok=True)
        raw = png(rgb=(iid % 256, 10, 60))
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        images[str(iid)] = dict(image_id=iid, name=rel, camera_id=77, points2D=coords, point3D_ids=pids)
        frames.append(dict(input_index=index, frame_id="frame_" + str(index), image_output=f"canonical/{index}.png",
                           source_image_sha256=digest, photo_io_image_sha256="b" * 64,
                           width=3, height=2, status="OK", mask_bytes=bytes(pixels), image_domain="photo_io_canonical"))
        mapping.append(dict(input_index=index, frame_id="frame_" + str(index), upstream_io_run_id="io-test",
                            upstream_image_output=f"canonical/{index}.png", canonical_image_sha256=digest,
                            representative_input_index=index, staged_image_name=rel, database_image_id=iid,
                            registrations=[dict(model_id="chosen", image_id=iid, camera_id=77)]))
    model = dict(num_cameras=1, num_images=3, num_points3D=3, num_reg_images=3,
                 registered_image_ids=[17, 51, 904], cameras={"77": dict(camera_id=77, width=3, height=2, params=[3., 1.5, 1.])},
                 images=images, points3D={
                     "100": dict(xyz=[1., 2., 3.], track=[[17, 2], [904, 0], [51, 0]]),
                     "300": dict(xyz=[1., 2., 3.], track=[[17, 0], [904, 1], [51, 1]]),
                     "901": dict(xyz=[3., 2., 1.], track=[[17, 4], [904, 2]]),
                 })
    run = dict(upstream_io_run_id="io-test", coordinate_convention=tracks.CONVENTION,
               models=[dict(model_id="chosen", descriptions_path="models/chosen/descriptions.json",
                            registered_image_ids=[17, 51, 904], num_registered_images=3, num_points3D=3)])
    reads = []

    def write():
        (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
        description = tmp_path / "models/chosen/descriptions.json"
        description.parent.mkdir(parents=True, exist_ok=True)
        description.write_text(json.dumps({"binary": model, "source": {"not_consumed": True}}), encoding="utf-8")
        (tmp_path / "frames_map.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reversed(mapping)), encoding="utf-8")

    def read(path, maximum):
        assert maximum == 64 * 1024 * 1024
        reads.append(path)
        return path.read_bytes()

    def diagnose():
        write()
        return tracks.diagnose(tmp_path, "chosen", frames, "io-test", read)

    return dict(root=tmp_path, frames=frames, mapping=mapping, model=model, run=run,
                reads=reads, write=write, read=read, diagnose=diagnose)


def test_hand_counted_unordered_pairs_and_foreground_denominator(case):
    result = case["diagnose"]()
    assert result["observations"] == dict(foreground=4, background=4, missing_mask=0, out_of_bounds=0)
    assert result["pair_totals"]["FF"] == 1
    assert result["pair_totals"]["FB"] == 5
    assert result["pair_totals"]["BB"] == 1
    assert result["pair_totals"]["valid_shared_points"] == 7
    assert result["pair_totals"]["foreground_union"] == 6
    assert result["pair_totals"]["foreground_conditioned_disagreement"] == pytest.approx(5 / 6)
    assert result["points"]["mixed"] == result["points"]["comparable_foreground_union"] == 3
    assert result["points"]["foreground_conditioned_disagreement"] == 1
    assert {tuple(p["image_ids"]): p["valid_shared_points"] for p in result["pairs"]} == {(17, 51): 2, (17, 904): 3, (51, 904): 2}
    assert result["source_evidence"] == "SOURCE_SIDECAR_NOT_NATIVE_RELOADED"
    assert result["cross_view_status"] == result["quality_status"] == "NOT_EVALUATED"
    assert "PASS" not in result["status"]
    assert len(set(case["reads"])) == 6  # run, map, sidecar and three staged PNGs.
    json.dumps(result, allow_nan=False)


def test_explicit_mapping_survives_frame_and_row_reordering(case):
    case["frames"].reverse()
    result = case["diagnose"]()
    assert {r["input_index"]: r["image_id"] for r in result["bound_frames"]} == {8: 17, 2: 904, 23: 51}
    assert result["pair_totals"]["FB"] == 5


@pytest.mark.parametrize("mutation", ["name", "registration", "frame", "source_hash", "upstream", "domain", "camera_size", "convention"])
def test_wrong_join_or_domain_is_rejected(case, mutation):
    if mutation == "name": case["mapping"][0]["staged_image_name"] = "images/apple.png"
    elif mutation == "registration": case["mapping"][0]["registrations"][0]["image_id"] = 904
    elif mutation == "frame": case["frames"][0]["frame_id"] = "other"
    elif mutation == "source_hash": case["frames"][0]["source_image_sha256"] = "c" * 64
    elif mutation == "upstream": case["run"]["upstream_io_run_id"] = "wrong-run"
    elif mutation == "domain": case["frames"][0]["image_domain"] = "undistorted"
    elif mutation == "camera_size": case["frames"][0]["width"] = 4
    elif mutation == "convention": case["run"]["coordinate_convention"] = "pixel_center_zero"
    with pytest.raises(ValueError): case["diagnose"]()


def test_actual_staged_bytes_must_match_source_hash(case):
    (case["root"] / "images/zebra.png").write_bytes(png(rgb=(99, 99, 99)))
    with pytest.raises(ValueError, match="staged image hash"):
        case["diagnose"]()


@pytest.mark.parametrize("x, expected", [(0., True), (.49, True), (.5, True), (.99, True), (1., False),
                                         (1.49, False), (1.5, False), (2.99, True), (3., None), (-.01, None)])
def test_native_floor_half_pixel_and_border_sampling(case, x, expected):
    # Only source observation 17/2 changes. Other seven labels are hand-counted.
    case["model"]["images"]["17"]["points2D"][2] = [x, .49]
    result = case["diagnose"]()
    assert result["observations"]["foreground"] == 3 + int(expected is True)
    assert result["observations"]["background"] == 4 + int(expected is False)
    assert result["observations"]["out_of_bounds"] == int(expected is None)
    assert result["pair_totals"]["valid_shared_points"] == (5 if expected is None else 7)


def test_axes_are_not_transposed(case):
    case["frames"][0]["mask_bytes"] = bytes([255, 0, 255, 255, 255, 0])
    case["model"]["images"]["17"]["points2D"][2] = [.5, 1.5]
    result = case["diagnose"]()
    assert result["observations"]["foreground"] == 4
    assert result["observations"]["background"] == 4


def test_duplicate_same_view_excludes_entire_point_not_arbitrary_observation(case):
    image = case["model"]["images"]["17"]
    image["point3D_ids"][1] = 100
    image["points2D"][1] = [1.5, .5]  # opposing label in the same native view
    case["model"]["points3D"]["100"]["track"].append([17, 1])
    result = case["diagnose"]()
    assert result["excluded_duplicate_point_count"] == 1
    assert result["excluded_duplicate_observations"] == 4
    assert result["scene"]["eligible_points"] == 2
    assert result["observations"] == dict(foreground=2, background=3, missing_mask=0, out_of_bounds=0)
    assert result["pair_totals"]["valid_shared_points"] == 4


@pytest.mark.parametrize("mutation", ["track_wrong_point", "reverse_missing", "outside_index", "exact_duplicate", "nonfinite"])
def test_malformed_full_feature_track_topology_rejected(case, mutation):
    if mutation == "track_wrong_point": case["model"]["points3D"]["100"]["track"][0] = [17, 0]
    elif mutation == "reverse_missing": case["model"]["images"]["17"]["point3D_ids"][3] = 100
    elif mutation == "outside_index": case["model"]["points3D"]["100"]["track"][0] = [17, 999]
    elif mutation == "exact_duplicate": case["model"]["points3D"]["100"]["track"].append([17, 2])
    elif mutation == "nonfinite": case["model"]["images"]["17"]["points2D"][3] = [float("nan"), 0]
    with pytest.raises(ValueError): case["diagnose"]()


def add_alias(case, different=False):
    row, frame = deepcopy(case["mapping"][0]), deepcopy(case["frames"][0])
    row["input_index"] = frame["input_index"] = 80
    row["frame_id"] = frame["frame_id"] = "alias-frame"
    row["upstream_image_output"] = frame["image_output"] = "canonical/alias.png"
    row["representative_input_index"] = 8
    if different: frame["mask_bytes"] = bytes([0, 255, 255, 0, 255, 0])
    case["mapping"].append(row)
    case["frames"].append(frame)


def test_identical_alias_is_one_view_without_inflated_denominator(case):
    add_alias(case)
    result = case["diagnose"]()
    assert result["coverage"]["requested_frames"] == 4
    assert result["coverage"]["images_with_masks"] == 3
    assert result["pair_totals"]["valid_shared_points"] == 7


def test_conflicting_alias_masks_rejected(case):
    add_alias(case, different=True)
    with pytest.raises(ValueError, match="aliases"):
        case["diagnose"]()


def test_no_masks_are_unknown_not_background(case):
    for f in case["frames"]:
        f.update(status="NEEDS_SELECTION", mask_bytes=None)
    result = case["diagnose"]()
    assert result["observations"] == dict(foreground=0, background=0, missing_mask=8, out_of_bounds=0)
    assert result["pair_totals"]["excluded_unknown"] == 7
    assert result["pair_totals"]["agreement"] is None
    assert result["points"]["no_valid_observations"] == 3


def test_background_only_observations_have_no_foreground_rate_or_pass(case):
    # Nonempty foreground exists elsewhere but no observed track samples it.
    for f in case["frames"]:
        f["mask_bytes"] = bytes([0, 0, 0, 0, 255, 0])
    result = case["diagnose"]()
    assert result["pair_totals"]["BB"] == 7
    assert result["pair_totals"]["agreement"] == 1
    assert result["pair_totals"]["foreground_conditioned_disagreement"] is None
    assert result["points"]["all_background"] == 3
    assert result["points"]["foreground_conditioned_disagreement"] is None
    assert "NO_COMPARABLE_FOREGROUND_SUPPORT" in result["flags"]
    assert result["cross_view_status"] == "NOT_EVALUATED"


def test_one_valid_view_does_not_enter_point_comparison_denominator(case):
    case["frames"][:] = case["frames"][:1]
    result = case["diagnose"]()
    assert result["points"]["one_valid_observation"] == 3
    assert result["points"]["any_observed_foreground"] == 2
    assert result["points"]["comparable_foreground_union"] == 0
    assert result["points"]["foreground_conditioned_disagreement"] is None
    assert result["coverage"]["scene_images"] == 3
    assert result["coverage"]["scene_image_mask_fraction"] == pytest.approx(1 / 3)


def test_empty_scene_has_explicit_zero_denominators(case):
    model = case["model"]
    model.update(num_cameras=0, num_images=0, num_points3D=0, num_reg_images=0,
                 registered_image_ids=[], cameras={}, images={}, points3D={})
    case["run"]["models"][0].update(registered_image_ids=[], num_registered_images=0, num_points3D=0)
    case["mapping"].clear()
    case["frames"].clear()
    result = case["diagnose"]()
    assert result["pairs"] == []
    assert result["coverage"]["valid_pair_coverage"] is None
    assert result["points"]["mixed_fraction"] is None
    assert result["pair_totals"]["agreement"] is None


def test_pair_work_cap_is_enforced_before_pair_expansion(case, monkeypatch):
    monkeypatch.setattr(tracks, "MAX_PAIR_WORK", 6)
    with pytest.raises(ValueError, match="pair work limit"):
        case["diagnose"]()


def test_unsafe_description_path_never_reaches_reader(case):
    case["run"]["models"][0]["descriptions_path"] = "../outside.json"
    with pytest.raises(ValueError, match="unsafe"):
        case["diagnose"]()
    assert case["reads"] == [case["root"] / "run.json"]


@pytest.mark.parametrize("invalid", [png([255, 0, 255, 0, 255, 0]), b"\xff\0", b"\xff\0\0\0\0\0\0", b"\xff\0\0\0\0\x01"])
def test_mask_interface_rejects_encoded_png_wrong_length_and_nonbinary_labels(case, invalid):
    case["frames"][0]["mask_bytes"] = invalid
    with pytest.raises(ValueError, match="decoded mask"):
        case["diagnose"]()


def test_mask_plane_hash_is_explicitly_distinct_from_source_png_hash(case):
    result = case["diagnose"]()
    bound = next(r for r in result["bound_frames"] if r["input_index"] == 8)
    assert bound["selected_mask_pixels_sha256"] == hashlib.sha256(case["frames"][0]["mask_bytes"]).hexdigest()
    assert "selected_mask_sha256" not in bound
