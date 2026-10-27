"""Tests for the mvs-mask-v1 consumer and CLI (R3: real production identity).

Layers:

* ``integrations/mvs_masks/manifest_io.py`` - strict JSON (duplicate keys,
  non-finite numbers, ``1e999``, bool-as-int), schema/selection/counts
  consistency, path safety, mask label identity, OpenMVS naming collisions;
* the real CLI function chain (argparse -> selected-model set -> scene ->
  frames_map/T01 -> pycolmap models -> warp -> PNG write -> read-back ->
  manifest) against a small self-consistent synthetic world with **real byte
  semantics**: the T01 ``image_sha256`` is the original digest and deliberately
  differs from the canonical file, while the canonical file is a byte-exact copy
  of the staged file, exactly as the real producer behaves, and the frames_map
  carries rows the selected model does not contain.

Masks are synthetic geometric patterns, never SAM3 predictions or human truth,
and no test claims OpenMVS consumed them.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import struct
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI_PATH = pathlib.Path(os.environ.get("MVS_MASK_CLI_PATH", str(ROOT / "scripts" / "prepare_mvs_masks.py")))
INTEGRATIONS = pathlib.Path(os.environ.get("MVS_MASK_INTEGRATIONS_PATH", str(ROOT / "integrations")))
sys.path.insert(0, str(INTEGRATIONS))

from mvs_masks import manifest_io as mio  # noqa: E402

_spec = importlib.util.spec_from_file_location("prepare_mvs_masks_under_test", CLI_PATH)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

SRC_W, SRC_H = 64, 48
DST_W, DST_H = 60, 44
OBJECT_ID = "skull-01-synthetic"
ORIGINAL_SHA = "0" * 64  # the original input digest; never equal to any file here


# --------------------------------------------------------------------------- #
# minimal MVSI v7 writer (mirrors the frozen reader profile; test fixture only)
# --------------------------------------------------------------------------- #
def _s(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _f(values) -> bytes:
    return struct.pack("<%dd" % len(values), *values)


def _f32(values) -> bytes:
    return struct.pack("<%df" % len(values), *values)


def write_scene(path, names, width, height, *, global_ids=None, centers=None, focal=50.0,
                prefix="undistorted/images/images", principal_offset=(0.0, 0.0), rotate_first=False):
    """Write a structurally valid MVSI version 7 file holding the given images."""
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    blob = b"MVSI" + struct.pack("<II", 7, 0)
    blob += struct.pack("<Q", 1)
    blob += _s("platform-0")
    blob += struct.pack("<Q", 1)
    blob += _s("camera-0") + _s("")
    blob += struct.pack("<II", width, height)
    fx, fy = focal if isinstance(focal, tuple) else (focal, focal)
    blob += _f([fx, 0.0, width / 2 - 0.5 + principal_offset[0],
                0.0, fy, height / 2 - 0.5 + principal_offset[1], 0.0, 0.0, 1.0])
    blob += _f(identity)
    blob += _f([0.0, 0.0, 0.0])
    blob += struct.pack("<Q", len(names))
    for index in range(len(names)):
        center = centers[index] if centers is not None else [-float(index + 1), 0.0, 0.0]
        rotation = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] if rotate_first and index == 0 else identity
        blob += _f(rotation) + _f(center)
    blob += struct.pack("<Q", len(names))
    for index, name in enumerate(names):
        blob += _s(prefix + "/" + name) + _s("")
        blob += struct.pack("<IIII", 0, 0, index, global_ids[index] if global_ids is not None else index + 1)
        blob += _f32([0.0, 1.0, 2.0])
        blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 0)   # vertices
    blob += struct.pack("<Q", 0)   # verticesNormal
    blob += struct.pack("<Q", 0)   # verticesColor
    blob += struct.pack("<Q", 0)   # lines
    blob += struct.pack("<Q", 0)   # linesNormal
    blob += struct.pack("<Q", 0)   # linesColor
    blob += _f([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    blob += _f(identity) + _f([0.0, 0.0, 0.0]) + _f([1.0, 1.0, 1.0])
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_bytes(blob)


# --------------------------------------------------------------------------- #
# synthetic world
# --------------------------------------------------------------------------- #
def _rgb(i):
    arr = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
    arr[:, :, 0] = ((np.arange(SRC_W) * 4 + i * 8) % 256)[None, :]
    arr[:, :, 1] = ((np.arange(SRC_H) * 5 + i * 3) % 256)[:, None]
    arr[:, :, 2] = (i * 37) % 256
    return arr


def _mask(i):
    arr = np.zeros((SRC_H, SRC_W), dtype=np.uint8)
    arr[10:34, 8:48] = 255
    arr[4:8, 4 + i * 3:8 + i * 3] = 255
    return arr


def _sha(path):
    return mio.sha256_file(path)


def _write_model(path, model, width, height, params, names, poses):
    import pycolmap

    path.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera(model=model, width=width, height=height,
                                                   params=np.asarray(params, dtype=float), camera_id=1))
    for image_id, (name, pose) in enumerate(zip(names, poses), start=1):
        image = pycolmap.Image(name=name, camera_id=1, image_id=image_id)
        rec.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pycolmap.Rotation3d(), np.asarray(pose, dtype=float)))
    rec.write(str(path))


def world_manifest(root, *, n_ok=2, object_id=OBJECT_ID, t01_dir=None, directory=None,
                   name="manifest.json", mutate=None, source_status=None):
    """Write a production-dialect object-mask-v1 manifest for the synthetic world."""
    root = pathlib.Path(root)
    t01 = pathlib.Path(t01_dir or (root / "t01"))
    target = pathlib.Path(directory or (root / "objmask"))
    target.mkdir(parents=True, exist_ok=True)
    rows = {json.loads(line)["input_index"]: json.loads(line)
            for line in (t01 / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    frames = []
    for i in sorted(rows):
        ok = i < n_ok
        relative = f"masks/{i:06d}_c00.png"
        mask_path = target / relative
        if not mask_path.is_file():
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_mask(i), mode="L").save(mask_path, format="PNG")
        digest = _sha(mask_path)
        row = rows[i]
        frames.append({
            "input_index": i, "frame_id": row["frame_id"],
            "source_image": str((t01 / row["image_output"]).resolve()),
            "source_image_sha256": _sha(t01 / row["image_output"]),
            "photo_io_image_sha256": row["image_sha256"],
            "image_domain": "photo_io_canonical", "width": SRC_W, "height": SRC_H,
            "status": "OK" if ok else "NOT_RUN",
            "candidates": [{"candidate_id": "c00", "mask_path": relative, "mask_sha256": digest,
                            "foreground_pixels": int(np.count_nonzero(_mask(i))), "score": 0.9,
                            "score_semantics": "presence_calibrated_sigmoid_product_not_probability"}]
            if ok else [],
            "candidates_dropped_empty": 0,
            "selected_candidate_id": "c00" if ok else None,
            "selected_mask": {"path": relative, "sha256": digest} if ok else None,
            "selection": {"method": "unique_candidate", "source": "automatic",
                          "mask_source": "automatic_sam3"} if ok else None,
            "errors": [], "not_run_reason": None if ok else "deferred by the source run",
        })
    doc = {
        "contract_version": "object-mask-v1", "run_id": "20261007T185024Z_synthetic_fixture",
        "started_utc": "2026-10-07T18:50:24+00:00", "finished_utc": "2026-10-07T18:51:00+00:00",
        "status": source_status or ("SUCCESS" if n_ok == len(rows) else "RESOURCE_LIMIT"),
        "source_io": {"path": str(t01),
                      "run_id": json.loads((t01 / "run.json").read_text(encoding="utf-8"))["run_id"],
                      "run_json_sha256": _sha(t01 / "run.json"),
                      "frames_jsonl_sha256": _sha(t01 / "frames.jsonl")},
        "backend": {"name": "synthetic_interface_fixture", "code_commit": None, "checkpoint_path": None,
                    "checkpoint_sha256": None, "config_sha256": None, "python_version": None,
                    "torch_version": None, "cuda_runtime": None, "device": None},
        "object": {"object_id": object_id, "text_prompt": "synthetic", "visual_prompt": None,
                   "selection_policy": "unique_candidate_or_explicit"},
        "frames": frames,
        "counts": {"requested": len(frames), "success": n_ok, "no_match": 0, "needs_selection": 0,
                   "error": 0, "not_run": len(frames) - n_ok},
        "quality_status": "NOT_EVALUATED", "cross_view_status": "NOT_EVALUATED",
        "evidence_level": "synthetic_interface_fixture",
    }
    if mutate is not None:
        result = mutate(doc)
        if result is not None:
            doc = result
    path = target / name
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def build_world(root, *, model_images=2, extra_frames=2, n_ok=2):
    """Selected model set smaller than the frames_map; original digest != canonical."""
    root = pathlib.Path(root)
    t01, sfm, bridge, obj = root / "t01", root / "sfm", root / "bridge", root / "objmask"
    names = [f"images/{i:06d}.png" for i in range(model_images)]
    total = model_images + extra_frames
    for directory in (t01 / "images", sfm / "images", bridge / "undistorted" / "images" / "images",
                      bridge / "undistorted" / "sparse", obj / "masks"):
        directory.mkdir(parents=True, exist_ok=True)
    t01_rows, sfm_rows = [], []
    for i in range(total):
        name = f"images/{i:06d}.png"
        canonical, staged = t01 / name, sfm / name
        Image.fromarray(_rgb(i), mode="RGB").save(canonical, format="PNG", compress_level=6)
        staged.write_bytes(canonical.read_bytes())   # byte-exact copy, like the real producer
        assert _sha(staged) == _sha(canonical)
        frame_id = f"f_{i:040d}"
        t01_rows.append({"input_index": i, "frame_id": frame_id, "image_output": name, "image_path": name,
                         "image_sha256": ORIGINAL_SHA if i < model_images else "1" * 64, "status": "OK",
                         "canonical_width": SRC_W, "canonical_height": SRC_H,
                         "image_dtype": "uint8", "image_mode": "RGB"})
        sfm_rows.append({"upstream_io_run_id": "20261007T184900Z_aaaa", "input_index": i,
                         "frame_id": frame_id, "staged_image_name": name, "upstream_image_output": name,
                         "canonical_image_sha256": _sha(staged),
                         "status": "REGISTERED" if i < model_images else "UNREGISTERED",
                         "reason": None if i < model_images else "not registered in the selected model"})
    (t01 / "run.json").write_text(json.dumps({
        "contract_version": "photo-io-v1", "run_id": "20261007T184900Z_aaaa", "status": "SUCCESS",
        "input_count": total, "ok_count": total}, indent=2) + "\n", encoding="utf-8", newline="\n")
    (t01 / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in t01_rows) + "\n",
                                     encoding="utf-8", newline="\n")
    (sfm / "run.json").write_text(json.dumps({
        "contract_version": "sfm-native-object-v1", "run_id": "20261007T185000Z_bbbb",
        "upstream_io_run_id": "20261007T184900Z_aaaa", "status": "PARTIAL"}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    (sfm / "frames_map.jsonl").write_text("\n".join(json.dumps(r) for r in sfm_rows) + "\n",
                                         encoding="utf-8", newline="\n")
    poses = [[float(i + 1), 0.0, 0.0] for i in range(model_images)]
    _write_model(bridge / "undistorted" / "sparse", "PINHOLE", DST_W, DST_H,
                 [50.0, 50.0, DST_W / 2, DST_H / 2], names, poses)
    _write_model(root / "source_sparse", "SIMPLE_RADIAL", SRC_W, SRC_H,
                 [53.34, SRC_W / 2, SRC_H / 2, 0.0], names, poses)
    for name in names:
        Image.fromarray(np.zeros((DST_H, DST_W, 3), dtype=np.uint8), mode="RGB").save(
            bridge / "undistorted" / "images" / name, format="PNG")
    write_scene(bridge / "scene.mvs", [pathlib.Path(n).name for n in names], DST_W, DST_H)
    (bridge / "UNDISTORT_COMPARISON.json").write_text('{"status": "PASS"}\n', encoding="utf-8", newline="\n")
    (bridge / "PLAN.json").write_text(json.dumps({
        "source_sparse": str(root / "source_sparse"), "source_images": str(sfm),
        "tolerance": "synthetic"}, indent=2) + "\n", encoding="utf-8", newline="\n")
    (bridge / "RESULT.json").write_text(json.dumps({"status": "PASS"}, indent=2) + "\n",
                                       encoding="utf-8", newline="\n")
    bindings = [{"path": str(sfm / name), "bytes": (sfm / name).stat().st_size, "sha256": _sha(sfm / name)}
                for name in names]
    bindings += [{"path": str(p), "bytes": p.stat().st_size, "sha256": _sha(p)}
                 for p in sorted((root / "source_sparse").iterdir()) if p.is_file()]
    (bridge / "INPUT_BINDINGS.json").write_text(json.dumps(bindings, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    return {"root": root, "t01": t01, "sfm": sfm, "bridge": bridge, "obj": obj, "names": names,
            "manifest": world_manifest(root, n_ok=n_ok)}


def cli_args(world, manifest, output, *extra):
    root = world["root"] if isinstance(world, dict) else pathlib.Path(world)
    return ["--bridge", str(root / "bridge"), "--sfm-run", str(root / "sfm"),
            "--mask-manifest", str(manifest), "--output", str(output), *extra]


def parsed_map(path):
    return mio.parse_object_manifest(path)


@pytest.fixture()
def world(tmp_path):
    return build_world(tmp_path / "world")


# --------------------------------------------------------------------------- #
# 1. contract-conforming complete path through the real CLI chain
# --------------------------------------------------------------------------- #
def test_complete_success_path_with_scene_and_selected_set(tmp_path, world, capsys):
    out = tmp_path / "out_complete"
    code = cli.main(cli_args(world, world["manifest"], out))
    assert code == 0, capsys.readouterr().err
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCESS" and manifest["coverage_status"] == "COMPLETE"
    assert manifest["counts"] == {"expected": 2, "ok": 2, "missing": 0, "error": 0, "not_run": 0}
    assert manifest["publishable_arguments"] is True
    assert manifest["mvs_mask_arguments"]["available"] is True
    assert manifest["mvs_mask_arguments"]["verified"] is False
    assert manifest["mvs_mask_arguments"]["actual_mvs_invocations"] == 0
    assert manifest["inputs"]["input_bindings_unchanged"] is True
    assert manifest["scene"]["verified"] is True
    assert manifest["scene"]["version"] == 7 and manifest["scene"]["image_count"] == 2
    assert len(manifest["scene"]["mapping"]) == 2
    assert all(entry["scene_global_id"] for entry in manifest["scene"]["mapping"])
    assert manifest["unverified_edges"] == []
    summary = manifest["selection_summary"]
    assert summary["selected_model_images"] == 2
    assert summary["frames_map_rows"] == 4
    assert summary["frames_map_rows_unselected"] == 2
    assert summary["unregistered_input_indices"] == [2, 3]
    for frame in manifest["frames"]:
        assert frame["status"] == "OK" and frame["readback_verified"] is True
        assert frame["target_size"] == [DST_W, DST_H]
        assert frame["source_candidate_id"] == "c00"
        assert frame["selection"]["mask_source"] == "automatic_sam3"
        mask = np.asarray(Image.open(out / frame["mask"]["path"]), dtype=np.uint8)
        validity = np.asarray(Image.open(out / frame["validity"]["path"]), dtype=np.uint8)
        assert mask.shape == validity.shape == (DST_H, DST_W)
        assert set(int(v) for v in np.unique(mask)) <= {0, 255}
        assert not np.any((validity == 0) & (mask == 255))
        assert frame["core_counts"]["valid"] == int(np.count_nonzero(validity))
        assert frame["core_counts"]["unknown"] == int(np.count_nonzero(validity == 0))
        assert frame["mask"]["sha256"] == _sha(out / frame["mask"]["path"])
        assert frame["validity"]["sha256"] == _sha(out / frame["validity"]["path"])


def test_original_digest_differs_from_canonical_but_canonical_equals_staged(world):
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    entry = index["by_name"]["images/000000.png"]
    assert entry["original_image_sha256"] == ORIGINAL_SHA
    assert entry["canonical_actual_sha256"] == entry["staged_sha256"]
    assert entry["original_image_sha256"] != entry["canonical_actual_sha256"]
    assert entry["original_hash_is_a_reference"] is True


def test_canonical_and_staged_must_be_byte_identical(world):
    """Same pixels, different PNG encoding: rejected, not tolerated."""
    Image.fromarray(_rgb(0), mode="RGB").save(world["sfm"] / "images/000000.png", format="PNG", compress_level=1)
    changed = world["sfm"] / "images/000000.png"
    rows = [json.loads(line) for line in (world["sfm"] / "frames_map.jsonl").read_text().splitlines()]
    rows[0]["canonical_image_sha256"] = _sha(changed)
    (world["sfm"] / "frames_map.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    bindings = json.loads((world["bridge"] / "INPUT_BINDINGS.json").read_text())
    bindings[0].update(bytes=changed.stat().st_size, sha256=_sha(changed))
    (world["bridge"] / "INPUT_BINDINGS.json").write_text(json.dumps(bindings))
    with pytest.raises(mio.InputError, match="not byte-identical to the staged copy"):
        cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])


def test_frames_map_rows_outside_the_selected_model_are_ignored(world):
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    assert index["frames_map_rows"] == 4
    assert index["selected_rows"] == 2
    assert index["unselected_staged_names"] == ["images/000002.png", "images/000003.png"]
    assert index["unregistered_input_indices"] == [2, 3]
    assert sorted(index["by_name"]) == ["images/000000.png", "images/000001.png"]


# --------------------------------------------------------------------------- #
# 2. scene binding
# --------------------------------------------------------------------------- #
def test_placeholder_scene_is_rejected(tmp_path, world):
    (world["bridge"] / "scene.mvs").write_bytes(b"synthetic scene placeholder\n")
    assert cli.main(cli_args(world, world["manifest"], tmp_path / "never")) == 2
    assert not (tmp_path / "never").exists()


def test_scene_image_count_must_match_the_selected_model(world):
    write_scene(world["bridge"] / "scene.mvs", ["000000.png"], DST_W, DST_H)
    with pytest.raises(mio.InputError, match="scene image count"):
        cli.scene_binding(cli.bridge_artifacts(world["bridge"]),
                          cli.model_bindings(cli.bridge_artifacts(world["bridge"]))["records"])


def test_scene_images_must_correspond_to_the_selected_model(world):
    write_scene(world["bridge"] / "scene.mvs", ["000000.png", "999999.png"], DST_W, DST_H)
    with pytest.raises(mio.InputError, match="do not correspond to the selected model"):
        cli.scene_binding(cli.bridge_artifacts(world["bridge"]),
                          cli.model_bindings(cli.bridge_artifacts(world["bridge"]))["records"])


def test_scene_camera_size_must_match_the_target_camera(world):
    write_scene(world["bridge"] / "scene.mvs", ["000000.png", "000001.png"], DST_W + 2, DST_H)
    with pytest.raises(mio.InputError, match="the target camera is"):
        cli.scene_binding(cli.bridge_artifacts(world["bridge"]),
                          cli.model_bindings(cli.bridge_artifacts(world["bridge"]))["records"])


@pytest.mark.parametrize("change,match", [
    ({"global_ids": [2, 1]}, "global/native image identity"),
    ({"centers": [[-9.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]}, "scene C mismatch"),
    ({"focal": (51.0, 50.0)}, "scene K mismatch"),
    ({"focal": (50.0, 51.0)}, "scene K mismatch"),
    ({"principal_offset": (0.01, 0.0)}, "scene K mismatch"),
    ({"principal_offset": (0.0, 0.01)}, "scene K mismatch"),
    ({"rotate_first": True}, "scene R mismatch"),
    ({"prefix": "other/images"}, "do not correspond"),
])
def test_scene_identity_and_geometry_mismatch_are_rejected(world, change, match):
    write_scene(world["bridge"] / "scene.mvs", ["000000.png", "000001.png"], DST_W, DST_H, **change)
    bridge = cli.bridge_artifacts(world["bridge"])
    with pytest.raises(mio.InputError, match=match):
        cli.scene_binding(bridge, cli.model_bindings(bridge)["records"])


def test_nonselected_frame_still_requires_its_upstream_identity(world):
    path = world_manifest(world["root"], name="bad_ignored.json",
                          mutate=lambda d: d["frames"][2].__setitem__("source_image_sha256", "f" * 64))
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    with pytest.raises(mio.InputError, match="nonselected canonical hash conflict"):
        cli.bind_frames(world["names"], index, [parsed_map(path)])


def test_foreground_declaration_must_equal_actual_pixels(world):
    path = world_manifest(world["root"], name="bad_pixels.json",
                          mutate=lambda d: d["frames"][0]["candidates"][0].__setitem__("foreground_pixels", 1))
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    with pytest.raises(mio.InputError, match="does not match actual mask pixels"):
        cli.bind_frames(world["names"], index, [parsed_map(path)])


def test_canonical_pixels_are_decoded_before_consumption(world):
    path = world["t01"] / "images/000000.png"
    Image.fromarray(np.zeros((SRC_H, SRC_W), dtype=np.uint8), mode="L").save(path)
    (world["sfm"] / "images/000000.png").write_bytes(path.read_bytes())
    digest = _sha(path)
    rows = [json.loads(line) for line in (world["sfm"] / "frames_map.jsonl").read_text().splitlines()]
    rows[0]["canonical_image_sha256"] = digest
    (world["sfm"] / "frames_map.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    bindings = json.loads((world["bridge"] / "INPUT_BINDINGS.json").read_text())
    bindings[0].update(bytes=path.stat().st_size, sha256=digest)
    (world["bridge"] / "INPUT_BINDINGS.json").write_text(json.dumps(bindings))
    with pytest.raises(mio.InputError, match="RGB8 PNG"):
        cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])


def test_an_unverified_edge_blocks_publication(tmp_path, world, monkeypatch):
    real = cli.scene_binding

    def unverified(bridge, records):
        result = real(bridge, records)
        result["edges"][0]["agrees"] = False
        return result

    monkeypatch.setattr(cli, "scene_binding", unverified)
    out = tmp_path / "out_edge"
    assert cli.main(cli_args(world, world["manifest"], out)) == 2
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["publishable_arguments"] is False
    assert manifest["mvs_mask_arguments"]["available"] is False
    assert manifest["unverified_edges"]


# --------------------------------------------------------------------------- #
# 3. input drift and late failures
# --------------------------------------------------------------------------- #
def test_input_drift_after_write_refuses_to_publish(tmp_path, world, monkeypatch):
    out = tmp_path / "out_drift"
    real_warp = cli.warp_binary_mask
    state = {"done": False}

    def drifting_warp(*args, **kwargs):
        result = real_warp(*args, **kwargs)
        if not state["done"]:
            state["done"] = True
            (world["t01"] / "frames.jsonl").write_text("", encoding="utf-8", newline="\n")
        return result

    monkeypatch.setattr(cli, "warp_binary_mask", drifting_warp)
    assert cli.main(cli_args(world, world["manifest"], out)) == 2
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "INPUT_ERROR"
    assert manifest["coverage_status"] == "COMPLETE"
    assert manifest["publishable_arguments"] is False
    assert manifest["inputs"]["input_bindings_unchanged"] is False


def test_late_write_failure_keeps_all_frame_slots(tmp_path, world, monkeypatch):
    world = build_world(tmp_path / "three_selected", model_images=3, n_ok=3)
    out = tmp_path / "out_late"
    real_write = cli.mio.write_label_png
    calls = {"n": 0}

    def failing_write(path, array):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("synthetic disk failure")
        return real_write(path, array)

    monkeypatch.setattr(cli.mio, "write_label_png", failing_write)
    assert cli.main(cli_args(world, world["manifest"], out)) == 4
    failure = json.loads((out / "FAILURE.json").read_text(encoding="utf-8"))
    assert "synthetic disk failure" in failure["first_cause"]
    assert not (out / "manifest.json").exists()
    frames = failure["frames"]
    assert len(frames) == 3, "every target frame must keep a slot"
    assert [f["image_name"] for f in frames] == ["images/000000.png", "images/000001.png", "images/000002.png"]
    assert [f["status"] for f in frames] == ["OK", "INPUT_ERROR", "NOT_RUN"]
    assert frames[1]["failed_during_execution"] is True
    assert "synthetic disk failure" in frames[1]["reason"]
    assert failure["written_outputs"]


def test_cancellation_is_not_swallowed(tmp_path, world, monkeypatch):
    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "warp_binary_mask", cancelled)
    with pytest.raises(KeyboardInterrupt):
        cli.main(cli_args(world, world["manifest"], tmp_path / "out_cancel"))


def test_corrupted_validity_raster_is_detected(tmp_path, world, monkeypatch):
    out = tmp_path / "out_validity"
    real_write = cli.mio.write_label_png

    def corrupting_write(path, array):
        if path.name.endswith(".validity.png"):
            array = np.zeros_like(np.asarray(array))
        return real_write(path, array)

    monkeypatch.setattr(cli.mio, "write_label_png", corrupting_write)
    assert cli.main(cli_args(world, world["manifest"], out)) == 4
    failure = json.loads((out / "FAILURE.json").read_text(encoding="utf-8"))
    assert "array handed to the writer" in failure["first_cause"]


def test_read_model_failure_is_controlled(tmp_path, world):
    (world["bridge"] / "undistorted" / "sparse" / "images.bin").write_bytes(b"not a model")
    assert cli.main(cli_args(world, world["manifest"], tmp_path / "out_bad_model")) == 2
    assert not (tmp_path / "out_bad_model").exists()


# --------------------------------------------------------------------------- #
# 4. coverage, ownership, damaged masks
# --------------------------------------------------------------------------- #
def test_partial_coverage_strict_and_allow_partial(tmp_path, world):
    partial = world_manifest(world["root"], n_ok=1, directory=world["obj"], name="partial.json",
                             source_status="RESOURCE_LIMIT")
    strict_out, partial_out = tmp_path / "strict", tmp_path / "partial"
    assert cli.main(cli_args(world, partial, strict_out)) == 2
    strict = json.loads((strict_out / "manifest.json").read_text(encoding="utf-8"))
    assert strict["counts"] == {"expected": 2, "ok": 0, "missing": 1, "error": 0, "not_run": 1}
    assert not (strict_out / "masks").exists()
    assert cli.main(cli_args(world, partial, partial_out, "--allow-partial")) == 1
    kept = json.loads((partial_out / "manifest.json").read_text(encoding="utf-8"))
    assert kept["status"] == "PARTIAL" and kept["counts"]["ok"] == 1
    assert kept["publishable_arguments"] is False
    assert kept["object_mask_manifests"][0]["status"] == "RESOURCE_LIMIT"
    ok = [f for f in kept["frames"] if f["status"] == "OK"]
    assert len(ok) == 1 and ok[0]["input_index"] == 0


def test_empty_selected_mask_is_an_input_error(tmp_path, world):
    Image.fromarray(np.zeros((SRC_H, SRC_W), dtype=np.uint8), mode="L").save(
        world["obj"] / "masks/000000_c00.png", format="PNG")
    doc = json.loads(world["manifest"].read_text())
    digest = _sha(world["obj"] / "masks/000000_c00.png")
    doc["frames"][0]["selected_mask"]["sha256"] = digest
    doc["frames"][0]["candidates"][0]["mask_sha256"] = digest
    world["manifest"].write_text(json.dumps(doc))
    out = tmp_path / "out_empty_mask"
    assert cli.main(cli_args(world, world["manifest"], out, "--allow-partial")) == 2
    assert not out.exists(), "a damaged mask must not be laundered into MISSING"


def test_existing_output_directory_is_refused(tmp_path, world):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("mine\n", encoding="utf-8", newline="\n")
    assert cli.main(cli_args(world, world["manifest"], out)) == 2
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_early_input_error_does_not_claim_a_directory(tmp_path, world):
    broken = world_manifest(world["root"], directory=world["obj"], name="broken.json",
                            mutate=lambda d: d["frames"][0].__setitem__("photo_io_image_sha256", "b" * 64))
    out = tmp_path / "never"
    assert cli.main(cli_args(world, broken, out)) == 2
    assert not out.exists()


# --------------------------------------------------------------------------- #
# 5. manifest layer
# --------------------------------------------------------------------------- #
def test_production_selected_mask_fields(world):
    frame = parsed_map(world["manifest"])["frames"][0]
    assert set(frame["selected_mask"]) == {"path", "sha256"}
    assert frame["selected_mask"]["sha256"] == frame["chosen"]["mask_sha256"]
    assert frame["selection"]["method"] == "unique_candidate"


def test_legacy_selected_mask_dialect_is_rejected(world):
    def mutate(doc):
        candidate = doc["frames"][0]["candidates"][0]
        doc["frames"][0]["selected_mask"] = {"mask_path": candidate["mask_path"],
                                             "mask_sha256": candidate["mask_sha256"]}
        return doc

    path = world_manifest(world["root"], directory=world["obj"], name="legacy.json", mutate=mutate)
    with pytest.raises(mio.InputError, match="exactly path and sha256"):
        parsed_map(path)


def test_bad_image_domain_is_rejected(world):
    path = world_manifest(world["root"], directory=world["obj"], name="domain.json",
                          mutate=lambda d: d["frames"][0].__setitem__("image_domain", "colmap_undistorted"))
    with pytest.raises(mio.InputError, match="image_domain must be"):
        parsed_map(path)


def test_unique_candidate_method_requires_one_candidate(world):
    def mutate(doc):
        doc["frames"][0]["candidates"].append(dict(doc["frames"][0]["candidates"][0], candidate_id="c01"))
        return doc

    path = world_manifest(world["root"], directory=world["obj"], name="twocand.json", mutate=mutate)
    with pytest.raises(mio.InputError, match="requires exactly one candidate"):
        parsed_map(path)


def test_selected_path_must_match_the_candidate(world):
    path = world_manifest(world["root"], directory=world["obj"], name="wrongpath.json",
                          mutate=lambda d: d["frames"][0]["selected_mask"].__setitem__("path", "masks/other.png"))
    with pytest.raises(mio.InputError, match="does not match the selected candidate mask_path"):
        parsed_map(path)


def test_status_ok_without_selection_is_rejected(world):
    path = world_manifest(world["root"], directory=world["obj"], name="nosel.json",
                          mutate=lambda d: d["frames"][0].__setitem__("selected_mask", None))
    with pytest.raises(mio.InputError, match="status OK requires"):
        parsed_map(path)


def test_non_ok_frame_must_not_carry_a_selection(world):
    def mutate(doc):
        doc["frames"][1]["selection"] = {"method": "unique_candidate", "source": "automatic",
                                         "mask_source": "automatic_sam3"}
        return doc

    path = world_manifest(world["root"], n_ok=1, directory=world["obj"], name="badnotrun.json", mutate=mutate)
    with pytest.raises(mio.InputError, match="must not carry a selection"):
        parsed_map(path)


def test_counts_must_agree_with_statuses(world):
    def mutate(doc):
        doc["counts"]["success"] = 0
        doc["counts"]["no_match"] = 2
        return doc

    path = world_manifest(world["root"], directory=world["obj"], name="badcounts.json", mutate=mutate)
    with pytest.raises(mio.InputError, match="disagree with the per-frame statuses"):
        parsed_map(path)


def test_success_status_incompatible_with_non_ok_frames(world):
    path = world_manifest(world["root"], n_ok=1, directory=world["obj"], name="badsuccess.json",
                          source_status="SUCCESS")
    with pytest.raises(mio.InputError, match="declares SUCCESS but"):
        parsed_map(path)


def test_bool_is_not_an_integer(world):
    path = world_manifest(world["root"], directory=world["obj"], name="bool.json",
                          mutate=lambda d: d["counts"].__setitem__("success", True))
    with pytest.raises(mio.InputError, match="plain integer"):
        parsed_map(path)


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8", newline="\n")
    with pytest.raises(mio.InputError, match="duplicate JSON key"):
        mio.load_json_strict(path)


@pytest.mark.parametrize("literal", ["1e999", "-1e999", "NaN", "Infinity"])
def test_non_finite_json_numbers_are_rejected(tmp_path, literal):
    path = tmp_path / "nf.json"
    path.write_text('{"value": %s}\n' % literal, encoding="utf-8", newline="\n")
    with pytest.raises(mio.InputError, match="non-finite"):
        mio.load_json_strict(path)


def test_input_bindings_uses_the_same_strict_json(tmp_path, world):
    (world["bridge"] / "INPUT_BINDINGS.json").write_text('[{"path": "a", "path": "b"}]\n',
                                                        encoding="utf-8", newline="\n")
    with pytest.raises(mio.InputError, match="duplicate JSON key"):
        cli.bridge_artifacts(world["bridge"])
    (world["bridge"] / "INPUT_BINDINGS.json").write_text('[{"bytes": 1e999}]\n', encoding="utf-8", newline="\n")
    with pytest.raises(mio.InputError, match="non-finite"):
        cli.bridge_artifacts(world["bridge"])


def test_duplicate_input_binding_path_is_rejected(world):
    bindings = json.loads((world["bridge"] / "INPUT_BINDINGS.json").read_text(encoding="utf-8"))
    bindings.append(dict(bindings[0]))
    (world["bridge"] / "INPUT_BINDINGS.json").write_text(json.dumps(bindings, indent=2) + "\n",
                                                        encoding="utf-8", newline="\n")
    with pytest.raises(mio.InputError, match="repeats an already declared path"):
        cli.bridge_artifacts(world["bridge"])


def test_mask_path_escape_is_rejected(world):
    def mutate(doc):
        doc["frames"][0]["candidates"][0]["mask_path"] = "../../escape.png"
        doc["frames"][0]["selected_mask"]["path"] = "../../escape.png"
        return doc

    path = world_manifest(world["root"], directory=world["obj"], name="escape.json", mutate=mutate)
    with pytest.raises(mio.InputError, match="escapes its declared root"):
        parsed_map(path)


def test_source_image_must_be_the_t01_image_output(world):
    other = world["t01"] / "images/000001.png"

    def mutate(doc):
        doc["frames"][0]["source_image"] = str(other)
        doc["frames"][0]["source_image_sha256"] = _sha(other)
        return doc

    path = world_manifest(world["root"], directory=world["obj"], name="otherimg.json", mutate=mutate)
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    with pytest.raises(mio.InputError, match="must be the T01 line image_output"):
        cli.bind_frames(sorted(index["by_name"]), index, [parsed_map(path)])


def test_source_image_sha_must_match_the_canonical_bytes(world):
    path = world_manifest(world["root"], directory=world["obj"], name="badsrc.json",
                          mutate=lambda d: d["frames"][0].__setitem__("source_image_sha256", "c" * 64))
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    with pytest.raises(mio.InputError, match="source_image_sha256"):
        cli.bind_frames(sorted(index["by_name"]), index, [parsed_map(path)])


def test_photo_io_hash_must_equal_the_original_reference(world):
    path = world_manifest(world["root"], directory=world["obj"], name="badphoto.json",
                          mutate=lambda d: d["frames"][0].__setitem__("photo_io_image_sha256", "d" * 64))
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    with pytest.raises(mio.InputError, match="photo_io_image_sha256"):
        cli.bind_frames(sorted(index["by_name"]), index, [parsed_map(path)])


def test_mask_bytes_must_match_the_declared_selected_sha(world):
    parsed = parsed_map(world["manifest"])
    Image.fromarray(_mask(9), mode="L").save(world["obj"] / "masks/000000_c00.png", format="PNG")
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    with pytest.raises(mio.InputError, match="declared selected sha256"):
        cli.bind_frames(sorted(index["by_name"]), index, [parsed])


def test_mask_labels_size_and_mode(tmp_path):
    grey = tmp_path / "grey.png"
    Image.fromarray(np.full((SRC_H, SRC_W), 7, dtype=np.uint8), mode="L").save(grey, format="PNG")
    with pytest.raises(mio.InputError, match="labels must be 0/255"):
        mio.read_label_png(grey, SRC_W, SRC_H)
    small = tmp_path / "small.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(small, format="PNG")
    with pytest.raises(mio.InputError, match="does not equal canonical size"):
        mio.read_label_png(small, SRC_W, SRC_H)
    rgb = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)).save(rgb, format="PNG")
    with pytest.raises(mio.InputError, match="single-channel L mode"):
        mio.read_label_png(rgb, SRC_W, SRC_H)


def test_same_image_in_two_manifests_is_rejected(world):
    second = world_manifest(world["root"], t01_dir=world["t01"], directory=world["root"] / "obj2")
    index = cli.frame_index(cli.bridge_artifacts(world["bridge"]), world["sfm"], world["t01"], world["names"])
    with pytest.raises(mio.InputError, match="more than one manifest"):
        cli.bind_frames(sorted(index["by_name"]), index, [parsed_map(world["manifest"]), parsed_map(second)])


def test_pose_mismatch_between_models_is_rejected(world):
    poses = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    poses[0] = [9.0, 9.0, 9.0]
    _write_model(world["bridge"] / "undistorted" / "sparse", "PINHOLE", DST_W, DST_H,
                 [50.0, 50.0, DST_W / 2, DST_H / 2], world["names"], poses)
    with pytest.raises(mio.InputError, match="pose mismatch"):
        cli.model_bindings(cli.bridge_artifacts(world["bridge"]))


def test_openmvs_naming_and_collisions():
    assert mio.mask_output_name("images/000017.png") == "000017.mask.png"
    assert mio.mask_output_name("images/000017.jpeg") == "000017.mask.png"
    with pytest.raises(mio.InputError, match="collision"):
        mio.output_plan(["images/000017.png", "images/000017.PNG"])
    with pytest.raises(mio.InputError, match="collision"):
        mio.output_plan(["a/000017.png", "b/000017.jpeg"])


def test_classification_table_and_raster_readback(tmp_path):
    counts = mio.counts_from_frames([{"status": s} for s in ["OK", "OK", "MISSING"]])
    assert counts == {"expected": 3, "ok": 2, "missing": 1, "error": 0, "not_run": 0}
    assert mio.classify_run(counts, allow_partial=True) == ("PARTIAL", 1)
    assert mio.classify_run(counts, allow_partial=False) == ("INPUT_ERROR", 2)
    complete = mio.counts_from_frames([{"status": "OK"}, {"status": "OK"}])
    assert mio.classify_run(complete, allow_partial=False) == ("SUCCESS", 0)
    assert [mio.exit_code_for(s) for s in ("SUCCESS", "PARTIAL", "INPUT_ERROR", "BACKEND_ERROR")] == [0, 1, 2, 4]
    array = np.zeros((DST_H, DST_W), dtype=np.uint8)
    array[5:10, 5:10] = 255
    record = mio.write_label_png(tmp_path / "r.png", array)
    assert mio.verify_label_png(tmp_path / "r.png", array, DST_W, DST_H)["sha256"] == record["sha256"]
    array[0, 0] = 255
    with pytest.raises(mio.ProductError, match="does not match the array"):
        mio.verify_label_png(tmp_path / "r.png", array, DST_W, DST_H)
