import hashlib
import json
from pathlib import Path

import pytest

from integrations.demo_pipeline.segmentation_batches import merge_manifests


def batch(root, indices, states=None):
    root.mkdir()
    frames = []
    for idx, status in zip(indices, states or ["OK"] * len(indices)):
        candidates = []
        for ordinal in range(0 if status == "NO_MATCH" else 2 if status == "NEEDS_SELECTION" else 1):
            data = b"PNG exact original bytes" + bytes([idx, ordinal])
            rel = f"masks/{idx}_{ordinal}.png"
            p = root / rel
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(data)
            candidates.append(dict(candidate_id=f"c{ordinal}", mask_path=rel,
                                   mask_sha256=hashlib.sha256(data).hexdigest(), score=0.91))
        frames.append(dict(input_index=idx, frame_id=f"frame{idx}", status=status, candidates=candidates,
                           selected_candidate_id="c0" if status == "OK" else None,
                           selected_mask=dict(path=candidates[0]["mask_path"], sha256=candidates[0]["mask_sha256"]) if status == "OK" else None))
    doc = dict(contract_version="object-mask-v1", run_id=root.name,
               source_io=dict(path="same", run_id="io", run_json_sha256="r", frames_jsonl_sha256="f"),
               object=dict(object_id="object"), backend=dict(checkpoint_sha256="weights", config_sha256="config"),
               backend_bindings=dict(revision="pinned"), frames=frames)
    doc["counts"] = dict(requested=len(frames), success=sum(f["status"] == "OK" for f in frames),
                         no_match=sum(f["status"] == "NO_MATCH" for f in frames),
                         needs_selection=sum(f["status"] == "NEEDS_SELECTION" for f in frames), error=0, not_run=0)
    doc["status"] = "SUCCESS" if doc["counts"]["success"] == len(frames) else "PARTIAL"
    path = root / "manifest.json"
    write(path, doc)
    return path


def write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_lossless_partial_merge(tmp_path):
    a = batch(tmp_path / "a", [0, 1], ["OK", "NEEDS_SELECTION"])
    b = batch(tmp_path / "b", [2], ["NO_MATCH"])
    overlay = a.parent / "overlays" / "000000.png"
    overlay.parent.mkdir()
    overlay.write_bytes(b"overlay exact bytes")
    before = a.read_bytes()
    product = merge_manifests([a, b], tmp_path / "out", range(3))
    assert product["status"] == "PARTIAL" and product["exit_code"] == 1
    merged = json.loads(Path(product["manifest"]).read_text())
    assert [f["status"] for f in merged["frames"]] == ["OK", "NEEDS_SELECTION", "NO_MATCH"]
    assert a.read_bytes() == before
    assert (tmp_path / "out/sources/batch_000/manifest.json").read_bytes() == before
    assert (tmp_path / "out/overlays/000000.png").read_bytes() == overlay.read_bytes()
    for f in merged["frames"]:
        for c in f["candidates"]:
            assert hashlib.sha256((tmp_path / "out" / c["mask_path"]).read_bytes()).hexdigest() == c["mask_sha256"]
        if f["status"] == "OK":
            assert f["selected_mask"]["path"] == f["candidates"][0]["mask_path"]
    assert merged["quality_status"] == "NOT_EVALUATED"
    assert "resource_observation" not in merged
    assert merged["aggregation"]["sam_execution"] is False


@pytest.mark.parametrize("kind", ["duplicate", "missing", "sourcehash", "traversal", "drift", "backend", "error", "not_run", "selection"])
def test_reject_invalid_evidence(tmp_path, kind):
    a = batch(tmp_path / "a", [0])
    b = batch(tmp_path / "b", [0 if kind == "duplicate" else 1])
    doc = json.loads(b.read_text())
    if kind == "sourcehash":
        doc["source_io"]["frames_jsonl_sha256"] = "changed"
    elif kind == "traversal":
        doc["frames"][0]["candidates"][0]["mask_path"] = "../outside.png"
    elif kind == "drift":
        (b.parent / doc["frames"][0]["candidates"][0]["mask_path"]).write_bytes(b"changed")
    elif kind == "backend":
        doc["backend"]["config_sha256"] = "changed"
    elif kind == "error":
        doc["status"] = "ERROR"
    elif kind == "not_run":
        doc["frames"][0]["status"] = "NOT_RUN"
    elif kind == "selection":
        doc["frames"][0]["selected_mask"]["path"] = "wrong"
    write(b, doc)
    with pytest.raises(ValueError):
        merge_manifests([a, b], tmp_path / "out", range(3 if kind == "missing" else 2))
    assert not (tmp_path / "out").exists()


def test_success_and_existing_output(tmp_path):
    a = batch(tmp_path / "a", [0])
    result = merge_manifests([a], tmp_path / "out", [0])
    assert result["status"] == "SUCCESS" and result["exit_code"] == 0
    with pytest.raises(ValueError, match="new directory"):
        merge_manifests([a], tmp_path / "out", [0])


@pytest.mark.parametrize("bad", ['{"status":1,"status":2}', '{"x":NaN}', '{"x":1e999}'])
def test_reject_ambiguous_json(tmp_path, bad):
    path = tmp_path / "manifest.json"
    path.write_text(bad)
    with pytest.raises(ValueError):
        merge_manifests([path], tmp_path / "out", [0])


@pytest.mark.parametrize("kind", ["bool_count", "large_dimension", "non_dict_frame", "non_dict_candidate", "too_many_expected", "too_many_batches"])
def test_bounded_contract(tmp_path, kind):
    path = batch(tmp_path / "a", [0])
    doc = json.loads(path.read_text())
    if kind == "bool_count":
        doc["counts"]["requested"] = True
    elif kind == "large_dimension":
        doc["frames"][0]["width"] = 1025
    elif kind == "non_dict_frame":
        doc["frames"] = [None]
    elif kind == "non_dict_candidate":
        doc["frames"][0]["candidates"] = [None]
    write(path, doc)
    with pytest.raises(ValueError):
        merge_manifests([path] * (3 if kind == "too_many_batches" else 1), tmp_path / "out",
                        range(193) if kind == "too_many_expected" else [0])
