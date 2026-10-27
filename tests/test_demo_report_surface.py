"""Synthetic fixtures test report gating only, never reconstruction quality."""
import hashlib
import json

import pytest

from integrations.demo_pipeline.report import render_report


def fixture_run(tmp_path, *, status="SUCCESS", reopen=True, digest_ok=True, exists=True):
    mesh = tmp_path / "mesh.ply"
    mesh.write_bytes(b"synthetic report fixture")
    blend = tmp_path / "model.blend"
    blend.write_bytes(b"synthetic report fixture, not a Blender file")
    digest = hashlib.sha256(blend.read_bytes()).hexdigest()
    metadata = {
        "surface": {"status": "SUCCESS", "selected_model_id": "component-1",
                    "selected_input_indices": [0, 2], "missing_input_indices": [1],
                    "artifacts": {"mesh": {"path": str(mesh), "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}}},
        "blender": {"status": status, "blend": {"path": str(blend), "sha256": digest if digest_ok else "0" * 64},
                    "reopen_verified": reopen},
    }
    if not exists:
        blend.unlink()
    (tmp_path / "REPORT_DATA.json").write_text(json.dumps(metadata), encoding="utf-8")
    output = render_report(tmp_path)
    return output.read_text(encoding="utf-8"), json.loads((output.parent / "REPORT_VALIDATION.json").read_text(encoding="utf-8"))


def test_explicit_pass_and_hash_bound_files_are_linked(tmp_path):
    page, validation = fixture_run(tmp_path)
    assert "下载 .blend" in page
    assert validation["blender"]["available"] is True
    assert validation["surface"]["selected_input_indices"] == [0, 2]
    assert validation["surface"]["missing_input_indices"] == [1]
    assert "开放表面允许孔洞；真实尺度未知" in page
    assert validation["verified_committed_masks"] == 0
    assert validation["pointcloud"]["viewer_available"] is False
    assert validation["broken_links"] == []


@pytest.mark.parametrize("kwargs", [{"status": "FAILED"}, {"status": "OK"}, {"status": "PASS"}, {"status": "PARTIAL"}, {"reopen": False}, {"digest_ok": False}, {"exists": False}])
def test_failed_or_unverified_blend_never_has_download(tmp_path, kwargs):
    page, validation = fixture_run(tmp_path, **kwargs)
    assert "下载 .blend" not in page
    assert validation["blender"]["available"] is False


def test_surface_hash_mismatch_never_links(tmp_path):
    fixture_run(tmp_path)
    (tmp_path / "mesh.ply").write_bytes(b"modified")
    output = render_report(tmp_path)
    page = output.read_text(encoding="utf-8")
    assert "下载 mesh" not in page
    validation = json.loads((output.parent / "REPORT_VALIDATION.json").read_text(encoding="utf-8"))
    assert validation["surface"]["available"] is False
    assert validation["surface"]["artifacts"][0]["verified"] is False


def test_surface_metadata_cannot_bypass_blender_reopen_gate(tmp_path):
    fixture_run(tmp_path, reopen=False)
    metadata_path = tmp_path / "REPORT_DATA.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["surface"]["artifacts"] = [metadata["blender"]["blend"]]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    page = render_report(tmp_path).read_text(encoding="utf-8")
    assert 'href="../model.blend"' not in page
