"""Offline gates; native worker is verified separately by the delivery run."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integrations.blender_export import export as exporter


SCRATCH = exporter.ROOT / "outputs/delegation/PROJECT_FINISH/20261011_blender_export_B"


class BlenderExportTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH, prefix="offline_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ply = self.base / "source.ply"
        self.ply.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\nproperty double y\nproperty double z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n", encoding="ascii")
        self.spec = self.base / "spec.json"
        self.spec.write_text(json.dumps({"schema": "ply-viewer-scenes-v1", "scenes": [{
            "ply": str(self.ply), "sha256": exporter.digest(self.ply), "title": "test",
            "data_kind": "synthetic", "declared_unit": None, "note": "offline fixture"}]}), encoding="utf-8")
        self.output = self.base / "run"
        self.exe = self.base / "blender.exe"
        self.exe.write_bytes(b"mock executable never launched")

    def fake(self, change=None, fail=None, reclaim=True):
        def run(argv, **kwargs):
            mode, payload, expected, blend, result_path = argv[-5:]
            self.assertEqual(kwargs["wall_seconds"], 90)
            self.assertEqual(kwargs["output_bytes"], 128*1024**2)
            self.assertIn("--disable-autoexec", argv)
            if mode == "create":
                Path(blend).write_bytes(b"mock blend")
            elif change == "blend":
                Path(blend).write_bytes(b"changed mock blend")
            elif change == "source":
                self.ply.write_bytes(b"changed source")
            scenes = [{"title": "test", "vertices": 3, "faces": 1}]
            if mode == "verify" and change == "scenes":
                scenes[0]["faces"] = 0
            Path(result_path).write_text(json.dumps({"status": "PASS", "payload_sha256": expected, "scenes": scenes}), encoding="utf-8")
            return {"status": "completed", "root_exit_code": 7 if mode == fail else 0,
                    "reclaim_confirmed": reclaim}
        return run

    def test_success_two_independent_stages(self):
        with patch.object(exporter, "run_supervised", side_effect=self.fake()) as mocked:
            result = exporter.export_blend(self.spec, self.output, blender_exe=self.exe)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["blend"]["sha256"], exporter.digest(self.output / "model.blend"))
        self.assertTrue(Path(result["report_path"]).is_file())

    def test_stage_failure_stops_without_retry(self):
        with patch.object(exporter, "run_supervised", side_effect=self.fake(fail="create")) as mocked:
            result = exporter.export_blend(self.spec, self.output, blender_exe=self.exe)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(mocked.call_count, 1)
        self.assertIn("create stage failed", result["error"])

    def test_unconfirmed_reclamation_cannot_succeed(self):
        with patch.object(exporter, "run_supervised", side_effect=self.fake(reclaim=False)):
            self.assertEqual(exporter.export_blend(self.spec, self.output, blender_exe=self.exe)["status"], "PARTIAL")

    def test_changed_readback_or_sources_cannot_succeed(self):
        for change in ("blend", "scenes", "source"):
            with self.subTest(change=change), patch.object(exporter, "run_supervised", side_effect=self.fake(change=change)):
                result = exporter.export_blend(self.spec, self.base / change, blender_exe=self.exe)
                self.assertEqual(result["status"], "PARTIAL")
                self.assertNotIn("AssertionError", result["error"])
                self.assertNotIn("unpack", result["error"])

    def test_hash_mismatch_prevents_launch(self):
        self.ply.write_bytes(b"changed")
        with patch.object(exporter, "run_supervised") as mocked, self.assertRaises(ValueError):
            exporter.export_blend(self.spec, self.output, blender_exe=self.exe)
        mocked.assert_not_called()

    def test_invalid_spec_and_missing_blender(self):
        with self.assertRaises(FileNotFoundError):
            exporter.export_blend(self.spec, self.output, blender_exe=self.base / "absent.exe")
        self.spec.write_text('{"schema":"bad","scenes":[]}', encoding="utf-8")
        with self.assertRaises(ValueError):
            exporter.export_blend(self.spec, self.output, blender_exe=self.exe)

    def test_refuses_overwrite_and_outside_outputs(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            exporter.export_blend(self.spec, self.output, blender_exe=self.exe)
        with self.assertRaises(ValueError):
            exporter.export_blend(self.spec, exporter.ROOT / "forbidden", blender_exe=self.exe)

    def test_float32_new_degenerate_triangle_is_rejected(self):
        scene = {"vertices": [[1, 0, 0], [1 + 1e-9, 0, 0], [1, 1, 0]], "faces": [[0, 1, 2]]}
        with self.assertRaisesRegex(ValueError, "introduces 1 zero-area"):
            exporter._float32_geometry(scene)
        scene["faces"] = []
        self.assertEqual(exporter._float32_geometry(scene)["additional_duplicate_points"], 1)

    def test_float32_valid_triangle_reports_no_added_degeneracy(self):
        scene = {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]]}
        self.assertEqual(exporter._float32_geometry(scene),
                         {"new_zero_area_triangles": 0, "additional_duplicate_points": 0})


if __name__ == "__main__":
    unittest.main()
