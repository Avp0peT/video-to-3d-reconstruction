import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from integrations.video_masks.planning import make_plan


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class VideoMaskPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.io = self.root / "input/video/photo_io/run"
        self.io.mkdir(parents=True)
        self.manifest_path = self.root / "segmentation/recovered/manifest.json"
        self.video_path = self.root / "input/video/RESULT.json"
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"TEST_DATA_NOT_A_REAL_VIDEO")
        source_hash = digest(self.source)
        write_json(self.root / "PLAN.json", {"source": {"kind": "video", "path": str(self.source),
                   "files": [{"path": str(self.source), "sha256": source_hash}]}})
        write_json(self.io / "run.json", {"run_id": "test", "status": "SUCCESS", "input_count": 3})
        self.rows, io_rows, samples, timeline = [], [], [], []
        for index, decode in enumerate((10, 20, 30)):
            image = self.io / f"{index}.png"
            image.write_bytes(f"TEST_DATA_IMAGE_{index}".encode())
            time = {"decode_index": decode, "pts": decode * 100, "time_base": [1, 1000],
                    "time_seconds": decode / 10}
            timeline.append(time)
            io_rows.append({"input_index": index, "frame_id": f"f{index}", "image_sha256": f"pixel{index}",
                            "status": "OK", "canonical_width": 4, "canonical_height": 5,
                            "image_output": image.name})
            row = {"input_index": index, "frame_id": f"f{index}", "photo_io_image_sha256": f"pixel{index}",
                   "source_image": str(image), "source_image_sha256": digest(image),
                   "width": 4, "height": 5, "status": "NO_MATCH" if index == 1 else "OK"}
            if index != 1:
                mask = self.root / f"segmentation/recovered/mask{index}.png"
                mask.parent.mkdir(parents=True, exist_ok=True)
                mask.write_bytes(b"TEST_DATA_MASK")
                row["selected_mask"] = {"path": mask.name, "sha256": digest(mask)}
            self.rows.append(row)
            samples.append(dict(time, photo_input_index=index, photo_frame_id=f"f{index}",
                                sha256=f"pixel{index}", status="PHOTO_IO_OK", width=4, height=5,
                                export_pts_verified=True))
        (self.io / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in io_rows), encoding="utf-8")
        io_hash, frames_hash = digest(self.io / "run.json"), digest(self.io / "frames.jsonl")
        self.manifest = {"source_io": {"path": str(self.io), "run_id": "test",
                         "run_json_sha256": io_hash, "frames_jsonl_sha256": frames_hash}, "frames": self.rows}
        self.video = {"status": "SUCCESS", "source_video": str(self.source), "source_sha256": source_hash,
                      "source_sha256_after": source_hash, "frames": samples,
                      "photo_io": {"run": str(self.io / "run.json"), "sha256": io_hash,
                                   "frames_sha256": frames_hash}}
        write_json(self.root / "segmentation/PRODUCT_OUTCOME.json", {"manifest": str(self.manifest_path)})
        write_json(self.root / "input/video/timeline.json", {"frames": timeline})
        self.save()

    def save(self):
        write_json(self.manifest_path, self.manifest)
        write_json(self.video_path, self.video)

    def test_sample_index_is_not_decode_index(self):
        result = make_plan(self.root)
        self.assertEqual(result["status"], "PLANNED")
        self.assertEqual((result["start_decode_index"], result["end_decode_index"], result["frame_count"]),
                         (10, 30, 21))
        self.assertEqual([row["local_frame_index"] for row in result["targets"]], [0, 10, 20])
        self.assertEqual([row["input_index"] for row in result["seeds"]], [0, 2])
        self.assertEqual(result["seed_quality"], "NOT_EVALUATED")
        self.assertTrue(all(Path(row["mask_path"]).is_absolute() for row in result["seeds"]))

    def test_limits_and_no_bracket(self):
        self.assertEqual(make_plan(self.root, 20)["status"], "NOT_APPLICABLE")
        self.assertEqual(make_plan(self.root, 21)["status"], "PLANNED")
        for value in (True, False, 0, -1, 2, 3.5, "48", 97):
            with self.assertRaises(ValueError):
                make_plan(self.root, value)
        self.rows[1]["status"] = "NEEDS_SELECTION"
        self.save()
        self.assertEqual(make_plan(self.root)["status"], "NOT_APPLICABLE")

    def test_primary_run_directory_fallback(self):
        write_json(self.root / "segmentation/PRODUCT_OUTCOME.json",
                   {"run_dir": str(self.manifest_path.parent)})
        self.assertEqual(make_plan(self.root)["status"], "PLANNED")

    def test_wrong_frame_id(self):
        self.rows[1]["frame_id"] = "other"
        self.save()
        with self.assertRaisesRegex(ValueError, "frame_id mismatch"):
            make_plan(self.root)

    def test_source_image_and_mask_hash(self):
        for key in ("source_image_sha256", "mask"):
            with self.subTest(key=key):
                original = self.rows[0]["source_image_sha256"]
                if key == "mask":
                    self.rows[0]["selected_mask"]["sha256"] = "bad"
                else:
                    self.rows[0][key] = "bad"
                self.save()
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    make_plan(self.root)
                self.rows[0]["source_image_sha256"] = original

    def test_source_video_hash_and_mapping(self):
        self.video["source_sha256_after"] = "bad"
        self.save()
        with self.assertRaisesRegex(ValueError, "source video hash mismatch"):
            make_plan(self.root)
        self.video["source_sha256_after"] = digest(self.source)
        self.video["frames"][1]["decode_index"] = 21
        self.save()
        with self.assertRaisesRegex(ValueError, "decode index outside timeline"):
            make_plan(self.root)

    def test_io_binding_and_duplicate_index(self):
        self.manifest["source_io"]["run_id"] = "other"
        self.save()
        with self.assertRaisesRegex(ValueError, "run identity mismatch"):
            make_plan(self.root)
        self.manifest["source_io"]["run_id"] = "test"
        self.rows[1]["input_index"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate input_index"):
            make_plan(self.root)


if __name__ == "__main__":
    unittest.main()
