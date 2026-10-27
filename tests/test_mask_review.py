"""Offline review-consumer checks using tiny synthetic files, never inference.

The fixtures deliberately give original and canonical photos different file
hashes.  Morphology expectations are hand counted, not derived from the code
under test.  This file does not load SAM3, a checkpoint, or any native backend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.mask_review.review import create_review, mask_statistics  # noqa: E402


SIZE = (5, 5)
RING = ("00000", "01110", "01010", "01110", "00000")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pixels(rows: tuple[str, ...]) -> bytes:
    return bytes(255 if value == "1" else 0 for row in rows for value in row)


def _snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha(path.read_bytes())
            for path in root.rglob("*") if path.is_file()}


@dataclass
class ReviewInput:
    root: Path
    io: Path
    product: Path
    manifest: Path
    document: dict
    upstream: list[dict]

    def save(self) -> Path:
        _write_json(self.manifest, self.document)
        return self.manifest

    def bind_source(self) -> None:
        self.document["source_io"] = {
            "path": str(self.io.resolve()),
            "run_id": "synthetic-photo-io",
            "run_json_sha256": _sha((self.io / "run.json").read_bytes()),
            "frames_jsonl_sha256": _sha((self.io / "frames.jsonl").read_bytes()),
        }

    def candidate(self, index: int, ordinal: int = 0) -> dict:
        relative = f"masks/{index:06d}_{ordinal:02d}.png"
        path = self.product / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.frombytes("L", SIZE, _pixels(RING)).save(path, format="PNG")
        return {"candidate_id": f"c{ordinal:02d}", "mask_path": relative,
                "mask_sha256": _sha(path.read_bytes()), "foreground_pixels": 8,
                "score": 0.8, "score_semantics": "synthetic_test_score"}

    def frame(self, index: int) -> dict:
        row = next(row for row in self.upstream if row["input_index"] == index)
        image = (self.io / row["image_output"]).resolve()
        candidate = self.candidate(index)
        return {
            "input_index": index, "frame_id": row["frame_id"],
            "source_image": str(image), "source_image_sha256": _sha(image.read_bytes()),
            "photo_io_image_sha256": row["image_sha256"],
            "image_domain": "photo_io_canonical", "width": SIZE[0], "height": SIZE[1],
            "status": "OK", "candidates": [candidate],
            "selected_candidate_id": candidate["candidate_id"],
            "selected_mask": {"path": candidate["mask_path"], "sha256": candidate["mask_sha256"]},
            "selection": {"method": "unique_candidate", "source": "automatic",
                          "mask_source": "automatic_sam3"},
            "errors": [], "not_run_reason": None,
        }

    def recount(self) -> None:
        statuses = [row["status"] for row in self.document["frames"]]
        self.document["counts"] = {
            "requested": len(statuses), "success": statuses.count("OK"),
            "no_match": statuses.count("NO_MATCH"),
            "needs_selection": statuses.count("NEEDS_SELECTION"),
            "error": sum(statuses.count(s) for s in ("INPUT_ERROR", "BACKEND_ERROR", "RESOURCE_LIMIT")),
            "not_run": statuses.count("NOT_RUN"),
        }
        self.document["status"] = (
            "BACKEND_ERROR" if any(s in ("BACKEND_ERROR", "NOT_RUN") for s in statuses)
            else "PARTIAL" if any(s != "OK" for s in statuses) else "SUCCESS"
        )


def _make_input(root: Path) -> ReviewInput:
    io, product = root / "photo_io", root / "mask_run"
    (io / "images").mkdir(parents=True)
    (root / "raw").mkdir()
    product.mkdir()
    rows = []
    for index in (17, 34):
        image = Image.new("RGB", SIZE, (index, 91, 163))
        original = root / "raw" / f"{index}.png"
        image.save(original, format="PNG", compress_level=0)
        canonical = io / "images" / f"{index:06d}.png"
        image.save(canonical, format="PNG", compress_level=6)
        original_hash = _sha(original.read_bytes())
        image_path = f"raw/{index}.png"
        identity = json.dumps(["synthetic-review", image_path, original_hash],
                              ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        rows.append({
            "input_index": index, "input_image": image_path, "input_mask": None,
            "frame_id": "f_" + _sha(identity), "image_path": image_path, "mask_path": None,
            "image_sha256": original_hash, "mask_sha256": None, "duplicate_of_input_index": None,
            "status": "OK", "raw_width": 5, "raw_height": 5,
            "canonical_width": 5, "canonical_height": 5,
            "exif_orientation_original": None, "orientation_applied": 1,
            "raw_to_canonical": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "image_dtype": "uint8", "image_mode": "RGB", "mask_encoding": None,
            "mask_foreground_pixels": None, "image_output": f"images/{index:06d}.png",
            "mask_output": None, "cache_fingerprint": _sha(identity), "cache_hit": False,
            "camera": None, "distortion": None, "pose": None, "geometry_scale": None,
            "warnings": [], "errors": [],
        })
    _write_json(io / "run.json", {
        "contract_version": "photo-io-v1", "run_id": "synthetic-photo-io",
        "dataset_id": "synthetic-review", "data_kind": "synthetic", "status": "SUCCESS",
        "input_count": 2, "ok_count": 2, "error_count": 0, "warning_count": 0,
        "manifest_sha256": _sha(b"synthetic manifest"), "implementation_sha256": _sha(b"fixture"),
        "runtime": {"python": "fixture", "pillow": "fixture", "numpy": "fixture"}, "errors": [],
    })
    (io / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)),
                                     encoding="utf-8")
    result = ReviewInput(root, io, product, product / "manifest.json", {}, rows)
    result.document = {
        "contract_version": "object-mask-v1", "run_id": "synthetic-mask-run",
        "started_utc": "2026-10-07T00:00:00+00:00", "finished_utc": "2026-10-07T00:00:01+00:00",
        "status": "SUCCESS",
        "backend": {"name": "sam3", "code_commit": "synthetic-only",
                    "checkpoint_path": "fixture-no-checkpoint", "checkpoint_sha256": _sha(b"no model"),
                    "config_sha256": _sha(b"fixture config"), "python_version": "fixture",
                    "torch_version": "fixture", "cuda_runtime": None, "device": "cpu"},
        "object": {"object_id": "synthetic-ring", "text_prompt": "ring", "visual_prompt": None,
                   "selection_policy": "unique_candidate_or_explicit"},
        "frames": [result.frame(17)], "quality_status": "NOT_EVALUATED",
        "cross_view_status": "NOT_EVALUATED", "evidence_level": "synthetic_fixture_only",
    }
    result.bind_source()
    result.recount()
    result.save()
    return result


@pytest.fixture
def sample(tmp_path: Path) -> ReviewInput:
    return _make_input(tmp_path / "inputs")


def _call(sample: ReviewInput, tmp_path: Path, *, manifests: list[Path] | None = None):
    output = tmp_path / "review"
    report, code = create_review(manifests or [sample.save()], output)
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert report["quality_status"] == "NOT_EVALUATED"
    assert report["cross_view_status"] == "NOT_EVALUATED"
    return report, code, output


def _assert_invalid(sample: ReviewInput, tmp_path: Path, *, manifests=None) -> None:
    report, code, _ = _call(sample, tmp_path, manifests=manifests)
    assert (report["status"], code) == ("INPUT_ERROR", 2)
    assert report["errors"]


def test_bound_canonical_hash_explicit_index_and_readonly_artifacts(sample, tmp_path):
    frame = sample.document["frames"][0]
    assert frame["source_image_sha256"] != frame["photo_io_image_sha256"]
    before = _snapshot(sample.root)
    report, code, output = _call(sample, tmp_path)
    assert (report["status"], code) == ("REPORT_READY", 0)
    assert report["counts"] == {"requested": 1, "selected": 1, "candidates": 1, "unreviewable": 0}
    assert (output / "report.html").is_file()
    with Image.open(output / "contact_sheet.png") as image:
        image.load()
        assert image.width > 0 and image.height > 0
    panels = list((output / "panels").glob("*.png"))
    assert len(panels) == 1 and int(panels[0].stem.split("_")[0]) == 17
    assert _snapshot(sample.root) == before


def test_output_must_be_new_and_outside_inputs(sample, tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep.txt"
    sentinel.write_bytes(b"keep me")
    with pytest.raises(ValueError):
        create_review([sample.manifest], occupied)
    assert sentinel.read_bytes() == b"keep me"
    nested = sample.product / "review"
    with pytest.raises(ValueError):
        create_review([sample.manifest], nested)
    assert not nested.exists()


@pytest.mark.parametrize("status", ["NO_MATCH", "NEEDS_SELECTION", "BACKEND_ERROR", "NOT_RUN"])
def test_nonselected_rows_remain_visible_without_becoming_success(sample, tmp_path, status):
    row = sample.frame(34)
    row.update(status=status, selected_candidate_id=None, selected_mask=None, selection=None)
    row["candidates"] = [sample.candidate(34, 0), sample.candidate(34, 1)] if status == "NEEDS_SELECTION" else []
    row["errors"] = [{"code": "BACKEND_ERROR", "message": "synthetic failure"}] if status == "BACKEND_ERROR" else []
    row["not_run_reason"] = "synthetic stop before frame" if status == "NOT_RUN" else None
    sample.document["frames"].append(row)
    sample.recount()
    report, code, output = _call(sample, tmp_path)
    assert (report["status"], code) == ("PARTIAL_REVIEW", 1)
    assert report["counts"] == {"requested": 2, "selected": 1,
                                "candidates": 3 if status == "NEEDS_SELECTION" else 1,
                                "unreviewable": 1}
    assert status in (output / "report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("explicit", [True, False])
def test_multi_candidate_selection_is_explicit_or_unselected(sample, tmp_path, explicit):
    frame = sample.document["frames"][0]
    candidate = sample.candidate(17, 1)
    frame["candidates"].append(candidate)
    if explicit:
        frame["selected_candidate_id"] = candidate["candidate_id"]
        frame["selected_mask"] = {"path": candidate["mask_path"], "sha256": candidate["mask_sha256"]}
        frame["selection"] = {"method": "explicit_candidate", "source": "cli_argument",
                              "mask_source": "automatic_sam3"}
    else:
        frame.update(status="NEEDS_SELECTION", selected_candidate_id=None, selected_mask=None, selection=None)
    sample.recount()
    report, code, _ = _call(sample, tmp_path)
    assert (report["status"], code) == (("REPORT_READY", 0) if explicit else ("NO_REVIEWABLE_MASKS", 1))
    assert report["counts"] == {"requested": 1, "selected": int(explicit), "candidates": 2,
                                "unreviewable": int(not explicit)}


@pytest.mark.parametrize("value", [True, 2], ids=["bool-is-not-a-count", "wrong-count"])
def test_manifest_counts_are_strict_integers_and_recomputed(sample, tmp_path, value):
    sample.document["counts"]["success"] = value
    _assert_invalid(sample, tmp_path)


@pytest.mark.parametrize("field", ["hash", "path", "id"])
def test_selected_identity_must_match_a_candidate(sample, tmp_path, field):
    frame = sample.document["frames"][0]
    if field == "hash":
        frame["selected_mask"]["sha256"] = "0" * 64
    elif field == "path":
        other = sample.candidate(17, 1)
        frame["selected_mask"] = {"path": other["mask_path"], "sha256": other["mask_sha256"]}
    else:
        frame["selected_candidate_id"] = "absent-candidate"
    _assert_invalid(sample, tmp_path)


@pytest.mark.parametrize("kind", ["source-image", "upstream-metadata"])
def test_bound_input_drift_is_rejected(sample, tmp_path, kind):
    if kind == "source-image":
        Image.new("RGB", SIZE, (200, 10, 40)).save(sample.io / "images" / "000017.png")
    else:
        path = sample.io / "frames.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    _assert_invalid(sample, tmp_path)


@pytest.mark.parametrize("kind", ["same-index-conflict", "different-object", "different-source"])
def test_multiple_manifests_require_unambiguous_common_identity(sample, tmp_path, kind):
    first = sample.save()
    if kind == "different-source":
        other = _make_input(tmp_path / "other-inputs")
        other.document["frames"] = [other.frame(34)]
        other.recount()
        second = other.save()
    else:
        document = copy.deepcopy(sample.document)
        if kind == "same-index-conflict":
            candidate = sample.candidate(17, 1)
            document["frames"][0].update(
                candidates=[candidate], selected_candidate_id=candidate["candidate_id"],
                selected_mask={"path": candidate["mask_path"], "sha256": candidate["mask_sha256"]})
        else:
            document["object"]["object_id"] = "different-object"
            document["frames"] = [sample.frame(34)]
        second = sample.product / "second_manifest.json"
        _write_json(second, document)
    _assert_invalid(sample, tmp_path, manifests=[first, second])


@pytest.mark.parametrize("kind", ["renamed-jpeg", "rgba", "transparent-l", "apng",
                                   "labels", "hash", "size", "orientation"])
def test_authoritative_mask_validation_is_strict(sample, tmp_path, kind):
    frame = sample.document["frames"][0]
    candidate = frame["candidates"][0]
    path = sample.product / candidate["mask_path"]
    mask = Image.frombytes("L", SIZE, _pixels(RING))
    if kind == "renamed-jpeg":
        mask.save(path, format="JPEG", quality=100)
    elif kind == "rgba":
        mask.convert("RGBA").save(path, format="PNG")
    elif kind == "transparent-l":
        mask.save(path, format="PNG", transparency=0)
    elif kind == "apng":
        alternate = mask.copy()
        alternate.putpixel((0, 0), 255)
        mask.save(path, format="PNG", save_all=True, append_images=[alternate], duration=50, loop=0)
    elif kind == "labels":
        mask.putpixel((0, 0), 128)
        mask.save(path, format="PNG")
    elif kind == "size":
        Image.new("L", (4, 5), 255).save(path, format="PNG")
    elif kind == "orientation":
        exif = Image.Exif()
        exif[274] = 6
        mask.save(path, format="PNG", exif=exif)
    digest = "0" * 64 if kind == "hash" else _sha(path.read_bytes())
    candidate["mask_sha256"] = digest
    frame["selected_mask"]["sha256"] = digest
    _assert_invalid(sample, tmp_path)


@pytest.mark.parametrize("kind", ["mask", "source"])
def test_resolved_paths_cannot_escape_their_input_root(sample, tmp_path, kind):
    frame = sample.document["frames"][0]
    if kind == "mask":
        candidate = frame["candidates"][0]
        external = sample.root / "external.png"
        external.write_bytes((sample.product / candidate["mask_path"]).read_bytes())
        candidate["mask_path"] = "../external.png"
        frame["selected_mask"]["path"] = candidate["mask_path"]
    else:
        external = sample.root / "external-source.png"
        external.write_bytes(Path(frame["source_image"]).read_bytes())
        row = next(row for row in sample.upstream if row["input_index"] == 17)
        row["image_output"] = "../external-source.png"
        (sample.io / "frames.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in sample.upstream), encoding="utf-8")
        sample.bind_source()
        frame["source_image"] = str(external.resolve())
    _assert_invalid(sample, tmp_path)


def test_untrusted_text_is_escaped_and_candidate_id_never_names_outputs(sample, tmp_path):
    text = '<script>alert("synthetic")</script>'
    sample.document["object"]["text_prompt"] = text
    frame = sample.document["frames"][0]
    identifier = '../../candidate<script>&"'
    frame["candidates"][0]["candidate_id"] = identifier
    frame["selected_candidate_id"] = identifier
    report, code, output = _call(sample, tmp_path)
    assert (report["status"], code) == ("REPORT_READY", 0)
    html = (output / "report.html").read_text(encoding="utf-8")
    assert text not in html and "&lt;script&gt;" in html
    names = [path.name for path in (output / "panels").iterdir()]
    assert names and all(name.endswith(".png") and all(
        part.isdigit() for part in Path(name).stem.split("_")) for name in names)


def test_cli_main_directly_without_a_child_process(sample, tmp_path):
    from scripts.review_object_masks import main

    output = tmp_path / "cli-review"
    assert main(["--mask-manifest", str(sample.manifest), "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "REPORT_READY"
    assert report["quality_status"] == report["cross_view_status"] == "NOT_EVALUATED"


@pytest.mark.parametrize("rows,foreground,bbox,components,holes,largest,border", [
    (RING, 8, [1, 1, 4, 4], 1, 1, 8, False),
    (("10000", "00000", "00110", "00100", "00001"), 5, [0, 0, 5, 5], 3, 0, 3, True),
    (("010", "101", "010"), 4, [0, 0, 3, 3], 1, 1, 4, True),
    (("000", "000"), 0, None, 0, 0, 0, False),
], ids=["ring", "disconnected", "diagonal-fg8-bg4", "all-background"])
def test_morphology_hand_counted_shapes(rows, foreground, bbox, components, holes, largest, border):
    width, height = len(rows[0]), len(rows)
    result = mask_statistics(_pixels(rows), width, height)
    assert result["foreground_pixels"] == foreground
    assert result["foreground_fraction"] == pytest.approx(foreground / (width * height))
    assert result["bbox_xyxy_exclusive"] == bbox
    assert result["foreground_components_8"] == components
    assert result["background_holes_4"] == holes
    assert result["largest_component_pixels"] == largest
    assert result["touches_border"] is border
    assert isinstance(result["topology_status"], str) and result["topology_status"]


def _unavailable_frame(status: str, index: int = 999) -> dict:
    """A producer can preserve a requested index it could not resolve in T01."""
    return {
        "input_index": index, "frame_id": None, "source_image": None,
        "source_image_sha256": None, "photo_io_image_sha256": None,
        "image_domain": "photo_io_canonical", "width": None, "height": None,
        "status": status, "candidates": [], "selected_candidate_id": None,
        "selected_mask": None, "selection": None,
        "errors": ([{"code": "INPUT_NOT_IN_UPSTREAM", "message": "synthetic missing index"}]
                   if status == "INPUT_ERROR" else []),
        "not_run_reason": "synthetic stop before resolving index" if status == "NOT_RUN" else None,
    }


@pytest.mark.parametrize("status", ["INPUT_ERROR", "NOT_RUN"])
def test_unknown_failed_frame_is_preserved_without_a_fabricated_source_panel(sample, tmp_path, status):
    missing = _unavailable_frame(status)
    sample.document["frames"].append(missing)
    sample.recount()
    sample.document["status"] = "INPUT_ERROR" if status == "INPUT_ERROR" else "BACKEND_ERROR"
    report, code, output = _call(sample, tmp_path)
    assert (report["status"], code) == ("PARTIAL_REVIEW", 1)
    assert report["counts"] == {"requested": 2, "selected": 1, "candidates": 1, "unreviewable": 1}
    row = next(row for row in report["frames"] if row["input_index"] == 999)
    assert row["status"] == status and row["binding_status"] == "UNAVAILABLE"
    assert row["errors"] == missing["errors"]
    assert row["not_run_reason"] == missing["not_run_reason"]
    assert len(row['panels']) == 1
    assert row['panels'][0]['candidate_id'] is None and row['panels'][0]['selected'] is False
    with Image.open(output / row['panels'][0]['path']) as placeholder:
        assert placeholder.getpixel((100, 200)) == (25, 30, 37)  # blank, never a guessed photo
    assert list((output / "panels").glob("17_*.png"))
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "999" in html and status in html


def test_unavailable_binding_does_not_excuse_nonnull_identity_contradictions(sample, tmp_path):
    missing = _unavailable_frame("INPUT_ERROR", 34)
    missing["frame_id"] = "f_" + "0" * 64
    sample.document["frames"].append(missing)
    sample.recount()
    sample.document["status"] = "INPUT_ERROR"
    _assert_invalid(sample, tmp_path)


def test_output_image_failure_is_internal_and_still_has_a_json_report(sample, tmp_path, monkeypatch):
    output = (tmp_path / "review").resolve()
    original_save = Image.Image.save

    def fail_output_save(image, target, *args, **kwargs):
        if isinstance(target, (str, Path)) and Path(target).resolve().is_relative_to(output):
            raise OSError("synthetic output image failure")
        return original_save(image, target, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_output_save)
    report, code, _ = _call(sample, tmp_path)
    assert (report["status"], code) == ("INTERNAL_ERROR", 4)
    assert "synthetic output image failure" in json.dumps(report["errors"])


def test_unwritable_final_json_returns_internal_without_claiming_a_saved_report(sample, tmp_path, monkeypatch):
    output = (tmp_path / "review").resolve()
    original_write = Path.write_text

    def fail_report_write(path, *args, **kwargs):
        if path.resolve() == output / "report.json":
            raise OSError("synthetic report JSON write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_report_write)
    report, code = create_review([sample.save()], output)
    assert (report["status"], code) == ("INTERNAL_ERROR", 4)
    assert "synthetic report JSON write failure" in json.dumps(report["errors"])
    assert not (output / "report.json").exists()


def test_final_input_drift_is_visible_in_json_and_html(sample, tmp_path, monkeypatch):
    from integrations.mask_review import review

    def changed_inputs(_bindings):
        raise ValueError("synthetic drift")

    monkeypatch.setattr(review.Bindings, "verify", changed_inputs)
    report, code, output = _call(sample, tmp_path)
    assert (report["status"], code) == ("INPUT_ERROR", 2)
    assert "synthetic drift" in json.dumps(report["errors"])
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "INPUT_ERROR" in html and "synthetic drift" in html


@pytest.mark.parametrize("limit,value", [("MAX_PANELS", 1), ("MAX_TOTAL_SOURCE_PIXELS", 49)])
def test_input_budgets_are_cumulative_across_frames(sample, tmp_path, monkeypatch, limit, value):
    from integrations.mask_review import review

    sample.document["frames"].append(sample.frame(34))
    sample.recount()
    monkeypatch.setattr(review, limit, value)
    _assert_invalid(sample, tmp_path)


def test_render_output_budget_failure_is_internal(sample, tmp_path, monkeypatch):
    from integrations.mask_review import review

    monkeypatch.setattr(review, "MAX_OUTPUT_BYTES", 1)
    output = tmp_path / "review"
    report, code = create_review([sample.save()], output)
    assert (report["status"], code) == ("INTERNAL_ERROR", 4)
    assert report["errors"]
    # A metadata reserve may still permit the failure report; no saved report
    # is required when even the error artifact cannot be written.
    if (output / "report.json").exists():
        saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert saved["status"] == "INTERNAL_ERROR"


def test_prompt_and_selection_provenance_are_preserved_and_independently_displayed(sample, tmp_path):
    prompt = '<em data-case="prompt-only">target&part</em>'
    sample.document["object"]["text_prompt"] = prompt
    frame = sample.document["frames"][0]
    frame["selection"] = {"method": "explicit_candidate", "source": "synthetic-reviewer-selection",
                          "mask_source": "automatic_sam3"}
    assert frame["candidates"][0]["candidate_id"] == "c00"
    report, code, output = _call(sample, tmp_path)
    assert (report["status"], code) == ("REPORT_READY", 0)
    assert report["object"]["text_prompt"] == prompt
    assert "visual_prompt" in report["object"] and report["object"]["visual_prompt"] is None
    row = next(row for row in report["frames"] if row["input_index"] == 17)
    assert row["selection"] == frame["selection"]
    html = (output / "report.html").read_text(encoding="utf-8")
    assert prompt not in html and "&lt;em" in html and "target&amp;part" in html
    assert all(value in html for value in frame["selection"].values())
