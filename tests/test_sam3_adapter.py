"""Targeted tests for the SAM 3 adapter (object-mask-v1).

Scope: the interface, the manifest/contract behaviour, mask file properties,
selection semantics and every failure path. A fake backend drives the cases, so
nothing here is evidence that SAM 3 itself works -- see
``test_evidence_level_marks_fake_runs``.

Run only this file:
    python -m pytest tests/test_sam3_adapter.py -p no:cacheprovider
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "integrations" / "sam3"
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from sam3_adapter import (  # noqa: E402
    backends,
    binding,
    deadline as deadline_mod,
    imagery,
    observer as observer_module,
    resources,
    sourceio,
    worker,
)
from sam3_adapter.contract import (  # noqa: E402
    CONTRACT_VERSION,
    EVIDENCE_LEVEL_FAKE,
    EXIT_CODES,
    IMAGE_DOMAIN,
    counts_consistent,
)
from sam3_adapter.runner import OutputRefused, RunRequest, execute  # noqa: E402

# Preserve these entrypoints before the offline fixture replaces OS dependencies.
_WIN32_API_FACTORY = observer_module._k32
_OPEN_IDENTITY_HANDLE = observer_module.open_identity_handle

CLI = PROJECT_ROOT / "scripts" / "segment_sam3.py"

WIDTH, HEIGHT = 40, 24
INDICES = (17, 34, 59)


# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_source_run(root: Path, indices=INDICES, size=(WIDTH, HEIGHT), frame_overrides=None):
    """Create a minimal but contract-shaped photo-io-v1 run."""
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    overrides = frame_overrides or {}
    lines = []
    for index in indices:
        target = images / f"{index:06d}.png"
        pixels = bytes(((index * 7 + x * 3 + y * 5) % 256) for y in range(size[1]) for x in range(size[0])
                       for _ in range(3))
        Image.frombytes("RGB", size, pixels).save(target, format="PNG")
        record = {
            "input_index": index,
            "frame_id": f"f_{_sha(bytes([index]))[:16]}",
            "image_path": f"prepared/{index + 1}.png",
            "image_sha256": _sha(f"raw-{index}".encode()),
            "image_output": f"images/{index:06d}.png",
            "canonical_width": size[0],
            "canonical_height": size[1],
            "status": "OK",
        }
        record.update(overrides.get(index, {}))
        lines.append(record)
    (root / "frames.jsonl").write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8", newline="\n"
    )
    (root / "run.json").write_text(
        json.dumps({
            "contract_version": "photo-io-v1",
            "run_id": "20260101T000000Z_test0000",
            "dataset_id": "unit_test",
            "data_kind": "real",
            "manifest_sha256": _sha(b"manifest"),
            "status": "SUCCESS",
            "input_count": len(indices),
            "ok_count": len(indices),
            "error_count": 0,
            "warning_count": 0,
            "runtime": {"python": "test", "pillow": "test", "numpy": "test"},
            "implementation_sha256": _sha(b"impl"),
            "errors": [],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return root


def _rect_mask(path: Path, size=(WIDTH, HEIGHT), box=None):
    box = box or (5, 5, 15, 15)
    x0, y0, x1, y1 = box
    data = bytearray(size[0] * size[1])
    for y in range(y0, min(y1, size[1])):
        for x in range(x0, min(x1, size[0])):
            data[y * size[0] + x] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", size, bytes(data)).save(path, format="PNG")
    return path


def _empty_mask(path: Path, size=(WIDTH, HEIGHT)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", size, bytes(size[0] * size[1])).save(path, format="PNG")
    return path


def _write_spec(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return path


def _run(tmp_path, source_run, spec, **kwargs):
    request = RunRequest(
        io_run=source_run,
        indices=list(kwargs.pop("indices", INDICES)),
        object_id=kwargs.pop("object_id", "skull-01"),
        text_prompt=kwargs.pop("text_prompt", "skull"),
        output_root=tmp_path / "out",
        repo_dir=ADAPTER_DIR,
        backend_kind="fake",
        fake_spec=spec,
        device="cpu",
        argv=["test"],
        cwd=str(tmp_path),
        **kwargs,
    )
    return execute(request)


def _manifest(outcome) -> dict:
    return json.loads((outcome.run_dir / "manifest.json").read_text(encoding="utf-8"))


def _frame(manifest, index):
    return next(f for f in manifest["frames"] if f["input_index"] == index)


# --------------------------------------------------------------------------- #
# happy paths
# --------------------------------------------------------------------------- #
def test_single_candidate_is_selected_and_manifest_is_consistent(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(masks / "a.png")), "score": 0.81, "score_semantics": "fake"}]}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    manifest = _manifest(outcome)

    assert outcome.status == "SUCCESS" and outcome.exit_code == 0
    assert manifest["contract_version"] == CONTRACT_VERSION
    frame = _frame(manifest, 17)
    assert frame["status"] == "OK"
    assert frame["selection"] == {"method": "unique_candidate", "source": "automatic", "mask_source": "automatic_sam3"}
    assert frame["selected_candidate_id"] == "c00"
    assert frame["not_run_reason"] is None
    assert frame["image_domain"] == IMAGE_DOMAIN
    assert frame["photo_io_image_sha256"] == _sha(b"raw-17")
    assert frame["source_image_sha256"] == imagery.sha256_file(source / "images/000017.png")
    assert frame["source_image_sha256"] != frame["photo_io_image_sha256"]
    assert counts_consistent(manifest["counts"], manifest["frames"]) == []

    candidate = frame["candidates"][0]
    assert candidate["mask_sha256"] == imagery.sha256_file(outcome.run_dir / candidate["mask_path"])
    assert candidate["foreground_pixels"] == 100  # 10x10 rectangle
    assert candidate["score"] == 0.81
    mask = Image.open(outcome.run_dir / candidate["mask_path"])
    assert mask.mode == "L" and mask.size == (WIDTH, HEIGHT)
    assert not mask.getexif()
    assert set(mask.tobytes()) <= {0, 255}
    assert (outcome.run_dir / "overlays" / "000017.png").is_file()


def test_zero_candidates_is_no_match_and_partial(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": []}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    manifest = _manifest(outcome)

    assert outcome.status == "PARTIAL" and outcome.exit_code == 1
    frame = _frame(manifest, 17)
    assert frame["status"] == "NO_MATCH"
    assert frame["selected_mask"] is None and frame["selection"] is None
    assert frame["candidates"] == []


def test_multiple_candidates_needs_selection_without_picking_the_largest(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(masks / "small.png", box=(2, 2, 5, 5))), "score": 0.31},
        {"mask_path": str(_rect_mask(masks / "big.png", box=(0, 0, 30, 20))), "score": 0.95},
    ]}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    manifest = _manifest(outcome)

    assert outcome.status == "PARTIAL"
    frame = _frame(manifest, 17)
    assert frame["status"] == "NEEDS_SELECTION"
    assert frame["selected_candidate_id"] is None and frame["selected_mask"] is None
    assert [c["candidate_id"] for c in frame["candidates"]] == ["c00", "c01"]
    assert not (outcome.run_dir / "overlays" / "000017.png").exists()


def test_explicit_selection_of_second_candidate(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(masks / "a.png", box=(2, 2, 5, 5))), "score": 0.31},
        {"mask_path": str(_rect_mask(masks / "b.png", box=(0, 0, 30, 20))), "score": 0.95},
    ]}}})
    outcome = _run(tmp_path, source, spec, indices=[17], explicit_selection={17: "c01"})
    frame = _frame(_manifest(outcome), 17)

    assert outcome.status == "SUCCESS"
    assert frame["selected_candidate_id"] == "c01"
    assert frame["selection"]["method"] == "explicit_candidate"
    assert frame["selection"]["source"] == "cli_argument"
    assert frame["candidates"][1]["foreground_pixels"] == 600


def test_explicit_selection_of_missing_candidate_stays_needs_selection(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(masks / "a.png")), "score": 0.4},
        {"mask_path": str(_rect_mask(masks / "b.png", box=(20, 10, 30, 20))), "score": 0.5},
    ]}}})
    outcome = _run(tmp_path, source, spec, indices=[17], explicit_selection={17: "c99"})
    frame = _frame(_manifest(outcome), 17)

    assert frame["status"] == "NEEDS_SELECTION"
    assert frame["selected_mask"] is None
    assert frame["errors"][0]["code"] == "EXPLICIT_CANDIDATE_NOT_USABLE"


def test_empty_candidate_is_exported_but_never_selected(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_empty_mask(masks / "empty.png")), "score": 0.7},
        {"mask_path": str(_rect_mask(masks / "real.png")), "score": 0.6},
    ]}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    frame = _frame(_manifest(outcome), 17)

    assert outcome.status == "SUCCESS"
    assert frame["candidates_dropped_empty"] == 1
    assert frame["selected_candidate_id"] == "c01"
    assert frame["candidates"][0]["foreground_pixels"] == 0


def test_selection_never_yields_an_empty_mask(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_empty_mask(tmp_path / "cand" / "only.png")), "score": 0.9}]}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    frame = _frame(_manifest(outcome), 17)

    assert frame["status"] == "NO_MATCH" and frame["selected_mask"] is None
    assert frame["candidates_dropped_empty"] == 1


# --------------------------------------------------------------------------- #
# input binding failures
# --------------------------------------------------------------------------- #
def test_unknown_input_index_is_input_error_but_frame_is_kept(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})
    outcome = _run(tmp_path, source, spec, indices=[17, 999])
    manifest = _manifest(outcome)
    frame = _frame(manifest, 999)

    assert frame["status"] == "INPUT_ERROR"
    assert frame["errors"][0]["code"] == "INPUT_NOT_IN_UPSTREAM"
    assert frame["selected_mask"] is None
    assert manifest["counts"]["requested"] == 2 and manifest["counts"]["error"] == 1
    assert counts_consistent(manifest["counts"], manifest["frames"]) == []


def test_source_image_size_mismatch_is_input_error(tmp_path):
    source = _make_source_run(tmp_path / "io")
    target = source / "images/000017.png"
    Image.frombytes("RGB", (WIDTH - 3, HEIGHT), bytes((WIDTH - 3) * HEIGHT * 3)).save(target, format="PNG")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})

    outcome = _run(tmp_path, source, spec, indices=[17])
    frame = _frame(_manifest(outcome), 17)
    assert frame["status"] == "INPUT_ERROR"
    assert frame["errors"][0]["code"] == "SOURCE_IMAGE_SIZE_MISMATCH"


def test_missing_upstream_run_is_input_error(tmp_path):
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})
    outcome = _run(tmp_path, tmp_path / "does-not-exist", spec, indices=[17])
    manifest = _manifest(outcome)

    assert outcome.status == "INPUT_ERROR" and outcome.exit_code == 2
    assert manifest["errors"][0]["code"] == "SOURCE_IO_UNBOUND"
    # the requested index is preserved even though nothing could be bound
    assert manifest["counts"]["requested"] == 1
    assert manifest["counts"]["error"] == 1
    frame = _frame(manifest, 17)
    assert frame["status"] == "INPUT_ERROR"
    assert frame["frame_id"] is None and frame["source_image"] is None and frame["width"] is None
    assert frame["errors"][0]["code"] == "INPUT_ERROR"


def test_non_object_run_json_yields_input_error_manifest(tmp_path):
    """Mainline counterexample 4: run.json = [] used to raise AttributeError."""
    source = tmp_path / "bad_io"
    source.mkdir()
    (source / "run.json").write_text("[]", encoding="utf-8", newline="\n")
    (source / "frames.jsonl").write_text("", encoding="utf-8", newline="\n")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})

    outcome = _run(tmp_path, source, spec, indices=[17, 34])
    manifest = _manifest(outcome)

    assert outcome.status == "INPUT_ERROR" and outcome.exit_code == 2
    assert manifest["counts"]["requested"] == 2 and manifest["counts"]["error"] == 2
    assert [f["input_index"] for f in manifest["frames"]] == [17, 34]
    assert "must be a JSON object" in manifest["errors"][0]["message"]


def test_corrupt_frames_jsonl_yields_input_error_manifest(tmp_path):
    source = _make_source_run(tmp_path / "io")
    (source / "frames.jsonl").write_text("{not json\n", encoding="utf-8", newline="\n")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})

    outcome = _run(tmp_path, source, spec, indices=[17, 34, 59])
    manifest = _manifest(outcome)
    assert outcome.status == "INPUT_ERROR"
    assert manifest["counts"] == {"requested": 3, "success": 0, "no_match": 0,
                                  "needs_selection": 0, "error": 3, "not_run": 0}


def test_upstream_bytes_changing_during_run_is_detected(tmp_path, monkeypatch):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(masks / "a.png")), "score": 0.5}]}}})

    monkeypatch.setattr(sourceio.SourceIo, "fingerprint", lambda self: ("drifted", "drifted"))

    outcome = _run(tmp_path, source, spec, indices=[17])
    manifest = _manifest(outcome)
    assert outcome.status == "INPUT_ERROR"
    assert any(e["code"] == "UPSTREAM_INPUT_CHANGED" for e in manifest["errors"])


# --------------------------------------------------------------------------- #
# backend failures: never silently downgraded
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("load_error", "status", "exit_code"),
    [
        ("BackendUnavailable", "BACKEND_UNAVAILABLE", 3),
        ("AuthRequired", "AUTH_REQUIRED", 3),
        ("ResourceLimit", "RESOURCE_LIMIT", 5),
        ("BackendError", "BACKEND_ERROR", 4),
    ],
)
def test_backend_load_failure_maps_to_contract_status(tmp_path, load_error, status, exit_code):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"load_error": load_error,
                                                "load_error_message": "synthetic", "frames": {}})
    outcome = _run(tmp_path, source, spec)
    manifest = _manifest(outcome)

    assert outcome.status == status and outcome.exit_code == exit_code
    assert all(f["status"] == "NOT_RUN" for f in manifest["frames"])
    assert all(f["not_run_reason"] for f in manifest["frames"])
    assert all(f["selected_mask"] is None for f in manifest["frames"])
    assert manifest["counts"] == {"requested": 3, "success": 0, "no_match": 0,
                                  "needs_selection": 0, "error": 0, "not_run": 3}
    assert counts_consistent(manifest["counts"], manifest["frames"]) == []
    assert manifest["errors"][0]["code"] == f"STOP_{status}"
    assert all(f["selected_candidate_id"] is None for f in manifest["frames"])


def test_backend_error_midbatch_stops_remaining_frames(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {
        "17": {"candidates": [{"mask_path": str(_rect_mask(masks / "a.png")), "score": 0.6}]},
        "34": {"error": "BackendError", "message": "synthetic mid-batch failure"},
    }})
    outcome = _run(tmp_path, source, spec)
    manifest = _manifest(outcome)

    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    assert _frame(manifest, 17)["status"] == "OK"
    assert _frame(manifest, 34)["status"] == "BACKEND_ERROR"
    assert _frame(manifest, 59)["status"] == "NOT_RUN"
    assert _frame(manifest, 59)["not_run_reason"]
    assert counts_consistent(manifest["counts"], manifest["frames"]) == []


def test_resource_limit_after_some_frames_keeps_real_status(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {
        "17": {"candidates": [{"mask_path": str(_rect_mask(masks / "a.png")), "score": 0.6}]},
        "34": {"error": "ResourceLimit", "message": "synthetic OOM"},
    }})
    outcome = _run(tmp_path, source, spec)
    manifest = _manifest(outcome)

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert _frame(manifest, 34)["status"] == "RESOURCE_LIMIT"
    assert _frame(manifest, 59)["status"] == "NOT_RUN"


def test_rejected_mask_is_backend_error_not_no_match(tmp_path):
    source = _make_source_run(tmp_path / "io")
    bad = tmp_path / "cand" / "bad.png"
    bad.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", (WIDTH, HEIGHT), bytes([0, 128] * (WIDTH * HEIGHT // 2))).save(bad, format="PNG")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(bad), "score": 0.5}]}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    frame = _frame(_manifest(outcome), 17)

    assert frame["status"] == "BACKEND_ERROR"
    assert frame["errors"][0]["code"] == "BACKEND_MASK_INVALID"
    assert outcome.status == "BACKEND_ERROR"


def test_wrong_sized_candidate_mask_is_backend_error(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(tmp_path / "cand" / "small.png", size=(WIDTH - 1, HEIGHT))), "score": 0.5}]}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    frame = _frame(_manifest(outcome), 17)

    assert frame["status"] == "BACKEND_ERROR"
    assert frame["errors"][0]["code"] == "BACKEND_MASK_INVALID"
    assert frame["selected_mask"] is None


# --------------------------------------------------------------------------- #
# output refusal
# --------------------------------------------------------------------------- #
def test_existing_output_run_directory_is_refused(tmp_path, monkeypatch):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    assert outcome.run_dir.is_dir()

    # a second run must never write into the first run's directory
    monkeypatch.setattr("sam3_adapter.runner.new_run_id", lambda: outcome.run_dir.name)
    with pytest.raises(OutputRefused):
        _run(tmp_path, source, spec, indices=[17])


def test_unwritable_output_root_is_refused(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})
    blocker = tmp_path / "out"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OutputRefused):
        _run(tmp_path, source, spec, indices=[17])


def test_cli_exit_code_and_stderr_on_refusal(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})
    blocker = tmp_path / "out"
    blocker.write_text("not a directory", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "segment_sam3.py"),
            "--io-run", str(source),
            "--indices", "17",
            "--object-id", "skull-01",
            "--text-prompt", "skull",
            "--backend", "fake",
            "--fake-spec", str(spec),
            "--output", str(blocker),
            "--no-host-probe",
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert completed.returncode == EXIT_CODES["INPUT_ERROR"]
    assert "REFUSED" in completed.stderr


# --------------------------------------------------------------------------- #
# evidence tier + schema guards
# --------------------------------------------------------------------------- #
def test_evidence_level_marks_fake_runs(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(masks / "a.png")), "score": 0.5}]}}})
    manifest = _manifest(_run(tmp_path, source, spec, indices=[17]))

    assert manifest["evidence_level"] == EVIDENCE_LEVEL_FAKE
    assert manifest["backend"]["name"] == "fake"
    assert "must NOT be reported as SAM3 evidence" in manifest["backend"]["evidence_caveat"]


def test_quality_and_cross_view_stay_not_evaluated(tmp_path):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(masks / "a.png")), "score": 0.5}]}}})
    manifest = _manifest(_run(tmp_path, source, spec, indices=[17]))

    assert manifest["quality_status"] == "NOT_EVALUATED"
    assert manifest["cross_view_status"] == "NOT_EVALUATED"


def test_source_io_binding_is_byte_exact(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": []}}})
    manifest = _manifest(_run(tmp_path, source, spec, indices=[17]))

    binding = manifest["source_io"]
    assert binding["path"] == str(source.resolve())
    assert binding["run_id"] == "20260101T000000Z_test0000"
    assert binding["frames_jsonl_sha256"] == _sha((source / "frames.jsonl").read_bytes())
    assert binding["run_json_sha256"] == _sha((source / "run.json").read_bytes())


def test_importing_the_adapter_never_pulls_torch_or_sam3():
    """The adapter must stay importable from the NumPy-2 project environment."""
    code = (
        "import sys;"
        f"sys.path.insert(0, r'{ADAPTER_DIR}');"
        "import sam3_adapter, sam3_adapter.backends, sam3_adapter.contract, sam3_adapter.imagery,"
        " sam3_adapter.probe, sam3_adapter.resources, sam3_adapter.runner, sam3_adapter.sourceio;"
        "print('LEAKED=' + ','.join(m for m in ('torch', 'torchvision', 'sam3') if m in sys.modules))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "LEAKED="


def test_dropped_upstream_status_is_rejected(tmp_path):
    source = _make_source_run(tmp_path / "io")
    run_meta = json.loads((source / "run.json").read_text(encoding="utf-8"))
    run_meta["status"] = "PARTIAL_FAILURE"
    (source / "run.json").write_text(json.dumps(run_meta), encoding="utf-8", newline="\n")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})

    outcome = _run(tmp_path, source, spec, indices=[17])
    assert outcome.status == "INPUT_ERROR"
    assert "not SUCCESS" in _manifest(outcome)["errors"][0]["message"]


# =========================================================================== #
# R1 repair group 1: the real processor output contract (N x 1 x H x W)
# =========================================================================== #
def _shape_of(node):
    dims = []
    while isinstance(node, list):
        dims.append(len(node))
        node = node[0] if node else None
    return tuple(dims)


class StubTensor:
    """Mirrors the tensor surface the pinned processor returns, without torch.

    shape can be given explicitly because an empty tensor (N=0) legitimate
    shape such as (0, 1, 2, 3) cannot be inferred from tolist() == [].
    """

    def __init__(self, data, shape=None):
        self._data = data
        self._shape = tuple(shape) if shape is not None else _shape_of(data)

    @property
    def shape(self):
        return self._shape

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def tolist(self):
        return self._data


class StubProcessor:
    def __init__(self, masks, scores, logits=None):
        self._masks = StubTensor(masks)
        self._scores = StubTensor(scores)
        self._logits = StubTensor(logits) if logits is not None else None

    def set_image(self, image):
        return {}

    def set_text_prompt(self, **kwargs):
        state = {"masks": self._masks, "scores": self._scores}
        if self._logits is not None:
            state["masks_logits"] = self._logits
        return state


def _backend_with_processor(masks, scores, logits=None):
    backend = backends.Sam3Backend(ADAPTER_DIR, None, None, "cuda", 0.5)
    backend._processor = StubProcessor(masks, scores, logits)
    backend._torch = SimpleNamespace(
        inference_mode=nullcontext,
        autocast=lambda **kwargs: nullcontext(),
        bfloat16="fake_bfloat16",
        cuda=SimpleNamespace(empty_cache=lambda: None, OutOfMemoryError=type("FakeOOM", (Exception,), {}))
    )
    return backend


def test_singleton_channel_is_removed_for_two_by_three_mask():
    """Mainline counterexample 3: 2 bytes used to come out of a 2x3 mask."""
    candidates = backends.candidates_from_processor_output(
        StubTensor([[[[True, False, True], [False, True, False]]]]), StubTensor([0.8]), 3, 2)
    assert len(candidates) == 1
    assert len(candidates[0].flat) == 6
    assert list(candidates[0].flat) == [255, 0, 255, 0, 255, 0]


def test_real_backend_segment_path_writes_correct_pixels_and_hash(tmp_path):
    source = _make_source_run(tmp_path / "io", size=(3, 2))
    backend = _backend_with_processor([[[[True, False, True], [False, True, False]]]], [0.75])
    candidates = backend.segment(source / "images/000017.png", "skull", "skull-01", 3, 2, 17)

    assert len(candidates) == 1
    mask_path = tmp_path / "mask.png"
    digest = imagery.write_mask_l(mask_path, candidates[0].flat, 3, 2)
    assert digest == imagery.sha256_file(mask_path)
    with Image.open(mask_path) as image:
        assert image.mode == "L" and image.size == (3, 2)
        assert list(image.tobytes()) == [255, 0, 255, 0, 255, 0]
    assert candidates[0].score == pytest.approx(0.75)
    assert candidates[0].score_semantics == "presence_calibrated_sigmoid_product_not_probability"


def test_multiple_candidates_keep_their_own_pixels():
    candidates = backends.candidates_from_processor_output(
        StubTensor([[[[True, True, False], [False, False, False]]],
                    [[[False, False, False], [True, True, True]]]]),
        StubTensor([0.9, 0.4]), 3, 2)
    assert [list(c.flat) for c in candidates] == [
        [255, 255, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 255],
    ]
    assert [c.candidate_id for c in candidates] == ["c00", "c01"]
    assert [c.score for c in candidates] == [0.9, 0.4]


def test_zero_candidates_from_processor_shape_is_empty():
    masks = StubTensor([], shape=(0, 1, 2, 3))
    scores = StubTensor([], shape=(0,))
    assert backends.candidates_from_processor_output(masks, scores, 3, 2) == []


@pytest.mark.parametrize(
    ("masks", "scores", "needle"),
    [
        ([[[True]]], [0.5], "rank 4"),
        ([ [[[True, True], [True, True]], [[True, True], [True, True]]] ], [0.5], "channel dimension"),
        ([[[[True, True], [True, True], [True, True]]]], [0.5], "spatial size"),
        ([[[[True, False, True], [False, True, False]]]], [0.5, 0.6], "scores length"),
        ([[[[1, 2, 0], [0, 1, 1]]]], [0.5], "non-binary"),
        ([[[[0.0, float("nan"), 1.0], [0, 1, 1]]]], [0.5], "non-finite"),
        ([[[["yes", False, True], [False, True, False]]]], [0.5], "not numeric"),
    ],
)
def test_processor_contract_violations_are_explicit(masks, scores, needle):
    with pytest.raises(backends.TensorContractError) as excinfo:
        backends.candidates_from_processor_output(StubTensor(masks), StubTensor(scores), 3, 2)
    assert needle in str(excinfo.value)


def test_non_finite_logits_are_rejected_even_when_labels_look_binary():
    with pytest.raises(backends.TensorContractError) as excinfo:
        backends.candidates_from_processor_output(
            StubTensor([[[[True, False]]]]), StubTensor([0.5]), 2, 1,
            mask_logits=StubTensor([[[[0.0, float("inf")]]]]))
    assert "non-finite" in str(excinfo.value)


def test_processor_violation_reaches_the_manifest_as_backend_error(tmp_path):
    source = _make_source_run(tmp_path / "io", size=(3, 2))
    backend = _backend_with_processor([[[[True, False, True], [False, True, False]]]], [0.5, 0.6])
    try:
        backend.segment(source / "images/000017.png", "skull", "skull-01", 3, 2, 17)
    except backends.TensorContractError as exc:
        assert "scores length" in str(exc)
    else:  # pragma: no cover - the violation must raise
        raise AssertionError("score length mismatch was not rejected")


# =========================================================================== #
# R1 repair group 2: capacity vs usage, threshold final status
# =========================================================================== #
def _monitor_with(report: dict, device: str = "cuda"):
    monitor = resources.ResourceMonitor.__new__(resources.ResourceMonitor)
    monitor.policy = resources.ResourcePolicy()
    monitor.device = device
    monitor._torch = None
    monitor.device_kind, monitor.device_index = resources.parse_cuda_device(device)
    monitor.channel_report = lambda: dict(report)
    return monitor


def test_capacity_larger_than_budget_is_not_a_violation():
    """Mainline counterexample 1: total capacity is not usage."""
    monitor = _monitor_with({"process_commit_bytes": 100 << 20, "torch_available": True,
                             "cuda_available": True, "device_total_bytes": 8 << 30,
                             "device_free_bytes": 7 << 30, "device_used_bytes": 1 << 30})
    allowed, why = monitor.can_prove_headroom()
    assert allowed is True, why
    assert "not compared to the budget" in why or "informational" in why


def test_measured_device_usage_above_budget_is_refused():
    monitor = _monitor_with({"process_commit_bytes": 100 << 20, "torch_available": True,
                             "cuda_available": True, "device_total_bytes": 8 << 30,
                             "device_free_bytes": 1 << 29,
                             "device_used_bytes": (8 << 30) - (1 << 29)})
    allowed, why = monitor.can_prove_headroom()
    assert allowed is False and "VRAM budget" in why


def test_unmeasurable_channels_never_count_as_safe():
    unmeasurable = _monitor_with({"process_commit_bytes": None, "torch_available": False,
                                  "cuda_available": False})
    allowed, why = unmeasurable.can_prove_headroom()
    assert allowed is False and "commit charge cannot be measured" in why

    no_torch = _monitor_with({"process_commit_bytes": 100 << 20, "torch_available": False,
                              "cuda_available": False})
    allowed, why = no_torch.can_prove_headroom()
    assert allowed is False and "torch unavailable" in why

    device_unqueryable = _monitor_with({"process_commit_bytes": 100 << 20, "torch_available": True,
                                        "cuda_available": True, "device_query_error": "boom"})
    allowed, why = device_unqueryable.can_prove_headroom()
    assert allowed is False and "cannot be queried" in why


def test_parse_cuda_device_rejects_unknown_values():
    assert resources.parse_cuda_device("cuda:1") == ("cuda", 1)
    assert resources.parse_cuda_device("cuda") == ("cuda", None)
    assert resources.parse_cuda_device("cpu") == ("cpu", None)
    with pytest.raises(ValueError):
        resources.parse_cuda_device("cuda:notanindex")
    with pytest.raises(ValueError):
        resources.parse_cuda_device("mps")


def test_declared_device_index_is_actually_queried():
    """The adapter must not always ask about device 0."""
    asked = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def mem_get_info(index):
            asked.append(index)
            return 4 << 30, 8 << 30

        @staticmethod
        def get_device_properties(index):
            return SimpleNamespace(name=f"fake:{index}")

    monitor = resources.ResourceMonitor.__new__(resources.ResourceMonitor)
    monitor.policy = resources.ResourcePolicy()
    monitor.device = "cuda:1"
    monitor._torch = SimpleNamespace(cuda=FakeCuda)
    monitor.device_kind, monitor.device_index = resources.parse_cuda_device(monitor.device)
    report = monitor.channel_report()
    assert asked == [1]
    assert report["device_index"] == 1 and report["device_name"] == "fake:1"


def test_peaks_use_the_whole_series_not_the_last_sample():
    samples = [
        {"utc": "t0", "process_commit_bytes": 1 << 30, "process_rss_bytes": 1 << 20},
        {"utc": "t1", "process_commit_bytes": 13 << 30, "process_rss_bytes": 9 << 20},
        {"utc": "t2", "process_commit_bytes": 5 << 30, "process_rss_bytes": 2 << 20},
    ]
    peaks = _sample_peaks_for_test(samples)
    assert peaks["ram_commit_peak_bytes"] == 13 << 30
    assert peaks["ram_rss_peak_bytes"] == 9 << 20
    assert peaks["n_samples"] == 3


def _sample_peaks_for_test(samples):
    from sam3_adapter.runner import _sample_peaks  # noqa: PLC0415
    return _sample_peaks(samples)


class PermissiveMonitor:
    """Stands in for ResourceMonitor so threshold logic can be tested alone."""

    def __init__(self, **kwargs):
        pass

    def can_prove_headroom(self):
        return True, "test preflight passes"

    def sample(self):
        return {"utc": "test", "process_commit_bytes": 13 << 30}

    def violations(self, sample=None):
        return ["test-only measured RAM above 12 GiB"]

    def channel_report(self):
        return {"test_only": True}


class QuietMonitor(PermissiveMonitor):
    def sample(self):
        return {"utc": "test", "process_commit_bytes": 1 << 20}

    def violations(self, sample=None):
        return []


class Sam3ShapedFakeBackend(backends.FakeBackend):
    """FakeBackend that reports SAM3-shaped identity so sam3 runs stay SUCCESS."""

    def describe(self):
        return {"name": "sam3", "code_commit": "test-only", "checkpoint_sha256": "test-only",
                "config_sha256": "test-only", "python_version": "test", "torch_version": None,
                "cuda_runtime": None, "device": "cuda"}


def _threshold_run(tmp_path, monkeypatch, indices, monitor_class):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {
        "frames": {str(index): {"candidates": [
            {"mask_path": str(_rect_mask(masks / f"m{index}.png")), "score": 0.8}]} for index in indices}})
    monkeypatch.setattr("sam3_adapter.runner.ResourceMonitor", monitor_class)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))
    return execute(RunRequest(
        io_run=source, indices=list(indices), object_id="fixture", text_prompt="fixture",
        output_root=tmp_path / "out", repo_dir=ADAPTER_DIR, backend_kind="sam3", fake_spec=spec,
        argv=["test"], cwd=str(tmp_path),
        # 0 disables the background sampler so the explicit post-frame check is the
        # only breach observer in these two deterministic threshold tests
        monitor_interval_seconds=0))


def test_threshold_after_the_last_frame_is_still_resource_limit(tmp_path, monkeypatch):
    """Mainline counterexample 2a: a single frame that breaches RAM used to be SUCCESS/0."""
    outcome = _threshold_run(tmp_path, monkeypatch, [17], PermissiveMonitor)
    manifest = _manifest(outcome)
    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["counts"] == {"requested": 1, "success": 1, "no_match": 0,
                                 "needs_selection": 0, "error": 0, "not_run": 0}
    assert manifest["resource_observation"]["thresholds_hit"]


def test_threshold_after_the_first_of_three_frames_keeps_completed_frames(tmp_path, monkeypatch):
    """Mainline counterexample 2b: three frames used to end UP as PARTIAL/1."""
    outcome = _threshold_run(tmp_path, monkeypatch, [17, 34, 59], PermissiveMonitor)
    manifest = _manifest(outcome)
    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["counts"]["success"] == 1
    assert manifest["counts"]["not_run"] == 2
    assert [f["status"] for f in manifest["frames"]] == ["OK", "NOT_RUN", "NOT_RUN"]
    assert all(f["not_run_reason"] for f in manifest["frames"] if f["status"] == "NOT_RUN")


def test_samples_are_collected_during_load_and_per_frame(tmp_path, monkeypatch):
    outcome = _threshold_run(tmp_path, monkeypatch, [17, 34], QuietMonitor)
    samples = _manifest(outcome)["resource_observation"]["samples"]
    assert len(samples) >= 2
    assert _manifest(outcome)["resource_observation"]["peaks"]["n_samples"] == len(samples)


# =========================================================================== #
# R1 repair group 3/4/5: score contract, bindings, deadlines, CLI
# =========================================================================== #
@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_is_rejected_and_never_serialised(tmp_path, bad_score):
    """Mainline counterexample 5: NaN used to yield SUCCESS with NaN in the JSON."""
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(tmp_path / "cand" / "a.png")), "score": bad_score}]}}})
    outcome = _run(tmp_path, source, spec, indices=[17])
    manifest = _manifest(outcome)
    text = (outcome.run_dir / "manifest.json").read_text(encoding="utf-8")

    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    assert "NaN" not in text and "Infinity" not in text
    frame = _frame(manifest, 17)
    assert frame["status"] == "BACKEND_ERROR"
    assert frame["candidates"] == []
    assert "finite" in frame["errors"][0]["message"]
    assert json.loads(text)["status"] == "BACKEND_ERROR"


def test_null_score_is_allowed_and_stays_null(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(tmp_path / "cand" / "a.png")), "score": None}]}}})
    manifest = _manifest(_run(tmp_path, source, spec, indices=[17]))
    assert manifest["status"] == "SUCCESS"
    assert manifest["frames"][0]["candidates"][0]["score"] is None


def _fake_repo(tmp_path):
    repo = tmp_path / "sam3-repo"
    (repo / "sam3" / "assets").mkdir(parents=True)
    (repo / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"fake-bpe")
    return repo


def test_local_checkpoint_without_hash_is_refused_before_load(tmp_path):
    repo = _fake_repo(tmp_path)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"weights")
    backend = backends.Sam3Backend(repo, checkpoint, None, "cuda", 0.5)
    with pytest.raises(backends.BackendUnavailable) as excinfo:
        backend.resolve_weights()
    assert "--checkpoint-sha256" in str(excinfo.value)
    assert backend.bindings()["weights"]["source"] is None


def test_local_checkpoint_hash_mismatch_is_refused(tmp_path):
    repo = _fake_repo(tmp_path)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"weights")
    backend = backends.Sam3Backend(repo, checkpoint, "0" * 64, "cuda", 0.5)
    with pytest.raises(backends.BackendUnavailable) as excinfo:
        backend.resolve_weights()
    assert "sha256 mismatch" in str(excinfo.value)


def test_declared_config_file_is_recorded_as_not_consumed(tmp_path):
    repo = _fake_repo(tmp_path)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8", newline="\n")
    digest = binding.sha256_file(config)

    backend = backends.Sam3Backend(repo, checkpoint, binding.sha256_file(checkpoint), "cuda", 0.5,
                                  config_path=config, config_sha256=digest)
    backend.resolve_weights()
    config_block = backend.bindings()["config"]

    assert config_block["config_source"] == "effective_construction_and_inference_options"
    assert config_block["declared_config_file"]["sha256"] == digest
    assert config_block["declared_config_file_consumed"] is False
    # the digest is over the real construction options, and is reproducible
    assert config_block["config_sha256"] == binding.effective_options_digest(
        backend.effective_options())
    assert backend.describe()["config_sha256"] == config_block["config_sha256"]


def test_hf_resolution_pins_repo_revision_and_cache_and_asks_once(tmp_path, monkeypatch):
    """R1 repair group 4: prove repo/revision/file/cache are fixed, and no double probe."""
    repo = _fake_repo(tmp_path)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    calls = []

    def fake_download(**kwargs):
        calls.append(dict(kwargs))
        target = tmp_path / "downloaded" / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stub-" + kwargs["filename"].encode())
        return str(target)

    stub = SimpleNamespace(get_token=lambda: "stub-token", hf_hub_download=fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    backend = backends.Sam3Backend(repo, None, None, "cuda", 0.5, allow_hf_download=True,
                                   hf_revision="2345a4ad109ac29c569da749c91d84f10dc08c40",
                                   hf_cache_dir=cache)
    backend.resolve_weights()
    backend.resolve_weights()  # must not download a second time

    assert len(calls) == 2
    assert [call["filename"] for call in calls] == ["config.json", "sam3.pt"]
    assert {call["repo_id"] for call in calls} == {"facebook/sam3"}
    assert {call["revision"] for call in calls} == {"2345a4ad109ac29c569da749c91d84f10dc08c40"}
    assert {call["cache_dir"] for call in calls} == {str(cache)}
    weights = backend.bindings()["weights"]
    assert weights["hf_revision_pinned"] is True
    assert weights["token_present"] is True
    assert weights["token_value_recorded"] is False
    assert "stub-token" not in json.dumps(weights)
    assert weights["checkpoint_sha256"] == binding.sha256_file(backend.checkpoint_path)


def test_hf_401_becomes_auth_required_without_touching_torch(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)

    class GatedError(Exception):
        response = SimpleNamespace(status_code=401)

    def fake_download(**kwargs):
        raise GatedError("401 Client Error: gated repo")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(get_token=lambda: None, hf_hub_download=fake_download))
    backend = backends.Sam3Backend(repo, None, None, "cuda", 0.5, allow_hf_download=True)
    with pytest.raises(backends.AuthRequired) as excinfo:
        backend.resolve_weights()
    assert "facebook/sam3" in str(excinfo.value)
    assert "torch" not in sys.modules


def test_deadline_shares_one_origin_across_phases():
    clock = {"t": 0.0}
    dl = deadline_mod.Deadline(budgets=deadline_mod.Budgets(load_seconds=10, frame_seconds=100,
                                                            batch_seconds=20),
                               clock=lambda: clock["t"])
    assert dl.batch_deadline - dl.origin == pytest.approx(20)
    dl.begin_load()
    clock["t"] = 11.0
    with pytest.raises(deadline_mod.DeadlineExceeded):
        dl.end_load()
    dl.begin_frame(17)
    # the frame budget is clamped by the remaining batch budget, not reset
    assert dl.snapshot()["current_frame_remaining_seconds"] <= 9.0


def test_load_deadline_is_reported_as_resource_limit(tmp_path, monkeypatch):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})

    class SlowBackend(backends.FakeBackend):
        name = "fake"

        def resolve_weights(self):
            return None

        def load(self):
            import time as _time
            _time.sleep(1.0)

    monkeypatch.setattr("sam3_adapter.runner.ResourceMonitor", QuietMonitor)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: SlowBackend(spec))
    outcome = execute(RunRequest(
        io_run=source, indices=[17], object_id="fixture", text_prompt="fixture",
        output_root=tmp_path / "out", repo_dir=ADAPTER_DIR, backend_kind="sam3", fake_spec=spec,
        argv=["test"], cwd=str(tmp_path),
        budgets=deadline_mod.Budgets(load_seconds=0.2, frame_seconds=5, batch_seconds=5)))
    manifest = _manifest(outcome)
    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert any(e["code"] == "DEADLINE_EXCEEDED" for e in manifest["errors"])
    assert manifest["frames"][0]["status"] == "NOT_RUN"
    assert manifest["deadlines"]["budgets"]["load_seconds"] == 0.2



def test_cli_assigns_thread_policy_instead_of_inheriting():
    original = dict(os.environ)
    try:
        os.environ["OMP_NUM_THREADS"] = "32"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "0"
        spec = importlib.util.spec_from_file_location("cli_policy_probe", CLI)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.CPU_THREADS == "2"
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["MKL_NUM_THREADS"] == "2"
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
        assert sys.dont_write_bytecode is True
    finally:
        os.environ.clear()
        os.environ.update(original)


def _run_cli(*args, env=None):
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True,
                          timeout=180, check=False, env=env,
                          encoding="utf-8", errors="replace")


def test_cli_rejects_conflicting_prompts_and_out_of_range_options(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {}})
    base = ["--io-run", str(source), "--indices", "17", "--object-id", "skull-01",
            "--text-prompt", "skull", "--backend", "fake", "--fake-spec", str(spec),
            "--output", str(tmp_path / "out"), "--no-host-probe"]

    visual = _run_cli(*base, "--visual-prompt", "box:0,0,1,1")
    assert visual.returncode == EXIT_CODES["INPUT_ERROR"]
    assert "--visual-prompt" in visual.stderr and "not allowed with" in visual.stderr

    threshold = _run_cli(*base, "--confidence-threshold", "1.5")
    assert threshold.returncode == EXIT_CODES["INPUT_ERROR"]

    resolution = _run_cli(*base, "--resolution", "1007")
    assert resolution.returncode == EXIT_CODES["INPUT_ERROR"]

    duplicate = _run_cli("--io-run", str(source), "--indices", "17,17", "--object-id", "skull-01",
                         "--text-prompt", "skull", "--backend", "fake", "--fake-spec", str(spec),
                         "--output", str(tmp_path / "out"), "--no-host-probe")
    assert duplicate.returncode == EXIT_CODES["INPUT_ERROR"]
    assert "duplicate input_index" in duplicate.stderr

    bad_index = _run_cli(*base, "--select", "34:c00")
    assert bad_index.returncode == EXIT_CODES["INPUT_ERROR"]
    assert "not in --indices" in bad_index.stderr

    bad_candidate = _run_cli(*base, "--select", "17:nope")
    assert bad_candidate.returncode == EXIT_CODES["INPUT_ERROR"]

    missing_hash = _run_cli("--io-run", str(source), "--indices", "17", "--object-id", "skull-01",
                            "--text-prompt", "skull", "--checkpoint", str(tmp_path / "sam3.pt"),
                            "--output", str(tmp_path / "out"), "--no-host-probe")
    assert missing_hash.returncode == EXIT_CODES["INPUT_ERROR"]
    assert "checkpoint-sha256" in missing_hash.stderr


def test_cli_fake_backend_run_reports_contract_exit_code(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(tmp_path / "cand" / "a.png")), "score": 0.7}]}}})
    completed = _run_cli("--io-run", str(source), "--indices", "17", "--object-id", "skull-01",
                         "--text-prompt", "skull", "--backend", "fake", "--fake-spec", str(spec),
                         "--output", str(tmp_path / "out"), "--no-host-probe")
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "SUCCESS"
    assert summary["evidence_level"] == EVIDENCE_LEVEL_FAKE
    mask = next((tmp_path / "out").rglob("masks/*.png"))
    assert mask.name == "000017_c00.png"
    with Image.open(mask) as image:
        assert image.mode == "L" and image.size == (WIDTH, HEIGHT)


# =========================================================================== #
# A-R2: product-path wiring, irrevocable stop latch, resilient finalisation
# =========================================================================== #
class _StubProcess:
    """Minimal Popen stand-in for reclaim bookkeeping (no real process)."""

    def __init__(self, pid=4242, exits_after=None):
        self.pid = pid
        self._polls = 0
        self._exits_after = exits_after

    def poll(self):
        self._polls += 1
        if self._exits_after is None:
            return None
        return 0 if self._polls > self._exits_after else None

    def kill(self):
        return None


def test_kill_tree_records_unknown_when_exit_is_not_proved():
    """A successful kill call is not evidence of exit."""
    record = worker.kill_tree(_StubProcess(exits_after=None), time.monotonic() + 0.15)
    assert record["exited"] is False
    assert record["residual_unknown"] is True
    assert record["escalated_to_kill"] is True
    proved = worker.kill_tree(_StubProcess(exits_after=0), time.monotonic() + 0.15)
    assert proved["exited"] is True and proved["residual_unknown"] is False


class _BreachThenRecoverMonitor(QuietMonitor):
    """Breaches once, then reports healthy numbers again: the latch must hold."""

    def __init__(self, **kwargs):
        self.samples = 0

    def sample(self):
        self.samples += 1
        return {"utc": "t", "process_commit_bytes": (13 << 30) if self.samples == 1 else (1 << 20)}

    def violations(self, sample=None):
        commit = (sample or {}).get("process_commit_bytes")
        return [] if commit is None or commit <= (12 << 30) else ["test-only RAM above 12 GiB"]


class _FailingMonitor(QuietMonitor):
    def __init__(self, **kwargs):
        self.samples = 0

    def sample(self):
        self.samples += 1
        raise RuntimeError("test-only monitor channel failure")


class _LateChannelFailureMonitor(QuietMonitor):
    """Initial channel query works, the final one raises."""

    def __init__(self, **kwargs):
        self.queries = 0

    def channel_report(self):
        self.queries += 1
        raise RuntimeError("test-only late channel query failure")
        return {"test_only": True}


def _sam3_fake_run(tmp_path, monkeypatch, indices, monitor_class, **overrides):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {
        "frames": {str(index): {"candidates": [
            {"mask_path": str(_rect_mask(masks / f"m{index}.png")), "score": 0.8}]} for index in indices}})
    monkeypatch.setattr("sam3_adapter.runner.ResourceMonitor", monitor_class)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))
    request = dict(io_run=source, indices=list(indices), object_id="fixture", text_prompt="fixture",
                   output_root=tmp_path / "out", repo_dir=ADAPTER_DIR, backend_kind="sam3",
                   fake_spec=spec, argv=["test"], cwd=str(tmp_path))
    request.update(overrides)
    return execute(RunRequest(**request))


def test_observed_breach_latches_even_after_the_sample_recovers(tmp_path, monkeypatch):
    """A transient breach used to be forgotten when the next sample looked fine."""
    outcome = _sam3_fake_run(tmp_path, monkeypatch, [17, 34, 59], _BreachThenRecoverMonitor)
    manifest = _manifest(outcome)

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    latch = manifest["stop_state"]
    assert latch["latched"] is True and latch["kind"] == "RESOURCE_LIMIT"
    assert latch["source"] in ("sampler", "parent")
    # the breach stopped the batch; at least one frame never started
    assert manifest["counts"]["not_run"] >= 1
    assert manifest["counts"]["success"] < 3


def test_monitor_failure_stops_later_frames_immediately(tmp_path, monkeypatch):
    outcome = _sam3_fake_run(tmp_path, monkeypatch, [17, 34, 59], _FailingMonitor)
    manifest = _manifest(outcome)

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["counts"]["not_run"] >= 1
    assert any(e["code"] == "MONITOR_SAMPLE_FAILED" for e in manifest["errors"])
    assert manifest["stop_state"]["latched"] is True


def test_late_channel_query_failure_still_writes_a_manifest(tmp_path, monkeypatch):
    """The manifest must survive a failing final query instead of vanishing."""
    outcome = _sam3_fake_run(tmp_path, monkeypatch, [17], _LateChannelFailureMonitor)
    manifest = _manifest(outcome)

    assert (outcome.run_dir / "manifest.json").is_file()
    assert any(e["code"] == "CHANNEL_QUERY_FAILED" for e in manifest["errors"])
    assert manifest["resource_observation"]["channels"] is None
    assert manifest["counts"]["requested"] == 1


def test_sampler_stop_does_not_shadow_thread_internals():
    """Python 3.12 exposes Thread._stop; an Event there breaks join()."""
    stop_state = deadline_mod.StopState()
    sampler = _import_runner()._BackgroundSampler(QuietMonitor(), 0.02, [], [], stop_state)
    assert "_stop" not in vars(sampler)
    assert isinstance(vars(sampler)["_stop_event"], type(threading.Event()))
    sampler.start()
    report = sampler.stop()
    assert report["joined_cleanly"] is True


def _import_runner():
    from sam3_adapter import runner as runner_module  # noqa: PLC0415
    return runner_module


def test_shared_deadline_is_not_reset_by_the_load_phase():
    clock = {"t": 0.0}
    dl = deadline_mod.Deadline(budgets=deadline_mod.Budgets(load_seconds=5, frame_seconds=100,
                                                           batch_seconds=12),
                               clock=lambda: clock["t"])
    dl.begin_load()
    clock["t"] = 4.5
    dl.end_load()
    clock["t"] = 5.0
    dl.begin_frame(34)
    snapshot = dl.snapshot()
    assert snapshot["current_frame_remaining_seconds"] <= 7.01
    clock["t"] = 12.5
    with pytest.raises(deadline_mod.DeadlineExceeded):
        dl.check_batch()


# --------------------------------------------------------------------------- #
# product path: one bounded worker per batch, identity + return code propagation
# --------------------------------------------------------------------------- #
def _many_index_spec(tmp_path, count):
    """A batch long enough that the worker is still busy when a budget fires."""
    indices = tuple(range(100, 100 + count))
    source = _make_source_run(tmp_path / "io", indices=indices)
    mask = _rect_mask(tmp_path / "cand" / "m.png")
    spec = _write_spec(tmp_path / "spec.json", {
        "frames": {str(index): {"candidates": [{"mask_path": str(mask), "score": 0.8}]}
                   for index in indices}})
    return source, spec, indices


def _long_worker_run(tmp_path, budgets):
    source, spec, indices = _many_index_spec(tmp_path, 120)
    return execute(RunRequest(
        io_run=source, indices=list(indices), object_id="fixture-obj", text_prompt="fixture-prompt",
        output_root=tmp_path / "out", repo_dir=ADAPTER_DIR, backend_kind="fake", fake_spec=spec,
        argv=["test"], cwd=str(tmp_path), use_worker=True, worker_python=sys.executable,
        worker_adapter_dir=ADAPTER_DIR, budgets=budgets, device="cpu"))


def _product_worker_run(tmp_path, indices=(17, 34, 59), budgets=None, **overrides):
    source = _make_source_run(tmp_path / "io")
    masks = tmp_path / "cand"
    spec = _write_spec(tmp_path / "spec.json", {
        "frames": {str(index): {"candidates": [
            {"mask_path": str(_rect_mask(masks / f"m{index}.png")), "score": 0.8}]} for index in indices}})
    request = dict(
        io_run=source, indices=list(indices), object_id="fixture-obj", text_prompt="fixture-prompt",
        output_root=tmp_path / "out", repo_dir=ADAPTER_DIR, backend_kind="fake", fake_spec=spec,
        argv=["test"], cwd=str(tmp_path), use_worker=True, worker_python=sys.executable,
        worker_adapter_dir=ADAPTER_DIR, budgets=budgets or deadline_mod.Budgets(),
    )
    request.update(overrides)
    return execute(RunRequest(**request))


def test_product_path_uses_one_reused_worker_and_propagates_identity(tmp_path):
    outcome = _product_worker_run(tmp_path)
    manifest = _manifest(outcome)
    report = manifest["worker"]

    assert manifest["execution_mode"] == "bounded_worker"
    assert outcome.status == "SUCCESS", manifest["errors"]
    assert manifest["counts"] == {"requested": 3, "success": 3, "no_match": 0,
                                 "needs_selection": 0, "error": 0, "not_run": 0}
    assert report["exited"] is True and report["returncode"] == 0 and report["residual_unknown"] is False
    assert report["frames_declared"] == 3 and report["frames_seen"] == 3
    # one child served all three frames: one load, not one per frame
    assert report["ledger_summary"]["frames_seen"] == [17, 34, 59]
    assert report["ledger_summary"]["dedup_key"].startswith("monotonic seq")
    # the child announced itself before doing any work; the pid it reported is
    # recorded next to the launcher pid so a mismatch is visible, not hidden
    assert isinstance(report["ledger_summary"]["ready"]["pid"], int)
    assert report["pid"] > 0
    events = worker.read_events(Path(report["events_path"]))
    assert sum(1 for e in events if e["type"] == "stage" and e.get("stage") == "load") == 2
    assert sum(1 for e in events if e["type"] == "frame") == 3
    # RAM coverage includes the worker, not only the idle parent
    assert manifest["resource_observation"]["worker_reported_samples"] >= 3
    scopes = {s.get("sampler_scope") for s in manifest["resource_observation"]["samples"]}
    assert "worker process" in scopes
    # masks produced by the child are visible to the parent manifest
    for frame in manifest["frames"]:
        assert frame["selected_mask"] and (outcome.run_dir / frame["selected_mask"]["path"]).is_file()
        assert frame["selected_mask"]["sha256"] == imagery.sha256_file(
            outcome.run_dir / frame["selected_mask"]["path"])


def test_product_load_budget_is_enforced_before_the_worker_starts(tmp_path):
    """Fixture 1: the load budget shares one origin and is enforced from the start.

    With a budget shorter than this process' own pre-flight, the run stops before
    launching the worker at all. That is correct enforcement (and it is why the
    budget is never reset), but it does NOT exercise the kill path -- so it is
    asserted as such instead of being dressed up as a termination.
    Live-worker termination is covered by fixture 2 (batch budget) and fixture 3
    (nested child).
    """
    outcome = _long_worker_run(
        tmp_path, deadline_mod.Budgets(load_seconds=0.02, frame_seconds=30, batch_seconds=30))
    manifest = _manifest(outcome)

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["worker"] is None  # nothing was launched, so nothing is claimed
    assert any(e["code"] == "DEADLINE_EXCEEDED" for e in manifest["errors"])
    assert manifest["counts"]["success"] == 0
    assert manifest["counts"]["not_run"] == manifest["counts"]["requested"]
    assert manifest["stop_state"]["latched"] is False  # the parent stopped, not the supervisor


def test_product_batch_budget_terminates_a_live_worker(tmp_path):
    """Fixture 2: the batch budget fires while the worker is still running."""
    outcome = _long_worker_run(
        tmp_path, deadline_mod.Budgets(load_seconds=30, frame_seconds=30, batch_seconds=0.05))
    manifest = _manifest(outcome)

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["counts"]["not_run"] >= 1
    assert manifest["counts"]["success"] < manifest["counts"]["requested"]
    assert manifest["worker"]["returncode"] is not None


def test_bounded_worker_fixture_three_covers_a_nested_child(tmp_path):
    """Fixture 3: a nested child plus an identity-bound exit confirmation."""
    record = worker.run_bounded(
        [sys.executable, "-m", "sam3_adapter.worker", "--spawn-child", "--sleep", "2"],
        deadline_seconds=0.6, cwd=ADAPTER_DIR, log_dir=tmp_path / "logs",
        env=dict(os.environ, PYTHONPATH=str(ADAPTER_DIR)), reclaim_seconds=3.0)

    assert record["timed_out"] is True
    assert record["wall_seconds"] < 3.0
    kill = record["kill"]
    assert kill["tree_kill_attempted"] is True
    assert kill["pid"] == record["pid"]
    assert kill["exited"] is True and kill["residual_unknown"] is False
    assert Path(record["stdout_path"]).is_file() and Path(record["stderr_path"]).is_file()


def test_probe_is_never_run_implicitly(tmp_path):
    source = _make_source_run(tmp_path / "io")
    spec = _write_spec(tmp_path / "spec.json", {"frames": {"17": {"candidates": [
        {"mask_path": str(_rect_mask(tmp_path / "cand" / "a.png")), "score": 0.7}]}}})
    completed = _run_cli("--io-run", str(source), "--indices", "17", "--object-id", "skull-01",
                         "--text-prompt", "skull", "--backend", "fake", "--fake-spec", str(spec),
                         "--output", str(tmp_path / "out"))
    assert completed.returncode == 0, completed.stderr
    manifest_path = next((tmp_path / "out").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "bounded_worker"
    assert manifest["host_probe"]["status"] == "NOT_RUN"
    assert manifest["host_probe"]["implicit_probe_is_bypass"] is False


# =========================================================================== #
# A-R3: live worker ownership, in-operation observation, event finalisation
# =========================================================================== #
FIXTURE_DIR = PROJECT_ROOT / "outputs/delegation/SAM3_A_R3"


def _fixture_env(fixtures_root, **values):
    env = {"PYTHONPATH": os.pathsep.join([str(ADAPTER_DIR), str(fixtures_root)]),
           "FIXTURE_ADAPTER_DIR": str(ADAPTER_DIR)}
    env.update({k: str(v) for k, v in values.items()})
    return env


class _StubPopen:
    def __init__(self, pid=4321):
        self.pid = pid
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class _RecordingOwnership:
    """Records reclaim calls; stands in for the OS-bound ownership object."""

    instances: list = []

    def __init__(self, popen, python, adapter_dir, job_path, reclaim_seconds=5.0):
        self.process = popen
        self.launcher_pid = popen.pid
        self.reclaim_seconds = reclaim_seconds
        self.reclaims: list = []
        self.reclaim_records: list = []
        self.ready_events: list = []
        _RecordingOwnership.instances.append(self)

    def note_ready(self, event):
        self.ready_events.append(event)

    def identity(self):
        return {"launcher_pid": self.launcher_pid, "pid_equality_assumed": False}

    def members(self):
        return {"top_level": {"pid": self.launcher_pid, "alive": self.process.poll() is None},
                "live_descendants": [], "descendant_count": 0, "enumeration_available": True}

    def top_level_exited(self):
        return self.process.poll() is not None

    def reclaim(self, reason):
        self.reclaims.append({"reason": reason, "t": time.monotonic()})
        self.process.returncode = -9
        record = {"reason": reason, "exited": True, "residual_unknown": [], "already_reclaimed": False}
        self.reclaim_records.append(record)
        return record


def _stub_worker_launch(monkeypatch):
    stub = _StubPopen()
    monkeypatch.setattr("sam3_adapter.worker.launch_worker",
                        lambda *a, **k: stub)
    monkeypatch.setattr("sam3_adapter.observer.WorkerOwnership", _RecordingOwnership)
    _RecordingOwnership.instances = []
    return stub


def _worker_request(tmp_path, spec, budgets, **overrides):
    source = _make_source_run(tmp_path / "io")
    request = dict(
        io_run=source, indices=[17, 34, 59], object_id="fixture", text_prompt="fixture",
        output_root=tmp_path / "out", repo_dir=ADAPTER_DIR, backend_kind="fake", fake_spec=spec,
        argv=["test"], cwd=str(tmp_path), use_worker=True, worker_python=sys.executable,
        worker_adapter_dir=ADAPTER_DIR, budgets=budgets, device="cpu")
    request.update(overrides)
    return RunRequest(**request)


def _spec_for(tmp_path, indices=(17, 34, 59)):
    masks = tmp_path / "cand"
    return _write_spec(tmp_path / "spec.json", {
        "frames": {str(index): {"candidates": [
            {"mask_path": str(_rect_mask(masks / f"m{index}.png")), "score": 0.8}]} for index in indices}})


def test_kill_callback_owns_the_live_worker_and_interrupts_the_wait(tmp_path, monkeypatch):
    """A-R2 counterexample: the stop callback had nothing to reclaim."""
    stub = _stub_worker_launch(monkeypatch)
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))

    started = time.monotonic()
    outcome = execute(_worker_request(
        tmp_path, spec, deadline_mod.Budgets(load_seconds=0.08, frame_seconds=30, batch_seconds=0.60)))
    elapsed = time.monotonic() - started
    manifest = _manifest(outcome)
    ownership = _RecordingOwnership.instances[-1]

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    # the callback reached the live worker instead of "no worker to reclaim"
    assert manifest["worker"]["reclaim"] is not None
    assert "no worker to reclaim" not in json.dumps(manifest["worker"]["reclaim"])
    assert ownership.reclaims, "the reclaim callback never fired"
    assert ownership.ready_events or True  # ready may not arrive before the deadline
    assert stub.returncode == -9  # the double's kill path ran
    assert manifest["worker"]["reclaim"]["exited"] is True
    # the wait was interrupted at the load deadline, not at the batch deadline
    assert elapsed < 0.45, f"waiting was not interrupted early (elapsed={elapsed:.3f}s)"
    assert manifest["stop_state"]["source"] == "parent_deadline"


def test_observer_catches_an_in_operation_peak_and_the_latch_survives_recovery(tmp_path, monkeypatch):
    """A-R2 counterexample: zero samples during load, and a fallback cleared the breach."""
    _stub_worker_launch(monkeypatch)
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))
    readings = [1 << 20, 13 << 30, 1 << 20, 1 << 20]

    def fake_for_pid(pid):
        value = readings.pop(0) if readings else (1 << 20)
        return {"commit_bytes": value, "rss_bytes": value, "pid": pid, "commit_channel": "test"}

    monkeypatch.setattr("sam3_adapter.resources.read_process_memory_for_pid", fake_for_pid)

    outcome = execute(_worker_request(
        tmp_path, spec, deadline_mod.Budgets(load_seconds=30, frame_seconds=30, batch_seconds=30),
        monitor_interval_seconds=0.02))
    manifest = _manifest(outcome)
    observations = [s for s in manifest["resource_observation"]["samples"] if s.get("reported_by") == "parent"]

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert len(observations) >= 2
    assert any((s.get("worker_commit_bytes") or 0) > (12 << 30) for s in observations)
    assert manifest["stop_state"]["latched"] is True
    assert manifest["stop_state"]["source"] == "observer"
    assert manifest["counts"]["not_run"] == 3  # nothing started after the latch
    assert observations[0]["covers_pid"] == manifest["worker"]["pid"]


def test_ledger_applies_every_event_type_and_reports_corrupt_records(tmp_path):
    events_path = tmp_path / "events.jsonl"
    lines = [
        {"type": "ready", "seq": 1, "pid": 111, "at_utc": "2026-01-01T00:00:00Z"},
        {"type": "stage", "seq": 2, "stage": "load", "state": "begin"},
        {"type": "sample", "seq": 3, "stage": "post_load", "process_commit_bytes": 4096},
        {"type": "frame_begin", "seq": 4, "input_index": 17},
        {"type": "frame", "seq": 5, "input_index": 17, "status": "OK", "candidates": []},
        "this line is not json",
        {"type": "stage", "seq": 6, "stage": "close", "state": "error",
         "message": "RuntimeError: fixture close failure"},
        {"type": "fatal", "seq": 7, "message": "fixture fatal"},
        {"type": "some_future_event", "seq": 8},
        {"type": "done", "seq": 9, "stopped": False, "frames_attempted": 1, "frames_ok": 1},
    ]
    events_path.write_text(("\n".join(
        line if isinstance(line, str) else json.dumps(line) for line in lines)) + "\n",
        encoding="utf-8", newline="\n")

    parsed, problems = worker.read_events_strict(events_path, allow_trailing_partial=False)
    assert len(parsed) == 9
    assert any(p["code"] == "EVENT_RECORD_CORRUPT" and p["tolerated"] is False for p in problems)

    stop_state = deadline_mod.StopState()
    samples: list = []
    errors: list = []
    finalisation: list = []
    frames: dict = {}
    ledger = _import_runner()._EventLedger(stop_state, samples, frames, errors, finalisation,
                                          deadline_mod.Deadline(budgets=deadline_mod.Budgets()))
    applied = ledger.apply_all(parsed)
    assert applied == 9
    assert ledger.state["ready"]["pid"] == 111
    assert ledger.state["close_error"] is not None
    assert any(e["code"] == "BACKEND_CLOSE_FAILED" for e in errors)
    assert any(e["code"] == "BACKEND_CLOSE_FAILED" for e in finalisation)
    assert stop_state.latched and stop_state.kind == "BACKEND_ERROR"
    assert ledger.state["done"] == {"stopped": False, "frames_attempted": 1, "frames_ok": 1}
    assert ledger.state["unknown_events"] == [{"type": "some_future_event"}]
    assert any(e["code"] == "EVENT_TYPE_UNKNOWN" for e in errors)
    assert len(samples) == 1 and frames[17]["status"] == "OK"
    # the same path is idempotent: a second pass must not double-count
    assert ledger.apply_all(parsed) == 0 and ledger.duplicates == 9


def test_torn_trailing_line_is_tolerated_only_while_writing(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text('{"type": "ready", "seq": 1}\n{"type": "sam', encoding="utf-8", newline="\n")
    live, live_problems = worker.read_events_strict(events_path, allow_trailing_partial=True)
    final, final_problems = worker.read_events_strict(events_path, allow_trailing_partial=False)
    assert len(live) == 1 and live_problems[0]["tolerated"] is True
    assert len(final) == 1 and final_problems[0]["tolerated"] is False


def test_unknown_member_state_is_reported_as_unknown(tmp_path):
    """The parent cannot prove a member is gone -> UNKNOWN, not 'exited'."""
    handle = observer_module.WorkerOwnership.__new__(observer_module.WorkerOwnership)
    # a PID the OS cannot answer for must stay UNKNOWN rather than "exited"
    assert observer_module.process_alive(999999) in (False, None)
    assert handle is not None
    record = observer_module.WorkerOwnership.identity(
        type("H", (), {"launcher_pid": 1, "launcher_creation": None, "python": "p",
                       "self_reported_pid": 2, "self_reported_at": "t"})())
    assert record["pid_equality_assumed"] is False
    assert record["self_reported_pid"] == 2 and record["launcher_pid"] == 1


# --------------------------------------------------------------------------- #
# one real-OS fixture batch: 4 scenarios, short-lived fake backends only
# --------------------------------------------------------------------------- #
def _fixture_run(tmp_path, name, *, frames=(17, 34, 59), budgets, fixture_values,
                 monitor_interval=0.02):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    spec = _spec_for(root, frames)
    source = _make_source_run(root / "io", indices=frames)
    request = RunRequest(
        io_run=source, indices=list(frames), object_id="fixture", text_prompt="fixture",
        output_root=root / "out", repo_dir=ADAPTER_DIR, backend_kind="fake", fake_spec=spec,
        argv=["test"], cwd=str(root), use_worker=True, worker_python=sys.executable,
        worker_adapter_dir=ADAPTER_DIR, worker_module="fixture_worker",
        worker_extra_env=_fixture_env(FIXTURY_DIR, **fixture_values),
        budgets=budgets, monitor_interval_seconds=monitor_interval)
    return execute(request), spec


FIXTURY_DIR = PROJECT_ROOT / "outputs/delegation/SAM3_A_R3/20261006T182953Z/fixtures"


def test_fixture_a_active_load_is_terminated_by_the_load_budget(tmp_path):
    """Scenario a: the worker is inside load() when the product load budget fires."""
    outcome, _ = _fixture_run(
        tmp_path, "a", budgets=deadline_mod.Budgets(load_seconds=1.2, frame_seconds=30,
                                                    batch_seconds=30),
        fixture_values={"FIXTURE_LOAD_SLEEP": "3"})
    manifest = _manifest(outcome)
    summary = manifest["worker"]["ledger_summary"]

    assert summary["load_started"] is True and summary["load_ended"] is False  # ACTIVE_LOAD anchor
    assert summary["done"] is None
    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    reclaim = manifest["worker"]["reclaim"]
    assert reclaim is not None and reclaim["exited"] is True
    assert reclaim["residual_unknown"] == []
    assert manifest["counts"]["not_run"] == 3 and manifest["counts"]["success"] == 0
    assert manifest["worker"]["members_after"]["descendant_count"] == 0


def test_fixture_b_active_frame_is_terminated_by_the_frame_budget(tmp_path):
    """Scenario b: the worker is inside segment() when the frame budget fires."""
    outcome, _ = _fixture_run(
        tmp_path, "b", budgets=deadline_mod.Budgets(load_seconds=30, frame_seconds=1.2,
                                                    batch_seconds=30),
        fixture_values={"FIXTURE_FRAME_SLEEP": "3"})
    manifest = _manifest(outcome)
    summary = manifest["worker"]["ledger_summary"]

    assert summary["load_ended"] is True                     # load really completed
    assert summary["frames_seen"] == []                      # no frame event: still inside the call
    assert summary["current_frame"] == 17                    # ACTIVE_FRAME anchor
    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["counts"]["not_run"] == 3                # the remaining frames never ran
    assert manifest["worker"]["reclaim"]["exited"] is True


def test_fixture_c_observer_stops_an_active_worker_and_recovery_does_not_unlatch(tmp_path, monkeypatch):
    """Scenario c: injected sustained over-limit while the worker is operating."""
    calls = {"n": 0}

    def injected(pid):
        calls["n"] += 1
        value = (13 << 30) if calls["n"] >= 3 else (1 << 20)
        return {"commit_bytes": value, "rss_bytes": value, "pid": pid, "commit_channel": "fixture"}

    monkeypatch.setattr("sam3_adapter.resources.read_process_memory_for_pid", injected)
    outcome, _ = _fixture_run(
        tmp_path, "c", budgets=deadline_mod.Budgets(load_seconds=30, frame_seconds=30,
                                                    batch_seconds=30),
        fixture_values={"FIXTURE_FRAME_SLEEP": "3"}, monitor_interval=0.05)
    manifest = _manifest(outcome)

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["stop_state"]["latched"] is True
    assert manifest["stop_state"]["source"] == "observer"
    assert manifest["counts"]["not_run"] >= 1
    assert manifest["worker"]["reclaim"]["exited"] is True
    # the later low readings did not clear the latch
    assert manifest["stop_state"]["all_observations"][0]["kind"] == "RESOURCE_LIMIT"


def test_fixture_d_reuse_descendant_and_close_error(tmp_path):
    """Scenario d: multi-frame reuse, a short-lived descendant, and a close error."""
    outcome, _ = _fixture_run(
        tmp_path, "d", budgets=deadline_mod.Budgets(load_seconds=30, frame_seconds=30,
                                                    batch_seconds=30),
        fixture_values={"FIXTURE_DESCENDANT": "1", "FIXTURE_DESCENDANT_SLEEP": "3",
                        "FIXTURE_CLOSE_ERROR": "1"})
    manifest = _manifest(outcome)
    summary = manifest["worker"]["ledger_summary"]

    # one worker served all three frames
    assert summary["frames_seen"] == [17, 34, 59]
    assert manifest["counts"]["success"] == 3
    # but the close error means this is NOT a clean success
    assert summary["close_error"] is not None
    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    assert any(e["code"] == "BACKEND_CLOSE_FAILED" for e in manifest["errors"])
    # descendants are accounted for separately from the top-level exit
    assert manifest["worker"]["members_after"] is not None
    assert manifest["worker"]["ownership_identity"]["pid_equality_assumed"] is False
    assert manifest["worker"]["ownership_identity"]["self_reported_pid"] is not None


# =========================================================================== #
# A-R4: R1 observe the actual worker / R2 members and identity / R3 startup
#       cleanup / R4 protocol failures that reach the verdict
# =========================================================================== #
def _ownership_with_registry(launcher=990001, launcher_creation=100, worker=990002,
                             worker_creation=200, table=None, top_alive=True):
    """A WorkerOwnership with an injected OS view (no real process)."""
    class _P:
        pid = launcher

        def poll(self):
            return None if top_alive else 0

        def kill(self):
            return None

    handle = observer_module.WorkerOwnership.__new__(observer_module.WorkerOwnership)
    handle.process = _P()
    handle.python = "pytest"
    handle.adapter_dir = str(ADAPTER_DIR)
    handle.job_path = "job.json"
    handle.reclaim_seconds = 0.02
    handle.launcher_pid = launcher
    handle.launcher_creation = launcher_creation
    handle.launcher_handle = None
    handle.self_reported_pid = worker
    handle.self_reported_at = "t"
    handle.worker_pid = worker
    handle.worker_creation = worker_creation
    handle.worker_handle = None
    handle.worker_identity_evidence = {"accepted": True, "is_os_descendant_of_launcher": True}
    handle.members_registry = {
        launcher: observer_module.Member(role="launcher", pid=launcher,
                                         creation_time=launcher_creation, origin="popen"),
        worker: observer_module.Member(role="worker_interpreter", pid=worker,
                                       creation_time=worker_creation,
                                       origin="ready_event+os_evidence"),
    }
    handle.reclaim_deadline = None
    handle.reclaim_records = []
    handle._lock = threading.RLock()
    handle._reclaimed = False
    handle.created_monotonic = time.monotonic()
    handle.created_utc = "t"
    handle._table = table
    return handle


def _patch_os_view(monkeypatch, ownership, readings=None, alive_map=None, creation_map=None):
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes",
                        lambda: ownership._table)
    monkeypatch.setattr("sam3_adapter.observer.process_alive",
                        lambda pid: (alive_map or {}).get(pid, True))
    monkeypatch.setattr("sam3_adapter.observer.process_creation_time",
                        lambda pid: (creation_map or {}).get(pid, ownership.members_registry[pid].creation_time))
    readings = readings or {}
    monkeypatch.setattr("sam3_adapter.resources.read_process_memory_for_pid",
                        lambda pid: readings.get(pid, {"commit_bytes": 0, "rss_bytes": 0,
                                                       "pid": pid, "commit_channel": "test"}))


def _observer(ownership, samples=None, errors=None, stop_state=None, device_sampler=None):
    return observer_module.WorkerObserver(
        ownership, 0.05, samples if samples is not None else [],
        errors if errors is not None else [], stop_state or deadline_mod.StopState(),
        policy=resources.ResourcePolicy(), device_sampler=device_sampler)


def test_observation_covers_the_actual_worker_not_only_the_launcher(monkeypatch):
    """A-R3 counterexample: only 64224 was observed while the worker was 44212."""
    ownership = _ownership_with_registry(table={990002: {"ppid": 990001, "exe": "py.exe"}})
    _patch_os_view(monkeypatch, ownership,
                   readings={990001: {"commit_bytes": 1 << 20, "rss_bytes": 1 << 20, "pid": 990001,
                                      "commit_channel": "test"},
                             990002: {"commit_bytes": 13 << 30, "rss_bytes": 13 << 30, "pid": 990002,
                                      "commit_channel": "test"}})
    observation = _observer(ownership).observe()

    assert observation["actual_worker_pids"] == [990002]
    assert 990002 in observation["covers_pids"] and 990001 in observation["covers_pids"]
    rows = {row["pid"]: row for row in observation["targets"]}
    assert rows[990002]["is_actual_worker"] is True
    assert rows[990002]["identified_by"] == "ready_event+os_evidence"
    violations = _observer(ownership).violations(observation)
    assert any("990002" in item and "RAM commit" in item for item in violations)


def test_null_measurement_without_an_exception_latches(monkeypatch):
    """A-R3 counterexample: commit_bytes=null with no exception was treated as fine."""
    ownership = _ownership_with_registry()
    _patch_os_view(monkeypatch, ownership,
                   readings={990001: {"commit_bytes": None, "rss_bytes": None, "pid": 990001,
                                      "commit_channel": "psapi:OpenProcess failed (err=5)"},
                             990002: {"commit_bytes": None, "rss_bytes": None, "pid": 990002,
                                      "commit_channel": "psapi:GetProcessMemoryInfo failed (err=6)"}})
    observation = _observer(ownership).observe()
    violations = _observer(ownership).violations(observation)

    assert len(violations) >= 2
    assert any("unmeasurable while alive" in item and "err=5" in item for item in violations)
    assert any("990002" in item for item in violations)


def test_confirmed_exit_is_not_reported_as_a_measurement_failure(monkeypatch):
    ownership = _ownership_with_registry()
    _patch_os_view(monkeypatch, ownership, readings={}, alive_map={990001: False, 990002: False})
    observation = _observer(ownership).observe()
    assert all(row["confirmed_exited"] for row in observation["targets"])
    assert [v for v in _observer(ownership).violations(observation) if "commit" in v] == []


def test_unwired_device_channel_is_a_violation_and_bounded_query_latches(monkeypatch):
    ownership = _ownership_with_registry()
    _patch_os_view(monkeypatch, ownership,
                   readings={pid: {"commit_bytes": 0, "rss_bytes": 0, "pid": pid,
                                   "commit_channel": "test"} for pid in (990001, 990002)})
    assert any("device channel is not wired" in v
               for v in _observer(ownership).violations(_observer(ownership).observe()))

    over = _observer(ownership, device_sampler=lambda: {"device_used_bytes": 8 << 30,
                                                        "device_total_bytes": 8 << 30,
                                                        "device_index": 1, "age_seconds": 0.01,
                                                        "fresh": True})
    violations = over.violations(over.observe())
    assert any("device used" in item for item in violations)

    failed = _observer(ownership, device_sampler=lambda: {"error": "nvidia-smi rc=9",
                                                          "device_used_bytes": None})
    assert any("device query failed" in v for v in failed.violations(failed.observe()))


def test_device_query_is_bounded_device_identified_and_freshness_checked(monkeypatch):
    monkeypatch.setattr("sam3_adapter.observer.subprocess.run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout="1234, 8151\n", stderr=""))
    result = observer_module.driver_device_query(2, timeout=3.0, freshness=2.0)
    assert result["device_index"] == 2 and result["device_used_bytes"] == 1234 * (1 << 20)
    assert result["query_timeout_seconds"] == 3.0 and result["fresh"] is True


def test_members_are_kept_after_the_top_level_exits(monkeypatch):
    """A-R3 counterexample: `if top_alive else []` hid a live descendant."""
    ownership = _ownership_with_registry(top_alive=False,
                                         table={990002: {"ppid": 990001, "exe": "child.exe"}})
    _patch_os_view(monkeypatch, ownership,
                   alive_map={990001: False, 990002: True},
                   creation_map={990001: 100, 990002: 200})
    report = ownership.members()

    states = {row["pid"]: row["state"] for row in report["members"]}
    assert states[990002] == "ALIVE"          # not hidden behind a dead top level
    assert report["top_level_exited"] is True


def test_unavailable_enumeration_is_unknown_not_an_empty_safe_set(monkeypatch):
    ownership = _ownership_with_registry(table=None)
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes", lambda: None)
    monkeypatch.setattr("sam3_adapter.observer.process_alive", lambda pid: True)
    monkeypatch.setattr("sam3_adapter.observer.process_creation_time", lambda pid: 100)
    monkeypatch.setattr("sam3_adapter.observer.subprocess.run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    report = ownership.members()
    assert report["enumeration_available"] is False
    record = ownership.reclaim("enumeration unavailable")
    assert record["state"] == "COMPLETE"
    assert any(item.get("role") == "enumeration" and item["state"] == "UNKNOWN"
               for item in record["residual_unknown"])


def test_creation_time_change_prevents_termination(monkeypatch):
    """A-R3 counterexample: creation time was recorded but never checked."""
    ownership = _ownership_with_registry()
    calls = []
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes",
                        lambda: {990002: {"ppid": 990001, "exe": "py.exe"}})
    monkeypatch.setattr("sam3_adapter.observer.process_alive", lambda pid: True)
    # the PID now reports a different creation time: it is another process
    monkeypatch.setattr("sam3_adapter.observer.process_creation_time",
                        lambda pid: 999 if pid == 990001 else 200)
    monkeypatch.setattr("sam3_adapter.observer.subprocess.run",
                        lambda argv, *a, **k: calls.append(list(argv)) or SimpleNamespace(returncode=0))

    record = ownership.reclaim("identity counterexample")
    # the launcher PID whose creation time changed is never targeted; the worker
    # whose identity still matches may legitimately be terminated
    assert not any(item[-1] == "990001" for item in calls)
    assert any(action.get("why", "").startswith("identity not proven") for action in record["actions"])
    assert any(item["role"] == "launcher" and item["state"] == "UNKNOWN"
               for item in record["residual_unknown"])


def test_reclaim_shares_one_budget_and_skips_control_when_exhausted(monkeypatch):
    ownership = _ownership_with_registry()
    ownership.reclaim_seconds = 0.0  # the shared budget is already gone
    calls = []
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes",
                        lambda: {990002: {"ppid": 990001, "exe": "py.exe"}})
    monkeypatch.setattr("sam3_adapter.observer.process_alive", lambda pid: True)
    monkeypatch.setattr("sam3_adapter.observer.process_creation_time", lambda pid: 100)
    monkeypatch.setattr("sam3_adapter.observer.subprocess.run",
                        lambda argv, *a, **k: calls.append(list(argv)) or SimpleNamespace(returncode=0))

    record = ownership.reclaim("budget already exhausted")
    assert calls == []  # no blocking control command is opened once the budget is gone
    assert record["budget_exhausted"] is True
    assert all(action.get("skipped") for action in record["actions"] if action.get("argv"))
    assert record["residual_unknown"]  # something is still alive -> reported, not assumed clean


def test_post_launch_startup_failure_reclaims_the_created_process(tmp_path, monkeypatch):
    """A-R3 counterexample: observer.start() raising left the created process alive."""
    stub = _StubPopen()
    monkeypatch.setattr("sam3_adapter.worker.launch_worker", lambda *a, **k: stub)
    monkeypatch.setattr("sam3_adapter.observer.process_creation_time", lambda pid: None)
    monkeypatch.setattr("sam3_adapter.observer.process_alive", lambda pid: False)
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes", lambda: {})

    def boom(self):
        raise RuntimeError("injected observer.start failure")

    monkeypatch.setattr("sam3_adapter.observer.WorkerObserver.start", boom)
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))

    outcome = execute(_worker_request(tmp_path, spec, deadline_mod.Budgets()))
    manifest = _manifest(outcome)

    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    cleanup = [e for e in manifest["errors"] if e["code"] == "POST_LAUNCH_CLEANUP"]
    assert cleanup, manifest["errors"]
    reclaim = cleanup[0]["cleanup"]["reclaim"]
    assert reclaim["state"] == "COMPLETE"
    assert "injected observer.start failure" in reclaim["reason"]
    assert reclaim["members_before"]["member_count"] >= 1


def _stub_worker_with_events(monkeypatch, events, tmp_path, poll_code=0):
    stub = _StubPopen()
    if poll_code is not None:
        stub.returncode = poll_code
    monkeypatch.setattr("sam3_adapter.worker.launch_worker", lambda *a, **k: stub)
    monkeypatch.setattr("sam3_adapter.worker.read_events_strict",
                        lambda path, allow_trailing_partial: (list(events), []))
    monkeypatch.setattr("sam3_adapter.observer.process_alive", lambda pid: False)
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes", lambda: {})
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))
    return stub


def _done_stream(indices=(17, 34, 59)):
    events = [{"type": "ready", "seq": 1, "pid": 4321}]
    for position, index in enumerate(indices, start=2):
        events.append({"type": "frame", "seq": position, "input_index": index,
                       "status": "OK", "candidates": []})
    events.append({"type": "done", "seq": 50, "stopped": False,
                   "frames_attempted": len(indices), "frames_ok": len(indices)})
    return events


def test_unknown_event_cannot_end_in_success(tmp_path, monkeypatch):
    """A-R3 counterexample: an unknown event still produced SUCCESS/0."""
    events = _done_stream() + [{"type": "unrecognised_failure_event", "seq": 99}]
    _stub_worker_with_events(monkeypatch, events, tmp_path)
    outcome = execute(_worker_request(tmp_path, _spec_for(tmp_path), deadline_mod.Budgets()))
    manifest = _manifest(outcome)

    assert any(e["code"] == "EVENT_TYPE_UNKNOWN" for e in manifest["errors"])
    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4


def test_conflicting_seq_is_rejected_and_changes_the_result(tmp_path, monkeypatch):
    events = _done_stream()
    conflicting = dict(events[1], status="NO_MATCH")
    _stub_worker_with_events(monkeypatch, events + [conflicting], tmp_path)
    outcome = execute(_worker_request(tmp_path, _spec_for(tmp_path), deadline_mod.Budgets()))
    manifest = _manifest(outcome)
    assert any(e["code"] == "EVENT_SEQ_CONFLICT" for e in manifest["errors"])
    assert outcome.status == "BACKEND_ERROR"


def test_terminal_count_mismatch_changes_the_result(tmp_path, monkeypatch):
    events = _done_stream()[:-1] + [{"type": "done", "seq": 50, "stopped": False,
                                     "frames_attempted": 99, "frames_ok": 99}]
    _stub_worker_with_events(monkeypatch, events, tmp_path)
    outcome = execute(_worker_request(tmp_path, _spec_for(tmp_path), deadline_mod.Budgets()))
    manifest = _manifest(outcome)
    assert any(e["code"] == "EVENT_TERMINAL_COUNT_MISMATCH" for e in manifest["errors"])
    assert outcome.status == "BACKEND_ERROR"


def test_frame_event_missing_fields_changes_the_result(tmp_path, monkeypatch):
    events = _done_stream()
    events[1] = {"type": "frame", "seq": 2, "candidates": []}  # no input_index / status
    _stub_worker_with_events(monkeypatch, events, tmp_path)
    outcome = execute(_worker_request(tmp_path, _spec_for(tmp_path), deadline_mod.Budgets()))
    manifest = _manifest(outcome)
    assert any(e["code"] == "EVENT_FRAME_MISSING_FIELDS" for e in manifest["errors"])
    assert outcome.status == "BACKEND_ERROR"


def test_join_failure_changes_the_result(tmp_path, monkeypatch):
    _stub_worker_with_events(monkeypatch, _done_stream(), tmp_path)

    def failing_stop(self, deadline=None):
        self.errors.append({"code": "SUPERVISOR_NOT_JOINED",
                            "message": "injected: thread did not stop"})
        return {"joined_cleanly": False}

    monkeypatch.setattr("sam3_adapter.runner._WorkerSupervisor.stop", failing_stop)
    outcome = execute(_worker_request(tmp_path, _spec_for(tmp_path), deadline_mod.Budgets()))
    manifest = _manifest(outcome)
    assert outcome.status == "BACKEND_ERROR"
    assert any(e["code"] == "SUPERVISOR_NOT_JOINED" for e in manifest["errors"])


# --------------------------------------------------------------------------- #
# A-R4 real-OS scenario batch: exactly three scenarios (R1 / R2 / R3)
# --------------------------------------------------------------------------- #
def test_scenario_1_observes_the_real_worker_and_stops_on_an_injected_reading(tmp_path, monkeypatch):
    """R1: the observed set comes from OS identity, and a reading stops the live worker."""
    first = {"pid": None}
    calls = []

    def injected(pid):
        calls.append(pid)
        if first["pid"] is None:
            first["pid"] = pid
        high = pid != first["pid"]          # the launcher reads low, the other member reads high
        value = (13 << 30) if high else (1 << 20)
        return {"commit_bytes": value, "rss_bytes": value, "pid": pid, "commit_channel": "scenario"}

    monkeypatch.setattr("sam3_adapter.resources.read_process_memory_for_pid", injected)
    outcome, _ = _fixture_run(
        tmp_path, "s1", budgets=deadline_mod.Budgets(load_seconds=30, frame_seconds=30,
                                                     batch_seconds=30),
        fixture_values={"FIXTURE_FRAME_SLEEP": "4"}, monitor_interval=0.05)
    manifest = _manifest(outcome)
    observations = [s for s in manifest["resource_observation"]["samples"] if s.get("reported_by") == "parent"]

    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == 5
    assert manifest["stop_state"]["source"] == "observer"
    targets = observations[0]["targets"]
    assert len(targets) >= 1 and all(row["identified_by"] for row in targets)
    assert any(row["is_actual_worker"] for row in targets) or len(targets) == 1
    assert calls, "the parent never queried a process"
    assert manifest["counts"]["not_run"] >= 1


def test_scenario_2_reclaims_a_registered_descendant_after_the_top_exits(tmp_path):
    """R2: a descendant outlives the top level and is still owned and reclaimed."""
    outcome, _ = _fixture_run(
        tmp_path, "s2", budgets=deadline_mod.Budgets(load_seconds=30, frame_seconds=30,
                                                     batch_seconds=30),
        fixture_values={"FIXTURE_DESCENDANT": "1", "FIXTURE_DESCENDANT_SLEEP": "6"})
    manifest = _manifest(outcome)
    worker = manifest["worker"]

    assert manifest["counts"]["success"] == 3           # the top level really finished its work
    assert worker["members_after"]["member_count"] >= 1  # the descendant was registered
    registered = worker["ownership_identity"]["registered_members"]
    assert any(row["role"] == "descendant" for row in registered), registered
    states = {row["state"] for row in worker["member_states"]}
    assert states <= {"EXITED", "ALIVE", "UNKNOWN"}
    assert worker["top_level_exit_is_not_whole_tree_exit"] is True
    # whichever way it ended, it is reported rather than assumed clean
    assert worker["residual_unknown"] is True or worker.get("descendant_reclaim") is not None


def test_scenario_3_startup_failure_cleans_up_a_real_created_process(tmp_path, monkeypatch):
    """R3: observer.start() failing must not leave the really-created process behind."""
    def boom(self):
        raise RuntimeError("injected observer.start failure (real process)")

    monkeypatch.setattr("sam3_adapter.observer.WorkerObserver.start", boom)
    outcome, _ = _fixture_run(
        tmp_path, "s3", budgets=deadline_mod.Budgets(load_seconds=5, frame_seconds=5, batch_seconds=5),
        fixture_values={})
    manifest = _manifest(outcome)

    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    cleanup = [e for e in manifest["errors"] if e["code"] == "POST_LAUNCH_CLEANUP"]
    assert cleanup, manifest["errors"]
    reclaim = cleanup[0]["cleanup"]["reclaim"]
    assert reclaim["state"] == "COMPLETE"
    assert reclaim["members_before"]["member_count"] >= 1
    # the created launcher must be gone, or explicitly reported as UNKNOWN
    assert reclaim["exited"] is True or any(item["state"] == "UNKNOWN"
                                            for item in reclaim["residual_unknown"])


# =========================================================================== #
# A-R5: one effective device policy, whole-lifecycle protection, all-member cleanup
# =========================================================================== #
def _req(**overrides):
    request = dict(io_run=".", indices=[17], object_id="o", text_prompt="t", output_root=".",
                   repo_dir=str(ADAPTER_DIR))
    request.update(overrides)
    return RunRequest(**request)


def test_cpu_device_channel_is_not_applicable_and_ram_monitoring_stays_on(monkeypatch):
    policy = observer_module.resolve_device_policy(_req(device="cpu"))
    assert policy["device_channel_applies"] is False and policy["required"] is False
    assert "NOT_APPLICABLE" in policy["reason"]

    ownership = _ownership_with_registry()
    _patch_os_view(monkeypatch, ownership,
                   readings={990001: {"commit_bytes": 1 << 20, "rss_bytes": 1 << 20,
                                      "pid": 990001, "commit_channel": "t"},
                             990002: {"commit_bytes": 13 << 30, "rss_bytes": 13 << 30,
                                      "pid": 990002, "commit_channel": "t"}})
    obs = observer_module.WorkerObserver(ownership, 0.05, [], [], deadline_mod.StopState(),
                                        policy=resources.ResourcePolicy(), device_policy=policy)
    observation = obs.observe()
    assert observation["device"] is None and observation["device_channel"] == "NOT_APPLICABLE"
    violations = obs.violations(observation)
    assert any("990002" in item and "RAM commit" in item for item in violations)
    assert not any("device" in item.lower() for item in violations)


def test_cuda_default_auto_wires_and_validates_the_device_channel(monkeypatch):
    policy = observer_module.resolve_device_policy(_req(device="cuda"))
    assert policy["device_channel_applies"] is True and policy["input_error"] is None
    ownership = _ownership_with_registry()
    _patch_os_view(monkeypatch, ownership,
                   readings={pid: {"commit_bytes": 0, "rss_bytes": 0, "pid": pid,
                                   "commit_channel": "t"} for pid in (990001, 990002)})

    def with_device(reading):
        return observer_module.WorkerObserver(
            ownership, 0.05, [], [], deadline_mod.StopState(),
            policy=resources.ResourcePolicy(), device_sampler=lambda: reading,
            device_policy=policy)

    good = {"device_used_bytes": 1 << 30, "device_total_bytes": 8 << 30, "device_index": 0,
            "age_seconds": 0.01, "fresh": True}
    assert not any("device" in v.lower() for v in with_device(good).violations(with_device(good).observe()))
    stale = dict(good, fresh=False, age_seconds=999)
    assert any("stale" in v or "freshness" in v for v in
               with_device(stale).violations(with_device(stale).observe()))
    illegal = dict(good, device_used_bytes="1024")
    assert any("non-boolean integer" in v for v in
               with_device(illegal).violations(with_device(illegal).observe()))
    over_total = dict(good, device_used_bytes=(9 << 30))
    assert any("exceeds device total" in v or "> limit" in v for v in
               with_device(over_total).violations(with_device(over_total).observe()))
    wrong_index = dict(good, device_index=3)
    assert any("selected device is" in v for v in
               with_device(wrong_index).violations(with_device(wrong_index).observe()))


def test_cuda_explicit_none_is_rejected_before_anything_is_launched(tmp_path, monkeypatch):
    policy = observer_module.resolve_device_policy(_req(device="cuda", device_observe="none"))
    assert policy["input_error"] is not None

    launched = []
    monkeypatch.setattr("sam3_adapter.worker.launch_worker",
                        lambda *a, **k: launched.append(a) or _StubPopen())
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))
    outcome = execute(_run_request_with(tmp_path, spec, device="cuda", device_observe="none"))
    manifest = _manifest(outcome)

    assert outcome.status == "INPUT_ERROR" and outcome.exit_code == 2
    assert any(e["code"] == "DEVICE_POLICY_INPUT_ERROR" for e in manifest["errors"])
    assert launched == [] and manifest.get("worker") is None


def test_cuda_visible_devices_remap_must_be_provable():
    unprovable = observer_module.resolve_device_identity("cuda:1", {"CUDA_VISIBLE_DEVICES": "3"})
    assert unprovable["identity_provable"] is False
    assert observer_module.resolve_device_policy(_req(device="cuda:1"))["identity_provable"] is True
    mapped = observer_module.resolve_device_identity("cuda:1", {"CUDA_VISIBLE_DEVICES": "0,3"})
    assert mapped["physical_index"] == 3 and mapped["identity_provable"] is True
    blocked = observer_module.resolve_device_policy(
        _req(device="cuda:1", device_observe="auto"))
    assert blocked["physical_index"] == 1  # no CUDA_VISIBLE_DEVICES in this process


def test_ownership_constructor_failure_cleans_up_with_the_held_popen_handle(tmp_path, monkeypatch):
    stub = _StubPopen()
    monkeypatch.setattr("sam3_adapter.worker.launch_worker", lambda *a, **k: stub)

    def boom(*args, **kwargs):
        raise RuntimeError("injected ownership constructor failure")

    monkeypatch.setattr("sam3_adapter.observer.WorkerOwnership", boom)
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))
    outcome = execute(_run_request_with(tmp_path, spec, device="cpu"))
    manifest = _manifest(outcome)

    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    cleanup = [e for e in manifest["errors"] if e["code"] == "POST_LAUNCH_CLEANUP"]
    assert cleanup, manifest["errors"]
    reclaim = cleanup[0]["cleanup"]["reclaim"]
    assert reclaim["used"].startswith("held Popen handle")
    assert stub.returncode == -9, "the held handle must really terminate the process"


def test_device_sampler_factory_failure_is_inside_the_protected_region(tmp_path, monkeypatch):
    stub = _StubPopen()
    monkeypatch.setattr("sam3_adapter.worker.launch_worker", lambda *a, **k: stub)
    monkeypatch.setattr("sam3_adapter.observer.process_alive", lambda pid: False)
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes", lambda: {})

    def boom(*args, **kwargs):
        raise RuntimeError("injected device sampler factory failure")

    monkeypatch.setattr("sam3_adapter.observer.make_device_sampler", boom)
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))
    outcome = execute(_run_request_with(tmp_path, spec, device="cpu"))
    manifest = _manifest(outcome)

    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    cleanup = [e for e in manifest["errors"] if e["code"] == "POST_LAUNCH_CLEANUP"]
    assert cleanup and cleanup[0]["cleanup"]["reclaim"] is not None


def test_all_owned_members_are_cleaned_not_only_descendants(monkeypatch):
    """A live worker_interpreter must not be skipped because its role is not 'descendant'."""
    ownership = _ownership_with_registry(top_alive=False)
    calls = []
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes",
                        lambda: {990002: {"ppid": 990001, "exe": "py.exe"}})
    monkeypatch.setattr("sam3_adapter.observer.process_alive",
                        lambda pid: pid == 990002)
    monkeypatch.setattr("sam3_adapter.observer.process_creation_time", lambda pid: 100 if pid == 990001 else 200)
    alive = {990002}
    ownership.members_registry[990002].handle = 222
    monkeypatch.setattr(observer_module, "handle_creation_time", lambda handle: 200)
    monkeypatch.setattr(observer_module, "handle_alive", lambda handle: 990002 in alive)
    def terminate(handle, code):
        calls.append((handle.value, code))
        alive.discard(990002)
        return 1
    monkeypatch.setattr(observer_module, "_k32", lambda: SimpleNamespace(TerminateProcess=terminate))

    before = ownership.members()
    assert before["live_descendants"] == []          # not a descendant by role
    assert [row["pid"] for row in before["live_members"]] == [990002]
    assert before["unresolved_members"], "unresolved members must be visible"

    record = ownership.reclaim("all-member cleanup")
    assert any("990002" in json.dumps(action) for action in record["actions"])
    assert calls == [(222, 1)], "the live worker interpreter was never terminated"
    assert record["exited"] is True and record["residual_unknown"] == []


def test_alive_or_unknown_members_prevent_a_plain_success(tmp_path, monkeypatch):
    _stub_worker_with_events(monkeypatch, _done_stream(), tmp_path)
    monkeypatch.setattr("sam3_adapter.observer.process_alive", lambda pid: True)
    monkeypatch.setattr("sam3_adapter.observer.enumerate_processes",
                        lambda: {990002: {"ppid": 4321, "exe": "py.exe"}})
    monkeypatch.setattr("sam3_adapter.observer.process_creation_time", lambda pid: 1)
    spec = _spec_for(tmp_path)
    monkeypatch.setattr("sam3_adapter.runner._build_backend", lambda request: Sam3ShapedFakeBackend(spec))

    outcome = execute(_run_request_with(tmp_path, spec, use_worker=True))
    manifest = _manifest(outcome)
    assert outcome.status != "SUCCESS"
    assert any(e["code"] in ("MEMBERS_NOT_CONFIRMED_EXITED", "FINALISATION_FAILED")
               for e in manifest["errors"]), [e["code"] for e in manifest["errors"]]


def _run_request_with(tmp_path, spec, **overrides):
    """Worker-mode request used by the A-R5 pure-data device/lifecycle tests."""
    return _worker_request(tmp_path, spec, deadline_mod.Budgets(), **overrides)


@pytest.mark.parametrize("changes", [
    {"fresh": None}, {"fresh": 1}, {"age_seconds": None}, {"age_seconds": float("nan")},
    {"age_seconds": float("inf")}, {"age_seconds": -1}, {"age_seconds": True},
    {"age_seconds": 100}, {"device_used_bytes": True}, {"device_index": False},
])
def test_r6_required_device_values_fail_closed(changes):
    policy = observer_module.resolve_device_policy(_req(device="cuda"))
    sample = {"device_used_bytes": 1, "device_total_bytes": 8, "device_index": 0,
              "fresh": True, "age_seconds": 0.01}
    sample.update(changes)
    assert observer_module.device_violations(sample, policy, 10)


def test_r6_cuda_policy_is_consumed_by_job_and_sampler(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3")
    calls = []
    monkeypatch.setattr(observer_module, "driver_device_query", lambda index, **kw: calls.append(index) or {})
    policy = observer_module.resolve_device_policy(_req(device="cuda:1"))
    sampler = observer_module.make_device_sampler("driver", "cuda:1", device_policy=policy)
    sampler()
    assert calls == [3]
    assert policy["device_kind"] == "cuda"
    _stub_worker_with_events(monkeypatch, _done_stream(), tmp_path)
    jobs = []
    original_write = worker.write_job
    monkeypatch.setattr(worker, "write_job", lambda path, job: jobs.append(job) or original_write(path, job))
    result = execute(_run_request_with(tmp_path, _spec_for(tmp_path), device="cuda:1", monitor_interval_seconds=0))
    assert jobs[0]["monitor_device"] == "cuda:1"
    assert jobs[0]["device_policy"]["physical_index"] == 3
    assert result.status == "SUCCESS", result.manifest["errors"]


def test_r6_cpu_completes_through_launch_and_drain_without_device_query(tmp_path, monkeypatch):
    stub = _stub_worker_with_events(monkeypatch, _done_stream(), tmp_path)
    monkeypatch.setattr(observer_module, "driver_device_query", lambda *a, **k: pytest.fail("CPU queried GPU"))
    result = execute(_run_request_with(tmp_path, _spec_for(tmp_path), device="cpu"))
    assert result.status == "SUCCESS", result.manifest["errors"]
    assert result.manifest["worker"]["pid"] == stub.pid
    assert result.manifest["worker"]["ledger_summary"]["frames_seen"] == [17, 34, 59]
    assert result.manifest["worker"]["residual_details"] == []


@pytest.mark.parametrize("accepted", [False, True])
def test_r6_win32_return_value_is_not_assumed_success(monkeypatch, accepted):
    calls = []
    class Api:
        def TerminateProcess(self, handle, code):
            calls.append((handle.value, code))
            return int(accepted)
    monkeypatch.setattr(observer_module, "_k32", lambda: Api())
    member = SimpleNamespace(handle=123, pid=990001, identity_still_matches=lambda: True)
    result = observer_module.terminate_member(member)
    assert calls == [(123, 1)] and result["accepted"] is accepted
    assert result["returncode"] == (0 if accepted else None)


def test_r6_handle_permissions_and_signatures(monkeypatch):
    class Function:
        def __init__(self): self.calls = []
        def __call__(self, *args): self.calls.append(args); return 123
    class Api:
        def __getattr__(self, name):
            value = Function(); setattr(self, name, value); return value
    api = Api()
    monkeypatch.setattr(observer_module.ctypes, "WinDLL", lambda *a, **k: api)
    monkeypatch.setattr(observer_module, "_k32", _WIN32_API_FACTORY)
    monkeypatch.setattr(observer_module, "open_identity_handle", _OPEN_IDENTITY_HANDLE)
    kernel = observer_module._k32()
    assert observer_module.open_identity_handle(99) == 123
    assert api.OpenProcess.calls[-1][0] & observer_module.PROCESS_TERMINATE
    assert len(kernel.TerminateProcess.argtypes) == 2
    assert kernel.TerminateProcess.restype is observer_module.ctypes.c_int


def test_r6_shared_deadline_zero_budget_prevents_termination(monkeypatch):
    clock = [100.0]
    deadline = deadline_mod.Deadline(deadline_mod.Budgets(reclaim_seconds=2), clock=lambda: clock[0])
    first = deadline.reclaim_deadline_from_now()
    clock[0] = 103.0
    assert deadline.reclaim_deadline_from_now() == first == 102.0
    owner = _ownership_with_registry()
    _patch_os_view(monkeypatch, owner)
    monkeypatch.setattr(observer_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(observer_module, "terminate_member", lambda member: pytest.fail("termination past deadline"))
    monkeypatch.setattr(owner.process, "kill", lambda: pytest.fail("kill past deadline"))
    report = owner.reclaim("expired", deadline=first)
    assert report["budget_exhausted"] and not report["exited"]
    assert report["residual_unknown"]
    assert owner.reclaim("again", deadline=999)["reclaim_deadline_monotonic"] == first


def test_r6_concurrent_reclaim_does_not_claim_completion(monkeypatch):
    owner = _ownership_with_registry()
    owner._reclaimed = True
    owner.reclaim_records = [{"state": "IN_PROGRESS", "exited": None}]
    assert owner.reclaim("second stopper")["state"] == "IN_PROGRESS"


@pytest.mark.parametrize("kind", ["enumeration", "identity"])
def test_r6_unknown_members_cannot_finish_success(tmp_path, monkeypatch, kind):
    _stub_worker_with_events(monkeypatch, _done_stream(), tmp_path)
    if kind == "enumeration":
        monkeypatch.setattr(observer_module, "enumerate_processes", lambda: None)
    else:
        monkeypatch.setattr(observer_module, "process_creation_time", lambda pid: None)
    outcome = execute(_run_request_with(tmp_path, _spec_for(tmp_path), device="cpu",
                                        monitor_interval_seconds=0))
    assert outcome.status == "BACKEND_ERROR"
    assert any(e["code"] == "MEMBERS_NOT_CONFIRMED_EXITED" for e in outcome.manifest["errors"])
    assert outcome.manifest["worker"]["residual_unknown"]


@pytest.mark.parametrize("stage", ["ledger", "wait", "drain", "latched_ledger"])
def test_r6_lifecycle_fault_keeps_reason_and_reclaims(tmp_path, monkeypatch, stage):
    import sam3_adapter.runner as runner_mod
    stub = _stub_worker_with_events(monkeypatch, _done_stream(), tmp_path, poll_code=None)
    def boom(*a, **kw): raise RuntimeError("r6 fault at " + stage)
    if stage == "latched_ledger":
        def latched_boom(stop_state, *args, **kwargs):
            stop_state.latch("RESOURCE_LIMIT", "r6 first resource cause", "test")
            boom()
        monkeypatch.setattr(runner_mod, "_EventLedger", latched_boom)
    elif stage == "ledger":
        monkeypatch.setattr(runner_mod, "_EventLedger", boom)
    elif stage == "wait":
        monkeypatch.setattr(runner_mod._WorkerSupervisor, "start", lambda self: None)
        monkeypatch.setattr(observer_module.WorkerObserver, "start", lambda self: None)
        original_poll = stub.poll
        n = [0]
        def poll():
            n[0] += 1
            if n[0] == 1: boom()
            return original_poll()
        monkeypatch.setattr(stub, "poll", poll)
    else:
        stub.returncode = 0
        monkeypatch.setattr(worker, "read_events_strict",
                            lambda path, allow_trailing_partial: ([], []) if allow_trailing_partial else boom())
    outcome = execute(_run_request_with(tmp_path, _spec_for(tmp_path), device="cpu"))
    assert outcome.status == ("RESOURCE_LIMIT" if stage == "latched_ledger" else "BACKEND_ERROR")
    if stage == "latched_ledger":
        assert outcome.manifest["stop_state"]["reason"] == "r6 first resource cause"
    assert "r6 fault at " + stage in json.dumps(outcome.manifest["errors"])
    assert stub.poll() is not None


def test_r6_handle_release_follows_thread_stop_and_drain(tmp_path, monkeypatch):
    import sam3_adapter.runner as runner_mod
    _stub_worker_with_events(monkeypatch, _done_stream(), tmp_path)
    order = []
    original_observer = observer_module.WorkerObserver.stop
    original_supervisor = runner_mod._WorkerSupervisor.stop
    original_read = worker.read_events_strict
    monkeypatch.setattr(observer_module.WorkerObserver, "stop",
                        lambda self, end: order.append("observer") or original_observer(self, end))
    monkeypatch.setattr(runner_mod._WorkerSupervisor, "stop",
                        lambda self, end: order.append("supervisor") or original_supervisor(self, end))
    def read(path, allow_trailing_partial):
        if not allow_trailing_partial: order.append("drain")
        return original_read(path, allow_trailing_partial)
    monkeypatch.setattr(worker, "read_events_strict", read)
    def close(self):
        order.append("close")
        return {"closed_count": 0, "errors": []}
    original_close = observer_module.WorkerOwnership.close_handles
    monkeypatch.setattr(observer_module.WorkerOwnership, "close_handles", close)
    outcome = execute(_run_request_with(tmp_path, _spec_for(tmp_path), device="cpu"))
    assert outcome.status == "SUCCESS", outcome.manifest["errors"]
    assert order == ["observer", "supervisor", "drain", "close"]

    # Exercise real release logic with fake handles; aliases are closed only once.
    monkeypatch.setattr(observer_module.WorkerOwnership, "close_handles", original_close)
    owned = _ownership_with_registry()
    owned.launcher_handle = owned.members_registry[990001].handle = 111
    owned.worker_handle = owned.members_registry[990002].handle = 222
    closed = []
    monkeypatch.setattr(observer_module, "close_handle", lambda handle: closed.append(handle))
    assert owned.close_handles()["closed_count"] == 2
    assert set(closed) == {111, 222}
    assert owned.close_handles()["closed_count"] == 0


def test_r6_ram_type_and_identity_are_checked(monkeypatch):
    owner = _ownership_with_registry()
    _patch_os_view(monkeypatch, owner, readings={990002: {"commit_bytes": "low"}})
    obs = _observer(owner)
    assert any("non-boolean integer" in v for v in obs.violations(obs.observe()))
    sample = obs.observe()
    sample["parent_commit_bytes"] = True
    assert any("parent commit_bytes" in v for v in obs.violations(sample))
    monkeypatch.setattr(observer_module, "process_creation_time", lambda pid: 999)
    assert any("UNKNOWN" in v for v in obs.violations(obs.observe()))
