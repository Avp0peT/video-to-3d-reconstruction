"""photo-io-v1 run reader.

Identity is taken exclusively from ``frames.jsonl`` (explicit ``input_index``
lookup); nothing is inferred from directory order or filename sorting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import imagery
from .contract import IMAGE_DOMAIN, PHOTO_IO_CONTRACT_VERSION


class SourceIoError(ValueError):
    """Raised when the upstream photo-io-v1 run cannot be bound."""


@dataclass
class SourceFrame:
    input_index: int
    frame_id: str | None
    image_output: str | None
    source_image: Path | None
    source_image_sha256: str | None
    photo_io_image_sha256: str | None
    width: int | None
    height: int | None
    error: dict | None = None


@dataclass
class SourceIo:
    path: Path
    run_id: str | None
    run_json_bytes: bytes
    frames_jsonl_bytes: bytes
    lines: list[dict] = field(default_factory=list)

    @property
    def run_json_sha256(self) -> str:
        return imagery.sha256_bytes(self.run_json_bytes)

    @property
    def frames_jsonl_sha256(self) -> str:
        return imagery.sha256_bytes(self.frames_jsonl_bytes)

    def binding(self) -> dict:
        return {
            "path": str(self.path),
            "run_id": self.run_id,
            "run_json_sha256": self.run_json_sha256,
            "frames_jsonl_sha256": self.frames_jsonl_sha256,
        }

    def fingerprint(self) -> tuple[str, str]:
        """Current on-disk digest pair, used for the before/after invariant."""
        return (
            imagery.sha256_file(self.path / "run.json"),
            imagery.sha256_file(self.path / "frames.jsonl"),
        )


def load_source_io(io_run: str | Path) -> SourceIo:
    root = Path(io_run).expanduser().resolve()
    if not root.is_dir():
        raise SourceIoError(f"io-run directory not found: {root}")
    run_json = root / "run.json"
    frames_jsonl = root / "frames.jsonl"
    for required in (run_json, frames_jsonl):
        if not required.is_file():
            raise SourceIoError(f"missing required upstream file: {required}")

    run_json_bytes = run_json.read_bytes()
    frames_jsonl_bytes = frames_jsonl.read_bytes()
    try:
        run_meta = json.loads(run_json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIoError(f"run.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(run_meta, dict):
        raise SourceIoError(
            f"run.json top level must be a JSON object, got {type(run_meta).__name__}"
        )

    version = run_meta.get("contract_version")
    if version != PHOTO_IO_CONTRACT_VERSION:
        raise SourceIoError(f"run.json contract_version is {version!r}, expected {PHOTO_IO_CONTRACT_VERSION!r}")
    if run_meta.get("status") != "SUCCESS":
        raise SourceIoError(f"upstream photo-io-v1 run status is {run_meta.get('status')!r}, not SUCCESS")

    lines: list[dict] = []
    for lineno, raw_line in enumerate(frames_jsonl_bytes.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SourceIoError(f"frames.jsonl line {lineno} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("input_index"), int):
            raise SourceIoError(f"frames.jsonl line {lineno} has no integer input_index")
        lines.append(record)

    indices = [record["input_index"] for record in lines]
    if len(set(indices)) != len(indices):
        raise SourceIoError("frames.jsonl contains duplicate input_index values")

    return SourceIo(
        path=root,
        run_id=run_meta.get("run_id"),
        run_json_bytes=run_json_bytes,
        frames_jsonl_bytes=frames_jsonl_bytes,
        lines=lines,
    )


def _contained(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_frame(source: SourceIo, input_index: int) -> SourceFrame:
    """Resolve one requested input index into a bound SourceFrame."""
    record = next((line for line in source.lines if line["input_index"] == input_index), None)
    if record is None:
        return SourceFrame(
            input_index=input_index,
            frame_id=None,
            image_output=None,
            source_image=None,
            source_image_sha256=None,
            photo_io_image_sha256=None,
            width=None,
            height=None,
            error={"code": "INPUT_NOT_IN_UPSTREAM", "message": f"input_index {input_index} absent from frames.jsonl"},
        )

    frame_id = record.get("frame_id")
    photo_hash = record.get("image_sha256")
    width = record.get("canonical_width")
    height = record.get("canonical_height")
    image_output = record.get("image_output")
    if record.get("status") != "OK":
        return SourceFrame(input_index, frame_id, image_output, None, None, photo_hash, width, height,
                           {"code": "UPSTREAM_FRAME_NOT_OK", "message": f"frames.jsonl status={record.get('status')!r}"})
    if not isinstance(image_output, str) or not image_output:
        return SourceFrame(input_index, frame_id, image_output, None, None, photo_hash, width, height,
                           {"code": "UPSTREAM_NO_IMAGE_OUTPUT", "message": "frames.jsonl image_output is null"})
    if not isinstance(width, int) or not isinstance(height, int):
        return SourceFrame(input_index, frame_id, image_output, None, None, photo_hash, width, height,
                           {"code": "UPSTREAM_NO_CANONICAL_SIZE", "message": "canonical_width/height missing"})

    candidate = (source.path / image_output).resolve()
    if not _contained(source.path, candidate):
        return SourceFrame(input_index, frame_id, image_output, None, None, photo_hash, width, height,
                           {"code": "INPUT_PATH_ESCAPES_IO_RUN", "message": image_output})
    if not candidate.is_file():
        return SourceFrame(input_index, frame_id, image_output, None, None, photo_hash, width, height,
                           {"code": "MISSING_SOURCE_IMAGE", "message": str(candidate)})

    try:
        actual_w, actual_h, mode = imagery.read_canonical_rgb(candidate)
    except imagery.ImageError as exc:
        return SourceFrame(input_index, frame_id, image_output, None, None, photo_hash, width, height,
                           {"code": "SOURCE_IMAGE_NOT_CANONICAL_RGB8", "message": str(exc)})
    if (actual_w, actual_h) != (width, height):
        return SourceFrame(input_index, frame_id, image_output, None, None, photo_hash, width, height,
                           {"code": "SOURCE_IMAGE_SIZE_MISMATCH",
                            "message": f"file {(actual_w, actual_h)} != frames.jsonl {(width, height)}"})

    return SourceFrame(
        input_index=input_index,
        frame_id=frame_id,
        image_output=image_output,
        source_image=candidate,
        source_image_sha256=imagery.sha256_file(candidate),
        photo_io_image_sha256=photo_hash,
        width=width,
        height=height,
        error=None,
    )


def frame_domain() -> str:
    return IMAGE_DOMAIN
