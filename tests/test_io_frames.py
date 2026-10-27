"""``photo-io-v1`` photo-side contract tests (T01A).

What this file is trying to prove
---------------------------------
Not "the code runs", but that the *contract* holds — and specifically that the
checks could have failed.  Three techniques are used deliberately:

* **Independent expressions of the same rule.**  The EXIF orientation table is
  expressed three ways: the stored matrix, the NumPy pixel transform, and a
  literal table transcribed here.  Any single implementation could be wrong in a
  self-consistent way; three are much harder to get wrong identically.  The
  oracle test also asserts it *can* return a negative, so "the oracle agrees" is
  not a constant ``True``.
* **Asymmetric fixtures.**  A symmetric image (or a uniform one) survives a
  wrong rotation and would prove nothing.  Every geometric assertion uses a
  gradient frame with one unique marker pixel.
* **Error codes, not just failures.**  Asserting only "it errored" would pass if
  the wrong branch fired.  Each rejection test pins the contract code.

Sandbox projects
----------------
Most tests build a throwaway *project root* (a directory with ``pyproject.toml``
and ``outputs/delegation/T01A/fixtures/<id>/``) and point the CLI at it.  This
keeps the real repository clean and, because the contract pins ``dataset_root``
to an exact string, means nothing has to be relaxed to make tests possible.
"""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import zlib
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from kozlov_lab.io import cli
from kozlov_lab.io import runner as io_runner
from kozlov_lab.io.hashing import (
    CONTRACT_VERSION,
    compute_cache_fingerprint,
    compute_frame_id,
)
from kozlov_lab.io.images import SUPPORTED_FORMATS, as_uint8_rgb, decode_image, encode_rgb_png
from kozlov_lab.io.manifest import ManifestError, expected_dataset_root, parse_manifest
from kozlov_lab.io.orientation import (
    VALID_ORIENTATIONS,
    apply_orientation,
    canonical_size,
    matrix_matches_transform,
    orientation_matrix,
)
from kozlov_lab.io.paths import is_valid_dataset_id
from kozlov_lab.io.source_encoding import (
    SourceEncodingError,
    assert_eight_bit_source,
    jpeg_frame,
    png_header,
    source_bit_depth,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

MARKER = 255


# --------------------------------------------------------------------------- #
# helpers: fixtures, contract-table transcription, CLI driving
# --------------------------------------------------------------------------- #
def gradient_image(width: int, height: int, marker: tuple[int, int] | None, mode: str) -> np.ndarray:
    """Asymmetric frame: a low-valued gradient plus one unique marker pixel."""
    rows, cols = np.indices((height, width))
    grey = ((rows * width + cols) % 200).astype(np.uint8)
    if marker is not None:
        grey[marker[1], marker[0]] = MARKER
    if mode == "RGB":
        return np.repeat(grey[:, :, None], 3, axis=2)
    return grey


def encode_with_exif(array: np.ndarray, container: str = "PNG", exif: int | None = None) -> bytes:
    """Serialise ``array`` to image bytes, optionally attaching EXIF Orientation.

    The single place that knows how to write an orientation tag; every fixture
    builder routes through it so two copies cannot drift apart.
    """
    image = Image.fromarray(array)
    kwargs = {}
    if exif is not None:
        block = Image.Exif()
        block[274] = exif
        kwargs["exif"] = block
    buffer = BytesIO()
    image.save(buffer, format=container, **kwargs)
    return buffer.getvalue()


def save_image(path: pathlib.Path, array: np.ndarray, container: str = "PNG", exif: int | None = None) -> None:
    path.write_bytes(encode_with_exif(array, container, exif))


def save_mask(
    path: pathlib.Path,
    width: int,
    height: int,
    marker: tuple[int, int] | None = None,
    *,
    full: int = 255,
) -> None:
    """A manual mask in raw coordinates; ``full`` selects the foreground value."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    if marker is not None:
        canvas[marker[1], marker[0]] = 1
    if full == 255:
        canvas = np.where(canvas != 0, 255, 0).astype(np.uint8)
    Image.fromarray(canvas).save(path, format="PNG")


def contract_raw_to_canonical(
    orientation: int, x: int, y: int, width: int, height: int
) -> tuple[int, int, tuple[int, int]]:
    """Literal transcription of the contract's orientation table (§3).

    Written out longhand on purpose: it is the *expected* value that the stored
    matrix and the NumPy transform are each compared against.
    """
    if orientation == 1:
        return x, y, (width, height)
    if orientation == 2:
        return width - 1 - x, y, (width, height)
    if orientation == 3:
        return width - 1 - x, height - 1 - y, (width, height)
    if orientation == 4:
        return x, height - 1 - y, (width, height)
    if orientation == 5:
        return y, x, (height, width)
    if orientation == 6:
        return height - 1 - y, x, (height, width)
    if orientation == 7:
        return height - 1 - y, width - 1 - x, (height, width)
    if orientation == 8:
        return y, width - 1 - x, (height, width)
    raise AssertionError(f"unexpected orientation {orientation}")


@pytest.fixture
def sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    """A throwaway project root shaped like the real one."""
    root = tmp_path / "proj"
    (root / "outputs").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "sandbox"\n', encoding="utf-8")
    return root


def dataset_dir(sandbox: pathlib.Path, dataset_id: str) -> pathlib.Path:
    directory = sandbox / "outputs/delegation/T01A/fixtures" / dataset_id
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "masks").mkdir(parents=True, exist_ok=True)
    return directory


def write_manifest(
    sandbox: pathlib.Path,
    dataset_id: str,
    frames: list[dict],
    *,
    data_kind: str = "synthetic",
    overrides: dict | None = None,
    name: str = "manifest.json",
) -> pathlib.Path:
    document = {
        "contract_version": CONTRACT_VERSION,
        "dataset_id": dataset_id,
        "data_kind": data_kind,
        "dataset_root": expected_dataset_root(data_kind, dataset_id),
        "frames": frames,
    }
    if overrides is not None:
        document.update(overrides)
    path = sandbox / name
    # ``ensure_ascii=True`` on purpose.  A manifest may legitimately contain an
    # unpaired surrogate (e.g. ``"\ud800"`` as a path), and only the escaped form
    # can be written as UTF-8 at all: with ``ensure_ascii=False`` this helper
    # would raise before the CLI ever saw the file, so the case could not be
    # tested.  ``json.loads`` decodes the escapes back to the same strings.
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def run_cli(
    sandbox: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: pathlib.Path,
    *,
    data_kind: str = "synthetic",
    output: str = "outputs/delegation/T01A/runs",
) -> tuple[int, pathlib.Path | None]:
    """Invoke the CLI in ``sandbox``; return ``(exit_code, new_run_dir)``."""
    base = sandbox / output
    before = set(base.glob("*")) if base.exists() else set()
    monkeypatch.chdir(sandbox)
    code = cli.main(["--manifest", str(manifest), "--output", output, "--data-kind", data_kind])
    after = set(base.glob("*")) if base.exists() else set()
    created = sorted(after - before)
    assert len(created) <= 1, f"a single run must create at most one directory, got {created}"
    return code, (created[0] if created else None)


def read_records(run_dir: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]


def read_run_json(run_dir: pathlib.Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def only_record(run_dir: pathlib.Path) -> dict:
    records = read_records(run_dir)
    assert len(records) == 1
    return records[0]


# --------------------------------------------------------------------------- #
# EXIF orientation: three independent expressions of one rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("orientation", VALID_ORIENTATIONS)
@pytest.mark.parametrize(("width", "height"), [(5, 4), (4, 5), (4, 4), (7, 2), (2, 7)])
def test_matrix_and_array_transform_agree_on_every_pixel(orientation, width, height) -> None:
    """The stored matrix and the NumPy transform must place every pixel alike."""
    assert matrix_matches_transform(orientation, width, height)


def test_orientation_oracle_can_actually_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against a vacuous oracle.

    ``matrix_matches_transform`` is only evidence if it is capable of returning
    ``False``.  Substituting a deliberately transposed matrix for orientation 1
    must break the agreement, otherwise every "it agrees" assertion above would
    hold no information.
    """
    original = orientation_matrix
    monkeypatch.setattr(
        "kozlov_lab.io.orientation.orientation_matrix",
        lambda orientation, width, height: original(2 if orientation == 1 else 1, width, height),
    )
    assert matrix_matches_transform(1, 5, 4) is False
    assert matrix_matches_transform(2, 5, 4) is False


@pytest.mark.parametrize("orientation", VALID_ORIENTATIONS)
def test_marker_lands_where_the_contract_table_says(orientation: int) -> None:
    """Marker pixel placement, checked against the literal table above."""
    width, height, marker = 5, 4, (3, 1)
    array = gradient_image(width, height, marker, "RGB")
    moved = apply_orientation(array, orientation)

    expected_x, expected_y, expected_size = contract_raw_to_canonical(orientation, *marker, width, height)
    assert canonical_size(orientation, width, height) == expected_size
    assert moved.shape[:2] == (expected_size[1], expected_size[0])

    found = np.argwhere(moved[:, :, 0] == MARKER)
    assert len(found) == 1, "the marker must stay unique for this question to be well posed"
    assert tuple(found[0][::-1]) == (expected_x, expected_y)


def test_orientation_matrix_is_integral_and_matches_table() -> None:
    for orientation in VALID_ORIENTATIONS:
        matrix = orientation_matrix(orientation, 5, 4)
        assert all(isinstance(value, int) for row in matrix for value in row)
        assert matrix[2] == [0, 0, 1], "homogeneous row must be preserved"

    assert orientation_matrix(2, 5, 4) == [[-1, 0, 4], [0, 1, 0], [0, 0, 1]]
    assert orientation_matrix(6, 5, 4) == [[0, -1, 3], [1, 0, 0], [0, 0, 1]]
    assert orientation_matrix(8, 5, 4) == [[0, 1, 0], [-1, 0, 4], [0, 0, 1]]


def test_orientation_one_is_returned_unchanged() -> None:
    """v1 must be lossless: orientation 1 is a pure pass-through."""
    array = gradient_image(5, 4, (1, 1), "RGB")
    assert apply_orientation(array, 1) is array


# --------------------------------------------------------------------------- #
# decoding and refusal rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("container", ["PNG", "JPEG"])
@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_supported_containers_and_modes_decode(container: str, mode: str) -> None:
    decoded = decode_image(encode_with_exif(gradient_image(5, 4, (2, 2), mode), container))

    assert decoded.source_format == container
    assert decoded.source_mode == mode
    assert (decoded.raw_width, decoded.raw_height) == (5, 4)
    assert decoded.rgb.dtype == np.uint8
    assert decoded.rgb.shape == (4, 5, 3)


def test_supported_formats_list_matches_contract() -> None:
    assert SUPPORTED_FORMATS == ("JPEG", "PNG")


def test_l_mode_expands_to_rgb_without_changing_values() -> None:
    grey = gradient_image(5, 4, (4, 0), "L")

    rgb = decode_image(encode_with_exif(grey)).rgb
    assert np.array_equal(rgb[:, :, 0], grey)
    assert np.array_equal(rgb[:, :, 1], grey)
    assert np.array_equal(rgb[:, :, 2], grey)


def test_missing_exif_records_null_and_applies_orientation_one() -> None:
    loaded = decode_image(_png_bytes(gradient_image(5, 4, None, "RGB")))
    assert loaded.exif_orientation_original is None


@pytest.mark.parametrize("bad_orientation", [0, 9, 17])
def test_out_of_range_exif_orientation_is_rejected(bad_orientation: int) -> None:
    payload = _png_bytes(gradient_image(5, 4, None, "RGB"), exif=bad_orientation)
    with pytest.raises(Exception) as caught:
        decode_image(payload)
    assert getattr(caught.value, "issue").code == "INVALID_ORIENTATION"


def _png_bytes(array: np.ndarray, exif: int | None = None) -> bytes:
    return encode_with_exif(array, "PNG", exif)


@pytest.mark.parametrize(
    ("label", "builder"),
    [
        ("rgba", lambda: _buffer(Image.fromarray(np.zeros((4, 5, 4), dtype=np.uint8)))),
        ("palette", lambda: _buffer(Image.fromarray(gradient_image(5, 4, None, "L")).convert("P"))),
        ("16-bit-grey", lambda: _buffer(Image.fromarray(np.zeros((4, 5), dtype=np.uint16)))),
        ("gif", lambda: _buffer(Image.fromarray(gradient_image(5, 4, None, "RGB")), "GIF")),
        ("cmyk", lambda: _buffer(Image.fromarray(gradient_image(5, 4, None, "RGB")).convert("CMYK"), "JPEG")),
        (
            "apng-two-frames",
            lambda: _multiframe_png(),
        ),
    ],
)
def test_unsupported_inputs_are_refused(label: str, builder) -> None:
    """Every rejection must be explicit; no silent alpha drop or bit-depth cast."""
    with pytest.raises(Exception) as caught:
        decode_image(builder())
    assert getattr(caught.value, "issue").code == "UNSUPPORTED_IMAGE", label


def _buffer(image: Image.Image, container: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=container)
    return buffer.getvalue()


def _multiframe_png() -> bytes:
    frame = Image.fromarray(gradient_image(5, 4, None, "RGB"))
    buffer = BytesIO()
    frame.save(buffer, format="PNG", save_all=True, append_images=[frame])
    return buffer.getvalue()


def test_transparency_is_refused_even_when_mode_stays_l() -> None:
    """A tRNS chunk is rejected on its own, not merely via a mode change."""
    buffer = BytesIO()
    Image.fromarray(gradient_image(5, 4, None, "L")).save(buffer, format="PNG", transparency=0)
    payload = buffer.getvalue()
    assert "transparency" in Image.open(BytesIO(payload)).info, "fixture must really carry tRNS"

    with pytest.raises(Exception) as caught:
        decode_image(payload)
    assert getattr(caught.value, "issue").code == "UNSUPPORTED_IMAGE"


def test_corrupt_bytes_are_a_decode_error() -> None:
    with pytest.raises(Exception) as caught:
        decode_image(b"this is not an image at all")
    assert getattr(caught.value, "issue").code == "DECODE_ERROR"


def test_as_uint8_rgb_refuses_conversion_opportunities() -> None:
    """Out-of-range or wrongly typed input must raise, never be cast."""
    with pytest.raises(ValueError):
        as_uint8_rgb(np.zeros((4, 5, 3), dtype=np.int16))
    with pytest.raises(ValueError):
        as_uint8_rgb(np.zeros((4, 5, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        as_uint8_rgb(np.zeros((4, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        as_uint8_rgb(np.zeros((4, 5, 4), dtype=np.uint8))

    good = np.zeros((4, 5, 3), dtype=np.uint8)
    assert as_uint8_rgb(good) is good


def test_encoded_png_roundtrips_exactly_for_orientation_one() -> None:
    array = gradient_image(5, 4, (0, 0), "RGB")
    encoded = encode_rgb_png(array)
    decoded = decode_image(encoded)
    assert np.array_equal(decoded.rgb, array)
    assert Image.open(BytesIO(encoded)).mode == "RGB"


# --------------------------------------------------------------------------- #
# manifest: run-level vs row-level severity
# --------------------------------------------------------------------------- #
def _minimal_manifest(**overrides):
    document = {
        "contract_version": CONTRACT_VERSION,
        "dataset_id": "demo01",
        "data_kind": "synthetic",
        "dataset_root": expected_dataset_root("synthetic", "demo01"),
        "frames": [{"image": "images/a.png", "mask": None}],
    }
    document.update(overrides)
    return document


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("not-json", b"{ this is not json"),
        ("not-utf8", b"\xff\xfe\x00"),
        ("top-level-list", b"[]"),
        ("missing-fields", json.dumps({"dataset_id": "demo01"}).encode()),
        ("wrong-contract-version", json.dumps(_minimal_manifest(contract_version="photo-io-v2")).encode()),
        ("bad-dataset-id", json.dumps(_minimal_manifest(dataset_id="bad id!")).encode()),
        ("bad-data-kind", json.dumps(_minimal_manifest(data_kind="whatever")).encode()),
        ("empty-frames", json.dumps(_minimal_manifest(frames=[])).encode()),
        ("frames-not-a-list", json.dumps(_minimal_manifest(frames={"image": "a"})).encode()),
        (
            "dataset-root-mismatch",
            json.dumps(_minimal_manifest(dataset_root="data/real/demo01")).encode(),
        ),
    ],
)
def test_run_level_manifest_problems_are_invalid_manifest(label: str, payload: bytes) -> None:
    from kozlov_lab.io.manifest import ManifestError

    with pytest.raises(ManifestError) as caught:
        parse_manifest(payload, "synthetic")
    assert caught.value.code == "INVALID_MANIFEST", label


def test_data_kind_must_agree_with_the_cli_argument() -> None:
    from kozlov_lab.io.manifest import ManifestError

    payload = json.dumps(_minimal_manifest()).encode()
    with pytest.raises(ManifestError) as caught:
        parse_manifest(payload, "real")
    assert caught.value.code == "INVALID_MANIFEST"


def test_real_data_kind_requires_the_real_root_layout() -> None:
    """A fixture must not be passable as real data by relabelling it."""
    from kozlov_lab.io.manifest import ManifestError

    document = _minimal_manifest(
        data_kind="real",
        dataset_root=expected_dataset_root("synthetic", "demo01"),
    )
    with pytest.raises(ManifestError) as caught:
        parse_manifest(json.dumps(document).encode(), "real")
    assert caught.value.code == "INVALID_MANIFEST"


def test_row_field_problems_become_row_errors_without_dropping_rows() -> None:
    frames = [
        {"image": "images/a.png", "mask": None},
        {"image": "images/b.png"},
        {"mask": None},
        "not-an-object",
    ]
    manifest = parse_manifest(json.dumps(_minimal_manifest(frames=frames)).encode(), "synthetic")
    assert len(manifest.rows) == len(frames), "one row per input line, always"

    by_index = {row.input_index: row for row in manifest.rows}
    assert sorted(by_index) == [0, 1, 2, 3]
    assert by_index[0].row_error is None
    assert by_index[1].row_error.code == "INVALID_MANIFEST"
    assert by_index[2].row_error.code == "INVALID_MANIFEST"
    assert by_index[3].row_error.code == "INVALID_MANIFEST"
    # The usable half of a rejected row is still recorded for audit.
    assert by_index[1].image_canonical == "images/b.png"


@pytest.mark.parametrize(
    "bad_path",
    ["../escape.png", "/absolute.png", "C:/drive.png", "//server/share.png", "a/\x00b.png", "", "..", "."],
)
def test_illegal_relative_paths_are_rejected(bad_path: str) -> None:
    frames = [{"image": bad_path, "mask": None}]
    manifest = parse_manifest(json.dumps(_minimal_manifest(frames=frames)).encode(), "synthetic")
    assert manifest.rows[0].row_error is not None
    assert manifest.rows[0].row_error.code == "INVALID_PATH"


def test_backslashes_are_canonicalised_but_case_is_preserved() -> None:
    frames = [{"image": "images\\Sub\\A.PNG", "mask": None}]
    manifest = parse_manifest(json.dumps(_minimal_manifest(frames=frames)).encode(), "synthetic")
    row = manifest.rows[0]
    assert row.row_error is None
    assert row.image_canonical == "images/Sub/A.PNG"


# --------------------------------------------------------------------------- #
# CLI: exit codes, artifact completeness, one record per line
# --------------------------------------------------------------------------- #
def _ok_dataset(sandbox: pathlib.Path, dataset_id: str = "demo01") -> list[dict]:
    directory = dataset_dir(sandbox, dataset_id)
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    save_mask(directory / "masks/a.png", 5, 4, (0, 0))
    return [{"image": "images/a.png", "mask": "masks/a.png"}]


def test_cli_success_exit_zero_and_complete_artifacts(sandbox, monkeypatch) -> None:
    frames = _ok_dataset(sandbox)
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 0
    assert run_dir is not None
    run_json = read_run_json(run_dir)
    assert run_json["status"] == "SUCCESS"
    assert run_json["input_count"] == run_json["ok_count"] == 1
    assert run_json["error_count"] == 0
    assert run_json["contract_version"] == CONTRACT_VERSION
    assert run_json["manifest_sha256"] and len(run_json["manifest_sha256"]) == 64
    assert run_json["implementation_sha256"] and len(run_json["implementation_sha256"]) == 64
    assert set(run_json["runtime"]) == {"python", "pillow", "numpy"}
    assert run_json["errors"] == []

    record = only_record(run_dir)
    assert record["status"] == "OK"
    assert record["image_output"] == "images/000000.png"
    assert record["mask_output"] == "masks/000000.png"
    assert (run_dir / record["image_output"]).is_file()
    assert (run_dir / record["mask_output"]).is_file()
    assert record["cache_hit"] is False
    # Geometry is undefined at this stage and must stay null, never defaulted.
    for field in ("camera", "distortion", "pose", "geometry_scale"):
        assert record[field] is None


def test_cli_row_error_exit_one_keeps_every_record(sandbox, monkeypatch) -> None:
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    frames = [
        {"image": "images/a.png", "mask": None},
        {"image": "images/missing.png", "mask": None},
    ]
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    run_json = read_run_json(run_dir)
    assert run_json["status"] == "PARTIAL_FAILURE"
    assert (run_json["input_count"], run_json["ok_count"], run_json["error_count"]) == (2, 1, 1)

    records = read_records(run_dir)
    assert len(records) == len(frames)
    assert records[0]["status"] == "OK"
    assert records[1]["status"] == "ERROR"
    assert records[1]["errors"][0]["code"] == "MISSING_FILE"
    assert records[1]["image_output"] is None and records[1]["mask_output"] is None
    assert records[1]["cache_fingerprint"] is None


def test_cli_manifest_failure_exit_two_still_writes_run_record(sandbox, monkeypatch) -> None:
    manifest = sandbox / "broken.json"
    manifest.write_text("{ not json", encoding="utf-8")

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 2
    run_json = read_run_json(run_dir)
    assert run_json["status"] == "INPUT_ERROR"
    assert run_json["dataset_id"] is None
    assert (run_json["input_count"], run_json["ok_count"], run_json["error_count"]) == (0, 0, 0)
    assert run_json["errors"][0]["code"] == "INVALID_MANIFEST"
    assert read_records(run_dir) == []


def test_missing_manifest_reports_null_digest(sandbox, monkeypatch) -> None:
    code, run_dir = run_cli(sandbox, monkeypatch, sandbox / "does_not_exist.json")

    assert code == 2
    run_json = read_run_json(run_dir)
    assert run_json["manifest_sha256"] is None
    assert run_json["errors"][0]["code"] == "MISSING_FILE"


def test_output_outside_project_outputs_is_refused(sandbox, monkeypatch) -> None:
    frames = _ok_dataset(sandbox)
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest, output="elsewhere/runs")

    assert code == 2
    assert run_dir is None, "a parameter error must not leave artifacts behind"


def test_output_inside_the_dataset_root_is_refused(sandbox, monkeypatch) -> None:
    frames = _ok_dataset(sandbox)
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(
        sandbox, monkeypatch, manifest, output="outputs/delegation/T01A/fixtures/demo01/runs"
    )

    assert code == 2
    assert run_dir is None


def test_cli_exit_codes_from_a_real_subprocess(tmp_path) -> None:
    """The exit codes must be the process's, not just a returned integer."""
    sandbox = tmp_path / "proj"
    (sandbox / "outputs").mkdir(parents=True)
    (sandbox / "pyproject.toml").write_text('[project]\nname = "sandbox"\n', encoding="utf-8")
    directory = sandbox / "outputs/delegation/T01A/fixtures/demo01"
    (directory / "images").mkdir(parents=True)
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    good = write_manifest(sandbox, "demo01", [{"image": "images/a.png", "mask": None}])
    bad = sandbox / "row_error.json"
    bad.write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset_id": "demo01",
                "data_kind": "synthetic",
                "dataset_root": expected_dataset_root("synthetic", "demo01"),
                "frames": [{"image": "images/absent.png", "mask": None}],
            }
        ),
        encoding="utf-8",
    )

    env = {"PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
    common = [sys.executable, "-m", "kozlov_lab.io", "--output", "outputs/delegation/T01A/runs"]
    zero = subprocess.run(
        [*common, "--manifest", str(good), "--data-kind", "synthetic"],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
    )
    one = subprocess.run(
        [*common, "--manifest", str(bad), "--data-kind", "synthetic"],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
    )
    two = subprocess.run(
        [*common, "--manifest", str(good), "--data-kind", "real"],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
    )

    assert (zero.returncode, one.returncode, two.returncode) == (0, 1, 2), (
        zero.stderr + one.stderr + two.stderr
    )


def test_unlisted_files_in_the_dataset_produce_no_records(sandbox, monkeypatch) -> None:
    """Only explicit indices are read; nothing is auto-paired or auto-scanned."""
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    save_image(directory / "images/b.png", gradient_image(5, 4, (1, 1), "RGB"))
    save_mask(directory / "masks/a.png", 5, 4, (0, 0))
    save_mask(directory / "masks/b.png", 5, 4, (1, 1))
    manifest = write_manifest(sandbox, "demo01", [{"image": "images/a.png", "mask": "masks/a.png"}])

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 0
    assert len(read_records(run_dir)) == 1
    assert read_run_json(run_dir)["input_count"] == 1


# --------------------------------------------------------------------------- #
# output ordering and duplication
# --------------------------------------------------------------------------- #
def test_output_order_follows_contract_sort_and_keeps_input_index(sandbox, monkeypatch) -> None:
    directory = dataset_dir(sandbox, "demo01")
    names = ["c", "a", "d", "b"]
    for index, name in enumerate(names):
        save_image(directory / f"images/{name}.png", gradient_image(5, 4, (index % 5, 0), "RGB"))

    frames = [{"image": f"images/{name}.png", "mask": None} for name in names]
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 0
    records = read_records(run_dir)
    assert [record["image_path"] for record in records] == [
        "images/a.png",
        "images/b.png",
        "images/c.png",
        "images/d.png",
    ]
    assert [record["input_index"] for record in records] == [1, 3, 0, 2], (
        "sorting must reorder records without renumbering inputs"
    )
    assert {record["image_output"] for record in records} == {
        "images/000000.png",
        "images/000001.png",
        "images/000002.png",
        "images/000003.png",
    }, "artifact names must follow input_index, not output position"


def test_duplicate_content_is_reported_but_each_row_is_processed(sandbox, monkeypatch) -> None:
    directory = dataset_dir(sandbox, "demo01")
    payload = gradient_image(5, 4, (2, 2), "RGB")
    save_image(directory / "images/first.png", payload)
    save_image(directory / "images/second.png", payload)
    # Only the second row has a mask: each row must handle its own independently.
    save_mask(directory / "masks/second.png", 5, 4, (2, 2))
    frames = [
        {"image": "images/first.png", "mask": None},
        {"image": "images/second.png", "mask": "masks/second.png"},
    ]
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 0
    first, second = read_records(run_dir)
    assert first["image_sha256"] == second["image_sha256"]
    assert first["duplicate_of_input_index"] is None
    assert second["duplicate_of_input_index"] == first["input_index"]
    assert second["mask_output"] == "masks/000001.png", "a duplicate is reported, never skipped"
    assert first["mask_output"] is None


def test_identical_hash_never_marks_a_row_whose_hash_is_unavailable(sandbox, monkeypatch) -> None:
    """A missing file has no digest, so it can never be reported as a duplicate."""
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    frames = [
        {"image": "images/a.png", "mask": None},
        {"image": "images/gone.png", "mask": None},
    ]
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    gone = read_records(run_dir)[1]
    assert gone["image_sha256"] is None
    assert gone["duplicate_of_input_index"] is None


# --------------------------------------------------------------------------- #
# identity and cache invalidation
# --------------------------------------------------------------------------- #
def test_frame_id_ignores_root_and_mask_but_follows_identity_inputs() -> None:
    digest = "a" * 64
    base = compute_frame_id("demo01", "images/a.png", digest)

    assert base == compute_frame_id("demo01", "images/a.png", digest)
    assert base != compute_frame_id("demo02", "images/a.png", digest)
    assert base != compute_frame_id("demo01", "images/b.png", digest)
    assert base != compute_frame_id("demo01", "images/a.png", "b" * 64)
    assert base.startswith("f_") and len(base) == 66


def test_frame_id_survives_relocating_the_dataset(tmp_path, monkeypatch) -> None:
    """Same dataset_id, relative path and bytes => same identity, anywhere."""
    identities = []
    for variant in ("one", "two"):
        sandbox = tmp_path / variant
        (sandbox / "outputs").mkdir(parents=True)
        (sandbox / "pyproject.toml").write_text('[project]\nname = "s"\n', encoding="utf-8")
        directory = sandbox / "outputs/delegation/T01A/fixtures/demo01/images"
        directory.mkdir(parents=True)
        save_image(directory / "a.png", gradient_image(5, 4, (0, 0), "RGB"))
        manifest = write_manifest(sandbox, "demo01", [{"image": "images/a.png", "mask": None}])

        code, run_dir = run_cli(sandbox, monkeypatch, manifest)
        assert code == 0
        identities.append(only_record(run_dir)["frame_id"])

    assert identities[0] == identities[1]


def test_mask_change_keeps_frame_id_but_invalidates_the_cache(sandbox, monkeypatch) -> None:
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    save_mask(directory / "masks/a.png", 5, 4, (0, 0))
    save_mask(directory / "masks/a2.png", 5, 4, (4, 3))
    frames = [
        {"image": "images/a.png", "mask": "masks/a.png"},
        {"image": "images/a.png", "mask": "masks/a2.png"},
    ]
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)
    assert code == 0

    first, second = read_records(run_dir)
    assert first["frame_id"] == second["frame_id"], "only the mask changed, so identity is stable"
    assert first["mask_sha256"] != second["mask_sha256"]
    assert first["cache_fingerprint"] != second["cache_fingerprint"]


def test_cache_fingerprint_follows_every_documented_input() -> None:
    base = dict(
        dataset_id="demo01",
        data_kind="synthetic",
        image_path="images/a.png",
        mask_path="masks/a.png",
        image_sha256="a" * 64,
        mask_sha256="b" * 64,
        implementation_sha256="c" * 64,
        runtime={"python": "3.13.14", "pillow": "12.3.0", "numpy": "2.5.3"},
    )
    reference = compute_cache_fingerprint(**base)

    for field, new_value in (
        ("image_sha256", "d" * 64),
        ("mask_sha256", "e" * 64),
        ("mask_sha256", None),
        ("mask_path", None),
        ("image_path", "images/other.png"),
        ("dataset_id", "demo02"),
        ("data_kind", "real"),
        ("implementation_sha256", "f" * 64),
        ("runtime", {**base["runtime"], "pillow": "12.3.1"}),
    ):
        changed = {**base, field: new_value}
        assert compute_cache_fingerprint(**changed) != reference, field

    assert compute_cache_fingerprint(**base, contract_version="photo-io-v2") != reference


# --------------------------------------------------------------------------- #
# failure isolation
# --------------------------------------------------------------------------- #
def test_input_changed_is_detected_and_reported(sandbox, monkeypatch) -> None:
    """A file edited mid-run must not be reported with its earlier hash."""
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    manifest = write_manifest(sandbox, "demo01", [{"image": "images/a.png", "mask": None}])

    calls = {"count": 0}
    real_sha256_file = io_runner.sha256_file

    def flaky(path):
        calls["count"] += 1
        # The first re-verification call reports a different digest than the one
        # that was decoded, i.e. the file "changed" between hashing and finishing.
        if calls["count"] == 1:
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr("kozlov_lab.io.runner.sha256_file", flaky)
    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    record = only_record(run_dir)
    assert record["status"] == "ERROR"
    assert record["errors"][0]["code"] == "INPUT_CHANGED"
    assert record["image_output"] is None and record["mask_output"] is None


def test_a_failed_row_never_points_at_an_earlier_runs_artifact(sandbox, monkeypatch) -> None:
    """The strongest form of "failed rows must not reuse old results"."""
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    manifest = write_manifest(sandbox, "demo01", [{"image": "images/a.png", "mask": None}])

    good_code, good_run = run_cli(sandbox, monkeypatch, manifest)
    assert good_code == 0
    good_output = good_run / only_record(good_run)["image_output"]
    assert good_output.is_file(), "a reuse bug would have a real earlier artifact to point at"

    real_sha256_file = io_runner.sha256_file
    calls = {"count": 0}

    def flaky(path):
        calls["count"] += 1
        return "0" * 64 if calls["count"] == 1 else real_sha256_file(path)

    monkeypatch.setattr("kozlov_lab.io.runner.sha256_file", flaky)
    bad_code, bad_run = run_cli(sandbox, monkeypatch, manifest)

    assert bad_code == 1
    assert bad_run != good_run
    record = only_record(bad_run)
    assert record["status"] == "ERROR"
    assert record["image_output"] is None
    assert record["mask_output"] is None
    assert record["cache_fingerprint"] is None
    assert not list(bad_run.rglob("*.png")), "a failed row must not leave a partial artifact"


def test_internal_errors_are_never_rewritten_as_success(sandbox, monkeypatch) -> None:
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    manifest = write_manifest(sandbox, "demo01", [{"image": "images/a.png", "mask": None}])

    def explode(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr("kozlov_lab.io.runner.decode_image", explode)
    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    record = only_record(run_dir)
    assert record["status"] == "ERROR"
    assert record["errors"][0]["code"] == "INTERNAL_ERROR"


def test_runs_never_overwrite_an_earlier_run_directory(sandbox, monkeypatch) -> None:
    frames = _ok_dataset(sandbox)
    manifest = write_manifest(sandbox, "demo01", frames)

    code_a, run_a = run_cli(sandbox, monkeypatch, manifest)
    code_b, run_b = run_cli(sandbox, monkeypatch, manifest)

    assert (code_a, code_b) == (0, 0)
    assert run_a != run_b
    assert run_a.is_dir() and run_b.is_dir()


# =========================================================================== #
# R1 regressions
# =========================================================================== #
class _NonStandardConstant(ValueError):
    """A distinct type from ``json.JSONDecodeError``, which is also a ``ValueError``.

    Catching ``ValueError`` around ``json.loads`` would treat a perfectly
    ordinary multi-line JSONL file (not one JSON document) as containing a
    non-standard constant — a false positive in the dangerous direction.
    """


def _reject_constant(name: str) -> object:
    raise _NonStandardConstant(name)


def assert_strict_json(path: pathlib.Path, *, jsonl: bool) -> None:
    """Every JSON document in ``path`` must parse under a strict reader.

    Deliberately not a substring search for ``NaN``: a legal artifact can contain
    that text inside an error *message*, e.g. ``non-standard JSON constant 'NaN'
    is not allowed``, and would then be reported as malformed.
    """
    text = path.read_text(encoding="utf-8")
    chunks = [line for line in text.splitlines() if line.strip()] if jsonl else [text]
    for chunk in chunks:
        try:
            json.loads(chunk, parse_constant=_reject_constant)
        except _NonStandardConstant as exc:
            raise AssertionError(f"{path.name} contains a non-standard constant: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path.name} is not valid JSON: {exc}") from exc


def _crc_chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    )


def raw_png(width: int, height: int, bit_depth: int, colour_type: int, scanline: bytes) -> bytes:
    """A PNG assembled from raw bytes, so the *encoded* depth is under our control.

    An ordinary encoder cannot build these fixtures: asked for 16-bit colour it
    reports mode ``RGB``, and asked for 2-bit greyscale it reports ``L`` — which is
    exactly the property under test, so using one would defeat the whole point.

    The IHDR bit depth and colour type bytes are asserted here, so a fixture that
    silently became 8-bit fails immediately instead of quietly weakening every
    test that consumes it.
    """
    header = struct.pack(">IIBBBBB", width, height, bit_depth, colour_type, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _crc_chunk(b"IHDR", header)
        + _crc_chunk(b"IDAT", zlib.compress(b"\x00" + scanline))
        + _crc_chunk(b"IEND", b"")
    )
    assert payload[24] == bit_depth, "fixture IHDR bit depth is not what was requested"
    assert payload[25] == colour_type, "fixture IHDR colour type is not what was requested"
    return payload


@pytest.mark.parametrize(
    ("label", "depth", "colour_type", "scanline", "pillow_mode"),
    [
        # The three that slip past a mode-based check: mode is RGB / L either way.
        ("rgb16", 16, 2, struct.pack(">HHH", 65535, 256, 1), "RGB"),
        ("l2", 2, 0, bytes([0x30]), "L"),
        ("l4", 4, 0, bytes([0x0F]), "L"),
        # Also not 8 bits per channel, though this one a mode check would catch.
        ("l16", 16, 0, struct.pack(">H", 40000), "I;16"),
    ],
)
def test_sources_that_are_not_eight_bit_per_channel_are_refused(
    label: str, depth: int, colour_type: int, scanline: bytes, pillow_mode: str
) -> None:
    """The decoded mode says nothing about the source depth.

    A valid 16-bit RGB PNG opens as mode ``RGB`` and a 2/4-bit greyscale PNG as
    mode ``L``, so a check written against ``mode`` accepts every one of these.
    The accepted array is well formed and its values have already been reduced —
    ``(65535, 256, 1)`` arrives as ``(255, 1, 0)``.
    """
    payload = raw_png(2 if depth in (2, 4) else 1, 1, depth, colour_type, scanline)
    assert Image.open(BytesIO(payload)).mode == pillow_mode, "fixture must exercise the trap"

    with pytest.raises(Exception) as caught:
        decode_image(payload)

    assert caught.value.issue.code == "UNSUPPORTED_IMAGE", label


def test_eight_bit_rgb_and_grey_sources_are_still_accepted() -> None:
    """The success controls: refusing every PNG would also 'pass' the tests above."""
    rgb = decode_image(raw_png(1, 1, 8, 2, bytes([10, 20, 30])))
    assert rgb.source_mode == "RGB"
    assert rgb.rgb.tolist() == [[[10, 20, 30]]]

    grey = decode_image(raw_png(1, 1, 8, 0, bytes([200])))
    assert grey.source_mode == "L"
    assert grey.rgb.tolist() == [[[200, 200, 200]]]


def test_ordinary_encoders_still_produce_accepted_files() -> None:
    """The realistic success path, so the rule is not only satisfied by raw bytes."""
    for container in ("PNG", "JPEG"):
        for mode in ("RGB", "L"):
            payload = encode_with_exif(gradient_image(5, 4, (1, 1), mode), container)
            decoded = decode_image(payload)
            assert decoded.source_mode == mode
            assert decoded.rgb.dtype == np.uint8


def test_png_header_reader_reports_the_encoded_fields() -> None:
    header = png_header(raw_png(3, 2, 16, 2, struct.pack(">HHH", 1, 2, 3)))
    assert (header.width, header.height) == (3, 2)
    assert (header.bit_depth, header.colour_type) == (16, 2)

    assert png_header(raw_png(1, 1, 8, 0, bytes([7]))).bit_depth == 8


@pytest.mark.parametrize(
    "payload",
    [b"", b"not a png at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 10],
)
def test_png_header_reader_refuses_unreadable_input(payload: bytes) -> None:
    with pytest.raises(SourceEncodingError):
        png_header(payload)


def test_jpeg_reader_reads_ordinary_jpegs_as_eight_bit() -> None:
    """Positive control: the reader must not simply be hard-coded to 'not 8'."""
    for mode in ("RGB", "L"):
        payload = encode_with_exif(gradient_image(5, 4, (1, 1), mode), "JPEG")
        frame = jpeg_frame(payload)
        assert frame.precision == 8
        assert 0xC0 <= frame.marker <= 0xCF


def _synthetic_jpeg_frame(precision: int) -> bytes:
    """A JPEG header stream declaring ``precision``, stopping at the scan.

    Pillow cannot produce or decode a 12-bit JPEG, so this is the only way to
    show the reader reports what the file says rather than what it usually sees.
    """
    frame_body = struct.pack(">BHHB", precision, 4, 5, 1) + b"\x01\x11\x00"
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", len(frame_body) + 2)
        + frame_body
        + b"\xff\xda"
        + struct.pack(">H", 2)
    )


def test_jpeg_reader_reports_a_non_eight_bit_precision() -> None:
    payload = _synthetic_jpeg_frame(12)
    assert jpeg_frame(payload).precision == 12
    assert source_bit_depth(payload, "JPEG") == 12

    with pytest.raises(Exception) as caught:
        assert_eight_bit_source(payload, "JPEG", code="UNSUPPORTED_IMAGE", subject="image")
    assert caught.value.issue.code == "UNSUPPORTED_IMAGE"


def test_bit_depth_reader_refuses_an_unknown_container() -> None:
    """'Unreadable' must not be treated as 'fine'."""
    with pytest.raises(SourceEncodingError):
        source_bit_depth(b"whatever", "GIF")


# --------------------------------------------------------------------------- #
# R1: failure isolation for a path that cannot be encoded as UTF-8
# --------------------------------------------------------------------------- #
def test_unencodable_path_is_a_row_error_and_does_not_lose_the_run(sandbox, monkeypatch) -> None:
    """An unpaired surrogate in a path must not abort the artifact writing.

    ``"\\ud800"`` is a string Python holds but UTF-8 cannot encode.  It used to
    reach ``json.dumps(..., ensure_ascii=False)`` and raise ``UnicodeEncodeError``
    while writing ``frames.jsonl``, which lost **both** records and left no
    ``run.json`` at all — the legal row in the same manifest was collateral
    damage.  The row must be refused, and the run must still be reported.
    """
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    frames = [
        {"image": "\ud800", "mask": None},
        {"image": "images/a.png", "mask": None},
    ]
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    run_json = read_run_json(run_dir)
    assert run_json["status"] == "PARTIAL_FAILURE"
    assert (run_json["input_count"], run_json["ok_count"], run_json["error_count"]) == (2, 1, 1)

    records = read_records(run_dir)
    assert len(records) == 2, "the rejected row must be kept, never dropped"

    good = next(record for record in records if record["status"] == "OK")
    bad = next(record for record in records if record["status"] == "ERROR")

    assert good["image_path"] == "images/a.png"
    assert good["image_output"] == "images/000001.png"
    assert (run_dir / good["image_output"]).is_file()

    assert bad["errors"][0]["code"] == "INVALID_PATH"
    assert bad["input_image"] == "\ud800", "the original value is preserved for audit"
    assert bad["image_path"] is None
    assert bad["frame_id"] is None, "no identity can be built from an unusable path"
    assert bad["image_output"] is None and bad["mask_output"] is None
    assert bad["cache_fingerprint"] is None

    raw = (run_dir / "frames.jsonl").read_bytes()
    # Decoding must not raise: the artifact is UTF-8, and the unencodable value
    # was written as a JSON escape rather than as a raw surrogate.
    assert raw.decode("utf-8")
    assert b"\\ud800" in raw, "the value is preserved as a JSON escape, not rewritten"

    assert_strict_json(run_dir / "frames.jsonl", jsonl=True)
    assert_strict_json(run_dir / "run.json", jsonl=False)


def test_identity_and_cache_algorithms_are_unchanged_by_the_escaping_fix() -> None:
    """The safe output escaping must not have touched the contract hash inputs.

    ``frame_id`` and ``cache_fingerprint`` are computed from *canonical* JSON with
    ``ensure_ascii=False``; only the artifact writer falls back to escaping.
    Recomputed here from the published formula so a change in either would show.
    """
    import hashlib

    identity = json.dumps(
        ["demo01", "images/a.png", "a" * 64], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert compute_frame_id("demo01", "images/a.png", "a" * 64) == "f_" + hashlib.sha256(
        identity
    ).hexdigest()

    fingerprint = compute_cache_fingerprint(
        dataset_id="demo01",
        data_kind="synthetic",
        image_path="images/a.png",
        mask_path=None,
        image_sha256="a" * 64,
        mask_sha256=None,
        implementation_sha256="c" * 64,
        runtime={"python": "3.13.14"},
    )
    from kozlov_lab.io.paths import canonical_json

    expected = hashlib.sha256(
        canonical_json(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset_id": "demo01",
                "data_kind": "synthetic",
                "image_path": "images/a.png",
                "mask_path": None,
                "image_sha256": "a" * 64,
                "mask_sha256": None,
                "implementation_sha256": "c" * 64,
                "runtime": {"python": "3.13.14"},
            }
        ).encode("utf-8")
    ).hexdigest()
    assert fingerprint == expected


# --------------------------------------------------------------------------- #
# R1: strict JSON — no NaN / Infinity / -Infinity in a manifest or an artifact
# --------------------------------------------------------------------------- #
def _manifest_text_with_raw_extra(raw_extra: str, frames_json: str | None = None) -> str:
    """An otherwise *valid* manifest, with one raw JSON fragment spliced in.

    Validity matters: because nothing else is wrong with the document, a
    rejection can only have come from the non-finite rule.  That is what makes
    this a check rather than an assertion about an error message.
    """
    frames = (
        frames_json
        if frames_json is not None
        else '[{"image": "images/a.png", "mask": null}]'
    )
    return (
        "{"
        f'"contract_version": "{CONTRACT_VERSION}", '
        '"dataset_id": "demo01", '
        '"data_kind": "synthetic", '
        '"dataset_root": "outputs/delegation/T01A/fixtures/demo01", '
        f'"extra": {raw_extra}, '
        f'"frames": {frames}'
        "}"
    )


@pytest.mark.parametrize(
    ("label", "raw_extra"),
    [
        ("nan_extra_field", "NaN"),
        ("infinity_extra_field", "Infinity"),
        ("negative_infinity_extra_field", "-Infinity"),
        ("nested_nan", '{"outer": {"inner": [1, {"deep": NaN}]}}'),
        ("finite_literal_that_overflows_a_double", "1e400"),
        ("negative_finite_literal_that_overflows", "-1e400"),
    ],
)
def test_non_finite_numbers_make_the_manifest_invalid(label: str, raw_extra: str) -> None:
    payload = _manifest_text_with_raw_extra(raw_extra).encode("utf-8")

    with pytest.raises(ManifestError) as caught:
        parse_manifest(payload, "synthetic")

    assert caught.value.code == "INVALID_MANIFEST", label


def test_non_finite_number_as_a_row_value_makes_the_manifest_invalid() -> None:
    """The rule is about the document, not only about unknown fields."""
    payload = _manifest_text_with_raw_extra('"unused"', frames_json='[{"image": NaN, "mask": null}]')

    with pytest.raises(ManifestError) as caught:
        parse_manifest(payload.encode("utf-8"), "synthetic")

    assert caught.value.code == "INVALID_MANIFEST"


@pytest.mark.parametrize(
    ("label", "raw_extra"),
    [
        ("nan_extra_field", "NaN"),
        ("infinity_nested", '{"deep": [Infinity]}'),
        ("finite_overflow", "1e400"),
    ],
)
def test_non_finite_manifest_is_a_run_level_input_error(
    sandbox, monkeypatch, label: str, raw_extra: str
) -> None:
    """Recorded as ``INPUT_ERROR`` with evidence written, and no bad artifact."""
    manifest = sandbox / f"{label}.json"
    manifest.write_text(_manifest_text_with_raw_extra(raw_extra), encoding="utf-8")

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 2, label
    run_json = read_run_json(run_dir)
    assert run_json["status"] == "INPUT_ERROR"
    assert run_json["errors"][0]["code"] == "INVALID_MANIFEST"
    assert (run_json["input_count"], run_json["ok_count"], run_json["error_count"]) == (0, 0, 0)

    assert_strict_json(run_dir / "run.json", jsonl=False)
    assert_strict_json(run_dir / "frames.jsonl", jsonl=True)


# --------------------------------------------------------------------------- #
# R1: a non-object row contributes no sort keys
# --------------------------------------------------------------------------- #
def test_non_object_row_contributes_empty_sort_keys(sandbox, monkeypatch) -> None:
    """A row that is not an object has no ``image``/``mask`` fields at all.

    Serialising the row into the image key (the literal text ``null``) invents an
    image value the line never had and moves the row in the contract ordering,
    which previously produced ``[1, 2, 0]`` instead of ``[0, 1, 2]``.
    """
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))
    frames = [None, {"image": None, "mask": None}, {"image": "images/a.png", "mask": None}]
    manifest = write_manifest(sandbox, "demo01", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    records = read_records(run_dir)
    assert [record["input_index"] for record in records] == [0, 1, 2]
    assert len(records) == len(frames), "every line keeps exactly one record"

    assert records[0]["status"] == "ERROR"
    assert records[0]["errors"][0]["code"] == "INVALID_MANIFEST"
    assert records[1]["status"] == "ERROR"
    assert records[1]["errors"][0]["code"] == "INVALID_PATH"
    assert records[2]["status"] == "OK"


def test_non_object_row_sort_keys_are_both_empty() -> None:
    """Pinned at the parser, where the keys are produced."""
    rows = parse_manifest(
        json.dumps(
            _minimal_manifest(
                frames=[None, {"image": None, "mask": None}, {"image": "a.png", "mask": None}]
            )
        ).encode(),
        "synthetic",
    ).rows

    by_index = {row.input_index: row for row in rows}
    assert (by_index[0].image_key, by_index[0].mask_key) == ("", "")
    assert (by_index[1].image_key, by_index[1].mask_key) == ("", "")
    assert (by_index[2].image_key, by_index[2].mask_key) == ("a.png", "")
    assert [row.input_index for row in rows] == [0, 1, 2]

    # A non-object row must never acquire an image value it did not have.
    assert by_index[0].image_key != "null"


# --------------------------------------------------------------------------- #
# R1: dataset_id is matched against the whole string
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        "demo01\n",
        "demo01\r\n",
        "\ndemo01",
        "demo01 ",
        " demo01",
        "demo 01",
        "demo.01",
        "demo/01",
        "demo01;",
        "デモ01",
        "",
        " ",
    ],
)
def test_dataset_id_with_extra_characters_is_refused(bad: str) -> None:
    assert is_valid_dataset_id(bad) is False, repr(bad)


@pytest.mark.parametrize("good", ["demo01", "demo_01", "demo-01", "D", "9x", "aB3_-", "a" * 40])
def test_ordinary_dataset_ids_are_still_accepted(good: str) -> None:
    assert is_valid_dataset_id(good) is True, repr(good)


def test_manifest_dataset_id_with_a_trailing_newline_is_rejected(sandbox, monkeypatch) -> None:
    frames = _ok_dataset(sandbox)
    manifest = write_manifest(sandbox, "demo01\n", frames)

    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 2
    run_json = read_run_json(run_dir)
    assert run_json["status"] == "INPUT_ERROR"
    assert run_json["dataset_id"] is None
    assert run_json["errors"][0]["code"] == "INVALID_MANIFEST"
