"""``photo-io-v1`` manual-mask contract tests (T01A).

A mask is ground truth for the foreground boundary, so the interesting failures
are not crashes but *silently plausible* results: a thresholded greyscale, a
mixture of ``1`` and ``255``, a mask that was already rotated by EXIF and gets
rotated a second time.  Each of those yields a well-formed boolean array whose
boundary is wrong.  Every test here therefore pins the contract *code*, and the
orientation tests use a single unique foreground pixel so "where did the
foreground land?" has exactly one right answer.

The size rule gets its own trap test: the mask is compared against the photo's
**raw** size, never the canonical one.  A mask authored in canonical coordinates
for a rotated photo has the swapped dimensions, which is precisely the mistake
that a "compare canonical sizes" check would wave through.
"""

from __future__ import annotations

import json
import pathlib
import struct
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from kozlov_lab.io.hashing import CONTRACT_VERSION
from kozlov_lab.io.manifest import expected_dataset_root
from kozlov_lab.segmentation.manual import (
    MASK_ENCODINGS,
    canonical_mask_array,
    decode_manual_mask,
    encode_mask_png,
    expected_canonical_mask_size,
)

from test_io_frames import (  # noqa: E402 - shared sandbox helpers
    MARKER,
    assert_strict_json,
    contract_raw_to_canonical,
    dataset_dir,
    gradient_image,
    only_record,
    raw_png,
    read_records,
    read_run_json,
    run_cli,
    save_image,
    write_manifest,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    """A throwaway project root.

    Duplicated from ``test_io_frames.py`` on purpose: pytest fixtures are only
    visible within their own module unless declared in a ``conftest``, and this
    task's write scope covers the two test files only.
    """
    root = tmp_path / "proj"
    (root / "outputs").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "sandbox"\n', encoding="utf-8")
    return root


def mask_bytes(
    canvas: np.ndarray,
    *,
    container: str = "PNG",
    exif: int | None = None,
    transparency: int | None = None,
) -> bytes:
    image = Image.fromarray(canvas)
    kwargs = {}
    if exif is not None:
        block = Image.Exif()
        block[274] = exif
        kwargs["exif"] = block
    if transparency is not None:
        kwargs["transparency"] = transparency
    if container == "JPEG":
        image = image.convert("L")
    buffer = BytesIO()
    image.save(buffer, format=container, **kwargs)
    return buffer.getvalue()


def mask_canvas(width: int, height: int, marker: tuple[int, int] | None, full: int = 255) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    if marker is not None:
        canvas[marker[1], marker[0]] = 1
    if full == 255:
        canvas = np.where(canvas != 0, 255, 0).astype(np.uint8)
    return canvas


def write_mask(path: pathlib.Path, canvas: np.ndarray, **kwargs) -> None:
    path.write_bytes(mask_bytes(canvas, **kwargs))


def _decode_error_code(payload: bytes) -> str:
    with pytest.raises(Exception) as caught:
        decode_manual_mask(payload)
    issue = getattr(caught.value, "issue", None)
    assert issue is not None, f"expected a contract-coded failure, got {caught.value!r}"
    return issue.code


def one_row(sandbox, monkeypatch, image_spec, mask_entry, *, dataset_id="demo01"):
    """Build a single-row dataset and run the CLI. Returns ``(exit_code, record, run_dir)``."""
    directory = dataset_dir(sandbox, dataset_id)
    image_spec(directory)
    if mask_entry is not None:
        mask_entry(directory)
    frames = [{"image": "images/a.png", "mask": "masks/a.png" if mask_entry is not None else None}]
    manifest = write_manifest(sandbox, dataset_id, frames)
    code, run_dir = run_cli(sandbox, monkeypatch, manifest)
    return code, only_record(run_dir), run_dir


# --------------------------------------------------------------------------- #
# encoding classification
# --------------------------------------------------------------------------- #
def test_zero_255_encoding_is_recognised() -> None:
    decoded = decode_manual_mask(mask_bytes(mask_canvas(5, 4, (1, 1), full=255)))
    assert decoded.encoding == "zero_255"
    assert decoded.foreground_pixels == 1
    assert decoded.foreground.dtype == np.bool_
    assert decoded.foreground[1, 1]


def test_zero_one_encoding_is_recognised() -> None:
    decoded = decode_manual_mask(mask_bytes(mask_canvas(5, 4, (3, 2), full=1)))
    assert decoded.encoding == "zero_one"
    assert decoded.foreground_pixels == 1
    assert decoded.foreground[2, 3]


def test_all_zero_mask_is_its_own_encoding() -> None:
    """All-zero belongs to both encodings, so it is not silently claimed by one."""
    decoded = decode_manual_mask(mask_bytes(np.zeros((4, 5), dtype=np.uint8)))
    assert decoded.encoding == "all_zero"
    assert decoded.foreground_pixels == 0
    assert not decoded.foreground.any()


def test_all_foreground_is_legal() -> None:
    decoded = decode_manual_mask(mask_bytes(np.full((3, 4), 255, dtype=np.uint8)))
    assert decoded.encoding == "zero_255"
    assert decoded.foreground_pixels == 12
    assert decoded.foreground.all()


def test_every_declared_encoding_is_reachable() -> None:
    """Guard against a label that no input can ever produce."""
    observed = {
        decode_manual_mask(mask_bytes(mask_canvas(4, 3, (1, 1), full=255))).encoding,
        decode_manual_mask(mask_bytes(mask_canvas(4, 3, (1, 1), full=1))).encoding,
        decode_manual_mask(mask_bytes(np.zeros((3, 4), dtype=np.uint8))).encoding,
    }
    assert observed == set(MASK_ENCODINGS)


def test_mixed_foreground_values_are_refused() -> None:
    """``{0, 1, 255}`` lies in neither encoding and is refused.

    A note on what this test does *not* claim.  The dedicated mixture branch is
    behaviourally equivalent to the general value-domain check: deleting it still
    rejects ``{0, 1, 255}``, so the contract-level assertion is the error code,
    and the two paths are not distinguishable by outcome.  An earlier revision of
    this test asserted the branch's particular wording to make that deletion
    detectable; the review ruled that out, correctly, because it promotes a
    diagnostic detail into a contract requirement.  The branch stays for message
    quality, and the wording is not pinned here.
    """
    mixed = np.zeros((4, 5), dtype=np.uint8)
    mixed[0, 0] = 1
    mixed[0, 1] = 255

    with pytest.raises(Exception) as caught:
        decode_manual_mask(mask_bytes(mixed))

    assert caught.value.issue.code == "INVALID_MASK"


@pytest.mark.parametrize("bad_value", [2, 3, 100, 128, 200, 254])
def test_values_outside_the_two_domains_are_refused(bad_value: int) -> None:
    canvas = np.zeros((4, 5), dtype=np.uint8)
    canvas[2, 2] = bad_value
    assert _decode_error_code(mask_bytes(canvas)) == "INVALID_MASK"


def test_no_thresholding_happens() -> None:
    """A 0/128 image must fail, not become 0/1 by a convenient threshold."""
    canvas = np.zeros((4, 5), dtype=np.uint8)
    canvas[1, 1] = 128
    with pytest.raises(Exception) as caught:
        decode_manual_mask(mask_bytes(canvas))
    assert "128" in caught.value.issue.message


# --------------------------------------------------------------------------- #
# container, mode and metadata rules
# --------------------------------------------------------------------------- #
def _palette_png() -> bytes:
    """A genuine mode-``P`` PNG.

    Note the trap this avoids: ``np.asarray`` of a ``P`` image is a plain 2-D
    array, and rebuilding an image from it yields mode ``L`` — so the payload
    would silently stop testing the palette rule.
    """
    buffer = BytesIO()
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).convert("P").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("label", "builder", "expected_mode", "expected_format"),
    [
        ("rgb", lambda: mask_bytes(np.zeros((4, 5, 3), dtype=np.uint8)), "RGB", "PNG"),
        ("rgba", lambda: mask_bytes(np.zeros((4, 5, 4), dtype=np.uint8)), "RGBA", "PNG"),
        ("la", lambda: mask_bytes(np.zeros((4, 5, 2), dtype=np.uint8)), "LA", "PNG"),
        ("16-bit-grey", lambda: mask_bytes(np.zeros((4, 5), dtype=np.uint16)), "I;16", "PNG"),
        ("palette", _palette_png, "P", "PNG"),
        # Mode is legal here; the container is not.  Keeping the mode assertion
        # proves the rejection came from the format rule, not from the mode rule.
        ("jpeg-container", lambda: mask_bytes(np.zeros((4, 5), dtype=np.uint8), container="JPEG"), "L", "JPEG"),
    ],
)
def test_mask_modes_and_containers_are_refused(
    label: str, builder, expected_mode: str, expected_format: str
) -> None:
    payload = builder()
    opened = Image.open(BytesIO(payload))
    assert opened.mode == expected_mode, f"{label}: fixture is mode {opened.mode}, not {expected_mode}"
    assert opened.format == expected_format, f"{label}: fixture is {opened.format}"

    assert _decode_error_code(payload) == "INVALID_MASK", label


def test_mask_transparency_is_refused() -> None:
    payload = mask_bytes(np.zeros((4, 5), dtype=np.uint8), transparency=0)
    assert "transparency" in Image.open(BytesIO(payload)).info, "fixture must really carry tRNS"
    assert _decode_error_code(payload) == "INVALID_MASK"


def test_multiframe_mask_is_refused() -> None:
    frame = Image.fromarray(np.zeros((4, 5), dtype=np.uint8))
    buffer = BytesIO()
    frame.save(buffer, format="PNG", save_all=True, append_images=[frame])
    assert _decode_error_code(buffer.getvalue()) == "INVALID_MASK"


def test_corrupt_mask_is_a_decode_error() -> None:
    assert _decode_error_code(b"definitely not a png") == "DECODE_ERROR"


def test_mask_orientation_one_is_accepted() -> None:
    decoded = decode_manual_mask(mask_bytes(mask_canvas(5, 4, (0, 0)), exif=1))
    assert decoded.exif_orientation_original == 1


@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_rotated_mask_orientation_is_refused(orientation: int) -> None:
    """A pre-rotated mask would otherwise be rotated a second time."""
    payload = mask_bytes(mask_canvas(5, 4, (0, 0)), exif=orientation)
    assert _decode_error_code(payload) == "INVALID_ORIENTATION"


# --------------------------------------------------------------------------- #
# size rule: raw against raw
# --------------------------------------------------------------------------- #
def test_size_rule_compares_raw_sizes_not_canonical(sandbox, monkeypatch) -> None:
    """A mask authored in canonical coordinates must be rejected, not accepted.

    The photo is 5x4 with EXIF 6, so its *canonical* size is 4x5.  A mask whose
    raw size is 4x5 has exactly the canonical dimensions - and is still wrong,
    because masks are drawn in raw pixel coordinates.
    """

    def image(directory):
        save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"), exif=6)

    def mask(directory):
        write_mask(directory / "masks/a.png", mask_canvas(4, 5, (0, 0)))

    code, record, run_dir = one_row(sandbox, monkeypatch, image, mask)

    assert code == 1
    assert record["status"] == "ERROR"
    assert record["errors"][0]["code"] == "MASK_SIZE_MISMATCH"
    assert record["mask_output"] is None and record["image_output"] is None
    # The photo itself was fine and its metadata is kept for audit.
    assert record["raw_width"] == 5 and record["raw_height"] == 4
    assert record["canonical_width"] == 4 and record["canonical_height"] == 5


def test_raw_sized_mask_is_accepted_and_becomes_canonical(sandbox, monkeypatch) -> None:
    def image(directory):
        save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"), exif=6)

    def mask(directory):
        write_mask(directory / "masks/a.png", mask_canvas(5, 4, (0, 0)))

    code, record, run_dir = one_row(sandbox, monkeypatch, image, mask)

    assert code == 0
    assert record["status"] == "OK"
    written = Image.open(run_dir / record["mask_output"])
    assert written.size == (4, 5), "the mask must undergo the same transform as the photo"


# --------------------------------------------------------------------------- #
# the transform is shared with the photo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("orientation", range(1, 9))
def test_mask_and_photo_land_on_the_same_canonical_pixel(sandbox, monkeypatch, orientation: int) -> None:
    """Same raw marker in both => same canonical pixel in both, for all 8 cases."""
    marker = (3, 1)

    def image(directory):
        save_image(directory / "images/a.png", gradient_image(5, 4, marker, "RGB"), exif=orientation)

    def mask(directory):
        write_mask(directory / "masks/a.png", mask_canvas(5, 4, marker))

    code, record, run_dir = one_row(sandbox, monkeypatch, image, mask)
    assert code == 0, record["errors"]

    expected_x, expected_y, expected_size = contract_raw_to_canonical(orientation, *marker, 5, 4)
    assert (record["canonical_width"], record["canonical_height"]) == expected_size

    # The recorded matrix must itself map the raw marker onto the contract's
    # canonical marker.  Checked as a literal property rather than by comparing
    # against the implementation's own matrix builder.
    matrix = np.asarray(record["raw_to_canonical"])
    assert matrix.shape == (3, 3)
    mapped = matrix @ np.array([marker[0], marker[1], 1])
    assert tuple(int(value) for value in mapped[:2]) == (expected_x, expected_y)

    photo = np.asarray(Image.open(run_dir / record["image_output"]))
    found = np.argwhere(photo[:, :, 0] == MARKER)
    assert len(found) == 1
    assert tuple(found[0][::-1]) == (expected_x, expected_y)

    written_mask = np.asarray(Image.open(run_dir / record["mask_output"]))
    foreground = np.argwhere(written_mask != 0)
    assert len(foreground) == 1, "exactly one foreground pixel was drawn"
    assert tuple(foreground[0][::-1]) == (expected_x, expected_y)


def test_expected_canonical_mask_size_matches_the_shared_transform() -> None:
    for orientation in range(1, 9):
        expected = (4, 5) if orientation >= 5 else (5, 4)
        assert expected_canonical_mask_size(orientation, 5, 4) == expected


# --------------------------------------------------------------------------- #
# canonical output form
# --------------------------------------------------------------------------- #
def test_canonical_mask_output_is_l_mode_and_only_zero_or_255(sandbox, monkeypatch) -> None:
    def image(directory):
        save_image(directory / "images/a.png", gradient_image(5, 4, (2, 1), "RGB"))

    def mask(directory):
        write_mask(directory / "masks/a.png", mask_canvas(5, 4, (2, 1), full=1))

    code, record, run_dir = one_row(sandbox, monkeypatch, image, mask)
    assert code == 0
    assert record["mask_encoding"] == "zero_one", "the input encoding is reported as read"

    written = Image.open(run_dir / record["mask_output"])
    assert written.format == "PNG"
    assert written.mode == "L"
    assert written.getexif().get(274) is None, "output must not be re-rotatable"
    assert set(np.unique(np.asarray(written)).tolist()) <= {0, 255}
    assert record["mask_foreground_pixels"] == 1


def test_encode_mask_png_refuses_wrong_dtype_and_shape() -> None:
    with pytest.raises(ValueError):
        encode_mask_png(np.zeros((4, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_mask_png(np.zeros((4, 5, 1), dtype=np.bool_))


def test_encode_mask_png_roundtrips_through_the_reader() -> None:
    foreground = np.zeros((4, 5), dtype=np.bool_)
    foreground[2, 3] = True
    payload = encode_mask_png(foreground)

    decoded = decode_manual_mask(payload)
    assert decoded.encoding == "zero_255"
    assert np.array_equal(decoded.foreground, foreground)


# --------------------------------------------------------------------------- #
# absent vs broken masks
# --------------------------------------------------------------------------- #
def test_null_mask_yields_null_fields_and_no_artifact(sandbox, monkeypatch) -> None:
    def image(directory):
        save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))

    code, record, run_dir = one_row(sandbox, monkeypatch, image, None)

    assert code == 0
    assert record["status"] == "OK"
    for field in ("input_mask", "mask_path", "mask_sha256", "mask_encoding", "mask_foreground_pixels", "mask_output"):
        assert record[field] is None, field
    assert not (run_dir / "masks").exists(), "no mask was asked for, so none may be invented"


def test_empty_foreground_warns_but_still_succeeds(sandbox, monkeypatch) -> None:
    def image(directory):
        save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))

    def mask(directory):
        write_mask(directory / "masks/a.png", np.zeros((4, 5), dtype=np.uint8))

    code, record, run_dir = one_row(sandbox, monkeypatch, image, mask)

    assert code == 0, "an empty mask is a warning, not a failure"
    assert record["status"] == "OK"
    assert [warning["code"] for warning in record["warnings"]] == ["EMPTY_MASK"]
    assert record["mask_encoding"] == "all_zero"
    assert record["mask_foreground_pixels"] == 0
    assert (run_dir / record["mask_output"]).is_file()
    assert read_run_json(run_dir)["warning_count"] == 1


def test_a_missing_mask_file_is_an_error_not_an_absent_mask(sandbox, monkeypatch) -> None:
    """A path that does not resolve must never degrade into "no mask provided"."""

    def image(directory):
        save_image(directory / "images/a.png", gradient_image(5, 4, (0, 0), "RGB"))

    directory = dataset_dir(sandbox, "demo01")
    image(directory)
    manifest = write_manifest(sandbox, "demo01", [{"image": "images/a.png", "mask": "masks/absent.png"}])
    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    record = only_record(run_dir)
    assert record["status"] == "ERROR"
    assert record["errors"][0]["code"] == "MISSING_FILE"
    assert record["mask_encoding"] is None, "a failed mask must not be reported as an absent one"
    assert record["image_output"] is None


def test_mixed_encoding_row_fails_while_its_neighbour_succeeds(sandbox, monkeypatch) -> None:
    """Row failures are isolated: one broken mask must not sink the run output."""
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/good.png", gradient_image(5, 4, (0, 0), "RGB"))
    save_image(directory / "images/bad.png", gradient_image(5, 4, (1, 0), "RGB"))
    write_mask(directory / "masks/good.png", mask_canvas(5, 4, (0, 0)))

    mixed = np.zeros((4, 5), dtype=np.uint8)
    mixed[0, 0] = 1
    mixed[0, 1] = 255
    write_mask(directory / "masks/bad.png", mixed)

    frames = [
        {"image": "images/good.png", "mask": "masks/good.png"},
        {"image": "images/bad.png", "mask": "masks/bad.png"},
    ]
    code, run_dir = run_cli(sandbox, monkeypatch, write_manifest(sandbox, "demo01", frames))

    assert code == 1
    records = read_records(run_dir)
    assert len(records) == 2
    by_image = {record["image_path"]: record for record in records}
    assert by_image["images/good.png"]["status"] == "OK"
    assert by_image["images/bad.png"]["status"] == "ERROR"
    assert by_image["images/bad.png"]["errors"][0]["code"] == "INVALID_MASK"
    assert by_image["images/bad.png"]["mask_output"] is None
    assert by_image["images/good.png"]["mask_output"] == "masks/000000.png"
    run_json = read_run_json(run_dir)
    assert (run_json["status"], run_json["ok_count"], run_json["error_count"]) == ("PARTIAL_FAILURE", 1, 1)


def test_canonical_mask_array_uses_the_photo_orientation() -> None:
    canvas = mask_canvas(5, 4, (0, 0))
    decoded = decode_manual_mask(mask_bytes(canvas))
    canonical = canonical_mask_array(decoded, 6)
    assert canonical.shape == (5, 4)
    assert canonical[0, 3], "raw (0,0) under orientation 6 lands at canonical (3,0)"


# --------------------------------------------------------------------------- #
# R1: the mask's source depth comes from the encoded PNG, not the decoded mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "width", "depth", "scanline", "pillow_mode"),
    [
        ("l2_mask", 2, 2, bytes([0x30]), "L"),
        ("l4_mask", 2, 4, bytes([0x0F]), "L"),
        ("l16_mask", 1, 16, struct.pack(">H", 40000), "I;16"),
    ],
)
def test_masks_that_are_not_eight_bit_per_channel_are_refused(
    label: str, width: int, depth: int, scanline: bytes, pillow_mode: str
) -> None:
    """A 2/4-bit greyscale mask decodes to mode ``L``, so mode cannot show this.

    Such a mask is arguably worse than a reduced photo: its values land on the
    foreground boundary, and the ``{0,1}``/``{0,255}`` domain check can still
    accept the expanded result while the boundary a human drew has moved.
    """
    payload = raw_png(width, 1, depth, 0, scanline)
    assert Image.open(BytesIO(payload)).mode == pillow_mode, "fixture must exercise the trap"

    assert _decode_error_code(payload) == "INVALID_MASK", label


def test_eight_bit_masks_in_both_encodings_are_still_accepted() -> None:
    """Success controls for both encodings, so 'refuse everything' cannot pass."""
    zero_255 = decode_manual_mask(raw_png(2, 1, 8, 0, bytes([0, 255])))
    assert zero_255.encoding == "zero_255"
    assert [bool(value) for value in zero_255.foreground.reshape(-1)] == [False, True]

    zero_one = decode_manual_mask(raw_png(2, 1, 8, 0, bytes([0, 1])))
    assert zero_one.encoding == "zero_one"
    assert [bool(value) for value in zero_one.foreground.reshape(-1)] == [False, True]

    all_zero = decode_manual_mask(raw_png(1, 1, 8, 0, bytes([0])))
    assert all_zero.encoding == "all_zero"


def test_low_depth_mask_is_a_row_error_through_the_cli(sandbox, monkeypatch) -> None:
    """Read from disk, the low-depth mask still fails only its own row."""
    directory = dataset_dir(sandbox, "demo01")
    save_image(directory / "images/a.png", gradient_image(2, 1, (0, 0), "RGB"))
    (directory / "masks/a.png").write_bytes(raw_png(2, 1, 4, 0, bytes([0x0F])))

    manifest = write_manifest(
        sandbox, "demo01", [{"image": "images/a.png", "mask": "masks/a.png"}]
    )
    code, run_dir = run_cli(sandbox, monkeypatch, manifest)

    assert code == 1
    record = only_record(run_dir)
    assert record["status"] == "ERROR"
    assert record["errors"][0]["code"] == "INVALID_MASK"
    assert record["mask_output"] is None and record["image_output"] is None
    assert read_run_json(run_dir)["status"] == "PARTIAL_FAILURE"
    assert_strict_json(run_dir / "frames.jsonl", jsonl=True)
