"""Video entry image identity and argument boundaries; synthetic only."""
from argparse import Namespace
from pathlib import Path
import pytest
from PIL import Image
from scripts.prepare_video import inspect_image, expected_orientation, execute

def test_decoded_rgb_identity_ignores_png_compression(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    im = Image.new("RGB", (8, 6), (12, 22, 33))
    im.save(a, compress_level=0)
    im.save(b, compress_level=9)
    left, right = inspect_image(a), inspect_image(b)
    assert left["sha256"] != right["sha256"]
    assert left["decoded_rgb_sha256"] == right["decoded_rgb_sha256"]
    assert left["sharpness_laplacian_variance"] == 0

def test_pixel_change_changes_identity_and_sharpness(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    im = Image.new("RGB", (8, 6), (0, 0, 0))
    im.save(a)
    im.putpixel((4, 3), (255, 255, 255))
    im.save(b)
    assert inspect_image(a)["decoded_rgb_sha256"] != inspect_image(b)["decoded_rgb_sha256"]
    assert inspect_image(b)["sharpness_laplacian_variance"] > 0

@pytest.mark.parametrize("mode", ["L", "RGBA"], ids=["gray", "alpha"])
def test_non_rgb_rejected(tmp_path, mode):
    p = tmp_path / "bad.png"
    Image.new(mode, (8, 6)).save(p)
    with pytest.raises(ValueError, match="RGB8"):
        inspect_image(p)

def test_orientation_swaps_axes_once():
    timeline = dict(width=1920, height=1080, rotation_degrees=90)
    assert expected_orientation(timeline, 576, 1024)
    assert not expected_orientation(timeline, 1024, 576)
    assert not expected_orientation(timeline, 1080, 1920)

@pytest.mark.parametrize("updates", [
    {"count":1}, {"dataset_id":"../bad"}, {"min_sharpness":float("nan")},
    {"min_sharpness":-1}, {"stream_index":-1}
], ids=["count", "path", "nan", "negative", "stream"])
def test_bad_arguments_rejected_before_io(updates):
    args = Namespace(video="DOES_NOT_EXIST", dataset_id="valid", data_kind="synthetic",
                     output="UNWRITTEN", count=6, stream_index=None, min_sharpness=0,
                     ffmpeg=None, ffprobe=None)
    vars(args).update(updates)
    with pytest.raises(ValueError):
        execute(args)
