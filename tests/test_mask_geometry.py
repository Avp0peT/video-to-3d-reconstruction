"""Tiny native-camera checks, not photo reconstruction or MVS consumption."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pycolmap
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mask_geometry_under_test",
    ROOT / "integrations/mvs_masks/mask_geometry.py")
geometry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geometry
spec.loader.exec_module(geometry)


def cam(w=4, h=3, fx=4, fy=4, cx=2, cy=1.5):
    return pycolmap.Camera(model="PINHOLE", width=w, height=h, params=[fx, fy, cx, cy])


def pattern():
    return np.array([[0,255,0,255],[255,0,255,0],[0,0,255,255]], dtype=np.uint8)


def test_identity_keeps_every_pixel_and_source_bytes():
    source = pattern()
    before = source.tobytes()
    camera = cam()
    prior = camera.params.copy()
    result = geometry.warp_binary_mask(source, camera, camera, tile_rows=1)
    np.testing.assert_array_equal(result.mask, source)
    assert np.all(result.valid == 255)
    assert result.counts == dict(total=12,valid=12,unknown=0,invalid_projection=0,outside_source=0,foreground=6)
    assert source.tobytes() == before
    np.testing.assert_array_equal(camera.params, prior)


def test_downscale_center_and_tie_rule_use_columns_one_and_three():
    source = pattern()
    result = geometry.warp_binary_mask(source, cam(), cam(w=2,fx=2,cx=1))
    np.testing.assert_array_equal(result.mask, source[:, [1,3]])
    assert np.all(result.valid == 255)


def test_translation_has_unknown_border_without_edge_clamp():
    source = np.full((3,4),255,dtype=np.uint8)
    result = geometry.warp_binary_mask(source,cam(),cam(cx=3))
    np.testing.assert_array_equal(result.mask, [[0,255,255,255]]*3)
    np.testing.assert_array_equal(result.valid, result.mask)
    assert result.counts["unknown"] == result.counts["outside_source"] == 3


def test_complete_radial_model_changes_support_not_just_intrinsics():
    source = np.full((3,5),255,dtype=np.uint8)
    distorted = pycolmap.Camera(model="SIMPLE_RADIAL",width=5,height=3,params=[2,2.5,1.5,0.5])
    result = geometry.warp_binary_mask(source,distorted,cam(w=5,fx=2,fy=2,cx=2.5))
    np.testing.assert_array_equal(result.mask, [[0,255,255,255,0]]*3)
    assert result.counts["valid"] == 9 and result.counts["unknown"] == 6


def test_tile_size_does_not_change_labels_or_validity():
    a=geometry.warp_binary_mask(pattern(),cam(),cam(w=8,h=6,fx=8,fy=8,cx=4,cy=3),tile_rows=1)
    b=geometry.warp_binary_mask(pattern(),cam(),cam(w=8,h=6,fx=8,fy=8,cx=4,cy=3),tile_rows=64)
    np.testing.assert_array_equal(a.mask,np.repeat(np.repeat(pattern(),2,axis=0),2,axis=1))
    np.testing.assert_array_equal(a.mask,b.mask)
    np.testing.assert_array_equal(a.valid,b.valid)


@pytest.mark.parametrize("bad", [np.ones((3,4),dtype=np.uint8),np.zeros((3,4),dtype=bool),
                                 np.zeros((3,4),dtype=np.float32),np.zeros((4,3),dtype=np.uint8)])
def test_labels_dtype_and_size_are_strict(bad):
    with pytest.raises(ValueError): geometry.warp_binary_mask(bad,cam(),cam())


@pytest.mark.parametrize("kwargs", [dict(tile_rows=0),dict(tile_rows=True),dict(max_pixels=11)])
def test_resource_arguments_are_checked_before_mapping(kwargs):
    with pytest.raises(ValueError): geometry.warp_binary_mask(pattern(),cam(),cam(),**kwargs)


def test_distorted_destination_is_rejected():
    target=pycolmap.Camera(model="SIMPLE_RADIAL",width=4,height=3,params=[4,2,1.5,0.01])
    with pytest.raises(ValueError,match="target"):geometry.warp_binary_mask(pattern(),cam(),target)


def test_invalid_calibration_is_rejected():
    with pytest.raises(ValueError,match="focal"):geometry.warp_binary_mask(pattern(),cam(fx=0),cam())


def test_nonfinite_projection_is_unknown_without_integer_cast_or_clamp():
    class ProjectionFailure:
        def __init__(self): self.inner=cam()
        def __getattr__(self,name): return getattr(self.inner,name)
        def img_from_cam(self,rays):
            xy=self.inner.img_from_cam(rays)
            xy[0]=[np.nan,np.nan]
            xy[1]=[np.inf,0]
            return xy
    result=geometry.warp_binary_mask(np.full((3,4),255,dtype=np.uint8),ProjectionFailure(),cam())
    assert result.counts["invalid_projection"] == result.counts["unknown"] == 2
    np.testing.assert_array_equal(result.valid[0], [0,0,255,255])
    assert result.mask[0,0] == result.mask[0,1] == 0
