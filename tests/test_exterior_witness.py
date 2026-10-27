import json
import numpy as np
import pytest
import cv2 as cv
from kozlov_lab.stereo.exterior_witness import track_reference_exterior

cv.setNumThreads(2)


def scene():
    rng = np.random.default_rng(20261017)
    first = rng.integers(0, 256, (144, 192), dtype=np.uint8)
    images = [cv.warpAffine(first, np.float32([[1, 0, t], [0, 1, 0]]), (192, 144)) for t in range(3)]
    mask = np.zeros_like(first); mask[48:96, 72:120] = 255
    after = cv.warpAffine(mask, np.float32([[1, 0, 2], [0, 1, 0]]), (192, 144))
    return images, mask, after


def test_real_opencv_translation_and_no_mutation():
    images, before, after = scene()
    snapshots = [v.copy() for v in [*images, before, after]]
    out = track_reference_exterior(images, before, after, [80, 81, 82], cv)
    assert out['counts']['seeds'] > 50
    assert out['counts']['accepted'] > 40
    assert out['counts']['seeds'] == out['counts']['rejected_continuous'] + out['counts']['continuous_survivors']
    assert out['counts']['continuous_survivors'] == out['counts']['accepted'] + out['counts']['rejected_endpoint']
    for witness in out['accepted_witnesses']:
        points = witness['observations']
        assert [p['frame_index'] for p in points] == [80, 81, 82]
        assert np.allclose(np.subtract(points[-1]['xy'], points[0]['xy']), [2, 0], atol=.2)
        assert witness['verified_background'] is False
        assert witness['physical_identity'] == 'UNVERIFIED'
        assert witness['endpoint_distance_px'] >= 8
    for seed in out['seed_ledger']:
        x, y = seed['point']; assert 5 <= x <= 186 and 5 <= y <= 138
        assert 8 <= seed['before_exterior_distance_px'] <= 48
    assert all(np.array_equal(a,b) for a,b in zip([*images,before,after],snapshots))
    json.dumps(out, allow_nan=False)


def test_real_uniform_no_corners():
    images, before, after = scene()
    out = track_reference_exterior([np.zeros_like(images[0]) for _ in range(3)], before, after, [1,2,3], cv)
    assert out['counts']['seeds'] == 0
    assert out['step_records'] == out['accepted_witnesses'] == []


def test_real_endpoint_inside_rejected():
    images, before, after = scene()
    out = track_reference_exterior(images, before, np.full_like(after,255), [1,2,3], cv)
    assert out['counts']['continuous_survivors'] > 40
    assert out['counts']['accepted'] == 0
    assert all(c['reasons'] == ['endpoint_inside_original_after_mask'] for c in out['all_chains'] if c['continuous'])


class DropShim:
    """Real OpenCV except explicit injected forward failure for every seed."""
    def __init__(self): self.calls = 0
    def __getattr__(self, name): return getattr(cv, name)
    def calcOpticalFlowPyrLK(self, a, b, p, unused, **kwargs):
        self.calls += 1
        return np.full_like(p,np.nan), np.zeros((len(p),1),np.uint8), np.full((len(p),1),np.nan)


def test_injected_nan_dropout_never_backqueries_or_reseeds():
    images, before, after = scene(); shim = DropShim()
    out = track_reference_exterior(images,before,after,[83,84,85],shim)
    assert shim.calls == 1
    assert len(out['step_records']) == out['counts']['seeds']
    assert out['counts']['continuous_survivors'] == 0
    assert all(r['forward_point'] is None and r['backward_point'] is None for r in out['step_records'])
    json.dumps(out,allow_nan=False)


class FBShim:
    """Real OpenCV with injected backward displacement to test irreversible FB gate."""
    def __init__(self): self.calls=0
    def __getattr__(self,name): return getattr(cv,name)
    def calcOpticalFlowPyrLK(self,*args,**kwargs):
        self.calls+=1
        q,s,e=cv.calcOpticalFlowPyrLK(*args,**kwargs)
        if self.calls == 2: q=q+np.float32([[[3,0]]])
        return q,s,e


def test_injected_fb_failure():
    images,before,after=scene(); shim=FBShim()
    out=track_reference_exterior(images,before,after,[83,84,85],shim)
    assert shim.calls == 2
    assert out['counts']['accepted'] == 0
    assert any('fb_error' in r['reasons'] for r in out['step_records'])


def test_real_occlusion_stops_chains():
    images,before,after=scene(); images[1][:]=0
    out=track_reference_exterior(images,before,after,[1,2,3],cv)
    assert out['counts']['accepted'] == 0
    assert out['counts']['rejected_continuous'] == out['counts']['seeds']


@pytest.mark.parametrize('change', ['bool_ids','gap','duplicate','float_ids','short','long','dtype','shape','empty_mask','mask_dtype','nonbinary','large'])
def test_invalid_inputs(change):
    images,before,after=scene(); ids=[1,2,3]
    if change=='bool_ids': ids=[True,2,3]
    elif change=='gap': ids=[1,3,4]
    elif change=='duplicate': ids=[1,1,2]
    elif change=='float_ids': ids=[1.,2.,3.]
    elif change=='short': images=images[:2]; ids=ids[:2]
    elif change=='long': images=images*3; ids=list(range(9))
    elif change=='dtype': images[0]=images[0].astype(float)
    elif change=='shape': images[0]=images[0][1:]
    elif change=='empty_mask': before[:]=0
    elif change=='mask_dtype': before=before.astype(bool)
    elif change=='nonbinary': before[0,0]=2
    elif change=='large': images=[np.zeros((12,1025),np.uint8)]*3
    with pytest.raises(ValueError): track_reference_exterior(images,before,after,ids,cv)

class ClipShim:
    """Inject finite camera-clipped forward coordinates; never send them back."""
    def __init__(self): self.calls=0
    def __getattr__(self,name): return getattr(cv,name)
    def calcOpticalFlowPyrLK(self,a,b,p,unused,**kwargs):
        self.calls+=1
        if self.calls > 1:
            assert np.isfinite(p).all() and (p[...,0] >= 4).all()
        q,s,e=cv.calcOpticalFlowPyrLK(a,b,p,unused,**kwargs)
        if self.calls == 1: q[0,0]=[-1,20]
        return q,s,e


def test_injected_camera_clip_is_stopped_with_actual_coordinate():
    images,before,after=scene(); shim=ClipShim()
    out=track_reference_exterior(images,before,after,[80,81,82],shim)
    chain=out['all_chains'][0]
    assert chain['continuous'] is False and chain['accepted'] is False
    assert len(chain['steps']) == 1
    assert chain['steps'][0]['forward_point'] == [-1.,20.]
    assert chain['steps'][0]['backward_point'] is None
    assert 'forward_nonfinite_or_patch_bounds' in chain['reasons']


def test_rejects_one_valued_binary_mask():
    images,before,after=scene()
    before[before != 0]=1
    with pytest.raises(ValueError,match='binary uint8'):
        track_reference_exterior(images,before,after,[80,81,82],cv)


def test_rejects_negative_frame_indices():
    images,before,after=scene()
    with pytest.raises(ValueError,match='integers'):
        track_reference_exterior(images,before,after,[-1,0,1],cv)
