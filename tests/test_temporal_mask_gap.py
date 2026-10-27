"""Injected flow tests; native synthesis is explicitly gated."""
import json
import os

import numpy as np
import pytest

from integrations.demo_pipeline import temporal_mask_gap as gap


def fixture_inputs():
    images = [np.zeros((48,64), np.uint8) for _ in range(3)]
    first = np.zeros(images[0].shape, np.uint8)
    first[12:36,16:40] = 255
    last = np.zeros_like(first)
    last[12:36,20:44] = 255
    return images, first, last


def injected(monkeypatch, dx=2., bad_cycle=False):
    calls = []
    def flow(a,b,cv):
        sign = 1 if len(calls)%2 == 0 else -1
        out = np.zeros((*a.shape,2), np.float32)
        out[...,0] = dx*sign if not (bad_cycle and sign < 0) else dx
        calls.append((a.shape,b.shape))
        return out
    monkeypatch.setattr(gap, '_flow', flow)
    return calls


def test_inverse_warp_direction_and_input_preservation(monkeypatch):
    cv = pytest.importorskip('cv2')
    images, first, last = fixture_inputs()
    copies = [a.copy() for a in images+[first,last]]
    calls = injected(monkeypatch)
    result = gap.propagate_gap(images,first,last,cv)
    expected = np.zeros_like(first, bool)
    expected[12:36,18:42] = True
    assert np.array_equal(result['support'][0], expected)
    assert len(calls) == result['diagnostics']['flow_calls'] == 4
    assert result['diagnostics']['gaps'][0]['reasons'] == []
    assert not result['actual_SAM'] and not result['tracking_identity']
    json.dumps(result['diagnostics'],allow_nan=False)
    for before, after in zip(copies, images+[first,last]):
        assert np.array_equal(before, after)


def test_cycle_inconsistency_rejects_support(monkeypatch):
    cv = pytest.importorskip('cv2')
    images, first, last = fixture_inputs()
    injected(monkeypatch,bad_cycle=True)
    result = gap.propagate_gap(images,first,last,cv)
    assert not result['support'][0].any()
    assert not result['valid']['forward'][1].any()
    assert result['uncertainty'][0].any()
    assert 'ENDPOINT_CHECK_FAILED' in result['diagnostics']['gaps'][0]['reasons']


def test_out_of_bounds_is_unknown(monkeypatch):
    cv = pytest.importorskip('cv2')
    images, first, last = fixture_inputs()
    injected(monkeypatch)
    result = gap.propagate_gap(images,first,last,cv)
    assert not result['valid']['forward'][1][:,:2].any()
    assert not result['valid']['backward'][1][:,-2:].any()
    assert not result['support'][0][:,:2].any()
    # A zero mask sample outside valid is not certified background.
    assert not result['predictions']['forward'][1][:,:2].any()


def test_conflicting_anchors_reject_without_erasing_predictions(monkeypatch):
    cv = pytest.importorskip('cv2')
    images, first, last = fixture_inputs()
    last[:] = 0
    last[1:9,45:58] = 255
    injected(monkeypatch)
    result = gap.propagate_gap(images,first,last,cv)
    assert not result['support'][0].any()
    assert result['predictions']['forward'][1].any()
    assert result['predictions']['backward'][1].any()
    assert result['uncertainty'][0].any()
    assert 'ENDPOINT_CHECK_FAILED' in result['diagnostics']['gaps'][0]['reasons']


def test_working_bound_and_original_resolution_return(monkeypatch):
    cv = pytest.importorskip('cv2')
    images = [np.zeros((600,1000),np.uint8) for _ in range(3)]
    mask = np.zeros_like(images[0])
    mask[150:450,250:750] = 255
    calls = injected(monkeypatch,dx=0.)
    result = gap.propagate_gap(images,mask,mask,cv)
    assert len(calls) == 4
    assert all(max(a) <= 512 and a == b for a,b in calls)
    assert result['support'][0].shape == (600,1000)
    assert result['support'][0].dtype == bool


def test_endpoint_tiny_valid_domain_cannot_hide_missing_actual_foreground(monkeypatch):
    cv = pytest.importorskip('cv2')
    images, first, _ = fixture_inputs()
    injected(monkeypatch,dx=0.)
    def tiny_warp(mask, valid, forward, backward, cv):
        small = np.zeros_like(valid)
        small[12:16,16:40] = True
        # Perfect agreement on the tiny domain used to appear sufficient.
        return mask.copy(), small, dict(valid_fraction=float(small.mean()),
            inverse_map_inbounds_fraction=1.,cycle_consistent_fraction=1.,
            valid_foreground_fraction=gap._retained(mask,small))
    monkeypatch.setattr(gap,'_warp',tiny_warp)
    result = gap.propagate_gap(images,first,first,cv)
    assert not result['support'][0].any()
    for row in result['diagnostics']['directions'].values():
        assert row['endpoint_iou_valid'] == 1.
        assert row['endpoint_iou_full'] < .65
        assert row['actual_endpoint_foreground_valid_fraction'] < .80
        assert 'ENDPOINT_IOU_LOW' in row['endpoint_reasons']
        assert 'ACTUAL_ENDPOINT_FOREGROUND_VALID_LOW' in row['endpoint_reasons']


@pytest.mark.parametrize('missing', [None, np.zeros((48,64),np.uint8)])
def test_missing_anchor_rejected_before_flow(monkeypatch,missing):
    images, first, last = fixture_inputs()
    def forbidden(*args):
        raise AssertionError('flow must not run')
    monkeypatch.setattr(gap,'_flow',forbidden)
    with pytest.raises(ValueError,match='two nonempty actual'):
        gap.propagate_gap(images,first,missing,None)


@pytest.mark.skipif(os.environ.get('GAP_SUPERVISED') != '1',reason='explicit CPU native supervision required')
def test_actual_cpu_synthetic_translation():
    cv = pytest.importorskip('cv2')
    rng = np.random.default_rng(7)
    image = rng.integers(0,256,(96,128),dtype=np.uint8)
    images = [cv.warpAffine(image,np.float32([[1,0,2*i],[0,1,0]]),(128,96)) for i in range(3)]
    first = np.zeros_like(image)
    first[24:72,32:88] = 255
    last = cv.warpAffine(first,np.float32([[1,0,4],[0,1,0]]),(128,96),flags=cv.INTER_NEAREST)
    result = gap.propagate_gap(images,first,last,cv)
    assert result['diagnostics']['gaps'][0]['status'] == 'ACCEPTED_TEMPORAL_SUPPORT'
    assert result['support'][0].any()
