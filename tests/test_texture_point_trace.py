import json
import numpy as np
import pytest
from integrations.demo_pipeline.texture_point_trace import trace


def data():
    rng = np.random.default_rng(402)
    frame = rng.integers(35, 180, (220, 260), dtype=np.uint8)
    second = np.full_like(frame, 20)
    second[:, 3:] = frame[:, :-3]
    return [frame, second], [np.full_like(frame, 255) for _ in range(2)]


def cv_api():
    import cv2
    assert hasattr(cv2, 'findTransformECCWithMask'), 'Pinned masked ECC API required'
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    return cv2


def test_real_translation_brightness_and_fractional_seed():
    frames, masks = data()
    frames[1] = np.rint(frames[1].astype(float)*1.1+4).astype(np.uint8)
    result = trace(frames, masks, [555, 556], [0., .04], [110.2, 100.7], cv_api())
    assert result['frames'][0]['accepted_xy'] == [110.2, 100.7]
    assert result['frames'][1]['status'] == 'ACCEPTED', result['frames'][1]['reasons']
    np.testing.assert_allclose(result['frames'][1]['accepted_xy'], [113.2, 100.7], atol=.25)
    assert result['frames'][1]['forward']['holdout']['ncc'] > .9
    assert result['frames'][1]['cycle_error_px'] < 1.5
    json.dumps(result, allow_nan=False)


def test_center_hole_stops_and_never_recovers():
    frames, masks = data()
    frames.append(frames[0].copy()); masks.append(masks[0].copy())
    masks[1][100:103, 112:115] = 0
    result = trace(frames, masks, [0, 1, 2], [0., 1., 2.], [110.2, 100.7], cv_api())
    assert result['frames'][1]['accepted_xy'] is None
    assert result['frames'][2]['status'] == 'NOT_RUN_AFTER_STOP'
    assert result['frames'][2]['candidate_xy'] is None


def test_unknown_mask_no_native_and_permanent_stop():
    frames, masks = data()
    class NoNative:
        def __getattr__(self, name):
            raise AssertionError('native call after missing source mask')
    result = trace(frames, [None, masks[1]], [0, 1], [0., 1.], [110.2, 100.7], NoNative())
    assert result['frames'][0]['reasons'] == ['UNKNOWN_MASK']
    assert result['frames'][1]['status'] == 'NOT_RUN_AFTER_STOP'
    assert result['native_calls']['matchTemplate'] == 0


def test_repeated_texture_stops():
    frames, masks = data()
    # Two exact copies more than 12px apart inside the same search ROI.
    patch = frames[0][85:118, 94:127].copy()
    frames[1][85:118, 97:130] = patch
    frames[1][85:118, 140:173] = patch
    result = trace(frames, masks, [0, 1], [0., 1.], [110.2, 100.7], cv_api())
    assert result['frames'][1]['accepted_xy'] is None
    assert 'AMBIGUOUS_REPEATED_TEXTURE' in result['frames'][1]['reasons']


def test_malformed_mask_throws():
    frames, masks = data(); masks[1] = masks[1].astype(float)
    with pytest.raises(ValueError, match='MASK'):
        trace(frames, masks, [0, 1], [0., 1.], [110.2, 100.7], object())


@pytest.mark.parametrize('missing', [False, True])
def test_occlusion_or_unknown_target_stops_without_later_native(missing):
    frames, masks = data()
    frames.append(frames[0].copy()); masks.append(masks[0].copy())
    masks[1] = None if missing else np.zeros_like(masks[1])
    class Count:
        def __init__(self, cv):
            self.cv = cv; self.calls = 0
        def __getattr__(self, name):
            return getattr(self.cv, name)
        def matchTemplate(self, *args, **kwargs):
            self.calls += 1
            return self.cv.matchTemplate(*args, **kwargs)
        def findTransformECCWithMask(self, *args, **kwargs):
            self.calls += 1
            return self.cv.findTransformECCWithMask(*args, **kwargs)
    cv = Count(cv_api())
    result = trace(frames, masks, [0, 1, 2], [0., 1., 2.], [110.2, 100.7], cv)
    assert result['frames'][1]['accepted_xy'] is None
    assert result['frames'][2]['status'] == 'NOT_RUN_AFTER_STOP'
    assert sum(result['native_calls'].values()) == cv.calls
    if missing:
        assert cv.calls == 0
        assert result['frames'][1]['reasons'] == ['UNKNOWN_MASK']
