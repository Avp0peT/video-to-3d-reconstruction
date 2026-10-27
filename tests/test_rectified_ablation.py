import numpy as np
import pytest
from kozlov_lab.stereo.baseline import rectified_ablation as ab
from kozlov_lab.stereo.baseline.rectified_parallel import calibrated_rectification


def fixture(planar=False, negative=False):
    k = np.array(((700, 0, 320), (0, 700, 240), (0, 0, 1)), float)
    rec = calibrated_rectification(k, np.eye(3), [0, 0, 0], k, np.eye(3), [-1, 0, 0])
    # Binary rational surrogate control avoids accidental geometric rank changes.
    left = np.array(((.125,.125),(.25,.125),(.125,.25),(.3,.3),(.4,.2),(.2,.4)))
    left = np.round(left*262144)/262144
    depth = np.array((.125,.125,.125,.25,.0625,.1875))
    if planar: depth[:] = .125
    if negative: depth = -depth
    right = left.copy(); right[:,0] -= depth
    return left*700+[320,240], right*700+[320,240], rec


def test_correct_control_bc_equal_and_all_labels_retained():
    a,b,rec = fixture(); result = ab.evaluate_six(a,b,rec,[9,3,6,2,8,1])
    assert result['C']['independent']['status'] == 'VERIFIED'
    assert result['C']['algorithm_called'] == 'AlgT_prime_both_views'
    assert result['C']['projection_readback_exact']
    np.testing.assert_array_equal(result['B']['world_xyz'], result['C']['world_xyz'])
    assert result['errors']['B_vs_C']['max_abs'] == 0
    np.testing.assert_array_equal(result['raw_rectified1'], a)
    assert result['labels'] == (9,3,6,2,8,1)


def test_common_y_and_quantization_difference():
    a,b,rec = fixture(); a += [.001,.001]; b += [.002,-.003]
    result = ab.evaluate_six(a,b,rec,range(6))
    assert result['errors']['A_vs_B']['max_abs'] > 0
    np.testing.assert_array_equal(result['A']['first_pixels'][:,1], result['A']['second_pixels'][:,1])
    np.testing.assert_array_equal(result['B']['world_xyz'], result['C']['world_xyz'])


def test_planar_and_negative_preserve_alg_status():
    for planar, negative in ((True,False),(False,True)):
        a,b,rec = fixture(planar,negative)
        result = ab.evaluate_six(a,b,rec,range(6))
        if planar:
            assert result['C']['status'] == 'UNSUPPORTED_PLANAR'
            assert result['A']['world_xyz'] is not None
        else:
            assert result['C']['exact_status'] == 'CONSTRUCTED'
            assert result['C']['status'] == 'WORLD_CONVERSION_REFUSED'
            assert result['A']['world_xyz'] is None
        assert result['C']['world_xyz'] is None


def test_tampered_certificate_fails_independent_verification(monkeypatch):
    original = ab.given.assess_given_parallel_input
    def tamper(*args, **kwargs):
        result = original(*args, **kwargs)
        result['certificate']['verdict'] = 'UNKNOWN'
        return result
    monkeypatch.setattr(ab.given, 'assess_given_parallel_input', tamper)
    a,b,rec = fixture(); result = ab.evaluate_six(a,b,rec,range(6))
    assert result['C']['status'] == 'INDEPENDENT_INVALID'
    assert result['C']['world_xyz'] is None
    assert result['B']['world_xyz'] is not None


@pytest.mark.parametrize('stage', ['construct', 'verify'])
def test_budget_failure_cannot_borrow_baseline_world(monkeypatch, stage):
    if stage == 'construct':
        monkeypatch.setattr(ab.given, 'assess_given_parallel_input',
            lambda *a, **k: dict(status='UNKNOWN',reason='budget',certificate=None))
    else:
        monkeypatch.setattr(ab.given, 'assess_given_parallel_certificate',
            lambda *a, **k: dict(status='UNKNOWN',reason='budget'))
    a,b,rec = fixture(); result = ab.evaluate_six(a,b,rec,range(6))
    assert result['C']['status'] in ('UNKNOWN','INDEPENDENT_UNKNOWN')
    assert result['C']['world_xyz'] is None
    assert result['A']['world_xyz'] is not None and result['B']['world_xyz'] is not None


def test_construction_exception_keeps_baselines(monkeypatch):
    def fail(*a, **k): raise ValueError('construction failed')
    monkeypatch.setattr(ab.given, 'assess_given_parallel_input', fail)
    a,b,rec = fixture(); result = ab.evaluate_six(a,b,rec,range(6))
    assert result['C']['status'] == 'C_STAGE_INVALID'
    assert result['C']['world_xyz'] is None and result['B']['world_xyz'] is not None


def test_projection_failure_retains_constructed_verified_facts(monkeypatch):
    def fail(*a, **k): raise ValueError('projection readback failed')
    monkeypatch.setattr(ab, '_projection_readback', fail)
    a,b,rec = fixture(); result = ab.evaluate_six(a,b,rec,range(6))
    assert result['C']['status'] == 'C_STAGE_INVALID'
    assert result['C']['exact_status'] == 'CONSTRUCTED'
    assert result['C']['independent']['status'] == 'VERIFIED'
    assert result['C']['world_xyz'] is None
    assert result['B']['world_xyz'] is not None


@pytest.mark.parametrize('labels', [[1,2,3,4,5], [1,2,3,4,5,5], [True,2,3,4,5,6], [np.int64(1),2,3,4,5,6]])
def test_full_pure_int_labels(labels):
    a,b,rec = fixture()
    with pytest.raises(ValueError): ab.evaluate_six(a,b,rec,labels)


def test_complete_pixels_required():
    a,b,rec = fixture()
    with pytest.raises(ValueError): ab.evaluate_six(a[:5],b[:5],rec,range(6))
