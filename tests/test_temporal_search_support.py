import numpy as np
import pytest

from integrations.demo_pipeline.temporal_search_support import derive


def arrays():
    f = np.zeros((20,30),bool)
    f[4:16,6:24] = True
    return f,f.copy(),np.ones_like(f),np.ones_like(f)


def test_missing_control_rejects():
    result = derive(*arrays(),leaveout_passed=False)
    assert not result['allowed'] and not result['search'].any()
    assert 'LEAVEOUT_CONTROL_NOT_PASSED' in result['reasons']


def test_conflict_rejects():
    f,b,fv,bv = arrays()
    b[:] = False
    b[:3,:3] = True
    result = derive(f,b,fv,bv,leaveout_passed=True)
    assert not result['allowed'] and not result['search'].any()
    assert result['uncertain'].sum() == (f|b).sum()


def test_invalid_support_is_not_a_search_proposal():
    f,b,fv,bv = arrays()
    fv[:] = False
    result = derive(f,b,fv,bv,leaveout_passed=True)
    assert not result['allowed'] and not result['search'].any()
    assert 'EMPTY_VALID_INTERSECTION' in result['reasons']


def test_local_region_allowed_without_promoting_fullmask_and_inputs_preserved():
    f,b,fv,bv = arrays()
    fv[:] = False
    fv[8:10,10:12] = True
    originals = [a.copy() for a in (f,b,fv,bv)]
    result = derive(f,b,fv,bv,leaveout_passed=True,
        metrics=dict(bidirectional_iou=1.,area_ratio_forward_backward=1.,
                     area_ratio_backward_forward=1.,full_shape_status='REJECTED'))
    assert result['allowed'] and result['search'].sum() == 4
    assert not result['full_mask_accepted'] and not result['sam_mask']
    assert not result['identity_verified'] and result['search_only']
    assert result['full_shape_status'] == 'REJECTED'
    assert result['usage'] == 'SEARCH_PROPOSAL_ONLY'
    assert np.array_equal(result['uncertain'],(f|b)&~result['search'])
    for original,current in zip(originals,(f,b,fv,bv)):
        assert np.array_equal(original,current)


def test_forged_metrics_raise():
    with pytest.raises(ValueError,match='metric differs'):
        derive(*arrays(),leaveout_passed=True,metrics={'bidirectional_iou':.9})


def test_uint8_and_metric_omission():
    source = arrays()
    result = derive(*[a.astype(np.uint8)*255 for a in source],leaveout_passed=True)
    assert result['allowed'] and result['search'].dtype == bool


@pytest.mark.parametrize('value',[np.ones((20,30),np.uint8),np.ones((20,30),np.float32)])
def test_nonbinary_dtype_or_encoding_rejected(value):
    f,b,fv,bv = arrays()
    with pytest.raises(ValueError,match='binary'):
        derive(value,b,fv,bv,leaveout_passed=True)
