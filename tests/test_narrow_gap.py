import json
import math
import numpy as np
import pytest
from integrations.video_masks.narrow_gap import (
    GapPolicyNew,detect_narrow_gaps,evaluate_gap_preservation,PRESERVED,REJECTED)


def u_shape(width=6):
    mask=np.zeros((120,120),bool)
    mask[15:100,25:85]=True
    x=50
    mask[15:85,x:x+width]=False
    return mask,np.ones_like(mask),(55,x+width//2)


@pytest.mark.parametrize('width',[1,3,6,12])
def test_u_opening_width_is_bank_centres_and_not_eroded(width):
    mask,valid,p=u_shape(width)
    d=detect_narrow_gaps(mask,valid,GapPolicyNew(min_span_px=20,max_span_px=48))
    assert d['exterior'][p] and d['narrow_external'][p]
    assert d['width_upper_bound'][p]==width+1
    assert d['open_span'][p]==width
    assert d['subresolution'][p]==(width<3)
    assert d['protected_background'][p]
    assert d['uncertainty_margin'][p]==min(3,(width+1)*.2)
    a,b=d['bank_a_xy'][p],d['bank_b_xy'][p]
    assert mask[a[1],a[0]] and mask[b[1],b[0]]
    assert np.linalg.norm(a-b)==d['width_upper_bound'][p]


def test_default_relative_span_limit_is_not_a_universal_gap_width():
    mask,valid,p=u_shape(12)
    d=detect_narrow_gaps(mask,valid)
    assert d['stats']['maximum_bank_span_pixels']==6
    assert d['width_upper_bound'][p]==13 and not d['narrow_external'][p]
    assert not d['enclosed'][p]


def test_rotation_preserves_width_and_narrow_region():
    mask,valid,_=u_shape(3)
    a=detect_narrow_gaps(mask,valid)
    b=detect_narrow_gaps(np.rot90(mask),np.rot90(valid))
    assert np.array_equal(np.rot90(a['narrow_external']),b['narrow_external'])
    assert np.allclose(np.rot90(a['width_upper_bound']),b['width_upper_bound'],equal_nan=True)


def test_diagonal_bank_distance_is_euclidean_and_path_valid():
    mask=np.zeros((31,31),bool);mask[13,13]=True;mask[17,17]=True
    d=detect_narrow_gaps(mask,np.ones_like(mask))
    assert d['direction'][15,15]==2
    assert d['width_upper_bound'][15,15]==pytest.approx(4*math.sqrt(2))
    assert d['narrow_external'][15,15]
    valid=np.ones_like(mask);valid[14,14]=False
    blocked=detect_narrow_gaps(mask,valid)
    assert np.isnan(blocked['width_upper_bound'][15,15])


def test_enclosed_hole_protected_regardless_of_width_limit():
    mask=np.zeros((80,80),bool);mask[10:70,10:70]=True;mask[25:55,25:55]=False
    valid=np.ones_like(mask);d=detect_narrow_gaps(mask,valid)
    assert d['enclosed'][40,40] and not d['exterior'][40,40]
    assert d['protected_background'][40,40]
    proposal=mask.copy();proposal[40,40]=True
    result=evaluate_gap_preservation(mask,proposal,valid,d)
    assert not result['ok'] and 'PROTECTED_BACKGROUND_FILLED' in result['violations']


def test_four_connectivity_does_not_leak_through_diagonal_contact():
    mask=np.ones((7,7),bool);mask[0,0]=False;mask[1,1]=False
    d=detect_narrow_gaps(mask,np.ones_like(mask))
    assert d['exterior'][0,0] and d['enclosed'][1,1]
    assert d['labels'][0,0]!=d['labels'][1,1]


def test_no_known_exterior_path_and_unknown_adjacency_is_unresolved():
    mask=np.zeros((80,80),bool);mask[10:70,10:70]=True;mask[25:55,25:55]=False
    valid=np.ones_like(mask);valid[25,40]=False
    d=detect_narrow_gaps(mask,valid)
    assert d['unresolved'][40,40] and d['unresolved'][25,40]
    assert not d['enclosed'][40,40]
    # A different background region has an explicit valid route to the image edge.
    valid[2,2]=False
    d=detect_narrow_gaps(mask,valid)
    assert d['exterior'][3,2]
    assert d['component_touches_unknown'][d['labels'][3,2]]


def test_object_touching_image_border_does_not_invent_background_path():
    mask=np.ones((30,30),bool);mask[10:20,10:20]=False
    d=detect_narrow_gaps(mask,np.ones_like(mask))
    assert d['enclosed'][15,15] and not d['exterior'].any()
    assert d['stats']['exterior_components']==0


def test_one_pixel_gap_stays_protected_despite_subresolution_flag():
    source,valid,p=u_shape(1);d=detect_narrow_gaps(source,valid)
    candidate=source.copy();candidate[55,50]=True
    result=evaluate_gap_preservation(source,candidate,valid,d)
    assert d['subresolution'][p] and result['status']==REJECTED
    assert 'PROTECTED_BACKGROUND_FILLED' in result['violations']


def test_sealed_mouth_outside_narrow_mask_is_detected_as_path_loss():
    source=np.zeros((120,120),bool);source[10:110,20:100]=True
    source[10:50,35:85]=False   # wide open mouth (not narrow)
    source[50:95,55:58]=False   # narrow deeper recess
    valid=np.ones_like(source);d=detect_narrow_gaps(source,valid)
    assert d['narrow_external'][70,56]
    assert not d['narrow_external'][25,35:85].any()
    candidate=source.copy();candidate[25,35:85]=True
    result=evaluate_gap_preservation(source,candidate,valid,d)
    assert result['stats']['protected_filled_pixels']==0
    assert 'EXTERIOR_BACKGROUND_PATH_DISCONNECTED' in result['violations']
    assert result['path_disconnected'][70,56]
    assert result['stats']['erased_source_components']==0


def test_original_background_component_cannot_disappear():
    source=np.ones((20,20),bool);source[9:11,9:11]=False
    valid=np.ones_like(source);d=detect_narrow_gaps(source,valid)
    result=evaluate_gap_preservation(source,np.ones_like(source),valid,d)
    assert 'SOURCE_BACKGROUND_COMPONENT_ERASED' in result['violations']
    assert result['stats']['erased_source_components']==1


def test_identity_pass_and_json_stats_have_no_nan():
    source,valid,_=u_shape(3);d=detect_narrow_gaps(source,valid)
    result=evaluate_gap_preservation(source,source.copy(),valid,d)
    assert result['status']==PRESERVED and result['ok'] and result['preserved']
    json.dumps(d['stats'],allow_nan=False);json.dumps(result['stats'],allow_nan=False)
    empty=detect_narrow_gaps(np.zeros((20,20),bool),np.ones((20,20),bool))
    assert np.isnan(empty['width_upper_bound']).all()
    json.dumps(empty['stats'],allow_nan=False)


def test_sources_never_mutated_and_stale_diagnostics_rejected():
    source,valid,_=u_shape(3);candidate=source.copy();copies=[x.copy() for x in (source,valid,candidate)]
    d=detect_narrow_gaps(source,valid);evaluate_gap_preservation(source,candidate,valid,d)
    assert all(np.array_equal(a,b) for a,b in zip((source,valid,candidate),copies))
    changed=source.copy();changed[0,0]=True
    with pytest.raises(ValueError):evaluate_gap_preservation(changed,candidate,valid,d)


def test_limits_invalid_domain_and_check_callback():
    source,valid,_=u_shape(3)
    with pytest.raises(ValueError):detect_narrow_gaps(source,valid,GapPolicyNew(max_pixels=10))
    with pytest.raises(ValueError):detect_narrow_gaps(source,valid,GapPolicyNew(max_skew_pixels=10))
    invalid=valid.copy();invalid[20,30]=False
    with pytest.raises(ValueError):detect_narrow_gaps(source,invalid)
    def stop():raise TimeoutError('test')
    with pytest.raises(TimeoutError):detect_narrow_gaps(source,valid,check=stop)
    for args in ({'min_span_px':49},{'max_span_fraction':True},{'subresolution_px':float('nan')}):
        with pytest.raises(ValueError):GapPolicyNew(**args)
