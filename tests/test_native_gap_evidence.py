import json
import numpy as np
import pytest
from integrations.video_masks.narrow_gap import detect_narrow_gaps,GapPolicyNew
from integrations.video_masks.native_gap_evidence import assess_native_gap,_nearest_transfer


def fixture(width=12,scale=1.):
    mask=np.zeros((100,100),bool);mask[10:90,15:85]=True;mask[10:80,40:40+width]=False
    valid=np.ones_like(mask)
    diag=detect_narrow_gaps(mask,valid,GapPolicyNew(min_span_px=20,max_span_px=48))
    n=int(100*scale);lifted=_nearest_transfer(mask,n,n)
    rgb=np.repeat(np.where(lifted,200,20).astype(np.uint8)[:,:,None],3,axis=2)
    return mask,valid,diag,rgb


def assess(mask,diag,rgb,literal=None,candidate=None):
    return assess_native_gap(rgb,mask,mask.copy() if literal is None else literal,
                             mask.copy() if candidate is None else candidate,diag)


@pytest.mark.parametrize('scale',[1.,1.5,2.])
def test_two_clear_banks_support_rgb_only_with_pixel_center_mapping(scale):
    mask,_,diag,rgb=fixture(scale=scale)
    row=assess(mask,diag,rgb)
    assert row['maps']['evidence_code'][50,45]==1
    assert row['quality_status']=='NOT_ACCEPTED' and row['geometry_accepted'] is False
    assert row['stats']['measured_new_segmentation_pixels']==0
    assert row['method_origin']=='ENGINEERING_HEURISTIC_NOT_BOOK_ALGORITHM'
    json.dumps(row['stats'],allow_nan=False)
    i=np.flatnonzero(np.asarray(row['samples']['canonical_xy'])[:,1]==50)[0]
    assert row['samples']['bank_edges_native_xy'][i][0][0]==pytest.approx((39.5+.5)*scale-.5)
    json.dumps({key:value for key,value in row.items() if key!='maps'},allow_nan=False)


def test_same_color_is_weak_not_permission_to_fill():
    mask,_,diag,rgb=fixture();rgb[:]=100
    row=assess(mask,diag,rgb)
    assert row['maps']['evidence_code'][50,45]==2
    assert row['maps']['min_bank_contrast'][50,45]==0
    assert row['mask_changed'] is False and not row['stats']['geometry_background_claim']


def test_only_one_bank_edge_is_weak():
    mask,_,diag,rgb=fixture();rgb[:,52:]=20
    row=assess(mask,diag,rgb)
    assert row['maps']['evidence_code'][50,45]==2


def test_known_invalid_sampling_region_is_unknown_not_weak():
    mask,valid,_,rgb=fixture();valid[50,42]=False
    diag=detect_narrow_gaps(mask,valid,GapPolicyNew(min_span_px=20,max_span_px=48))
    row=assess(mask,diag,rgb)
    assert diag['narrow_external'][49,45]
    assert row['maps']['evidence_code'][49,45]==3
    assert row['maps']['reason_code'][49,45]==6


@pytest.mark.parametrize('width,scale,reason',[(1,4.,3),(2,4.,3),(3,1.,4)])
def test_upsampling_does_not_rescue_extremely_narrow_source_gap(width,scale,reason):
    mask,_,diag,rgb=fixture(width,scale)
    row=assess(mask,diag,rgb)
    assert row['maps']['evidence_code'][50,40]==3
    assert row['maps']['reason_code'][50,40]==reason
    json.dumps({key:value for key,value in row.items() if key!='maps'},allow_nan=False)


def test_native_border_without_foreground_patch_space_is_unknown():
    mask=np.zeros((100,100),bool);mask[:,0:2]=True;mask[:,12:40]=True
    diag=detect_narrow_gaps(mask,np.ones_like(mask),GapPolicyNew(min_span_px=20,max_span_px=48))
    rgb=np.repeat(np.where(mask,200,20).astype(np.uint8)[:,:,None],3,axis=2)
    row=assess(mask,diag,rgb)
    assert row['maps']['evidence_code'][50,5]==3
    assert row['maps']['reason_code'][50,5]==5


def test_ratio_mismatch_crop_and_downsample_rejected():
    mask,_,diag,rgb=fixture()
    with pytest.raises(ValueError):assess(mask,diag,np.zeros((100,101,3),np.uint8))
    with pytest.raises(ValueError):assess(mask,diag,np.zeros((50,50,3),np.uint8))
    with pytest.raises(ValueError):assess(mask,diag,rgb[:,:99])


@pytest.mark.parametrize('bad',[np.zeros((100,100,3),float),np.full((100,100,3),np.nan),np.zeros((100,100),np.uint8)])
def test_nan_or_wrong_rgb_dtype_shape_rejected(bad):
    mask,_,diag,_=fixture()
    with pytest.raises(ValueError):assess(mask,diag,bad)


def test_diag_source_binding_and_mask_order_contracts():
    mask,_,diag,rgb=fixture();wrong=mask.copy();wrong[0,0]=True
    with pytest.raises(ValueError):assess(wrong,diag,rgb)
    literal=mask.copy();literal[20,20]=False
    with pytest.raises(ValueError):assess(mask,diag,rgb,literal=literal)
    corrupt=dict(diag);corrupt['width_upper_bound']=diag['width_upper_bound'].copy();corrupt['width_upper_bound'][50,45]=np.nan
    with pytest.raises(ValueError):assess(mask,corrupt,rgb)


def test_union_proposal_does_not_choose_evidence_and_inputs_unchanged():
    mask,_,diag,rgb=fixture();literal=mask.copy();literal[30:60,40:52]=True
    candidate=mask.copy();saved=[a.copy() for a in (rgb,mask,literal,candidate)]
    original=assess(mask,diag,rgb)
    row=assess(mask,diag,rgb,literal,candidate)
    assert np.array_equal(row['maps']['evidence_code'],original['maps']['evidence_code'])
    assert row['stats']['literal_filled_source_target_pixels']>0 and row['stats']['candidate_filled_source_target_pixels']==0
    assert all(np.array_equal(a,b) for a,b in zip((rgb,mask,literal,candidate),saved))


def test_empty_gap_evidence_keeps_json_safe_unknown_scope():
    mask=np.zeros((30,30),bool);mask[5:20,5:20]=True
    diag=detect_narrow_gaps(mask,np.ones_like(mask));rgb=np.zeros((60,60,3),np.uint8)
    row=assess(mask,diag,rgb)
    assert row['stats']['sampled_bank_pairs']==0 and row['stats']['median_min_bank_contrast'] is None
    assert not row['maps']['target'].any()
    json.dumps(row['stats'],allow_nan=False)
