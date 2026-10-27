import copy
import json
import numpy as np
import pytest
from integrations.mvs_surface.boundary_image_evidence import inspect_mask_boundary as inspect


def scene():
    mask=np.zeros((120,120),bool); mask[30:90,30:90]=True
    image=np.repeat((mask.astype(np.uint8)*220)[:,:,None],3,axis=2)
    valid=np.ones(mask.shape,bool)
    region=np.zeros_like(mask); region[20:40,45:75]=True
    return image,mask,valid,region


def test_aligned_rectangle():
    image,mask,valid,region=scene(); result=inspect(image,mask,valid,region)
    assert result['stats']['region']['accepted_distinct']>=25
    assert result['stats']['region']['fraction_abs_peak_offset_le_2']==1
    assert result['stats']['region']['median_abs_peak_offset']<=1
    json.dumps(result['stats'],allow_nan=False)


def test_shrunken_mask_outward_offset():
    image,mask,valid,region=scene(); mask[:]=False; mask[35:85,35:85]=True
    result=inspect(image,mask,valid,region)
    assert 4<=result['stats']['region']['median_signed_peak_offset']<=6
    assert result['stats']['region']['fraction_abs_peak_offset_le_2']==0


def test_translated_mask_signed_offset():
    image,mask,valid,region=scene(); mask[:]=False; mask[34:94,30:90]=True
    result=inspect(image,mask,valid,region)
    assert 3<=result['stats']['region']['median_signed_peak_offset']<=5


def test_uniform_image_never_false_agreement():
    image,mask,valid,region=scene(); image[:]=75
    result=inspect(image,mask,valid,region)
    assert not result['accepted_distinct'].any()
    assert np.isnan(result['peak_offset']).all()
    assert result['stats']['global_stats']['no_peak']>0
    assert result['stats']['global_stats']['fraction_abs_peak_offset_le_2'] is None
    assert result['stats']['global_stats']['median_abs_peak_offset'] is None


def test_invalid_strip_entire_profile_unknown():
    image,mask,valid,region=scene(); valid[25,:]=False
    result=inspect(image,mask,valid,region)
    selected=result['region_hint_selected']
    assert selected.any() and (result['reason'][selected]==0).all()
    assert np.isnan(result['peak_offset'][selected]).all()


def test_multiple_similar_edges_ambiguous():
    image,mask,valid,region=scene()
    image[:]=0; image[25:34,:,:]=220; image[39:90,:,:]=220
    result=inspect(image,mask,valid,region)
    selected=result['region_hint_selected']
    assert selected.any() and (result['reason'][selected]==2).all()
    assert not result['accepted_distinct'][selected].any()


def test_nonmutation_and_hint_does_not_change_search():
    image,mask,valid,region=scene(); before=copy.deepcopy((image,mask,valid,region))
    a=inspect(image,mask,valid,region); b=inspect(image,mask,valid,~region)
    for key in ['boundary_xy','peak_xy','peak_offset','strength','accepted_distinct','reason']:
        np.testing.assert_array_equal(a[key],b[key])
    for old,new in zip(before,(image,mask,valid,region)): np.testing.assert_array_equal(old,new)


@pytest.mark.parametrize('shape',[(1,1),(2,2),(9,9)])
def test_small_border_unknown(shape):
    image=np.zeros((*shape,3),np.uint8); mask=np.ones(shape,bool); valid=mask.copy()
    result=inspect(image,mask,valid)
    assert not result['accepted_distinct'].any()
    assert (result['reason']==0).all()
    json.dumps(result['stats'],allow_nan=False)


def test_empty_and_sampling_cap():
    image=np.zeros((600,600,3),np.uint8); mask=np.zeros((600,600),bool); valid=np.ones_like(mask)
    empty=inspect(image,mask,valid)
    assert empty['boundary_xy'].shape==(0,2)
    mask[20:580,20:580]=True
    result=inspect(image,mask,valid)
    assert len(result['boundary_xy'])==2048
    xy=result['boundary_xy']; assert np.all(np.diff(xy[:,1]*600+xy[:,0])>0)
