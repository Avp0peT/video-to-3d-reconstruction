"""Independent all-pairs reference checks; no model/native or GPU execution."""
from copy import deepcopy
import numpy as np
import pytest
from kozlov_lab.stereo.point_coverage import greedy_r_cover,raster_radius_coverage


def brute_raster(mask,points,radius):
    result=np.zeros(mask.shape,dtype=np.uint8)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y,x]:
                result[y,x]=1 if any((x+.5-px)**2+(y+.5-py)**2<=radius**2 for px,py in points) else 2
    return result


def test_greedy_preserves_duplicate_rows_and_deterministic_nearest_ties():
    points=[[0.,0.],[0.,0.],[4.,0.],[2.,0.]];before=deepcopy(points)
    result=greedy_r_cover(points,2)
    assert result['status']=='COMPLETE' and result['selected_indices']==[0,2]
    assert result['nearest_selected_indices']==[0,0,2,0]
    assert result['nearest_distances']==[0.,0.,0.,2.]
    assert points==before and not result['minimum_cover_claim']


def test_farthest_tie_uses_smallest_original_row():
    result=greedy_r_cover([[0,0],[1,0],[-1,0],[0,1]],1)
    assert result['selected_indices']==[0]
    result=greedy_r_cover([[0,0],[1,0],[-1,0],[0,1]],.9,max_centers=2)
    assert result['selected_indices']==[0,1] and result['status']=='PARTIAL'


def test_greedy_independent_all_pairs_distance_and_budget():
    rng=np.random.default_rng(8);points=rng.normal(size=(40,2));before=points.copy()
    result=greedy_r_cover(points,.32)
    centers=np.array(result['selected_indices']);distance=np.linalg.norm(points[:,None,:]-points[centers][None,:,:],axis=2)
    truth=distance.min(axis=1)
    assert np.allclose(result['nearest_distances'],truth)
    assert result['max_distance']<=.32 and result['complete']
    assert np.array_equal(points,before)
    partial=greedy_r_cover(points,.001,max_centers=3)
    assert partial['status']=='PARTIAL' and not partial['complete'] and partial['stop_reason']=='MAX_CENTERS_EXHAUSTED'
    assert partial['max_distance']>.001 and len(partial['selected_indices'])==3


def test_zero_radius_empty_points_and_zero_budget():
    result=greedy_r_cover([[2,3],[2,3],[4,5]],0)
    assert result['selected_indices']==[0,2] and result['max_distance']==0
    assert result['nearest_selected_indices']==[0,0,2]
    assert greedy_r_cover([],0,max_centers=0)['complete']
    result=greedy_r_cover([[0,0]],0,max_centers=0)
    assert result['status']=='PARTIAL' and result['max_distance'] is None
    assert result['nearest_distances']==[None] and result['nearest_selected_indices']==[-1]


def test_greedy_translation_and_scale_preserve_indices_and_witnesses():
    points=np.array([[0.,0.],[0.,0.],[1.25,2.],[4.,3.],[-2.,1.]])
    a=greedy_r_cover(points,1.5);b=greedy_r_cover(points*4+[16,-8],6)
    assert a['selected_indices']==b['selected_indices']
    assert a['nearest_selected_indices']==b['nearest_selected_indices']
    assert np.allclose(np.array(a['nearest_distances'])*4,b['nearest_distances'])


def test_small_cover_independently_satisfies_existing_source_condition():
    from kozlov_lab.compression.official.kozlov_sketch import verify_sketch
    from kozlov_lab.compression.metrics import coverage_report
    points=np.array([[0.,0.],[0.,0.],[1.,1.],[3.,0.],[3.,2.]])
    result=greedy_r_cover(points,1.5);selected=points[result['selected_indices']]
    assert verify_sketch(points,selected,1.5)
    measured=coverage_report(points,selected)
    assert measured['directed_original_to_sketch']<=1.5
    assert measured['sketch_is_subset'] and measured['directed_sketch_to_original']==0


@pytest.mark.parametrize('points,radius',[
    ([[.5,.5]],0),([[.6,.5]],0),([[.25,.5]],.25),
    ([[.25,.5]],np.nextafter(.25,0)),([[1.2,2.7],[3.5,.5],[1.2,2.7]],1.1),([],1),([[0,0]],.8)])
def test_raster_matches_independent_brute_pixel_centres(points,radius):
    mask=np.array([[255,255,0,255],[0,255,255,255],[255,0,255,255]],dtype=np.uint8)
    before=mask.copy();old=deepcopy(points);result=raster_radius_coverage(mask,points,radius)
    truth=brute_raster(mask,points,radius)
    assert np.array_equal(result['coverage_map'],truth)
    assert result['covered_pixels']==np.count_nonzero(truth==1)
    assert result['uncovered_pixels']==np.count_nonzero(truth==2)
    assert result['target_pixels']==np.count_nonzero(mask)
    assert result['coverage_ratio']==result['covered_pixels']/result['target_pixels']
    assert np.array_equal(mask,before) and points==old


def test_empty_target_ratio_is_null_and_points_do_not_create_target():
    result=raster_radius_coverage(np.zeros((3,3),dtype=np.uint8),[[1.5,1.5]],2)
    assert result['status']=='NO_TARGET' and result['coverage_ratio'] is None
    assert result['target_pixels']==result['covered_pixels']==result['uncovered_pixels']==0
    assert not result['coverage_map'].any()


def test_raster_geometric_scale_at_corresponding_pixel_centres():
    # Odd scaling maps an old pixel centre onto a new pixel centre exactly.
    mask=np.full((3,4),255,dtype=np.uint8);points=np.array([[1.5,.5],[2.5,2.5]])
    a=raster_radius_coverage(mask,points,1.);b=raster_radius_coverage(np.repeat(np.repeat(mask,3,0),3,1),points*3,3.)
    assert np.array_equal(a['coverage_map'],b['coverage_map'][1::3,1::3])


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf')])
def test_nonfinite_and_bool_points_rejected(bad):
    with pytest.raises(ValueError):greedy_r_cover([[bad,0]],1)
    with pytest.raises(ValueError):raster_radius_coverage(np.full((2,2),255,np.uint8),[[bad,0]],1)


@pytest.mark.parametrize('xy',[[-.01,.5],[2,.5],[.5,2]])
def test_outside_points_refused_not_clipped(xy):
    with pytest.raises(ValueError,match='outside'):raster_radius_coverage(np.full((2,2),255,np.uint8),[xy],1)


def test_bounds_and_work_are_refused_before_disk_updates():
    mask=np.full((512,512),255,np.uint8);before=mask.copy()
    with pytest.raises(ValueError,match='100 million'):raster_radius_coverage(mask,[[256.5,256.5]]*2000,128)
    assert np.array_equal(mask,before)
    with pytest.raises(ValueError):raster_radius_coverage(mask,[],129)
    with pytest.raises(ValueError):raster_radius_coverage(np.zeros((2001,2000),np.uint8),[],1)
    with pytest.raises(ValueError):greedy_r_cover([[0,0]],1,max_centers=True)
    with pytest.raises(ValueError):greedy_r_cover([[0,0]],1,max_centers=4097)
    with pytest.raises(ValueError):greedy_r_cover([[0,0]]*20001,1)
    with pytest.raises(ValueError):raster_radius_coverage(np.full((2,2),1,np.uint8),[],1)
