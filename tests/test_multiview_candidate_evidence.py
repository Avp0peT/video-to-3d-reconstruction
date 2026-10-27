from copy import deepcopy
import numpy as np
import pytest
from kozlov_lab.stereo import multiview_candidate_evidence as kernel


def scene(extra=True):
    rng=np.random.default_rng(72);xyz=rng.uniform([-1,-1,4],[2,2,8],(32,3))
    centers={1:np.array([0.,0.,0.]),2:np.array([1.,0.,0.])}
    if extra:centers[3]=np.array([0.,1.,0.])
    return {pid:{i:(500*((p-c)[:2]/(p-c)[2])+[320,240]).tolist() for i,c in centers.items()}
            for pid,p in enumerate(xyz,1)}


def swaps():return [dict(image_ids=[1,2],source_point_ids=[1,2])]


def test_pinhole_extra_can_distinguish_frozen_candidate_and_inputs_unchanged():
    tracks=scene();before=deepcopy(tracks);request=swaps();original_request=deepcopy(request)
    plan=kernel.freeze_cases(tracks,request);frozen=deepcopy(plan);result=kernel.analyse_plan(plan)
    assert tracks==before and request==original_request and plan==frozen
    assert len(result['cases'])==2 and len(result['models'])==2
    for row in result['cases']:
        assert row['extra_summary']['paired_count']==1
        assert row['extra_summary']['support']=='original'
        assert row['extra_summary']['original_rms']<1e-8
        assert row['pair_plus_extra_summary']['paired_count']==2
        assert row['extra'][0]['evaluation']['model_key']==row['extra'][0]['model_key']
    assert all(m['status']=='FITTED' for m in result['models'])
    assert not result['correspondence_verified']


def test_global_alias_uses_same_image_identity_and_transitive_closure():
    tracks=scene()
    # 1--3 in view1; 3--4 in view3: alias chain spanning different images.
    tracks[3][1]=deepcopy(tracks[1][1]);tracks[4][3]=deepcopy(tracks[3][3])
    # Same numeric XY in a different view must not create an alias.
    tracks[5][2]=deepcopy(tracks[1][1])
    plan=kernel.freeze_cases(tracks,swaps())
    assert [1,3,4] in plan['global_alias_groups']
    assert [5] in plan['global_alias_groups']
    for row in plan['cases']:assert row['excluded_ids']==[1,2,3,4]
    for model in plan['models']:assert not set(model['train_ids'])&{1,2,3,4}


def test_training_uses_one_minimal_common_representative_per_global_alias():
    tracks=scene();tracks[5][3]=deepcopy(tracks[6][3])
    plan=kernel.freeze_cases(tracks,swaps())
    for model in plan['models']:
        assert 5 in model['train_ids'] and 6 not in model['train_ids']
        assert model['train_pixels'][model['train_ids'].index(5)]==tracks[5][model['image_ids'][0]]+tracks[5][model['image_ids'][1]]


def test_model_fitted_once_and_candidates_share_same_evaluable_views(monkeypatch):
    plan=kernel.freeze_cases(scene(),swaps());count=[];actual=kernel.fit_fundamental
    def fit(raw):count.append(len(raw));return actual(raw)
    monkeypatch.setattr(kernel,'fit_fundamental',fit)
    result=kernel.analyse_plan(plan)
    assert len(count)==len(plan['models'])==2
    for row in result['cases']:
        assert row['extra_summary']['paired_count']==len(row['extra'])
        assert row['extra'][0]['original_pixels'][2:]==row['extra'][0]['alternative_pixels'][2:]


def test_undefined_one_candidate_excludes_whole_extra_pair(monkeypatch):
    plan=kernel.freeze_cases(scene(),swaps());actual=kernel._residual
    special=plan['cases'][0]['extra'][0]['alternative_pixels']
    def residual(kind,model,points):
        if points.tolist()==[special]:raise ValueError('synthetic undefined alternative Sampson denominator')
        return actual(kind,model,points)
    monkeypatch.setattr(kernel,'_residual',residual)
    result=kernel.analyse_plan(plan);row=result['cases'][0]
    assert row['extra'][0]['evaluation']['status']=='UNAVAILABLE'
    assert row['extra'][0]['evaluation']['original'] is None and row['extra'][0]['evaluation']['alternative'] is None
    assert row['extra_summary']['paired_count']==0 and row['extra_summary']['support']=='unavailable'
    assert row['pair_plus_extra_summary']['paired_count']==1


def test_no_extra_is_explicit_unavailable():
    result=kernel.analyse_plan(kernel.freeze_cases(scene(False),swaps()))
    for row in result['cases']:
        assert row['extra']==[] and row['extra_summary']['status']=='UNAVAILABLE'
        assert row['pair_plus_extra_summary']['paired_count']==1


def test_insufficient_remaining_groups_retains_candidates():
    tracks={pid:views for pid,views in scene().items() if pid<=8}
    result=kernel.analyse_plan(kernel.freeze_cases(tracks,swaps()))
    assert all(m['status']=='NOT_EVALUATED' for m in result['models'])
    assert all(c['pair']['status']=='UNAVAILABLE' for c in result['cases'])


def test_freeze_tamper_cannot_change_exclusion_or_pixels():
    plan=kernel.freeze_cases(scene(),swaps());plan['models'][0]['train_ids'][0]=1
    with pytest.raises(ValueError,match='differs'):kernel.analyse_plan(plan)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf')])
def test_invalid_pixel_rejected(value):
    tracks=scene();tracks[1][1][0]=value
    with pytest.raises(ValueError,match='pixels'):kernel.freeze_cases(tracks,swaps())


def test_bool_ids_rejected():
    request=swaps();request[0]['image_ids']=[True,2]
    with pytest.raises(ValueError):kernel.freeze_cases(scene(),request)
