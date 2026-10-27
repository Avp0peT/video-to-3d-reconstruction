"""Full-view counterexamples for the saved-candidate diagnostic."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline.mismatch_identifiability import inspect_pair, full_observation_diagnostic
from kozlov_lab.stereo.baseline.multiview_fixed import project


def sample_pair():
    truth=[0.,0.,4.]; wrong=[0.,0.,8.]
    obs=[]
    # Left and heldout have exactly the same optical center; third view disambiguates depth.
    for image_id,center in enumerate(([0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]),1):
        o=dict(image_id=image_id,source_track_id=1,point2D_index=0,
            raw_pixels=[0.,0.],params=[800.,320.,240.,.01],R=np.eye(3).tolist(),
            t=(-np.array(center)).tolist(),max_ideal_radius=2.)
        o['raw_pixels']=project(truth,o)['pixels'].tolist(); obs.append(o)
    positive=dict(case_kind='positive',pair_index=0,point_id=1,right_source_track_id=1,image_pair=[1,2],
        split=dict(status='SPLIT_READY',pair=[1,2],additional_views=[3,4],train_extra=[3],heldout=[4]),
        observations=dict(pair=obs[:2],train_extra=[obs[2]],heldout=[obs[3]]),
        candidates=dict(A=dict(xyz=truth,status='EXISTS'),C=dict(xyz=truth,status='EXISTS')))
    negative=deepcopy(positive); negative['case_kind']='negative_control'; negative['right_source_track_id']=2
    negative['observations']['pair'][1]['source_track_id']=2
    negative['observations']['pair'][1]['raw_pixels']=project(wrong,obs[1])['pixels'].tolist()
    negative['candidates']['C']['xyz']=wrong
    return positive,negative


def test_near_duplicate_heldout_cannot_replace_full_view_check():
    positive,negative=sample_pair(); before=deepcopy((positive,negative))
    r=inspect_pair(positive,negative,include_segment=True)
    assert (positive,negative)==before
    assert r['diagnostics']['negative_C']['original_heldout']['max']==0
    assert r['diagnostics']['negative_C']['original_train_extra']['max']>90
    assert r['interpretation']=='ADDITIONAL_SUPPORT_EXCEEDS_DIAGNOSTIC_LINE'
    assert r['full_supplied_observation_diagnostic']['status']=='OBSERVATION_INCONSISTENCY_DETECTED'
    assert r['diagnostics']['reference_A']['left_plus_heldout_observability']['rank']==2
    assert r['diagnostics']['reference_A']['observability']['rank']==3
    assert len(r['segment_samples'])==5
    assert all(v.get('residual_identity_error',0)<1e-12 for v in r['predictions'])


def test_coordinate_alias_not_mislabeled_as_new_image_information():
    positive,negative=sample_pair()
    negative['observations']['pair'][1]['raw_pixels']=positive['observations']['pair'][1]['raw_pixels'][:]
    negative['candidates']['C']['xyz']=positive['candidates']['A']['xyz'][:]
    r=inspect_pair(positive,negative)
    assert r['right_raw_exactly_equal']
    assert r['interpretation']=='RIGHT_LABEL_CHANGE_WITH_IDENTICAL_PIXELS'
    assert full_observation_diagnostic(negative)['status']=='WITHIN_DIAGNOSTIC_LINE_NOT_VERIFIED'


@pytest.mark.parametrize('case', ['insufficient','missing','behind'])
def test_missing_invalid_or_partial_never_becomes_consistency(case):
    _,negative=sample_pair()
    if case=='insufficient': negative['split']['status']='INSUFFICIENT_EXTRA_VIEWS'
    elif case=='missing': negative['candidates']['C']['xyz']=None
    else: negative['candidates']['C']['xyz']=[0,0,-1]
    r=full_observation_diagnostic(negative)
    assert r['status']=={'insufficient':'NOT_EVALUATED_INSUFFICIENT_VIEWS','missing':'NO_CANDIDATE','behind':'PROJECTION_REFUSED'}[case]
    assert r['decision_is_production_acceptance'] is False


@pytest.mark.parametrize('field',['left','extra','split'])
def test_no_silent_identity_or_observation_repair(field):
    p,n=sample_pair()
    if field=='left': n['observations']['pair'][0]['raw_pixels'][0]+=1
    elif field=='extra': n['observations']['train_extra'][0]['raw_pixels'][1]+=1
    else: n['split']['heldout']=[3]
    with pytest.raises(ValueError): inspect_pair(p,n)
