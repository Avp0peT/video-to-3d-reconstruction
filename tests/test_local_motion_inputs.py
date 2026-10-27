"""Fake protocol regression checks only; no models or real clips."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('local_review',ROOT/'scripts/review_local_motion.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'])
def test_strict_json(tmp_path,text):
    p=tmp_path/'a';p.write_text(text)
    with pytest.raises(ValueError):m.read(p)

@pytest.mark.parametrize('where',['same','inside','ancestor','existing'])
def test_no_early_output_overlap(tmp_path,where):
    source=tmp_path/'source';source.mkdir();out={'same':source,'inside':source/'out','ancestor':tmp_path,'existing':tmp_path/'exists'}[where]
    if where=='existing':out.mkdir()
    with pytest.raises(ValueError):m.build(source,source,source,out)
    assert list(source.iterdir())==[]


def fake():
    row=dict(frame=dict(input_index=1,frame_id='f',original_mask_status='OK'),candidates=[])
    old=dict(status='UNASSESSED',input_index=1,frame_id='f',original_status='OK',before_index=0,after_index=2,eligible_track_count=2,track_ids=[2,3],geometry_segments=[],selection=None,publishable_for_geometry=False,candidates=[],kernel_evidence=dict(references=dict(before=dict(fit_track_ids=[3],holdout_track_ids=[2]),after=dict(fit_track_ids=[3],holdout_track_ids=[2])),fit_track_ids=[3],holdout_track_ids=[2]))
    return old,row

@pytest.mark.parametrize('change',['oldids','bool','sideids','status','refs'])
def test_wrong_prior_protocol(change):
    old,row=fake()
    if change=='oldids':old['track_ids']=[2,4]
    if change=='bool':old['track_ids']=[True,3]
    if change=='sideids':old['kernel_evidence']['references']['after']['holdout_track_ids']=[3]
    if change=='status':old['original_status']='NO_MATCH'
    if change=='refs':old['before_index']=False
    with pytest.raises(ValueError):m.check_prior(old,row,(0,2),[2,3],[])


def test_stale_binding(tmp_path):
    p=tmp_path/'file';p.write_text('old');ref=m.bind(p);p.write_text('changed')
    with pytest.raises(ValueError):m.verify(ref)


def test_global_comparison_uses_saved_matrix_and_same_points():
    ids=[1,2,3,4,5,6];p=np.arange(12,dtype=np.float32).reshape(6,2)
    side=dict(matrix=[[1,0,0],[0,1,0]],holdout_residuals_px=[0,0])
    evidence=dict(references=dict(before=deepcopy(side),after=deepcopy(side)))
    profile=dict(references=dict(before=dict(cells=[]),after=dict(cells=[])),common_supported_track_ids=[1],measured_query_evidence=[dict(track_id=1,query_point_xy=p[0].tolist(),before_residual_px=0,after_residual_px=0)])
    local=dict(profiles=dict(local_affine=profile));out=m.compare_global(evidence,local,[p,p,p],ids)
    assert out['local_affine']['references']['before']['common_global']['count']==1
    profile['measured_query_evidence'][0]['query_point_xy']=[99,99]
    with pytest.raises(ValueError):m.compare_global(evidence,local,[p,p,p],ids)


def test_seal_digest_must_match_before_output(tmp_path):
    association=tmp_path/'task/real01/association';association.mkdir(parents=True)
    (association.parent.parent/'SHA256SUMS.txt').write_text('stale')
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='drift'):m.build(association,tmp_path/'coverage',tmp_path/'review',out)
    assert not out.exists()


def test_rejected_bad_point_stays_in_all_prediction_comparison():
    ids=[1,2,3,4,5,6];p=np.zeros((6,2),np.float32);p[5]=[10,0]
    side=dict(matrix=[[1,0,0],[0,1,1]],holdout_residuals_px=[1,1])
    evidence=dict(references=dict(before=deepcopy(side),after=deepcopy(side)))
    cell=dict(cell_id=0,holdout_track_ids=[1,6],matrix=[[3,0,0],[0,1,0]],holdout_residuals_px=[0,20],passed=False,reasons=['holdout_residual_high'])
    profile=dict(references=dict(before=dict(cells=[deepcopy(cell)]),after=dict(cells=[deepcopy(cell)])),common_supported_track_ids=[1],measured_query_evidence=[dict(track_id=1,query_point_xy=[0,0],before_residual_px=0,after_residual_px=0)])
    out=m.compare_global(evidence,dict(profiles=dict(local_affine=profile)),[p,p,p],ids)['local_affine']
    before=out['references']['before'];assert before['common_local']['mean_px']<before['common_global']['mean_px']
    all_fit=before['all_fitted_local_predictions'];assert all_fit['local_stats']['mean_px']>all_fit['global_stats']['mean_px']
    assert all_fit['predicted_count']==2 and all_fit['selection_conditioned'] is False and out['selection_conditioned'] is True
