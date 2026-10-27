"""Fake CPU inputs only; never imports SAM or runs a real clip."""
import importlib.util
from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/toolchains/opencv-4.13.0.92'))
import cv2 as cv
spec=importlib.util.spec_from_file_location('association_cli',ROOT/'scripts/review_mask_association.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def track():
    return dict(track_id=4,observations=[dict(frame_index=i,xy=[15.5,15.5],identity_segment=i,label='foreground',target_verified=True,strong=True,fb_error_px=.1,lk_error=2.,temporal_support=dict(patch_ncc=.9,patch_std=10,reasons=['insufficient_neighbors'],supported=False)) for i in range(3)])

def collect(t):
    mask=np.full((32,32),255,np.uint8)
    return m.trajectories(dict(tracks=t),0,1,2,3,mask,mask,cv)

def test_independent_evidence_accepts_without_supported_and_retains_segments():
    points,ids,segments=collect([track()]);assert ids==[4] and points[1].tolist()==[[15.,15.]]
    assert len(segments[0]['geometry_segments'])==3

@pytest.mark.parametrize('change',['missing','patch','unverified','weak','seed'])
def test_unusable_track_never_reconstructed(change):
    t=track()
    if change=='missing':t['observations'].pop(1)
    if change=='patch':t['observations'][1]['temporal_support']['patch_ncc']=.1
    if change=='unverified':t['observations'][0]['target_verified']=False
    if change=='weak':t['observations'][1]['fb_error_px']=1.1
    if change=='seed':t['observations'][0]['fb_error_px']=None
    assert collect([t])[1]==[]

@pytest.mark.parametrize('change',['duplicate_ids','unordered','boolindex'])
def test_malformed_indices_rejected(change):
    t=track();tracks=[t]
    if change=='duplicate_ids':tracks.append(deepcopy(t))
    if change=='unordered':t['observations'].reverse()
    if change=='boolindex':t['observations'][1]['frame_index']=True
    with pytest.raises(ValueError):collect(tracks)

def test_derived_selection_never_becomes_reference():
    frames={0:dict(original_mask_status='OK'),1:dict(original_mask_status='NEEDS_SELECTION',status='TEMPORAL_CANDIDATE_SELECTED'),3:dict(original_mask_status='OK')}
    assert m.reference_pair(frames,2)==(0,3)
    frames[0]['original_mask_status']='NO_MATCH'
    assert m.reference_pair(frames,2) is None

def test_path_escape_and_unverified_root_rejected(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside';outside.write_text('x')
    with pytest.raises(ValueError):m.bounded(root,'../outside')
    with pytest.raises(FileNotFoundError):m.bounded(root/'unverified','x')

def test_duplicate_source_indices_and_bool_rejected():
    for rows in ([dict(input_index=True,frame_id='a')],[dict(input_index=1,frame_id='a'),dict(input_index=1,frame_id='b')]):
        with pytest.raises(ValueError):m.index_frames(rows)

@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}','{"a":1e999}'])
def test_strict_json_rejects_ambiguous_or_nonfinite(tmp_path,text):
    p=tmp_path/'bad.json';p.write_text(text)
    with pytest.raises(ValueError):m.read(p)


def test_expected_track_hash_and_audit_are_required(tmp_path):
    task=tmp_path/'task';review=task/'real01/comparison';tracks=review/'49_62/after/TRACKS.json';tracks.parent.mkdir(parents=True);m.save(tracks,dict(tracks=[]))
    audit=task/'INDEPENDENT_RESULT_AUDIT.json';m.save(audit,dict(status='PASS',failures=[],clips={'49_62':dict(tracks_sha256=m.bind(tracks)['sha256'])}))
    assert m.verified_tracks(review,'49_62')[0]==dict(tracks=[])
    tracks.write_text('{"tracks":[1]}')
    with pytest.raises(ValueError,match='drift'):m.verified_tracks(review,'49_62')
    m.save(audit,dict(status='FAIL',failures=['x']))
    with pytest.raises(ValueError,match='audit'):m.verified_tracks(review,'49_62')


def test_exclusion_counts_explain_missing_and_bad_patch():
    missing=track();missing['observations'].pop(1)
    bad=track();bad['track_id']=5;bad['observations'][1]['temporal_support']['patch_std']=1
    mask=np.full((32,32),255,np.uint8);diagnostics={}
    m.trajectories(dict(tracks=[missing,bad]),0,1,2,3,mask,mask,cv,diagnostics)
    assert diagnostics['eligible_tracks']==0
    assert diagnostics['exclusion_counts']==dict(missing_observation=1,independent_patch_chain=1)


def test_failure_leaves_new_artifact_but_existing_directory_untouched(tmp_path):
    output=tmp_path/'new'
    with pytest.raises(FileNotFoundError):m.build(tmp_path/'missing',tmp_path/'missingreview',output)
    assert m.read(output/'FAILURE.json')['status']=='FAILED'
    existing=tmp_path/'existing';existing.mkdir()
    with pytest.raises(ValueError):m.build(tmp_path/'missing',tmp_path/'missingreview',existing)
    assert list(existing.iterdir())==[]
