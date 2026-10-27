"""Synthetic adapter contracts; no model, reconstruction or native pipeline runs."""
from copy import deepcopy
import json
import numpy as np
from PIL import Image
import pytest
from integrations.demo_pipeline import automatic_point_review as auto


def test_canonical_noncontiguous_indices_and_no_mutation():
    frames=[dict(input_index=i,frame_id=f'f{i}') for i in (11,30,91)]
    tracks=dict(tracks=[dict(track_id=7,observations=[dict(frame_index=i,xy=[1,2]) for i in range(3)])])
    frozen=deepcopy(tracks)
    got=auto.canonical_tracks(tracks,frames)
    assert [o['frame_index'] for o in got['tracks'][0]['observations']]==[11,30,91]
    assert [o['sequence_position'] for o in got['tracks'][0]['observations']]==[0,1,2]
    assert tracks==frozen
    tracks['tracks'][0]['observations'][0]['frame_index']=True
    with pytest.raises(ValueError):auto.canonical_tracks(tracks,frames)


def test_reference_uses_positions_and_original_ok_only():
    frames=[dict(input_index=100+i*20,original_mask_status='NO_MATCH') for i in range(16)]
    for position in (0,5,9,15):frames[position]['original_mask_status']='OK'
    assert auto.reference_pair(frames,7)==(5,9)
    assert auto.reference_pair(frames,0) is None
    assert auto.reference_pair(frames,15) is None
    frames[5]['original_mask_status']='NEEDS_SELECTION'
    assert auto.reference_pair(frames,7) is None


def fixture_sequence(tmp_path,status='NEEDS_SELECTION',candidate_count=1,positive=True,exterior=True):
    cv=pytest.importorskip('cv2')
    frames=[];grays=[];masks=[];candidates={}
    mask=np.zeros((32,32),np.uint8);mask[8:24,8:24]=255
    for pos,index in enumerate((11,30,91)):
        path=tmp_path/f'f{index}.png';Image.fromarray(mask).save(path)
        original=status if pos==1 else 'OK'
        frames.append(dict(input_index=index,frame_id=f'f{index}',path=str(path),width=32,height=32,
                           original_mask_status=original,mask_binding=dict(path=str(path))))
        grays.append(mask.copy());masks.append(mask.copy() if original=='OK' else None)
        candidates[index]=[]
    if status!='NO_MATCH':
        for j in range(candidate_count):
            path=tmp_path/f'candidate{j}.png';Image.fromarray(mask).save(path)
            candidates[30].append(dict(candidate_id=f'c{j}',binding=dict(path=str(path))))
    tracking=dict(tracks=[dict(track_id=7,observations=[dict(frame_index=i,xy=[12.5,12.5],identity_segment=0,
        label='unknown' if i==1 else 'foreground',mask_status=status if i==1 else 'OK',
        target_verified=i!=1,strong=True,fb_error_px=0.1,lk_error=1.) for i in range(3)])])
    evidence=[dict(track_id=7,query_point_xy=[12.,12.])] if positive else []
    engines=dict(trajectories=lambda *a:(([[],[],[]]),[7],[]),associate=lambda *a:dict(status='UNRESOLVED'),
                 local=lambda *a:dict(profiles=dict(local_affine=dict(measured_query_evidence=evidence))),
                 exterior=lambda *a:dict(accepted_witnesses=[dict(witness_id='w',observations=[dict(frame_index=i,xy=[1.,1.]) for i in range(3)])] if exterior else []))
    return frames,grays,masks,candidates,tracking,cv,engines


@pytest.mark.parametrize('kind,expected', [('single','PROVISIONAL_POINT_SUPPORT_ONLY'),('ambiguous','AMBIGUOUS'),
    ('no_positive','INSUFFICIENT_EVIDENCE'),('no_exterior','INSUFFICIENT_EVIDENCE'),('no_match','NO_CANDIDATE')])
def test_real_boundary_policy_and_geometry_inputs(tmp_path,kind,expected):
    args=fixture_sequence(tmp_path,status='NO_MATCH' if kind=='no_match' else 'NEEDS_SELECTION',
                          candidate_count=2 if kind=='ambiguous' else 1,
                          positive=kind!='no_positive',exterior=kind!='no_exterior')
    frames,grays,masks,candidates,tracking,cv,engines=args
    snapshots=deepcopy((frames,candidates,tracking));arrays=[a.copy() if a is not None else None for a in grays+masks]
    output=tmp_path/'result';output.mkdir()
    got=auto.analyse_sequence(frames,grays,masks,candidates,tracking,[],cv,output,lambda:None,engines=engines)
    decision=got['decisions'][1]['point_decision']
    assert decision['status']==expected
    assert decision['original_status']==frames[1]['original_mask_status']
    assert all(decision[k] is False for k in ('publishable_for_geometry','full_mask_accepted','new_point_seeding_allowed'))
    assert json.loads((output/'MASK_SELECTION_LEDGER.json').read_text())==[]
    if kind=='single':
        group=got['geometry_groups'][0]
        assert group['status']=='NO_MODEL_WITH_ANCHORS'
        assert group['queries'][0]['input_index']==30
        assert group['query_checks'][0]['reason']=='NO_SAVED_CAMERA'
        assert group['geometry_accepted'] is False and not group['anchors']
        annotations=json.loads((output/'POINT_ANNOTATIONS.json').read_text())
        assert [r['frame_index'] for r in annotations['rows']]==[11,30,91]
        assert all(r['geometry_eligible'] is False for r in annotations['rows'])
    else:assert got['geometry_groups']==[]
    assert (frames,candidates,tracking)==snapshots
    for old,new in zip(arrays,grays+masks):
        if old is not None:np.testing.assert_array_equal(old,new)


def test_no_saved_model_explicit_without_native_reader():
    from integrations.demo_pipeline.point_review_cameras import load_registered_cameras
    refs=[]
    result=load_registered_cameras({'stages':{}},[],refs)
    assert result['status']=='NO_MODEL' and result['models']==[] and result['native_reads']==0
    assert refs==[]


def test_review_non_video_retains_bound_input_and_reports_insufficient(tmp_path):
    pipeline=tmp_path/'pipeline';pipeline.mkdir()
    source=pipeline/'pipeline.json';source.write_text(json.dumps(dict(source_kind='photos',stages={})),encoding='utf-8')
    original=source.read_bytes()
    result=auto.review(pipeline,pipeline/'point_review')
    assert result['status']=='INSUFFICIENT_DATA' and result['reason']=='NOT_APPLICABLE_NON_VIDEO'
    assert result['source_unchanged'] is True and source.read_bytes()==original
    assert result['geometry_accepted'] is False and result['neural_model_loads']==0
    assert (pipeline/'point_review/index.html').is_file()


def test_internal_timeout_cli_exit_blocks_next_diagnostic(tmp_path,monkeypatch):
    from scripts.review_point_sequence import main
    from integrations.demo_pipeline.pipeline import run_point_review
    class FakeCV:
        class OCL:
            def setUseOpenCL(self,*args):pass
        ocl=OCL()
        def setNumThreads(self,*args):pass
        def setRNGSeed(self,*args):pass
    monkeypatch.setattr(auto.inputs,'load_cv',lambda refs:FakeCV())
    def timeout(*args,**kwargs):raise TimeoutError('synthetic elapsed budget')
    monkeypatch.setattr(auto.inputs,'load_inputs',timeout)
    pipeline=tmp_path/'pipeline';pipeline.mkdir()
    (pipeline/'pipeline.json').write_text(json.dumps(dict(source_kind='video',stages=dict(segmentation=dict(run_dir=str(tmp_path/'seg'))))),encoding='utf-8')
    calls=[]
    def invoke(root,run,stage,argv,**limits):
        calls.append(stage)
        code=main(['--pipeline',str(run),'--output',str(run/'point_review')])
        assert code==5
        return dict(status='completed',root_exit_code=code,reclaim_confirmed=True)
    detail=run_point_review(tmp_path,pipeline,'unused',invoke)
    assert detail['blocks_post_review'] is True
    assert detail['status']=='RESOURCE_STOPPED'
    result=json.loads((pipeline/'point_review/RESULT.json').read_text(encoding='utf-8'))
    assert result['status']=='RESOURCE_LIMIT' and result['partial_evidence_retained'] is True
    assert calls==['point_review']


def test_long_sparse_reference_span_preserves_frames_without_exterior_extrapolation(tmp_path):
    cv=pytest.importorskip('cv2')
    mask=np.zeros((32,32),np.uint8);mask[8:24,8:24]=255
    frames=[];grays=[];masks=[];candidates={}
    for position in range(13):
        index=100+position*3
        status='OK' if position in (0,12) else 'NEEDS_SELECTION' if position==6 else 'NO_MATCH'
        path=tmp_path/f'long{index}.png';Image.fromarray(mask).save(path)
        frames.append(dict(input_index=index,frame_id=f'f{index}',path=str(path),width=32,height=32,
                           original_mask_status=status,mask_binding=dict(path=str(path)) if status=='OK' else None))
        grays.append(mask.copy());masks.append(mask.copy() if status=='OK' else None);candidates[index]=[]
    candidate_path=tmp_path/'candidate.png';Image.fromarray(mask).save(candidate_path)
    candidates[118]=[dict(candidate_id='c0',binding=dict(path=str(candidate_path)))]
    tracking=dict(tracks=[dict(track_id=7,observations=[dict(frame_index=position,xy=[12.5,12.5],
        identity_segment=0,label='foreground' if position in (0,12) else 'unknown',
        mask_status=frames[position]['original_mask_status'],target_verified=position in (0,12),
        strong=True,fb_error_px=0.1,lk_error=1.) for position in range(13)])])
    snapshots=deepcopy((frames,candidates,tracking));arrays=[a.copy() if a is not None else None for a in grays+masks]
    before_bytes={str(p):p.read_bytes() for p in tmp_path.glob('*.png')}
    exterior_calls=[]
    def exterior(*args):
        exterior_calls.append(args)
        raise AssertionError('long reference span must not call exterior kernel')
    engines=dict(trajectories=lambda *a:([[],[],[]],[7],[]),
        associate=lambda *a:dict(status='UNRESOLVED'),
        local=lambda *a:dict(profiles=dict(local_affine=dict(measured_query_evidence=[dict(track_id=7,query_point_xy=[12.,12.])]))),
        exterior=exterior)
    output=tmp_path/'review';output.mkdir()
    result=auto.analyse_sequence(frames,grays,masks,candidates,tracking,[],cv,output,lambda:None,engines=engines)
    assert result['counts']['processed_frames']==13 and len(result['decisions'])==13
    assert exterior_calls==[]
    query=result['decisions'][6]
    assert query['reference_positions']==(0,12)
    assert query['exterior_evidence']==[]
    assert query['point_decision']['status']=='INSUFFICIENT_EVIDENCE'
    assert query['point_decision']['publishable_for_geometry'] is False
    assert 'EXTERIOR_REFERENCE_SPAN_UNSUPPORTED' in json.dumps(query)
    assert result['geometry_groups']==[] and result['counts']['provisional_frames']==0
    assert json.loads((output/'MASK_SELECTION_LEDGER.json').read_text())==[]
    assert (frames,candidates,tracking)==snapshots
    assert all(__import__('pathlib').Path(path).read_bytes()==data for path,data in before_bytes.items())
    for old,new in zip(arrays,grays+masks):
        if old is not None:np.testing.assert_array_equal(old,new)
