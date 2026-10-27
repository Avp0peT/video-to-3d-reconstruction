"""Explicit 192-frame and independent SfM binding tests; synthetic only."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from test_temporal_review import chain, write, binding
from integrations.demo_pipeline import temporal_review as review
from integrations.demo_pipeline import coverage_expansion as expansion


def independent_sfm(c, tmp_path):
    folder=tmp_path/'override';folder.mkdir()
    manifest=c['manifest']
    run=dict(upstream_io_run_id='t01-fixture',models=[dict(model_id=0,registered_image_ids=[7,8])],
        effective_options={'foreground':{'requested':True,'mask_bindings':[
            {'input_index':r['input_index'],'sha256':r['selected_mask']['sha256']} for r in manifest['frames']]}})
    write(folder/'run.json',run);write(folder/'frames_map.jsonl',c['frows'],True)
    return folder,run


def test_override_independently_bound_and_original_discovery_unchanged(chain,tmp_path):
    folder,run=independent_sfm(chain,tmp_path)
    before=copy.deepcopy(chain['discovery']);refs=[]
    frames,_,_,meta=review.load_inputs(chain['pipeline'],chain['discovery'],refs,
        max_frame_profile='whole-video-incremental-v1',sfm_run=folder/'run.json')
    assert chain['discovery']==before
    assert meta['source_sfm_override']==binding(folder/'run.json')
    assert meta['pipeline_sfm_run']==before['stages']['sfm']['run_json']
    assert all(f['model_ids']==['0'] for f in frames)
    assert {str(folder/'run.json'),before['stages']['sfm']['run_json']}<={r['path'] for r in refs}


@pytest.mark.parametrize('damage',['upstream','mask','image','scope'])
def test_override_refuses_changed_provenance(chain,tmp_path,damage):
    folder,run=independent_sfm(chain,tmp_path)
    if damage=='upstream':run['upstream_io_run_id']='other'
    elif damage=='mask':run['effective_options']['foreground']['mask_bindings'][0]['sha256']='0'*64
    elif damage=='scope':run['effective_options']['foreground']['mask_bindings'].pop()
    else:
        rows=copy.deepcopy(chain['frows']);rows[0]['canonical_image_sha256']='0'*64
        write(folder/'frames_map.jsonl',rows,True)
    write(folder/'run.json',run)
    with pytest.raises(ValueError):
        review.load_inputs(chain['pipeline'],chain['discovery'],[],max_frame_profile='whole-video-incremental-v1',sfm_run=folder/'run.json')


def test_192_requires_explicit_profile_and_preserves_unknown(chain):
    c=chain;photo=c['receipt']['photo_io'];base=c['video']['frames'][1]
    rows=[json.loads(s) for s in (c['t01']/'frames.jsonl').read_text().splitlines()]
    for n in range(2,192):
        fid=f'frame-{n}'
        rows.append(dict(rows[1],input_index=n,frame_id=fid))
        photo['identities'].append(dict(photo['identities'][1],input_index=n,frame_id=fid))
        c['manifest']['frames'].append(dict(c['manifest']['frames'][1],input_index=n,frame_id=fid,status='NO_MATCH',selected_candidate_id=None,selected_mask=None,candidates=[]))
        c['video']['frames'].append(dict(base,photo_input_index=n,output_index=n,decode_index=n*3,pts=n*3,time_seconds=n/10,photo_frame_id=fid))
    write(c['t01']/'frames.jsonl',rows,True)
    run=json.loads((c['t01']/'run.json').read_text());run.update(input_count=192,ok_count=192)
    write(c['t01']/'run.json',run)
    photo.update(count=192,run=binding(c['t01']/'run.json'),frames=binding(c['t01']/'frames.jsonl'))
    c['video']['photo_io'].update(sha256=photo['run']['sha256'],frames_sha256=photo['frames']['sha256'])
    c['manifest']['source_io'].update(run_json_sha256=photo['run']['sha256'],frames_jsonl_sha256=photo['frames']['sha256'])
    c['discovery']['stages'].pop('sfm');c['commit']()
    with pytest.raises(ValueError,match='2..96'):review.load_inputs(c['pipeline'],c['discovery'],[])
    frames,_,masks,meta=review.load_inputs(c['pipeline'],c['discovery'],[],max_frame_profile='whole-video-incremental-v1')
    assert len(frames)==192 and frames[-1]['pts']==573
    assert masks[-1] is None and frames[-1]['original_mask_status']=='NO_MATCH'


def test_unknown_and_false_target_cannot_support_3d():
    masks=[np.full((10,10),255,np.uint8),None]
    obs=dict(frame_index=0,xy=[4,4],label='foreground',target_verified=True)
    assert expansion.verified_object_observation(obs,masks)
    for change in ({'label':'unknown'},{'target_verified':False},{'xy':[20,4]},{'frame_index':1}):
        assert not expansion.verified_object_observation(dict(obs,**change),masks)


def test_incremental_cannot_silently_enable_whole_frame_or_override_default(tmp_path):
    with pytest.raises(ValueError,match='explicit incremental'):
        expansion.run(tmp_path,tmp_path,tmp_path/'out',sfm_run='unused')
    with pytest.raises(ValueError,match='object selection'):
        expansion.run(tmp_path,tmp_path,tmp_path/'out',selection_scope='whole-frame',tracking_profile='whole-video-incremental-v1')


@pytest.mark.parametrize('per_segment,expected_calls',[(1,0),(3,2)])
def test_production_triangulation_never_combines_unknown_gap_segments(tmp_path,monkeypatch,per_segment,expected_calls):
    from types import SimpleNamespace
    from PIL import Image
    from kozlov_lab.stereo import coverage_tracks
    count=2*per_segment+1;frames=[];masks=[];cameras={};obs=[]
    image=tmp_path/'image.png';Image.new('RGB',(10,10),(30,40,50)).save(image)
    for i in range(count):
        unknown=i==per_segment
        segment=0 if i<=per_segment else 1
        frames.append(dict(input_index=i,path=str(image),original_mask_status='NO_MATCH' if unknown else 'OK'))
        masks.append(None if unknown else np.full((10,10),255,np.uint8))
        if not unknown:cameras[i]=dict(input_index=i,image_id=i+1)
        obs.append(dict(frame_index=i,xy=[4,4],label='unknown' if unknown else 'foreground',
            target_verified=not unknown,identity_segment=segment,strong=True,fb_error_px=0))
    track={'track_id':17,'observations':obs}
    monkeypatch.setattr(coverage_tracks,'profile_for',lambda scope,profile_id=None:dict(selection_scope=scope,id=profile_id))
    monkeypatch.setattr(coverage_tracks,'track_coverage',lambda *a,**k:dict(profile={'selection_scope':'object'},tracks=[track]))
    monkeypatch.setattr(expansion,'read_bound',lambda *a,**k:dict(stages={'sfm':{'run_json':'unused'}}))
    monkeypatch.setattr(expansion,'load_inputs',lambda *a,**k:(frames,[np.zeros((10,10),np.uint8)]*count,masks,{}))
    monkeypatch.setattr(expansion,'load_cv',lambda refs:SimpleNamespace(setNumThreads=lambda n:None,setRNGSeed=lambda n:None))
    original=dict(point_id=1,xyz=[0,0,1],color=[1,2,3],accepted_conditionally=True,observations=[])
    monkeypatch.setattr(expansion,'load_local_models',lambda *a:[dict(group_id=0,model_id=0,cameras=cameras,ledger=[original])])
    monkeypatch.setattr(expansion,'build_faces',lambda points:([],[],{}))
    monkeypatch.setattr(expansion,'render',lambda *a:[])
    from integrations.demo_pipeline import coverage_triangulation
    calls=[]
    def triangulate(observations):
        calls.append(observations)
        return dict(xyz=None,tier='REJECTED',reasons=['TEST_DATA'])
    monkeypatch.setattr(coverage_triangulation,'triangulate_candidate',triangulate)
    pipeline=tmp_path/'pipeline';pipeline.mkdir();local=tmp_path/'local';local.mkdir()
    result=expansion.run(pipeline,local,tmp_path/'out',tracking_profile='whole-video-incremental-v1')
    assert len(calls)==expected_calls
    for observations in calls:
        assert len(observations)==3
        assert len({o['identity_segment'] for o in observations})==1
    ledger=json.loads((tmp_path/'out/group_00_model_0/CANDIDATE_LEDGER.json').read_text())
    assert {r['identity_segment'] for r in ledger}==({0,1} if expected_calls else set())
    assert all(r['candidate_track_id']==17 and r['cross_segment_identity']=='UNCONFIRMED' for r in ledger)


@pytest.mark.parametrize('segment',[None,-1,True,1.5])
def test_incremental_missing_or_invalid_segment_is_rejected(segment):
    with pytest.raises(ValueError,match='identity_segment'):
        expansion.observation_segments({'observations':[{'identity_segment':segment}]},True)


def test_cross_gap_candidate_not_deduplicated_as_same_identity():
    points=[dict(point_id=1,candidate_track_id=7,identity_segment=0),dict(point_id=2,candidate_track_id=9,identity_segment=0)]
    assert expansion.independent_segment_points(points,7,1)==points[1:]
    assert expansion.observation_segments({'observations':[{'label':'foreground'}]})==[(0,[{'label':'foreground'}])]



def test_incremental_renderer_reports_unknown_lifecycle_and_first_known_mask(tmp_path):
    from PIL import Image
    path=tmp_path/'image.png';Image.new('RGB',(10,10),(20,30,40)).save(path)
    frames=[dict(input_index=i,path=str(path),original_mask_status='NO_MATCH' if i==0 else 'OK') for i in range(2)]
    masks=[None,np.full((10,10),255,np.uint8)]
    stats=dict(grid_bounds=None,selection_status='unknown',grid_domain='none',grid_counts=[],tracked=7,seeded=0,recovered=0,suspended=7)
    tracking=dict(profile={'selection_scope':'object'},tracks=[],coverage=[stats,dict(stats,tracked=6,seeded=2,recovered=3,suspended=0)])
    group={k:0 for k in ('group_id','original_points','added_strict','added_relaxed','supported_points','two_view_provisional','faces')}
    result=dict(profile={'tracking_profile':'whole-video-incremental-v1'},groups=[group]*2)
    rows=expansion.render(tmp_path,frames,masks,tracking,{0:[],1:[]},result)
    document=(tmp_path/'index.html').read_text(encoding='utf8')
    assert rows[0]['suspended']==7 and rows[1]['recovered']==3 and rows[1]['seeded']==2
    assert '黄点为未知帧暂存候选光流' in document and '跨段物理身份未确认' in document
    assert '2 个局部坐标系' in document and '之前三个局部' not in document
    assert 'f.value="000001"' in document and '000029' not in document
