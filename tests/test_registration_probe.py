"""Fake native API only: never imports installed pycolmap or estimates a pose."""
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace as S
import pytest
from integrations.demo_pipeline import registration_probe as probe
from integrations.demo_pipeline.registration_gate import bogus_camera


class Options:
    def __init__(self):
        self.num_threads=-1;self.random_seed=-1;self.min_focal_length_ratio=.1;self.max_focal_length_ratio=10.;self.max_extra_param=1.;self.abs_pose_min_num_inliers=30
    def check(self):return True
    def todict(self):return vars(self).copy()


class CacheOptions:
    def todict(self):return vars(self).copy()


class Camera:
    width=100;height=100
    model_name='SIMPLE_RADIAL'
    def __init__(self,k=2.):self.params=[100.,50.,50.,k]
    def has_bogus_params(self,*thresholds):return bogus_camera(self.width,self.height,self.params,*thresholds)['bogus']


class Image:
    def __init__(self,ident,posed):
        self.name='images/'+str(ident)+'.png';self.camera_id=1;self.has_pose=posed
        self.frame=S(rig_id=1)
        self.points2D=[S(point3D_id=7,has_point3D=lambda:posed)]
    def cam_from_world(self):return S(rotation=S(matrix=lambda:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),translation=[0.,0.,0.])


class Reconstruction:
    def __init__(self,path):
        self.cameras={1:Camera()};self.images={2:Image(2,True),3:Image(3,True)}
        self.points3D={7:S(xyz=[0.,0.,5.],color=[0,0,0],error=0.,track=S(elements=[S(image_id=2,point2D_idx=0),S(image_id=3,point2D_idx=0)]))}
    def image(self,i):return self.images[i]
    def exists_image(self,i):return i in self.images
    def reg_image_ids(self):return [i for i,v in self.images.items() if v.has_pose]
    def rig(self,ident):return S(num_sensors=lambda:1)


def setup(tmp_path,monkeypatch,mode='false'):
    root=tmp_path/'source';root.mkdir();binary=root/'model';binary.mkdir();logs=root/'logs';logs.mkdir()
    for n in ('cameras.bin','images.bin','points3D.bin','rigs.bin','frames.bin'):(binary/n).write_bytes(b'fake native source')
    row=dict(input_index=19,database_image_id=1,staged_image_name='images/1.png',upstream_io_run_id='io')
    (root/'frames_map.jsonl').write_text(json.dumps(row),encoding='utf-8')
    run=dict(contract_version='sfm-native-object-v1',upstream_io_run_id='io',backend=dict(name='pycolmap',version='4.2.1'),
        models=[dict(model_id='0',binary_path='model',usable=True,roundtrip_verified=True,registered_image_ids=[2,3],num_points3D=1)])
    run_path=root/'run.json';run_path.write_text(json.dumps(run),encoding='utf-8')
    pipeline=dict(num_threads=2,random_seed=0,min_focal_length_ratio=.1,max_focal_length_ratio=10.,max_extra_param=1.,
        min_num_matches=15,ignore_watermarks=False,image_names=['images/1.png','images/2.png','images/3.png'],mapper=Options().todict())
    if mode=='gate_pass':pipeline['mapper']['abs_pose_min_num_inliers']=1
    (logs/'incremental_mapping.result.json').write_text(json.dumps(dict(actual_options=dict(pipeline=pipeline))),encoding='utf-8')
    with sqlite3.connect(root/'database.db') as db:db.execute('CREATE TABLE sentinel(value INTEGER)');db.execute('INSERT INTO sentinel VALUES(42)')
    calls=[];handles=[]
    class Database:
        @staticmethod
        def open(path):
            assert Path(path)!=root/'database.db' and Path(path).name=='snapshot.db'
            with sqlite3.connect(path) as c:assert c.execute('SELECT value FROM sentinel').fetchone()[0]==42
            handles.append(path);return S(close=lambda:calls.append('close'))
    class Mapper:
        def __init__(self,cache):self.observation_manager=S(num_visible_points3D=lambda i:120,num_visible_correspondences=lambda i:385)
        def begin_reconstruction(self,r):
            self.r=r;r.images[1]=Image(1,False)
            if mode=='begin_mutation':r.cameras[1].params[0]+=1
            if mode=='good':r.cameras[1].params[3]=0
        def register_next_image(self,options,image_id):
            assert (tmp_path/'out/PREGATE.json').is_file();calls.append('register')
            if mode=='exception':raise RuntimeError('fake native failure')
            if mode=='drift':run_path.write_text(run_path.read_text()+' ',encoding='utf-8')
            return False
        def end_reconstruction(self,discard):calls.append(('cleanup',discard))
    graph=S(extract_correspondences=lambda i,row:[S(image_id=2,point2D_idx=0),S(image_id=3,point2D_idx=0)])
    def make_reconstruction(path):
        r=Reconstruction(path)
        if mode=='gate_pass':r.cameras[1].params[3]=0.
        return r
    pc=S(__version__='4.2.1',__file__='fake-pycolmap',logging=S(),Database=Database,DatabaseCacheOptions=CacheOptions,
        DatabaseCache=S(create=lambda db,options:S(correspondence_graph=graph)),Reconstruction=make_reconstruction,
        IncrementalMapperOptions=Options,IncrementalMapper=Mapper)
    monkeypatch.setattr(probe,'_load_pycolmap',lambda:pc)
    return run_path,calls,handles


def test_exactly_one_native_call_snapshot_only_pregate_flushed_and_sources_unchanged(tmp_path,monkeypatch):
    run,calls,handles=setup(tmp_path,monkeypatch);before=(run.parent/'database.db').read_bytes()
    r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='ONE_NATIVE_CALL_COMPLETED_NOT_GEOMETRY_ACCEPTED'
    assert r['registration_call_count']==1 and calls.count('register')==1
    assert r['fallback_call_count']==r['extra_estimator_call_count']==r['native_model_writes']==0
    assert r['stage_estimation']==r['stage_refinement']=='NOT_REACHED'
    assert r['source_unchanged'] and (run.parent/'database.db').read_bytes()==before
    assert r['gate_summary']['structure_based']['retained_count']==0
    assert r['gate_summary']['structure_based']['potential_unique_2D3D']==1
    assert r['target_pose_before'] is None and r['target_pose_after'] is None
    assert ('cleanup',False) in calls and 'close' in calls and len(handles)==1
    events=[json.loads(l) for l in (tmp_path/'out/EVENTS.jsonl').read_text().splitlines()]
    assert sum(e['phase']=='REGISTER_NEXT_IMAGE_ENTER' for e in events)==1
    assert events[-1]['phase']=='FINALIZED'
    json.dumps(r,allow_nan=False)


def test_begin_mutation_prevents_registration(tmp_path,monkeypatch):
    run,calls,_=setup(tmp_path,monkeypatch,'begin_mutation')
    r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='PROBE_FAILED' and r['registration_call_count']==0 and 'register' not in calls
    assert 'altered original' in r['error']


def test_native_exception_no_retry_and_cleanup(tmp_path,monkeypatch):
    run,calls,_=setup(tmp_path,monkeypatch,'exception');r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='PROBE_FAILED' and r['registration_call_count']==1 and calls.count('register')==1
    assert 'close' in calls and ('cleanup',False) in calls


def test_source_drift_refuses_manifest(tmp_path,monkeypatch):
    run,calls,_=setup(tmp_path,monkeypatch,'drift');r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='SOURCE_DRIFT_REFUSED' and not r['source_unchanged']


def test_version_refusal_before_native_import(tmp_path,monkeypatch):
    run,calls,_=setup(tmp_path,monkeypatch);value=json.loads(run.read_text());value['backend']['version']='other';run.write_text(json.dumps(value),encoding='utf-8')
    monkeypatch.setattr(probe,'_load_pycolmap',lambda:(_ for _ in ()).throw(AssertionError('should not import')))
    r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='PROBE_FAILED' and r['registration_call_count']==0


def test_fresh_output_precondition(tmp_path,monkeypatch):
    run,_,_=setup(tmp_path,monkeypatch);(tmp_path/'out').mkdir()
    with pytest.raises(ValueError):probe.run_probe(run,19,'0',tmp_path/'out')


def test_passed_collection_then_native_false_keeps_internal_stages_unknown(tmp_path,monkeypatch):
    run,calls,_=setup(tmp_path,monkeypatch,'gate_pass');r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='ONE_NATIVE_CALL_COMPLETED_NOT_GEOMETRY_ACCEPTED'
    assert r['gate_summary']['structure_based']['gate_reachable'] is True
    assert r['native_registration_result'] is False and r['stage_estimation']==r['stage_refinement']=='UNKNOWN'


def test_nonempty_source_wal_prevents_native_open(tmp_path,monkeypatch):
    run,calls,handles=setup(tmp_path,monkeypatch)
    Path(str(run.parent/'database.db')+'-wal').write_bytes(b'unfrozen transaction')
    r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='PROBE_FAILED' and r['registration_call_count']==0 and not handles


def test_generalized_rig_branch_refused_before_registration(tmp_path,monkeypatch):
    run,calls,_=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(Reconstruction,'rig',lambda self,ident:S(num_sensors=lambda:2))
    r=probe.run_probe(run,19,'0',tmp_path/'out')
    assert r['status']=='PROBE_FAILED' and r['registration_call_count']==0 and 'register' not in calls
    assert 'single-sensor' in r['error']
