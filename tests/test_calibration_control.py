"""Fake native calls only; no installed pycolmap import, BA or registration."""
import copy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace as S
import numpy as np
import pytest
from integrations.demo_pipeline import calibration_control as control
from integrations.demo_pipeline.registration_gate import bogus_camera


class Options:
    def __init__(self):
        self.num_threads=-1;self.random_seed=-1;self.min_focal_length_ratio=.1;self.max_focal_length_ratio=10.;self.max_extra_param=1.;self.abs_pose_min_num_inliers=30;self.init_min_tri_angle=16.
    def check(self):return True
    def todict(self):return vars(self).copy()


class DataOptions:
    def todict(self):return {k:v.todict() if hasattr(v,'todict') else v for k,v in vars(self).items()}


class PipelineOptions(DataOptions):
    def __init__(self):
        for k,v in Options().todict().items():setattr(self,k,v)
        self.ba_refine_extra_params=True;self.ba_refine_focal_length=True;self.ba_refine_principal_point=False
        self.triangulation=DataOptions();self.triangulation.min_angle=1.5;self.triangulation.random_seed=-1
    def get_triangulation(self):return copy.deepcopy(self.triangulation)
    def get_global_bundle_adjustment(self):return S(refine_extra_params=self.ba_refine_extra_params,todict=lambda:None)


class BA:
    def __init__(self):self.refine_extra_params=True;self.refine_focal_length=True;self.refine_principal_point=False
    def todict(self):return vars(self).copy()


class Pose:
    def __init__(self,translation):self.translation=np.array(translation,dtype=float);self.rotation=S(matrix=lambda:np.eye(3))
    def __mul__(self,xyz):return np.asarray(xyz)+self.translation


class Camera:
    model_name='SIMPLE_RADIAL';width=100;height=100
    def __init__(self):self.params=np.array([100.,50.,50.,0.])
    def has_bogus_params(self,*args):return bogus_camera(self.width,self.height,self.params.tolist(),*args)['bogus']
    def img_from_cam(self,xyz):return np.asarray(xyz[:2])/xyz[2]*100+50


class Pixel:
    def __init__(self,xy,pid=None):self.xy=np.array(xy,dtype=float);self.point3D_id=pid
    def has_point3D(self):return self.point3D_id is not None


class Image:
    def __init__(self,i):self.name='images/'+str(i)+'.png';self.camera_id=1;self.frame=S(rig_id=1);self.has_pose=False;self.points2D=[Pixel([50,50]) for _ in range(30)];self.pose=Pose([0,0,0])
    def cam_from_world(self):return self.pose


class Rec:
    def __init__(self,source=None):
        if source is not None:self.__dict__=copy.deepcopy(source.__dict__)
        else:self.cameras={};self.images={};self.points3D={}
    def reg_image_ids(self):return [i for i,v in self.images.items() if v.has_pose]
    def image(self,i):return self.images[i]
    def exists_image(self,i):return i in self.images
    def normalize(self):pass
    def rig(self,ident):return S(num_sensors=lambda:1)


def fixture(tmp_path,monkeypatch,mode='normal'):
    root=tmp_path/'source';root.mkdir();logs=root/'logs';logs.mkdir();run_path=root/'run.json'
    run_path.write_text(json.dumps(dict(backend=dict(name='pycolmap',version='4.2.1'),contract_version='sfm-native-object-v1',upstream_io_run_id='io')),encoding='utf-8')
    rows=[dict(input_index=index,database_image_id=ident,staged_image_name='images/'+str(ident)+'.png',upstream_io_run_id='io') for index,ident in ((20,12),(21,14),(19,11))]
    (root/'frames_map.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    recorded=dict(num_threads=2,random_seed=0,min_focal_length_ratio=.1,max_focal_length_ratio=10.,max_extra_param=1.,
        mapper=Options().todict(),triangulation=dict(min_angle=1.5,random_seed=-1),ba_refine_extra_params=True,
        ba_refine_focal_length=True,ba_refine_principal_point=False,min_num_matches=15,ignore_watermarks=False,image_names=['images/11.png','images/12.png','images/14.png'])
    (logs/'incremental_mapping.result.json').write_text(json.dumps(dict(actual_options=dict(pipeline=recorded))),encoding='utf-8')
    with sqlite3.connect(root/'database.db') as db:db.execute('CREATE TABLE witness(value INTEGER)');db.execute('INSERT INTO witness VALUES(42)')
    calls=[]
    class Mapper:
        def __init__(self,cache):self.observation_manager=S(num_visible_points3D=lambda i:30,num_visible_correspondences=lambda i:60)
        def begin_reconstruction(self,r):
            self.r=r
            if not r.images:r.cameras={1:Camera()};r.images={i:Image(i) for i in (11,12,14)}
            if mode=='nonzero_initial':r.cameras[1].params[3]=.1
        def end_reconstruction(self,discard):calls.append(('end',discard))
        def estimate_initial_two_view_geometry(self,opts,a,b):
            calls.append(('estimate',a,b))
            if mode=='bad_pair':return None
            return Pose([1,0,0])
        def register_initial_image_pair(self,opts,a,b,pose):
            calls.append(('initial',a,b));self.r.images[a].has_pose=True;self.r.images[b].has_pose=True;self.r.images[b].pose=pose
        def triangulate_image(self,opts,i):
            calls.append(('tri',i,opts.min_angle))
            if self.r.points3D:return 0
            for row in range(30):
                pid=row+1;xyz=np.array([row*.001,0,5.]);self.r.points3D[pid]=S(xyz=xyz,color=np.array([0,0,0]),error=0.,track=S(elements=[S(image_id=12,point2D_idx=row),S(image_id=14,point2D_idx=row)]))
                for iid in (12,14):self.r.images[iid].points2D[row]=Pixel(self.r.cameras[1].img_from_cam(self.r.images[iid].pose*xyz),pid)
            return 30
        def adjust_global_bundle(self,opts,ba):
            calls.append(('ba',ba.refine_extra_params))
            if ba.refine_extra_params:self.r.cameras[1].params[3]=2.
            if mode=='drift':run_path.write_text(run_path.read_text()+' ',encoding='utf-8')
            return True
        def filter_points(self,opts):return 0
        def filter_frames(self,opts):
            if mode=='filtered':
                for image in self.r.images.values():image.has_pose=False
            return 0
        def register_next_image(self,opts,i):calls.append(('register',i));return False
    class PO(PipelineOptions):
        def get_global_bundle_adjustment(self):return BA()
    class DB:
        @staticmethod
        def open(path):
            assert Path(path)!=root/'database.db'
            return S(close=lambda:calls.append(('close',)))
    graph=S(extract_correspondences=lambda i,row:[S(image_id=12,point2D_idx=row),S(image_id=14,point2D_idx=row)])
    class AliasedRec(Rec):
        def __init__(self,source=None):
            if source is not None:self.__dict__=source.__dict__
            else:super().__init__()
    pc=S(__version__='4.2.1',logging=S(),Database=DB,DatabaseCacheOptions=DataOptions,
        DatabaseCache=S(create=lambda db,opts:S(correspondence_graph=graph,cameras={1:Camera()})),IncrementalMapperOptions=Options,
        IncrementalPipelineOptions=PO,Reconstruction=AliasedRec if mode=='alias_clone' else Rec,IncrementalMapper=Mapper)
    monkeypatch.setattr(control.probe,'_load_pycolmap',lambda:pc)
    return run_path,calls


def test_shared_initial_state_only_extra_refinement_differs_and_bounded_counts(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch);before=(run.parent/'database.db').read_bytes()
    r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='CONTROL_COMPLETED_NOT_GEOMETRY_ACCEPTED'
    assert r['database_pair']==[12,14] and r['target_database_image_id']==11
    assert r['counts']==dict(initial_estimation=1,initial_registration=1,triangulation=2,global_ba=2,target_registration=2)
    assert [c for c in calls if c[0]=='ba']==[('ba',True),('ba',False)]
    assert [c for c in calls if c[0]=='tri']==[('tri',12,16.),('tri',14,16.)]
    a=json.loads((tmp_path/'out/FREE_K_PRE_BA.json').read_text());b=json.loads((tmp_path/'out/FIX_DB_K0_PRE_BA.json').read_text())
    assert a==b
    ba=[dict(arm['ba_options']) for arm in r['arms']]
    assert ba[0].pop('refine_extra_params') is True and ba[1].pop('refine_extra_params') is False and ba[0]==ba[1]
    assert r['source_unchanged'] and (run.parent/'database.db').read_bytes()==before
    assert not r['calibration_claim'] and r['native_model_writes']==0


def test_unsuitable_initial_pair_stops_no_alternate(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch,'bad_pair');r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='CONTROL_FAILED' and r['counts']['initial_estimation']==1 and r['counts']['global_ba']==0
    assert not any(c[0] in ('tri','ba','register') for c in calls)


def test_postfilter_model_invalid_skips_registration(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch,'filtered');r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['counts']['global_ba']==2 and r['counts']['target_registration']==0
    assert all(a['registration_status']=='NOT_RUN' for a in r['arms'])


def test_nonempty_wal_refused_before_native(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch);Path(str(run.parent/'database.db')+'-wal').write_bytes(b'not empty')
    monkeypatch.setattr(control.probe,'_load_pycolmap',lambda:(_ for _ in ()).throw(AssertionError('no native import')))
    r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='CONTROL_FAILED' and 'WAL' in r['error'] and not calls


def test_source_drift_refused(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch,'drift');r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='SOURCE_DRIFT_REFUSED' and not r['source_unchanged']


def test_nonzero_initial_k_refuses_without_clamping(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch,'nonzero_initial');r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='CONTROL_FAILED' and 'zero' in r['error']
    assert r['counts']['initial_estimation']==r['counts']['global_ba']==0


def test_aliasing_clone_refuses_second_arm_and_registration(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch,'alias_clone');r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='CONTROL_FAILED' and 'not isolated' in r['error']
    assert r['counts']['global_ba']==1 and r['counts']['target_registration']==0


def test_mapped_name_disagreement_refuses_before_estimation(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch)
    path=run.parent/'frames_map.jsonl';path.write_text(path.read_text().replace('images/12.png','images/wrong.png'))
    r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='CONTROL_FAILED' and r['counts']['initial_estimation']==0 and 'name differs' in r['error']


def test_generalized_rig_refuses_before_estimation(tmp_path,monkeypatch):
    run,calls=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(Rec,'rig',lambda self,ident:S(num_sensors=lambda:2))
    r=control.run_control(run,[20,21],19,tmp_path/'out')
    assert r['status']=='CONTROL_FAILED' and r['counts']['initial_estimation']==0 and 'single-sensor' in r['error']
