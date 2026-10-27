"""TEST_DATA offline native serialization and provenance; no solver or GPU."""
import json
import sqlite3
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import pycolmap._core as core
import pycolmap

from integrations.mvs_surface import experimental_inputs as ex
from integrations.mvs_masks import manifest_io as mio
from scripts import prepare_experimental_surface_inputs as cli
from test_native_subset import native
from test_mvs_mask_bridge import build_world, write_scene


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data),encoding='utf-8')


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    w=build_world(tmp_path/'TEST_DATA',model_images=2,extra_frames=0,n_ok=2)
    root=tmp_path/'producer';arm=root/'masked28';rgb=arm/'rgb';masks=arm/'feature_masks'
    model_root=arm/'models'/'0';model_root.mkdir(parents=True)
    model=native([1,2]);model.write_binary(str(model_root))
    inputs=[];manifest=ex.load(w['manifest'])
    for idx,name in enumerate(w['names']):
        for kind,source,dest,relative in (
            ('image',w['sfm']/name,rgb/name,name),
            ('mask',w['obj']/manifest['frames'][idx]['selected_mask']['path'],masks/(name+'.png'),name+'.png')):
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            inputs.append({'arm':'masked28','kind':kind,'input_index':idx,'relative':relative,
                           'source':ex.record(source),'copy':ex.record(dest)})
    dbpath=arm/'database.db'
    with sqlite3.connect(dbpath) as db:
        db.execute('CREATE TABLE images(image_id INTEGER,name TEXT,camera_id INTEGER)')
        db.execute('CREATE TABLE cameras(camera_id INTEGER,model INTEGER,width INTEGER,height INTEGER)')
        db.execute('CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB)')
        db.execute('INSERT INTO cameras VALUES (1,2,64,48)')
        for iid,image in model.images.items():
            db.execute('INSERT INTO images VALUES (?,?,?)',(iid,image.name,image.camera_id))
            xy=np.array([p.xy for p in image.points2D],dtype='<f4')
            db.execute('INSERT INTO keypoints VALUES (?,?,?,?)',(iid,len(xy),2,xy.tobytes()))
    worker=root/'fixture_worker.py';worker.write_bytes(b'# TEST_DATA no native producer execution\n')
    monkeypatch.setattr(ex,'PROFILE_WORKER_SHA256',ex.record(worker)['sha256'])
    options={'extract_features':{'reader':{'mask_path':str(masks)}}}
    config={'arm':'masked28','image_root':str(rgb),'mask_root':str(masks),'image_names':w['names'],
            'database_path':str(dbpath),'models_root':str(arm/'models'),'output':str(arm),'frozen_options':options}
    save(arm/'CONFIG.json',config)
    argv=['python','-B','-X','utf8',str(worker),str(arm/'CONFIG.json')]
    save(root/'PLAN.json',{'configs':[config],'config_hashes':[ex.record(arm/'CONFIG.json')],
         'source_bindings':[ex.record(worker),ex.record(core.__file__)],'inputs':inputs,'argv':[argv]})
    save(arm/'supervision'/'ENTRY.json',{'argv':argv,'cwd':str(arm),'status':'completed',
         'root_exit_code':0,'pid':123,'reclaim_confirmed':True,'errors':[],'log_errors':[]})
    save(arm/'RESULT.json',{'arm':'masked28','status':'COMPLETED','root_exit_code':0,
         'native_calls':['extract_features','match_exhaustive','incremental_mapping'],'actual_options':options,
         'models':[{'model_id':0,'path':str(model_root),'registered_native_ids':[1,2],'points3D':3}]})
    out=tmp_path/'identity'
    args=['identity','--producer',str(root),'--arm','masked28','--model-id','0',
          '--mask-manifest',str(w['manifest']),'--output',str(out)]
    return {'args':args,'out':out,'root':root,'arm':arm,'world':w}


def test_identity_reopens_native_db_and_original_masks(experiment):
    e=experiment
    assert cli.main(e['args'])==0
    link=ex.validate_identity(e['out']/'IDENTITY_SPEC.json')
    assert link['counts']=={'images':2,'points':3,'observations':6}
    assert [f['input_index'] for f in link['frames']]==[0,1]
    assert link['not_original_t01_sfm_run'] is True and link['full_object_accepted'] is False
    assert not Path(str(e['arm']/'database.db')+'-wal').exists()
    assert not Path(str(e['arm']/'database.db')+'-shm').exists()


@pytest.mark.parametrize('fault',['bytes','missing_binding','bad_entry','wrong_model','duplicate_selection','not_real','db_coordinate'])
def test_conflicting_or_missing_provenance_rejected(experiment,fault):
    e=experiment;assert cli.main(e['args'])==0
    path=e['out']/'IDENTITY_SPEC.json';spec=ex.load(path)
    if fault=='bytes':
        Path(spec['model_files'][0]['path']).write_bytes(b'changed')
    elif fault=='missing_binding':spec.pop('database')
    elif fault=='bad_entry':
        ep=Path(spec['producer_entry']['path']);entry=ex.load(ep);entry['reclaim_confirmed']=False
        save(ep,entry);spec['producer_entry']=ex.record(ep)
    elif fault=='wrong_model':spec['model_id']=9
    elif fault=='duplicate_selection':spec['mask_selections'][1]=spec['mask_selections'][0]
    elif fault=='not_real':spec['source_kind']='unverified'
    elif fault=='db_coordinate':
        dp=Path(spec['database']['path'])
        with sqlite3.connect(dp) as db:
            r=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=1').fetchone()
            xy=np.frombuffer(r[2],dtype='<f4').copy();xy[0]+=1
            db.execute('UPDATE keypoints SET data=? WHERE image_id=1',(xy.tobytes(),))
        spec['database']=ex.record(dp)
    save(path,spec)
    with pytest.raises((ValueError,KeyError,OSError)):
        ex.validate_identity(path)


def test_database_sidecar_rejected_without_open(tmp_path):
    path=tmp_path/'database.db'
    with sqlite3.connect(path) as db:db.execute('CREATE TABLE t(x)')
    Path(str(path)+'-wal').write_bytes(b'')
    with pytest.raises(ValueError,match='sidecars'):
        ex.immutable_database(path)
    assert not Path(str(path)+'-shm').exists()


def test_surface_spec_cannot_masquerade_as_standard_source(tmp_path):
    from integrations.mvs_surface.photo_inputs import validate
    path=tmp_path/'wrong.json';save(path,{'schema_version':ex.SURFACE_SCHEMA})
    with pytest.raises(ValueError):validate(path,'missing','missing')


def test_explicit_runner_origin_requires_source(tmp_path):
    from integrations.mvs_surface.runner import make_plan
    with pytest.raises(ValueError,match='photo-source'):
        make_plan('x','x','x','x','x',total_seconds=30,stage_seconds=20,
                  memory_bytes=2*1024**3,output_bytes=256*1024**2,input_origin='experimental-real')


def test_archive_cannot_replace_native_or_data(experiment):
    e=experiment;assert cli.main(e['args'])==0
    path=e['out']/'IDENTITY_SPEC.json';spec=ex.load(path)
    spec['historical_code_copies']=[{'original_path':str(e['arm']/'database.db'),
                                    'archive':spec['database']}]
    save(path,spec)
    with pytest.raises(ValueError,match='historical supervisor'):ex.validate_identity(path)


@pytest.mark.parametrize('fault',['mask_pixels','scene_xyz','bridge_exit','bridge_boolean'])
def test_existing_bridge_warp_and_independent_pixel_rejection(experiment,tmp_path,fault):
    e=experiment;assert cli.main(e['args'])==0
    identity_path=e['out']/'IDENTITY_SPEC.json';identity=ex.validate_identity(identity_path)
    bridge=tmp_path/'TEST_DATA_BRIDGE';target=native([1,2])
    camera=target.camera(1);camera.model=pycolmap.CameraModelId.PINHOLE
    camera.params=np.array([53.34,53.34,32.,24.])
    for rel in ('undistorted/sparse','roundtrip/sparse'):
        p=bridge/rel;p.mkdir(parents=True)
        if rel.startswith('roundtrip'):
            roundtrip=pycolmap.Reconstruction(target)
            for point in roundtrip.points3D.values():
                for track in point.track.elements:
                    image=roundtrip.images[track.image_id]
                    image.points2D[track.point2D_idx].xy=roundtrip.cameras[image.camera_id].img_from_cam(image.cam_from_world()*point.xyz)
            roundtrip.write_binary(str(p))
        else:target.write_binary(str(p))
    for name in e['world']['names']:
        p=bridge/'undistorted/images'/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(identity['source_images'])/name,p)
    scene=bridge/'scene.mvs';write_scene(scene,[Path(n).name for n in e['world']['names']],64,48,focal=53.34)
    # Replace the helper's deliberately empty vertex section with native records.
    from integrations.mvs_surface import scene as mvsi
    blob=scene.read_bytes();reader=mvsi._Reader(blob);reader.unpack('4sII','header')
    mvsi._read_images(reader,mvsi._read_platforms(reader));offset=reader.offset
    vertices=struct.pack('<Q',len(target.points3D))
    colors=b''
    for p in target.points3D.values():
        vertices+=struct.pack('<fffQ',*p.xyz,len(p.track.elements))
        for track in p.track.elements:vertices+=struct.pack('<If',track.image_id-1,1.)
        colors+=bytes(reversed(p.color))  # genuine MVSI BGR byte order
    section=vertices+struct.pack('<QQ',0,len(target.points3D))+colors
    scene.write_bytes(blob[:offset]+section+blob[offset+24:])
    assert {tuple(p['rgb']) for p in ex.scene_point_records(scene)} == {(10,20,30)}
    tool=bridge/'TEST_DATA_TOOL.bin';tool.write_bytes(b'TEST_DATA never executable')
    stages=[{'stage':name,'argv':[str(tool),name],'cwd':str(bridge)} for name in
            ('undistort','mvs_import','mvs_export','verify')]
    from scripts.prepare_mvs import sha
    save(bridge/'PLAN.json',{'source_sparse':identity['source_sparse'],'source_images':identity['source_images'],
         'script_sha256':sha(Path(__file__).resolve().parents[1]/'scripts/prepare_mvs.py'),
         'tool_sha256':sha(tool),'stages':stages})
    save(bridge/'STEPS.json',[{**s,'returncode':0,'pid':123} for s in stages])
    save(bridge/'RESULT.json',{'status':'PASS'})
    save(bridge/'UNDISTORT_COMPARISON.json',{'status':'PASS'})
    save(bridge/'INPUT_BINDINGS.json',[ex.record(Path(identity['source_images'])/n) for n in e['world']['names']]+
         [ex.record(p) for p in Path(identity['source_sparse']).iterdir() if p.is_file()])
    out=tmp_path/'warped'
    assert cli.main(['warp','--identity-spec',str(identity_path),'--bridge',str(bridge),'--output',str(out)])==0
    path=out/'SURFACE_SPEC.json'
    link=ex.validate(path,scene,bridge,out/'masks',out/'validity')
    assert link['status']=='EXPERIMENTAL_SOURCE_AND_WARP_BOUND'
    spec=ex.load(path)
    if fault=='mask_pixels':
        p=Path(spec['outputs'][0]['mask']['path'])
        from PIL import Image
        with Image.open(p) as handle:labels=np.array(handle)
        labels[0,0]=255 if labels[0,0]==0 else 0
        Image.fromarray(labels).save(p);spec['outputs'][0]['mask']=ex.record(p)
    elif fault=='scene_xyz':
        blob=bytearray(scene.read_bytes());struct.pack_into('<f',blob,offset+8,99.)
        scene.write_bytes(blob)
    elif fault=='bridge_exit':
        steps=ex.load(bridge/'PLAN.json')['stages']
        save(bridge/'STEPS.json',[{**s,'returncode':1 if i==1 else 0,'pid':123} for i,s in enumerate(steps)])
    else:
        steps=ex.load(bridge/'PLAN.json')['stages']
        for bad in ({'returncode':False,'pid':123},{'returncode':0,'pid':True}):
            records=[{**s,'returncode':0,'pid':123} for s in steps];records[1].update(bad)
            save(bridge/'STEPS.json',records)
            spec['bridge_bindings']=[ex.record(p) for p in sorted(bridge.rglob('*')) if p.is_file()]
            save(path,spec)
            with pytest.raises(ValueError,match='actual stage conflict'):
                ex.validate(path,scene,bridge,out/'masks',out/'validity')
        return
    spec['bridge_bindings']=[ex.record(p) for p in sorted(bridge.rglob('*')) if p.is_file()]
    save(path,spec)
    with pytest.raises((ValueError,mio.ProductError)):ex.validate(path,scene,bridge,out/'masks',out/'validity')


def test_historical_supervisor_is_exact_context_only(experiment,tmp_path):
    e=experiment;assert cli.main(e['args'])==0
    archive=tmp_path/'supervisor_original.py';archive.write_bytes(b'# TEST_DATA historical supervisor\n')
    original=str(Path(ex.__file__).resolve().parents[2]/'integrations/mvs_surface/supervision.py')
    bound=ex.record(archive);declared={**bound,'path':original}
    plan_path=e['root']/'PLAN.json';plan=ex.load(plan_path);plan['source_bindings'].append(declared);save(plan_path,plan)
    spec_path=e['out']/'IDENTITY_SPEC.json';spec=ex.load(spec_path);spec['producer_plan']=ex.record(plan_path)
    spec['historical_code_copies']=[{'original_path':original,'archive':bound}];save(spec_path,spec)
    link=ex.validate_identity(spec_path)
    assert bound in link['bindings'] and declared not in link['bindings']
    archive.write_bytes(b'# wrong archive\n');spec['historical_code_copies'][0]['archive']=ex.record(archive)
    save(spec_path,spec)
    with pytest.raises(ValueError,match='frozen original bytes'):ex.validate_identity(spec_path)


def test_producer_boolean_exit_codes_refused(experiment):
    e=experiment;assert cli.main(e['args'])==0
    path=e['out']/'IDENTITY_SPEC.json';original=ex.load(path)
    for key in ('producer_entry','producer_result'):
        spec=json.loads(json.dumps(original));p=Path(spec[key]['path']);doc=ex.load(p)
        doc['root_exit_code']=False;save(p,doc);spec[key]=ex.record(p);save(path,spec)
        with pytest.raises(ValueError):ex.validate_identity(path)
        doc['root_exit_code']=0;save(p,doc)
    save(path,original)


def test_derived_identity_recomputes_all_views_and_exact_points(experiment,tmp_path):
    e=experiment; assert cli.main(e['args'])==0
    out=tmp_path/'derived'
    assert cli.main(['derive','--parent-identity-spec',str(e['out']/'IDENTITY_SPEC.json'),'--output',str(out)])==0
    link=ex.validate_identity(out/'IDENTITY_SPEC.json')
    assert link['counts']=={'images':2,'points':3,'observations':6}
    assert link['derivation']['removed_points']==[]
    assert link['source_images']==ex.validate_identity(e['out']/'IDENTITY_SPEC.json')['source_images']


@pytest.mark.parametrize('fault',['legal_point','intrinsics','subset','certificate','code','mask'])
def test_derived_identity_rejects_refreshed_hash_tampering(experiment,tmp_path,fault):
    e=experiment; assert cli.main(e['args'])==0
    out=tmp_path/'derived'; assert cli.main(['derive','--parent-identity-spec',str(e['out']/'IDENTITY_SPEC.json'),'--output',str(out)])==0
    path=out/'IDENTITY_SPEC.json'; spec=ex.load(path)
    if fault in ('legal_point','intrinsics','subset'):
        m=pycolmap.Reconstruction(spec['derived_sparse'])
        if fault=='legal_point':m.delete_point3D(next(iter(m.points3D)))
        elif fault=='intrinsics':m.cameras[1].params[0]+=1
        else:m.deregister_frame(m.images[2].frame_id)
        m.write_binary(spec['derived_sparse'])
        spec['model_files']=[ex.record(p) for p in sorted(Path(spec['derived_sparse']).iterdir()) if p.is_file()]
    elif fault=='certificate':
        p=Path(spec['certificate']['path']); c=ex.load(p);c['relation']['removed_points']=[999];save(p,c);spec['certificate']=ex.record(p)
    elif fault=='code':spec['code_bindings'][0]['sha256']='0'*64
    else:
        parent=ex.load(e['out']/'IDENTITY_SPEC.json');p=Path(parent['mask_manifests'][0]['path']);p.write_text('{}')
    save(path,spec)
    with pytest.raises((ValueError,KeyError,OSError)):ex.validate_identity(path)


def test_whole_duplicate_point_deletion_relation(tmp_path):
    source=native([1,2],duplicates=2)
    p=tmp_path/'model';p.mkdir();source.write_binary(str(p));child=pycolmap.Reconstruction(str(p))
    removed=[]
    for pid,point in list(child.points3D.items()):
        ids=[e.image_id for e in point.track.elements]
        if len(set(ids))!=len(ids) or len(set(ids))<2: removed.append(pid);child.delete_point3D(pid)
    result=ex.derived_relation(source,child)
    assert result['removed_points']==sorted(removed) and len(removed)==2
    assert result['original_points']==5 and result['derived_points']==3


def test_derived_cli_fake_backend_exact_whole_deletion(tmp_path,monkeypatch):
    from types import SimpleNamespace as N
    parent=tmp_path/'parent.json';save(parent,{'schema_version':ex.SCHEMA})
    calls=[]
    model=N(points3D={1:N(track=N(elements=[N(image_id=1),N(image_id=1)])),
                      2:N(track=N(elements=[N(image_id=1)])),
                      3:N(track=N(elements=[N(image_id=1),N(image_id=2)]))},
            delete_point3D=lambda pid:calls.append(pid),write_binary=lambda p:None)
    monkeypatch.setattr(pycolmap,'Reconstruction',lambda p:model)
    monkeypatch.setattr(ex,'derived_relation',lambda a,b:{'removed_points':[1,2]})
    monkeypatch.setattr(ex,'validate_identity',lambda p,**kw:{'source_sparse':str(tmp_path/'source'),'source_images':'rgb','counts':{'points':1}})
    out=tmp_path/'child';out.mkdir()
    assert cli.derive(N(parent_identity_spec=str(parent)),out)['status']=='DERIVED_IDENTITY_PREPARED'
    assert calls==[1,2]


@pytest.mark.parametrize('fault',['none','xyz','reciprocal','xy'])
def test_derivation_parent_only_permits_whole_point_track_defects(experiment,tmp_path,fault):
    e=experiment; assert cli.main(e['args'])==0
    path=e['out']/'IDENTITY_SPEC.json';spec=ex.load(path)
    root=e['arm']/'models/0';m=native([1,2],duplicates=1)
    if fault=='xyz':m.points3D[1].xyz=np.array([np.nan,1.,2.])
    elif fault=='reciprocal':m.images[1].points2D[0].point3D_id=999
    elif fault=='xy':m.images[1].points2D[0].xy=np.array([np.nan,2.])
    m.write_binary(str(root)); spec['model_files']=[ex.record(p) for p in sorted(root.iterdir()) if p.is_file()]
    result_path=Path(spec['producer_result']['path']);result=ex.load(result_path);result['models'][0]['points3D']=4;save(result_path,result);spec['producer_result']=ex.record(result_path)
    with sqlite3.connect(spec['database']['path']) as db:
        for iid,image in m.images.items():
            xy=np.array([p.xy for p in image.points2D],dtype='<f4');db.execute('UPDATE keypoints SET rows=?,data=? WHERE image_id=?',(len(xy),xy.tobytes(),iid))
    spec['database']=ex.record(spec['database']['path']);save(path,spec)
    with pytest.raises(ValueError):ex.validate_identity(path)
    if fault=='none':
        authenticated=ex.validate_identity(path,_derivation_parent=True)
        assert authenticated['status']=='DERIVATION_PARENT_AUTHENTICATED_NOT_BRIDGE_ADMITTED'
        out=tmp_path/'derived'; assert cli.main(['derive','--parent-identity-spec',str(path),'--output',str(out)])==0
        assert ex.validate_identity(out/'IDENTITY_SPEC.json')['counts']['points']==3
    else:
        with pytest.raises(ValueError):ex.validate_identity(path,_derivation_parent=True)
