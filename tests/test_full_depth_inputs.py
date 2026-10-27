"""Real tiny PNG/binding contracts with fake scene, native-depth and baseline readers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from integrations.mvs_surface import full_depth_inputs as p


def fixture(tmp_path,monkeypatch,count=3,depth_count=2):
    parent=tmp_path/'parent';base=tmp_path/'baseline';base.mkdir()
    (parent/'surface/work').mkdir(parents=True);(parent/'export').mkdir();(parent/'masks').mkdir()
    scene_path=parent/'scene.mvs';scene_path.write_bytes(b'tiny synthetic scene')
    frames=[];mapping=[];cams=[];native={};baseline=[]
    for index in range(count):
        iid=11+index;name='image_'+str(iid)+'.png'
        image=parent/'images'/name;image.parent.mkdir(exist_ok=True)
        mask=parent/'mask'/('image_'+str(iid)+'.mask.png');mask.parent.mkdir(exist_ok=True)
        valid=parent/'valid'/('image_'+str(iid)+'.validity.png');valid.parent.mkdir(exist_ok=True)
        rgb=np.full((8,10,3),40+index,np.uint8)
        Image.fromarray(rgb).save(image);Image.fromarray(np.full((8,10),255,np.uint8)).save(mask);Image.fromarray(np.full((8,10),255,np.uint8)).save(valid)
        cam=dict(global_id=iid,name='images/'+name,width=10,height=8,K=[[20.,0,5],[0,20.,4],[0,0,1]],R=np.eye(3).tolist(),C=[index*.1,0.,0.])
        cams.append(cam)
        frames.append(dict(image=deepcopy(cam),image_file=p.binding(image),mask=p.binding(mask),validity=p.binding(valid),mask_name=mask.name,validity_name=valid.name))
        mapping.append(dict(export_image_id=iid,exported_name=name,decode_index=index*5,exported_png_path=str(image)))
        if index<min(2,count):
            baseline.append(dict(view_id=iid,image=rgb.copy(),mask=np.ones((8,10),bool),validity=np.ones((8,10),bool),depth=np.zeros((8,10),np.float32),K=np.asarray(cam['K']),R=np.eye(3),C=np.asarray(cam['C'])))
        if index<depth_count:
            file=parent/'surface/work'/('depth'+str(iid)+'.dmap');file.write_bytes(('depth'+str(iid)).encode())
            native[str(file.resolve())]=SimpleNamespace(view_ids=[iid],depth=np.full((8,10),2.,np.float32),K=np.asarray(cam['K']),R=np.eye(3),C=np.asarray(cam['C']))
    p.save(parent/'surface/plan.json',dict(scene=p.binding(scene_path),frames=frames,arguments=dict(image_root=str(parent),mask_root=str(parent/'mask'),validity_root=str(parent/'valid'))))
    p.save(parent/'surface/RESULT.json',dict(status='SURFACE_GENERATED_NOT_ACCEPTED',wall_seconds=1.))
    p.save(parent/'export/FRAME_MAP.json',dict(frames=mapping))
    p.save(parent/'masks/RESULT.json',dict(coverage_status='COMPLETE',mask_consumption='NOT_VERIFIED',frames=[dict(export_image_id=m['export_image_id'],decode_index=m['decode_index'],exported_name=m['exported_name'],mask=f['mask'],validity=f['validity']) for m,f in zip(mapping,frames)]))
    basefile=base/'native.bin';basefile.write_bytes(b'baseline sealed')
    data=dict(views=baseline,bindings=[p.binding(basefile)],reference_id=11,global_points=np.zeros((0,3)),global_colors=np.zeros((0,3),np.uint8))
    scene=dict(transform=np.eye(4).tolist(),images=cams)
    authenticated=[]
    monkeypatch.setattr(p,'read_scene',lambda path:deepcopy(scene))
    monkeypatch.setattr(p,'read_dmap',lambda path:native[str(Path(path).resolve())])
    def authenticate(depth,camera):
        assert depth.view_ids[0]==camera['global_id']
        for key in ('K','R','C'):np.testing.assert_allclose(getattr(depth,key),camera[key])
        authenticated.append(camera['global_id'])
    monkeypatch.setattr(p,'authenticate_depth',authenticate)
    monkeypatch.setattr(p,'load_model_inputs',lambda path,check:deepcopy(data))
    return parent,base,tmp_path/'full_spec.json',scene,data,native,authenticated


def test_prepare_serialized_full_load_common_coordinate_frame_and_missing_depth(tmp_path,monkeypatch):
    parent,base,spec,scene,data,native,auth=fixture(tmp_path,monkeypatch)
    before={path:path.read_bytes() for path in parent.rglob('*') if path.is_file()}
    r=p.prepare_full_inputs(parent,base,spec)
    assert r['view_count']==3 and r['depth_count']==2
    loaded=p.load_full_inputs(spec)
    assert [v['view_id'] for v in loaded['views']]==[11,12,13]
    assert auth==[11,12,11,12]
    assert loaded['full_input_provenance']['coordinate_transform']=='identity verified through common K/R/C'
    assert loaded['full_input_provenance']['mapping'][2]['decode_index']==10
    assert not loaded['views'][2]['depth_available'] and not loaded['views'][2]['depth'].any()
    assert loaded['full_input_provenance']['additional_views']==1
    for file,raw in before.items():assert file.read_bytes()==raw


@pytest.mark.parametrize('kind',['frame_id','plan_id','scene_transform','export_name'])
def test_identity_sets_and_export_names_fail_closed(tmp_path,monkeypatch,kind):
    parent,base,spec,scene,*_=fixture(tmp_path,monkeypatch)
    if kind=='scene_transform':scene['transform'][0][3]=1.
    elif kind=='frame_id':
        path=parent/'export/FRAME_MAP.json';value=p.load(path);value['frames'][2]['export_image_id']=99;p.save(path,value)
    elif kind=='plan_id':
        path=parent/'surface/plan.json';value=p.load(path);value['frames'][2]['image']['global_id']=99;p.save(path,value)
    else:
        path=parent/'export/FRAME_MAP.json';value=p.load(path);value['frames'][2]['exported_name']='wrong.png';p.save(path,value)
    with pytest.raises(ValueError):p.prepare_full_inputs(parent,base,spec)


@pytest.mark.parametrize('kind',['unbound_image','unbound_depth','changed_bytes'])
def test_serialized_spec_rejects_unbound_or_drift(tmp_path,monkeypatch,kind):
    parent,base,spec,*_=fixture(tmp_path,monkeypatch)
    p.prepare_full_inputs(parent,base,spec)
    value=p.load(spec)
    if kind=='changed_bytes':
        Path(value['depths'][0]['path']).write_bytes(b'drift')
    else:
        target=(str((parent/'images/image_13.png').resolve()) if kind=='unbound_image' else value['depths'][0]['path'])
        value['input_bindings']=[item for item in value['input_bindings'] if item['path']!=target];p.save(spec,value)
    with pytest.raises(ValueError):p.load_full_inputs(spec)


@pytest.mark.parametrize('kind',['K','R','C','image','mask','validity'])
def test_common_baseline_krc_or_pixels_cannot_be_silently_changed(tmp_path,monkeypatch,kind):
    parent,base,spec,scene,data,*_=fixture(tmp_path,monkeypatch)
    old=data['views'][0]
    if kind in ('K','R'):old[kind]=old[kind].copy();old[kind][0,0]+=.1
    elif kind=='C':old[kind]=old[kind]+.1
    elif kind=='image':old[kind]=old[kind].copy();old[kind][0,0,0]+=1
    else:old[kind]=old[kind].copy();old[kind][0,0]=False
    with pytest.raises(ValueError):p.prepare_full_inputs(parent,base,spec)


def test_duplicate_native_depth_and_capacity_no_silent_subset(tmp_path,monkeypatch):
    parent,base,spec,scene,data,native,*_=fixture(tmp_path,monkeypatch)
    native[str((parent/'surface/work/depth12.dmap').resolve())].view_ids=[11]
    with pytest.raises(ValueError,match='duplicate'):p.prepare_full_inputs(parent,base,spec)


def test_more_than_sixty_four_depths_rejected_before_native_read(tmp_path,monkeypatch):
    parent,base,spec,*_=fixture(tmp_path,monkeypatch,count=65,depth_count=65)
    def forbidden(*args):raise AssertionError('capacity rejection must precede reading native data')
    monkeypatch.setattr(p,'read_dmap',forbidden)
    with pytest.raises(ValueError,match='64'):p.prepare_full_inputs(parent,base,spec)


def test_additional_image_file_must_match_its_scene_frame_not_another_bound_png(tmp_path,monkeypatch):
    parent,base,spec,*_=fixture(tmp_path,monkeypatch)
    path=parent/'surface/plan.json';value=p.load(path)
    value['frames'][2]['image_file']=deepcopy(value['frames'][1]['image_file'])
    p.save(path,value)
    with pytest.raises(ValueError):p.prepare_full_inputs(parent,base,spec)



@pytest.mark.parametrize('kind',['decode','mask_binding','duplicate_mask_id','noncommon_image_size'])
def test_complete_mask_frame_mapping_and_native_image_size(tmp_path,monkeypatch,kind):
    parent,base,spec,*_=fixture(tmp_path,monkeypatch)
    if kind=='noncommon_image_size':
        image=parent/'images/image_13.png';Image.fromarray(np.full((9,10,3),42,np.uint8)).save(image)
        path=parent/'surface/plan.json';value=p.load(path);value['frames'][2]['image_file']=p.binding(image);p.save(path,value)
    else:
        path=parent/'masks/RESULT.json';value=p.load(path)
        if kind=='decode':value['frames'][2]['decode_index']+=1
        elif kind=='mask_binding':value['frames'][2]['mask']=deepcopy(value['frames'][1]['mask'])
        else:value['frames'][2]['export_image_id']=value['frames'][0]['export_image_id']
        p.save(path,value)
    with pytest.raises(ValueError):p.prepare_full_inputs(parent,base,spec)
