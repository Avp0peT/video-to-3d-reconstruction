from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from integrations.demo_pipeline import coverage_expansion as mod


def test_camera_copies_must_agree():
    obs=dict(image_id=1,input_index=4,params=[100,10,20,0],R=np.eye(3).tolist(),t=[0,0,0])
    assert mod.camera_table([{'observations':[obs]}])[4]==obs
    with pytest.raises(ValueError,match='disagree'):
        mod.camera_table([{'observations':[obs,dict(obs,t=[1,0,0])]}])


def test_dedup_requires_two_common_views_and_all_common_agree():
    a=[dict(image_id=i,raw_pixels=[10*i,20]) for i in (1,2,3)]
    p={'point_id':7,'observations':a}
    assert mod.duplicate_observations(a,[p])==7
    assert mod.duplicate_observations(a[:1],[p]) is None
    assert mod.duplicate_observations(a[:2]+[dict(a[2],raw_pixels=[100,50])],[p]) is None


def test_coverage_is_cell_count_and_excludes_background():
    mask=np.zeros((120,80),bool);mask[:60,:40]=True
    assert mod.occupancy([[1,1],[2,2],[75,110]],mask)==dict(
        occupied_cells=1,available_cells=24,occupied_fraction=1/24)


def test_real_tracker_to_triangulation_to_export(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'outputs/toolchains/opencv-4.13.0.92'))
    import cv2 as cv
    cv.setNumThreads(2)
    base=cv.GaussianBlur(np.random.default_rng(99).integers(0,256,(120,160),dtype=np.uint8),(3,3),.6)
    grays=[cv.warpAffine(base,np.float32([[1,0,6*i],[0,1,0]]),(160,120)) for i in range(3)]
    frames=[];masks=[np.full_like(base,255)]*3;cameras={}
    for i,gray in enumerate(grays):
        p=tmp_path/('image%d.png'%i);Image.fromarray(gray).save(p)
        frames.append(dict(input_index=i,path=str(p),original_mask_status='OK'))
        cameras[i]=dict(image_id=i+1,input_index=i,params=[200,80,60,0],R=np.eye(3).tolist(),t=[.3*i,0,0])
    original=dict(point_id=1,xyz=[0,0,10],color=[127]*3,accepted_conditionally=True,
        observations=[dict(cameras[i],feature_row=1,foreground=True,raw_pixels=[80+6*i,60]) for i in range(3)])
    pipeline=tmp_path/'pipeline';pipeline.mkdir();local=tmp_path/'local';local.mkdir()
    monkeypatch.setattr(mod,'read_bound',lambda *a,**k:{'stages':{'sfm':{'run_json':'fake'}}})
    monkeypatch.setattr(mod,'load_inputs',lambda *a: (frames,grays,masks,{}))
    monkeypatch.setattr(mod,'load_cv',lambda refs:cv)
    monkeypatch.setattr(mod,'load_local_models',lambda *a:[dict(group_id=0,model_id=0,cameras=cameras,ledger=[original])])
    # Surface generation already has its own real tests. This case exercises candidate-to-3D wiring.
    monkeypatch.setattr(mod,'build_faces',lambda p:([],[],{}))
    out=tmp_path/'output'
    result=mod.run(pipeline,local,out)
    assert result['status']=='PARTIAL_CONDITIONAL_EXPANSION'
    assert result['groups'][0]['added_strict']>20
    assert not result['background_in_object_model'] and not result['geometry_accepted']
    assert result['selection_scope']=='object' and result['fullframe_candidates'] is False
    inputs=json.loads((out/'INPUTS.json').read_text())
    tracks=json.loads((out/'TRACKS.json').read_text())
    assert inputs['track_profile']['selection_scope']=='object'
    assert tracks['profile']['selection_scope']=='object'
    assert all(o['label']=='foreground' for t in tracks['tracks'] for o in t['observations'])
    saved=json.loads((out/'group_00_model_0/SUPPORTED_POINTS.json').read_text())
    assert all(len(p['observations'])>=3 for p in saved)
    assert np.median([p['xyz'][2] for p in saved])==pytest.approx(10,abs=.03)
    assert (out/'viewer.html').is_file() and (out/'overlays/000001_candidates.png').is_file()
    assert result['counts']['observations']>result['counts']['tracks']


def test_cli_help_bootstrap():
    import subprocess,sys
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,'-B','-X','utf8',str(root/'scripts/expand_video_coverage.py'),'--help'],
        cwd=root,capture_output=True,text=True,timeout=10)
    assert p.returncode==0 and '--local-models' in p.stdout and '--selection-scope' in p.stdout


def test_object_bbox_occupancy_preserves_holes_and_fullframe_metric():
    mask=np.zeros((120,80),bool);mask[30:90,20:60]=True;mask[50:60,30:40]=False
    bounds=[20,30,60,90]
    points=[[20.5,30.5],[21.5,31.5],[35.5,55.5],[79.5,119.5]]
    local=mod.occupancy(points,mask,bounds=bounds)
    assert local['occupied_cells']==1
    assert local['available_cells']>24
    assert local['occupied_fraction']==pytest.approx(1/local['available_cells'])
    whole=mod.occupancy(points,mask)
    assert whole['available_cells']<local['available_cells']
