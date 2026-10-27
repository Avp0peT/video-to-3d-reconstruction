"""TEST_DATA: measured textures, no model fitting or mask inference."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.video_masks import region_tracks as rt
from integrations.video_masks.density_sampling import default_policy, seed_with_density
from integrations.video_masks.point_density import assess_density
from integrations.video_masks.density_report import observed_support


@pytest.fixture(scope='module')
def cv():
    result=rt.load_cv([])
    if result is None:pytest.skip('bound CPU OpenCV unavailable')
    result.setNumThreads(2)
    return result


def test_blank_regions_remain_deficient_and_no_fabrication(cv):
    gray=np.full((128,128),110,np.uint8);mask=np.full_like(gray,255)
    births,report=seed_with_density(gray,mask,[],cv)
    assert not births and report['status']=='BELOW_DENSITY_TARGET'
    assert report['met_area_fraction']==0
    assert report['attempted_quality_levels']==[.01,.003]


def test_object_relative_refill_measured_and_immutable(cv):
    rng=np.random.default_rng(13)
    gray=np.zeros((320,480),np.uint8);mask=np.zeros_like(gray)
    gray[40:280,180:420]=rng.integers(20,220,(240,240),dtype=np.uint8)
    mask[40:280,180:420]=255
    originals=gray.copy(),mask.copy()
    births,report=seed_with_density(gray,mask,[],cv)
    assert births and len(births)<=600
    assert report['bbox_xyxy']==[180,40,420,280]
    assert report['met_area_fraction']>.8
    assert all(rt.support(mask,p,cv) for p in births)
    assert report['geometry_accepted'] is False
    points=np.asarray(births)
    distance=np.sum((points[:,None]-points[None,:])**2,axis=2)
    np.fill_diagonal(distance,np.inf)
    assert distance.min()>=49
    assert np.array_equal(gray,originals[0]) and np.array_equal(mask,originals[1])


def test_frozen_policy_rejects_silent_matching_tuning(cv):
    policy=default_policy();policy['quality_levels']=[.00001]
    with pytest.raises(ValueError,match='policy'):
        seed_with_density(np.zeros((20,20),np.uint8),np.ones((20,20),np.uint8),[],cv,policy)


def test_new_policy_refills_intermediate_frames_without_reusing_identity(tmp_path,cv,monkeypatch):
    rng=np.random.default_rng(33);gray=rng.integers(20,220,(128,128),dtype=np.uint8)
    rows=[]
    for n in range(3):
        row=dict(decode_index=n,width=128,height=128)
        if n!=1:row['input_index']=n
        for kind,array in (('image',gray),('mask',np.full_like(gray,255))):
            path=tmp_path/f'{kind}{n}.png';Image.fromarray(array).save(path)
            row[kind+'_path']=str(path);row[kind+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    # Force real identity loss. Refilling must create fresh IDs on the non-endpoint too.
    monkeypatch.setattr(rt,'step',lambda a,b,c,d,p,cv:(np.asarray(p),['forward_lk']*len(p),[None]*len(p)))
    plan=tmp_path/'PLAN.json';plan.write_text(json.dumps(dict(schema='glove-region-plan-v1',timeline=rows,density_policy=default_policy())))
    result=rt.run(plan,tmp_path/'output')
    assert result['timeline_stats'][1]['births']
    assert set(result['timeline_stats'][0]['births']).isdisjoint(result['timeline_stats'][1]['births'])
    assert all('candidate_density' in row for row in result['timeline_stats'])
    assert {'density_sampling.py','point_density.py'} <= {Path(r['path']).name for r in result['source_bindings']}
    assert all(not track['observations'] or all(o['input_index']!=1 for o in track['observations']) for track in result['tracks'])


class Pose:
    def __mul__(self, xyz):return xyz


def test_only_observed_positive_reciprocal_low_error_support_counts():
    points={};observations=[]
    for row,(xyz,xy) in enumerate([([2,2,2],[1,1]),([2,2,-2],[1,1]),([2,2,2],[99,99])]):
        points[row]=SimpleNamespace(xyz=np.array(xyz),track=SimpleNamespace(elements=[
            SimpleNamespace(image_id=1,point2D_idx=row),SimpleNamespace(image_id=2,point2D_idx=row)]))
        observations.append(SimpleNamespace(has_point3D=lambda:True,point3D_id=row,xy=np.array(xy)))
    image=SimpleNamespace(image_id=1,camera_id=1,points2D=observations,cam_from_world=lambda:Pose())
    model=SimpleNamespace(points3D=points,cameras={1:SimpleNamespace(img_from_cam=lambda p:p[:2]/p[2])})
    result,rejected=observed_support(model,image)
    assert result==[dict(point3D_id=0,point2D_idx=0,xy=[.5,.5])]
    assert rejected==dict(depth=1,reprojection=1)
    for bad_projection in (None, [], [0,0,0], [float('nan'),0]):
        model.cameras[1].img_from_cam=lambda p:bad_projection
        result,rejected=observed_support(model,image)
        assert not result and rejected==dict(depth=1,reprojection=2)


def test_public_density_command_routes_to_export_without_native_pipeline(tmp_path,monkeypatch):
    from scripts import run_photo_demo as cli
    from integrations.video_masks import density_report
    calls=[]
    def audit(source,out):
        calls.append((source,out))
        return dict(status='BELOW_DENSITY_TARGET',frame_count=2,mean_frame_met_area_fraction={})
    monkeypatch.setattr(density_report,'audit_export',audit)
    monkeypatch.setattr(cli,'run_pipeline',lambda *a,**k:pytest.fail('model execution'))
    assert cli.main(['audit-point-density','--run',str(tmp_path/'surface'),'--output',str(tmp_path/'new')])==0
    assert calls==[(tmp_path/'surface/export',tmp_path/'new')]
