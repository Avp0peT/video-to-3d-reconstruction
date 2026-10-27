import numpy as np
import pytest
import sqlite3
import hashlib
from integrations.demo_pipeline.learned_identity_geometry_v2 import fit_query, read_keypoints_readonly
from integrations.demo_pipeline.learned_identity_geometry import effective_ray_angle_deg

pc=pytest.importorskip('pycolmap')


def test_keypoints_readonly_and_invalid_blob(tmp_path):
    path=tmp_path/'test.db'
    points=np.array([[.5,2.5],[5.5,8.5]],dtype='<f4')
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)')
        db.execute('INSERT INTO keypoints VALUES(1,2,2,?)',(points.tobytes(),))
        db.execute("INSERT INTO keypoints VALUES(2,2,2,X'00')")
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        assert np.array_equal(read_keypoints_readonly(db,1),points)
        with pytest.raises(sqlite3.OperationalError): db.execute('DELETE FROM keypoints')
        with pytest.raises(ValueError,match='blob size'): read_keypoints_readonly(db,2)
        with pytest.raises(ValueError,match='missing'): read_keypoints_readonly(db,3)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_nearly_opposite_rays_have_small_effective_angle():
    a=np.array([1.,0.,0.])
    b=np.array([-np.cos(np.deg2rad(.5)),np.sin(np.deg2rad(.5)),0.])
    assert np.rad2deg(np.arccos(a@b))>179.
    assert effective_ray_angle_deg(a,b)==pytest.approx(.5)
    assert effective_ray_angle_deg(a,b)<1.5


def observations(noise=0.):
    camera=pc.Camera.create_from_model_name(1,'SIMPLE_RADIAL',800,640,480)
    camera.params=np.array([800.,320.,240.,.03])
    xyz=np.array([.2,.1,5.])
    rng=np.random.default_rng(22)
    rows=[]
    for i in range(15):
        pose=pc.Rigid3d(pc.Rotation3d(),np.array([-i*.15,0.,0.]))
        xy=camera.img_from_cam(pose*xyz)+rng.normal(0,noise,2)
        rows.append(dict(image_id=i+1,feature_row=0,xy=xy.tolist(),camera=camera,pose=pose))
    return xyz,rows


def test_outlier_rejected_and_one_hypothesis():
    xyz,rows=observations()
    rows[2]['xy']=(np.asarray(rows[2]['xy'])+[100.,80.]).tolist()
    fit=fit_query(rows,pc,{5,10,15})
    assert fit['status']=='FITTED_NOT_IDENTITY_VERIFIED'
    assert np.allclose(fit['xyz'],xyz,atol=1e-6)
    assert 'reprojection_above_threshold' in fit['observations'][2]['rejection_reasons']
    assert fit['training_inliers']==11
    assert len(fit['observations'])==15


def test_heldout_cannot_change_fit():
    _,rows=observations(.2)
    before=fit_query(rows,pc,{5,10,15})
    for i in (4,9,14): rows[i]['xy']=[10000.,20000.]
    after=fit_query(rows,pc,{5,10,15})
    assert before['xyz']==after['xyz']
    assert after['heldout_error_px']['max']>1000
    assert all(r['split']=='heldout' for r in after['observations'] if r['image_id'] in {5,10,15})


def test_noisy_track_and_insufficient_training():
    xyz,rows=observations(.5)
    fit=fit_query(rows,pc,{5,10,15})
    assert fit['xyz'] is not None
    assert np.linalg.norm(np.asarray(fit['xyz'])-xyz)<.1
    assert fit['max_training_angle_deg']>=1.5
    short=fit_query(rows[:3],pc,{3})
    assert short['xyz'] is None
    assert short['rejection_reasons']==['fewer_than_three_training_views']
    assert len(short['observations'])==3


def test_no_baseline_rejected():
    _,rows=observations()
    for row in rows: row.update(pose=rows[0]['pose'],xy=rows[0]['xy'])
    fit=fit_query(rows,pc,{5,10,15})
    assert fit['xyz'] is None
