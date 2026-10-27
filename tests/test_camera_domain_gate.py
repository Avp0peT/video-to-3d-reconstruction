from types import SimpleNamespace as NS
import numpy as np
from integrations.demo_pipeline.camera_domain_gate import audit_reconstruction, audit_camera_domain
from kozlov_lab.reconstruction import profiles, colmap_adapter as adapter

class Camera:
    model_name='SIMPLE_PINHOLE'; width=100; height=80
    params=np.array([100.,50.,40.])
    def focal_length_idxs(self): return [0]
    def cam_from_img(self,p): return (p-self.params[1:])/self.params[0]
    def img_from_cam(self,p): return p[:2]*self.params[0]+self.params[1:]

def rec(camera=None):
    return NS(cameras={1:camera or Camera()},images={1:NS(camera_id=1,points2D=[NS(xy=np.array([2.,3.])),NS(xy=np.array([80.,70.]))])})

def test_all_rows_and_corners_checked():
    r=audit_reconstruction(rec())
    assert r['status']=='CAMERA_DOMAIN_VALID' and r['calibration_accuracy']=='NOT_EVALUATED'
    assert r['cameras'][0]['observation_count']==2 and r['cameras'][0]['grid_count']==81

def test_nonfinite_untriangulated_observation_rejected():
    class Broken(Camera):
        def cam_from_img(self,p):
            return np.array([np.nan,np.nan]) if p[0]==2 else super().cam_from_img(p)
    r=audit_reconstruction(rec(Broken()))
    assert r['status']=='CAMERA_DOMAIN_REJECTED'
    assert r['cameras'][0]['failures'][0]['location']['point2D_idx']==0

def test_corner_failure_and_roundtrip_rejected():
    class Broken(Camera):
        def img_from_cam(self,p):return super().img_from_cam(p)+[.01,0.]
    assert audit_reconstruction(rec(Broken()))['status']=='CAMERA_DOMAIN_REJECTED'

def test_positive_focal_required():
    c=Camera();c.params=np.array([-1.,50.,40.])
    assert audit_reconstruction(rec(c))['status']=='CAMERA_DOMAIN_REJECTED'

def test_radial_nonmonotonic_branch_rejected():
    class Radial(Camera):
        model_name='SIMPLE_RADIAL';params=np.array([100.,50.,40.,-1.])
        def cam_from_img(self,p):return np.array([1.,0.])
    r=audit_reconstruction(rec(Radial()))
    assert any('nonmonotone' in f['reason'] for f in r['cameras'][0]['failures'])

def test_source_hashes_all_files_and_detects_drift(tmp_path):
    (tmp_path/'cameras.bin').write_bytes(b'fixture');(tmp_path/'extra.txt').write_text('source')
    def load(path):
        (tmp_path/'extra.txt').write_text('changed')
        return rec()
    r=audit_camera_domain(tmp_path,pycolmap_module=NS(Reconstruction=load,__version__='fake'))
    assert r['status']=='SOURCE_DRIFT_REFUSED' and not r['source_unchanged']
    assert set(r['source_bindings_before'])=={'cameras.bin','extra.txt'}

def test_new_profile_is_explicit_pinhole_and_old_options_preserved():
    old=adapter.effective_options(profiles.PROFILE_CAD_OBJECT_V1)
    new=adapter.effective_options(profiles.PROFILE_CAD_OBJECT_PINHOLE_V1)
    assert old['camera_model']=='SIMPLE_RADIAL' and old['ba_refine_extra_params'] is True
    assert new['camera_model']=='SIMPLE_PINHOLE' and new['ba_refine_extra_params'] is False
    assert 'radial_param_init' not in new
    for k,v in old.items():
        if k not in ('profile_id','camera_model','ba_refine_extra_params','radial_param_init'):
            assert new[k]==v
    assert profiles.camera_mode(profiles.PROFILE_CAD_OBJECT_PINHOLE_V1)=='SINGLE'

def test_native_pinhole_and_hand_radial_counterexample():
    import pytest
    pc=pytest.importorskip('pycolmap')
    good=pc.Camera(model='SIMPLE_PINHOLE',width=1920,height=1080,params=[2096.6,960.,540.])
    assert audit_reconstruction(rec(good))['status']=='CAMERA_DOMAIN_VALID'
    bad=pc.Camera(model='SIMPLE_RADIAL',width=1920,height=1080,params=[2096.6,960.,540.,-2.9578])
    r=audit_reconstruction(rec(bad))
    assert r['status']=='CAMERA_DOMAIN_REJECTED'
    assert r['cameras'][0]['invalid_count']>0
