import copy
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from integrations.demo_pipeline import learned_mvs_masks as m
from integrations.mvs_masks.mask_geometry import warp_binary_mask

class Camera:
    def __init__(self,model='SIMPLE_RADIAL',params=(4.,3.,2.,-.1),size=(6,4)):
        self.model_name=model;self.params=np.array(params);self.width,self.height=size
    def verify_params(self):return True
    def calibration_matrix(self):
        if self.model_name=='PINHOLE':fx,fy,cx,cy=self.params
        else:fx,cx,cy,_=self.params;fy=fx
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    def cam_from_img(self,xy):
        K=self.calibration_matrix();return (xy-K[:2,2])/np.diag(K)[:2]
    def img_from_cam(self,xyz):
        xy=xyz[:,:2]/xyz[:,2,None]
        f,cx,cy,k=self.params
        return f*(1+k*(xy*xy).sum(axis=1))[:,None]*xy+[cx,cy]

def ledger():
    images=[dict(image_id=i,name=f'frame_{i:06d}.png',input_index=i,decode_index=100+i) for i in range(54)]
    native=[dict(database_image_id=i,image_name=f'frame_{i:06d}.png',input_index=i,decode_index=100+i) for i in range(54)]
    masks=[dict(decode_index=100+i,status='OK',current_observed=True) for i in range(54)]
    return images,native,masks

def test_identity_join_uses_names_ids_decode_not_order():
    a,b,c=ledger();c.reverse();b.reverse()
    assert m.join_identity(a,b,c)['frame_000005.png']['decode_index']==105

@pytest.mark.parametrize('kind',['duplicate','missing','no_match','decode_drift'])
def test_identity_rejections(kind):
    a,b,c=ledger()
    if kind=='duplicate':c.append(copy.copy(c[0]))
    if kind=='missing':c.pop()
    if kind=='no_match':c[0]['status']='NO_MATCH'
    if kind=='decode_drift':a[0]['decode_index']+=500
    with pytest.raises(ValueError):m.join_identity(a,b,c)

def test_radial_inverse_warp_matches_explicit_floor():
    source=Camera();target=Camera('PINHOLE',(4.,4.,3.,2.))
    mask=np.zeros((4,6),np.uint8);mask[1:3,2:4]=255
    result=warp_binary_mask(mask,source,target)
    yy,xx=np.mgrid[:4,:6]
    rays=target.cam_from_img(np.column_stack((xx.ravel()+.5,yy.ravel()+.5)))
    uv=source.img_from_cam(np.column_stack((rays,np.ones(len(rays)))))
    ij=np.floor(uv).astype(int)
    np.testing.assert_array_equal(result.mask,mask[ij[:,1],ij[:,0]].reshape(4,6))
    assert (result.valid==255).all()

def test_outside_source_unknown_not_foreground():
    source=Camera();target=Camera('PINHOLE',(4.,4.,30.,2.))
    result=warp_binary_mask(np.full((4,6),255,np.uint8),source,target)
    assert result.counts['unknown']>0
    assert not ((result.mask>0)&(result.valid==0)).any()

@pytest.mark.parametrize('bad',['rgb','labels','empty'])
def test_mask_rejects_implicit_conversion(tmp_path,bad):
    a=np.full((4,6),255,np.uint8)
    if bad=='rgb':a=np.repeat(a[:,:,None],3,axis=2)
    if bad=='labels':a[0,0]=127
    if bad=='empty':a[:]=0
    p=tmp_path/'mask.png';Image.fromarray(a).save(p)
    with pytest.raises(ValueError):m.binary_mask(p,(6,4))

def test_pose_rejects_translation_change():
    def image(t):
        return SimpleNamespace(cam_from_world=lambda:SimpleNamespace(rotation=SimpleNamespace(matrix=lambda:np.eye(3)),translation=np.array(t)),projection_center=lambda:-np.array(t))
    m.check_pose(image([0,0,0]),image([0,0,0]))
    with pytest.raises(ValueError):m.check_pose(image([0,0,0]),image([1,0,0]))
