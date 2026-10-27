from types import SimpleNamespace
import numpy as np
import pytest
from integrations.mvs_surface.mask_consumption_probe import depth_counts,verdict,resize_nearest,authenticate_depth


def rows():
    out={a:dict(reclaimed=True,execution_ok=True,counts=dict(available=True,positive_inside=0,positive_outside=60,positive_total=60)) for a in ('DISABLED','RECT','BLACK')}
    out['DISABLED']['counts']['positive_inside']=60
    out['BLACK']['counts']['positive_total']=0
    return out


def test_three_directional_verdicts():
    r=rows();assert verdict(r).startswith('REFERENCE_MASK_CONSUMPTION_VERIFIED')
    r['RECT']['counts']['positive_inside']=1;assert verdict(r)=='FAIL'
    r=rows();r['BLACK']['counts']['positive_total']=1;assert verdict(r)=='FAIL'
    r=rows();r['DISABLED']['counts']['positive_inside']=49;assert verdict(r)=='UNKNOWN'


@pytest.mark.parametrize('kind',['missing','reclaim','execution','outside'])
def test_unknown_not_pass_from_missing_or_exit(kind):
    r=rows()
    if kind=='missing':r['BLACK']['counts']['available']=False
    if kind=='reclaim':r['BLACK']['reclaimed']=False
    if kind=='execution':r['BLACK']['execution_ok']=False
    if kind=='outside':r['RECT']['counts']['positive_outside']=49
    assert verdict(r)=='UNKNOWN'


def test_resize_floor_and_eroded_interior():
    a=np.zeros((20,20),bool);a[4:16,4:16]=True
    b=resize_nearest(a,(10,10));assert b[2:8,2:8].all() and b.sum()==36
    d=np.ones((10,10));r=depth_counts(d,a)
    assert r['inside_pixels']==4 and r['positive_inside']==4
    assert depth_counts(None,a)['positive_inside'] is None


def identity():
    c=dict(name='x/image_00000080.png',global_id=80,width=20,height=20,K=np.array([[10.,0,9.5],[0,10,9.5],[0,0,1]]),R=np.eye(3),C=np.zeros(3))
    d=SimpleNamespace(depth_size=(10,10),image_size=(10,10),view_ids=[80],image_name_utf8='other/image_00000080.png',
        K=np.array([[5.,0,4.5],[0,5,4.5],[0,0,1]]),R=np.eye(3),C=np.zeros(3))
    return d,c


def test_resized_identity_and_denials():
    d,c=identity();authenticate_depth(d,c)
    for key,value in [('view_ids',[7]),('image_name_utf8','image_00000007.png'),('K',np.eye(3)),('C',np.ones(3))]:
        d,c=identity();setattr(d,key,value)
        with pytest.raises(ValueError):authenticate_depth(d,c)
