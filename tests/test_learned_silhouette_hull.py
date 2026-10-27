import numpy as np
from integrations.demo_pipeline import learned_silhouette_hull as h


def sphere_view(tx=0):
    size=(64,64)
    params=np.array([50.,32.,32.,0.])
    yy,xx=np.mgrid[:64,:64]
    rays=np.stack(((xx+.5-32)/50,(yy+.5-32)/50,np.ones_like(xx)),axis=-1)
    center=np.array([tx,0.,4.])
    dot=np.sum(rays*center,axis=-1)
    discriminant=dot**2-np.sum(rays*rays,axis=-1)*(center@center-1.)
    return dict(R=np.eye(3),t=np.array([tx,0,0]),params=params,size=size,mask=discriminant>=0)


def test_known_sphere_points_and_center_footprint():
    views=[sphere_view(t) for t in (-.4,0,.4)]
    points=np.array([[0,0,4],[.2,.1,4], [2,0,4]])
    strict,pos,observed,curves=h.support(points,views)
    assert strict.tolist()==[True,True,False]
    assert observed.tolist()==[3,3,3]
    assert curves['1.0']==2
    rendered,skipped=h.footprint(points[:1],np.array([.2,.2,.2]),views[1])
    assert rendered.sum()>1 and skipped==0
    assert rendered[32,32]


def test_occluded_mask_removes_real_sphere_center_and_remains_diagnostic():
    views=[sphere_view() for _ in range(3)]
    views[0]['mask'][30:35,30:35]=False
    strict,pos,observed,_=h.support(np.array([[0,0,4.]]),views)
    assert not strict[0] and pos[0]==2 and observed[0]==3
    assert h.STATUS=='OCCLUSION_UNMODELLED'


def test_outside_image_unknown_and_floor_convention():
    views=[sphere_view() for _ in range(3)]
    extra=sphere_view();extra['t']=np.array([100.,0,0]);extra['mask'][:]=False
    strict,pos,observed,_=h.support(np.array([[0,0,4.]]),views+[extra])
    assert strict[0] and pos[0]==observed[0]==3
    # COLMAP x=.9 belongs to pixel 0, not rounded pixel 1.
    v=sphere_view();v['params']=np.array([1.,0.,0.,0.]);v['mask'][:]=False;v['mask'][0,0]=True
    assert h.support(np.array([[.9,.9,1.]]),[v,v,v])[0][0]
    assert not h.project(np.array([[0,0,-1.]]),v['R'],v['t'],v['params'],v['size'])[1][0]


def test_domain_uses_frozen_sparse_extent():
    pts,lower,step=h.grid_domain(np.array([[-1,-2,-3],[1,2,3]]))
    np.testing.assert_allclose(lower,[-1.5,-3,-4.5])
    np.testing.assert_allclose(step*64,[3,6,9])
    assert pts.shape==(64**3,3)
