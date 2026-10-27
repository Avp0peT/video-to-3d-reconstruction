from copy import deepcopy
import numpy as np
import pytest
from integrations.mvs_surface.fused_mesh_review import review_mesh


@pytest.mark.parametrize('count',[17,64,67,128])
def test_extended_mesh_view_votes_and_centroid_state(count):
    base=views();selected=[]
    for index in range(count):
        v=deepcopy(base[index%3]);v['view_id']=index;selected.append(v)
    vertices,faces=square();r=review_mesh(vertices,faces,selected)
    assert r['centroid_states'].shape==(2,count)
    assert r['face_foreground_votes'].dtype==np.uint8
    assert (r['face_foreground_votes']==count).all()
    assert not r['face_known_background_votes'].any()
    assert r['topology']['faces']==2 and len(r['per_view'])==count


def test_129_mesh_review_views_rejected():
    vertices,faces=square()
    with pytest.raises(ValueError,match='128'):review_mesh(vertices,faces,[views()[0]]*129)


def views():
    out=[]
    for i,c in enumerate((-.1,0.,.1)):
        out.append(dict(view_id=i,image=np.full((32,32,3),100,np.uint8),mask=np.ones((32,32),bool),validity=np.ones((32,32),bool),depth=np.full((32,32),2.,np.float32),depth_available=True,K=np.array([[30.,0,16],[0,30.,16],[0,0,1]]),R=np.eye(3),C=np.array([c,0,0])))
    return out


def square():
    return np.array([[-.2,-.2,2.],[.2,-.2,2.],[.2,.2,2.],[-.2,.2,2.]]),np.array([[0,1,2],[0,2,3]],np.int64)


def test_shared_square_connected_boundary_and_readonly():
    vertices,faces=square();v=views();before=deepcopy((vertices,faces,v))
    r=review_mesh(vertices,faces,v)
    assert r['topology']['connected_components']==1
    assert r['topology']['boundary_edges']==4
    assert r['topology']['nonmanifold_edges']==0
    assert r['topology']['degenerate_faces']==0
    assert (r['face_foreground_votes']==3).all()
    assert (r['face_unknown_votes']==0).all()
    assert r['centroid_classification'].tolist()==[1,1]
    assert not r['mesh_modified'] and not r['geometry_accepted']
    np.testing.assert_array_equal(vertices,before[0]);np.testing.assert_array_equal(faces,before[1])
    for a,b in zip(v,before[2]):
        for key in ('image','mask','validity','depth','K','R','C'):np.testing.assert_array_equal(a[key],b[key])
    assert 'not an entire-face' in r['sampling']


def test_disjoint_indices_not_coordinate_welded():
    vertices,faces=square()
    vertices=np.vstack((vertices,vertices[:3]))
    faces=np.vstack((faces,[[4,5,6]]))
    r=review_mesh(vertices,faces,views())
    assert r['topology']['connected_components']==2
    assert r['face_component'][0]==r['face_component'][1]
    assert r['face_component'][2]!=r['face_component'][0]
    assert r['topology']['boundary_edges']==7


def test_known_background_and_unknown_are_separate_diagnostics():
    vertices,faces=square();v=views()
    v[0]['mask'][:]=False;v[1]['validity'][:]=False
    r=review_mesh(vertices,faces,v)
    assert r['face_known_background_votes'].tolist()==[1,1]
    assert r['face_unknown_votes'].tolist()==[1,1]
    assert r['face_foreground_votes'].tolist()==[1,1]
    assert r['per_view'][0]['background_samples']==8
    assert r['per_view'][1]['unknown_samples']==8
    assert r['per_view'][2]['foreground_samples']==8
    assert r['centroid_classification'].tolist()==[2,2]
    assert 'not proof' in r['centroid_semantics']


def test_out_of_view_is_unknown_not_background():
    vertices,faces=square();vertices[:,0]+=20
    r=review_mesh(vertices,faces,views())
    assert r['face_known_background_votes'].tolist()==[0,0]
    assert r['face_unknown_votes'].tolist()==[3,3]
    assert all(x['in_image_samples']==0 for x in r['per_view'])


def test_degenerate_duplicates_and_nonmanifold_are_reported_without_removal():
    vertices,faces=square();faces=np.vstack((faces,faces[0],[[0,0,1]]))
    r=review_mesh(vertices,faces,views())
    assert r['topology']['faces']==4
    assert r['topology']['duplicate_faces']==1
    assert r['topology']['degenerate_faces']==1
    assert r['topology']['nonmanifold_edges']>0
    assert len(r['face_component'])==4


def test_input_errors_and_empty_mesh():
    vertices,faces=square()
    with pytest.raises(ValueError):review_mesh(vertices,np.array([[0,1,99]]),views())
    with pytest.raises(ValueError):review_mesh(vertices,np.array([[-1,0,1]]),views())
    with pytest.raises(ValueError):review_mesh(vertices,faces.astype(float),views())
    with pytest.raises(ValueError):review_mesh(np.full((1,3),np.nan),np.empty((0,3),int),views())
    v=views();del v[0]['validity']
    with pytest.raises(ValueError):review_mesh(vertices,faces,v)
    r=review_mesh(np.empty((0,3)),np.empty((0,3),np.int64),views())
    assert r['topology']['connected_components']==0
    assert r['centroid_states'].shape==(0,3)
