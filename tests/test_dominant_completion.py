import json
import numpy as np
import pytest
from integrations.mvs_surface.dominant_completion import select_dominant_envelope as select


def octahedron():
    v=np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    return v,f


def test_dominant_closed_component_with_separate_isolated_contamination():
    v,f=octahedron();vertices=np.vstack((v,[[9.,9.,9.]]));before=vertices.copy()
    r=select(vertices,f,np.array([[-3.,-3,-3],[10,10,10]]),np.array([.1,.1,.1]))
    assert r['counts']['retained_faces']==8 and r['counts']['retained_face_fraction']==1
    assert r['discarded_vertex_ids'].tolist()==[6]
    assert r['discarded_faces'].shape==(0,3) and r['discarded_vertices'].shape==(1,3)
    assert r['audit']['closed_oriented_nonzero_volume']
    np.testing.assert_array_equal(r['vertices'][r['faces']],vertices[f])
    np.testing.assert_array_equal(before,vertices)
    # Metadata is JSON-serializable once all top-level arrays are stripped.
    json.dumps({k:value for k,value in r.items() if not isinstance(value,np.ndarray)},allow_nan=False)


def test_secondary_open_face_preserved_in_diagnostic_layer_at90percent():
    # Ten-face closed bipyramid: five equatorial points and two poles.
    angles=np.arange(5)*2*np.pi/5
    v=np.vstack((np.column_stack((np.cos(angles),np.sin(angles),np.zeros(5))),[[0,0,1],[0,0,-1]]))
    f=np.array([[i,(i+1)%5,5] for i in range(5)]+[[(i+1)%5,i,6] for i in range(5)])
    vertices=np.vstack((v,[[3,0,0],[3,1,0],[3,0,1]]));faces=np.vstack((f,[[7,8,9]]))
    r=select(vertices,faces,[[-5.,-5,-5],[5,5,5]],.1)
    assert r['counts']['retained_faces']==10 and r['counts']['discarded_faces']==1
    assert r['discarded_face_ids'].tolist()==[10]
    np.testing.assert_array_equal(r['discarded_vertices'][r['discarded_faces']],vertices[faces[[10]]])
    assert r['counts']['original_components']==2


def test_no_dominant_component_rejected():
    v,f=octahedron()
    with pytest.raises(ValueError,match='90percent'):select(np.vstack((v,v+[4,0,0])),np.vstack((f,f+6)),[[-5.,-5,-5],[7,5,5]],.1)


def test_open_or_inconsistent_dominant_rejected():
    v,f=octahedron()
    with pytest.raises(ValueError,match='not closed'):select(v,f[:-1],[[-3.,-3,-3],[3,3,3]],.1)
    bad=f.copy();bad[0]=bad[0,::-1]
    with pytest.raises(ValueError,match='not closed'):select(v,bad,[[-3.,-3,-3],[3,3,3]],.1)


def test_margin_exact_one_cell_pass_less_than_cell_reject():
    v,f=octahedron()
    assert len(select(v,f,[[-1.1,-1.1,-1.1],[1.1,1.1,1.1]],.1)['faces'])==8
    with pytest.raises(ValueError,match='one-cell'):select(v,f,[[-1.05,-1.1,-1.1],[1.1,1.1,1.1]],.1)


def test_validation_and_cancel():
    v,f=octahedron()
    for spacing in (0,True,[.1,.1],[-1,.1,.1]):
        with pytest.raises(ValueError):select(v,f,[[-3.,-3,-3],[3,3,3]],spacing)
    with pytest.raises(ValueError):select(v,f,np.zeros((2,3)),.1)
    with pytest.raises(ValueError):select(v,np.empty((0,3),np.int64),[[-3.,-3,-3],[3,3,3]],.1)
    def stop():raise RuntimeError('cancel')
    with pytest.raises(RuntimeError,match='cancel'):select(v,f,[[-3.,-3,-3],[3,3,3]],.1,check=stop)
