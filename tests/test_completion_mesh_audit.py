import copy
import numpy as np
import pytest
from integrations.mvs_surface.completion_mesh_audit import audit_closed_mesh,orient_components,split_by_face_labels


def tetra():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    return v,f


def test_closed_positive_tetra_and_centered_translation_stability():
    v,f=tetra();before=copy.deepcopy((v,f));r=audit_closed_mesh(v,f)
    assert r['closed'] and r['closed_oriented_nonzero_volume']
    assert r['counts']['two_face_edges']==6 and r['counts']['boundary_edges']==0
    assert r['components'][0]['signed_volume']==pytest.approx(1/6)
    assert r['components'][0]['volume_sign']=='POSITIVE'
    moved=audit_closed_mesh(v+[1e9,-1e9,1e9],f)
    assert moved['components'][0]['signed_volume']==pytest.approx(1/6)
    np.testing.assert_array_equal(v,before[0]);np.testing.assert_array_equal(f,before[1])
    assert r['self_intersections']=='NOT_EVALUATED' and not r['geometry_accepted']


def test_plane_open_cannot_be_closed_volume():
    v,f=tetra();r=audit_closed_mesh(v,f[:1])
    assert not r['closed'] and r['counts']['boundary_edges']==3
    assert r['counts']['isolated_vertices']==1
    assert not r['closed_oriented_nonzero_volume']


def test_duplicate_and_nonmanifold_not_accepted():
    v,f=tetra();r=audit_closed_mesh(v,np.concatenate((f,f[:1])))
    assert not r['closed'] and r['counts']['nonmanifold_edges']==3
    assert r['counts']['duplicate_faces']==1 and r['counts']['duplicate_group_faces']==2


def test_inconsistent_winding_not_silently_reoriented():
    v,f=tetra();f[0]=f[0,::-1];r=orient_components(v,f)
    assert r['before']['closed']
    assert not r['before']['edge_orientation_consistent']
    assert r['before']['counts']['inconsistent_two_face_edges']==3
    assert not len(r['flipped_component_ids'])
    np.testing.assert_array_equal(r['faces'],f)


def test_negative_component_flip_retains_before_evidence():
    v,f=tetra();negative=f[:,[0,2,1]];r=orient_components(v,negative)
    assert r['before']['components'][0]['volume_sign']=='NEGATIVE'
    assert r['after']['components'][0]['volume_sign']=='POSITIVE'
    assert r['flipped_component_ids'].tolist()==[0]
    np.testing.assert_array_equal(r['vertices'],v);np.testing.assert_array_equal(negative,f[:,[0,2,1]])


def test_two_indexed_bodies_and_isolated_vertex():
    v,f=tetra();vertices=np.concatenate((v,v+[3,0,0],[[99,99,99]]));faces=np.concatenate((f,f+4))
    r=audit_closed_mesh(vertices,faces)
    assert r['counts']['connected_components']==2 and r['counts']['isolated_vertices']==1
    assert r['closed'] and r['closed_oriented_nonzero_volume']
    split=split_by_face_labels(vertices,faces,np.array(['OBSERVED']*4+['INFERRED']*4))
    assert split['face_identity_coverage_exact'] and len(split['layers'])==2
    rows=[]
    for layer in split['layers']:
        ids=layer['original_face_ids'];rows.extend(ids)
        np.testing.assert_array_equal(layer['vertices'][layer['faces']],vertices[faces[ids]])
    assert sorted(rows)==list(range(8))


def test_layer_boundary_vertices_copy_without_missing_or_extra_faces():
    v,f=tetra();r=split_by_face_labels(v,f,np.array([0,1,0,1]))
    assert sum(len(layer['faces']) for layer in r['layers'])==4
    assert r['face_layer'].tolist()==[0,1,0,1]
    assert set(r['layers'][0]['original_vertex_ids'])&set(r['layers'][1]['original_vertex_ids'])
    assert r['face_identity_coverage_exact']


def test_degenerate_empty_and_exact_unwelded_coincident_components():
    v,f=tetra();degenerate=np.array([[0,0,1]])
    assert audit_closed_mesh(v,degenerate)['counts']['degenerate_faces']==1
    r=audit_closed_mesh(v,np.empty((0,3),np.int64))
    assert not r['closed'] and r['counts']['isolated_vertices']==4
    assert split_by_face_labels(v,np.empty((0,3),np.int64),np.empty(0,np.int64))['face_identity_coverage_exact']
    assert audit_closed_mesh(np.vstack((v,v)),np.vstack((f,f+4)))['counts']['connected_components']==2


def test_validation():
    v,f=tetra()
    with pytest.raises(ValueError):audit_closed_mesh(v,f.astype(float))
    with pytest.raises(ValueError):audit_closed_mesh(v,np.array([[0,1,4]]))
    bad=v.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):audit_closed_mesh(bad,f)
    with pytest.raises(ValueError):split_by_face_labels(v,f,np.array([0,1]))
    with pytest.raises(ValueError):split_by_face_labels(v,f,np.array([0.,1.,0.,1.]))
