from types import SimpleNamespace
import numpy as np
import pytest
from integrations.demo_pipeline import learned_surface_audit as a


def test_closed_tetra_vs_open_triangle_and_isolated_vertex():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[5,5,5]],float)
    f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    closed=a.topology(v,f)
    assert closed['boundary_edges']==0 and closed['nonmanifold_edges']==0
    assert closed['vertex_connected_components']==1 and closed['isolated_vertices']==1
    assert closed['watertight_acceptance'] is False
    opened=a.topology(v,f[:1])
    assert opened['boundary_edges']==3 and opened['isolated_vertices']==2


def test_disconnected_and_nonmanifold_are_distinguished():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1],[3,0,0],[4,0,0],[3,1,0]],float)
    f=np.array([[0,1,2],[1,0,3],[0,1,4],[5,6,7]])
    result=a.topology(v,f)
    assert result['nonmanifold_edges']==1
    assert result['vertex_connected_components']==2
    assert result['vertex_component_face_counts_descending']==[3,1]


def test_sample_is_deterministic_unique_and_budgeted():
    np.testing.assert_array_equal(a.sample_faces(10,3),[0,3,6])
    np.testing.assert_array_equal(a.sample_faces(2,3),[0,1])
    with pytest.raises(ValueError):a.sample_faces(10,0)


def test_projection_uses_world_to_camera_and_omits_behind_faces():
    pose=SimpleNamespace(rotation=SimpleNamespace(matrix=lambda:np.eye(3)),translation=np.array([0,0,2]))
    image=SimpleNamespace(cam_from_world=lambda:pose)
    camera=SimpleNamespace(width=20,height=20,img_from_cam=lambda q: q[:,:2]/q[:,2,None]*10+[10,10])
    v=np.array([[-.5,-.5,0],[.5,-.5,0],[0,.5,0],[-1,-1,-3],[1,-1,-3],[0,1,-3]])
    mask,diagnostic=a.projected_sample(v,np.array([[0,1,2],[3,4,5]]),camera,image)
    assert mask[10,10] and diagnostic['drawn_faces']==1
    assert diagnostic['skipped_near_camera_or_invalid_faces']==1
    assert 'sampled triangles' in diagnostic['raster']


def test_preview_reports_sampling_without_changing_geometry():
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],float);f=np.array([[0,1,2]])
    before=v.copy();im=a.preview(v,f,60)
    assert im.size==(800,800)
    np.testing.assert_array_equal(v,before)


def test_shared_single_vertex_is_not_edge_connected():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],float)
    result=a.topology(v,np.array([[0,1,2],[0,3,4]]))
    assert result['vertex_connected_components']==1
    assert result['face_edge_connected_components']==2
    assert result['face_edge_component_face_counts_descending']==[1,1]


def test_html_labels_full_triangle_scope_and_links(tmp_path):
    report={'visualization':{'triangle_scope':'all stored triangles'},'topology':{'face_edge_connected_components':2},'views':[{'name':f'frame_{i:06d}.png'} for i in range(6)]}
    a.write_html(tmp_path,report)
    text=(tmp_path/'index.html').read_text(encoding='utf-8')
    assert 'all stored triangles' in text and 'NOT QUALITY ACCEPTED' in text
    assert 'RESULT.json' in text and text.count('.rgb_projection.png')==6
    assert text.count('preview_yaw_')==6
