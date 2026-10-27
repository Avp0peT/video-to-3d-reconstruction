from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from PIL import Image

from integrations.video_masks import face_mask_adapter as a


def test_partition_preserves_coordinates_winding_and_identity(tmp_path):
    v = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 2.]])
    f = np.array([[0, 1, 2], [3, 2, 1]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30]], dtype=np.uint8)
    path = tmp_path/'part.ply'
    record = a.write_partition(path, v, f, colors, [1])
    assert record['exact_readback'] and record['faces'] == 1
    mapping = np.load(path.with_suffix('.mapping.npz'))
    mesh = a.read_ply(path, **a.LIMITS)
    assert mapping['source_face_ids'].tolist() == [1]
    assert np.array_equal(mapping['source_vertex_ids'][np.array(mesh['faces'])], f[[1]])
    assert np.array_equal(np.array(mesh['vertices']), v[mapping['source_vertex_ids']])
    assert np.array_equal(f, [[0, 1, 2], [3, 2, 1]])


def test_empty_candidate_is_explicit_not_placeholder_faces(tmp_path):
    record = a.write_partition(tmp_path/'empty.ply', np.array([[0., 0., 1.]]),
                               np.empty((0, 3), dtype=int), None, [])
    assert record['vertices'] == record['faces'] == 0
    assert a.read_ply(tmp_path/'empty.ply', **a.LIMITS)['faces'] == []


def test_partition_duplicate_ids_refused(tmp_path):
    with pytest.raises(ValueError, match='duplicate'):
        a.write_partition(tmp_path/'bad.ply', np.eye(3), np.array([[0, 1, 2]]), None, [0, 0])
    assert not (tmp_path/'bad.ply').exists()


def camera_pair():
    image = dict(name='undistorted/images/a.png', global_id=10, width=32, height=32,
                 K=np.eye(3).tolist(), R=np.eye(3).tolist(), C=[0, 0, 0])
    depth = SimpleNamespace(image_name_utf8='a.png', view_ids=(10,), image_size=(32, 32),
        depth_size=(32, 32), K=np.eye(3), R=np.eye(3), C=np.zeros(3))
    return image, depth


@pytest.mark.parametrize('change', ['shift', 'identity', 'scale', 'pose'])
def test_depth_binding_rejects_wrong_domain(change):
    image, depth = camera_pair()
    a.verify_depth(depth, image)
    if change == 'shift':
        depth.K[0, 2] += .5
    elif change == 'identity':
        depth.view_ids = (11,)
    elif change == 'scale':
        depth.depth_size = (16, 16)
    else:
        depth.C[0] = 1
    with pytest.raises(ValueError):
        a.verify_depth(depth, image)


def test_changed_mask_refused_even_if_pixels_are_same(tmp_path):
    p = tmp_path/'mask.png'
    Image.new('L', (4, 4), 255).save(p, compress_level=0)
    reference = {**a.bind(p), 'width': 4, 'height': 4}
    Image.new('L', (4, 4), 255).save(p, compress_level=9)
    with pytest.raises(ValueError, match='hash/size'):
        a._binary(reference, tmp_path, (4, 4), [])


def test_complete_adapter_never_turns_no_conflict_into_acceptance(tmp_path, monkeypatch):
    repo = a.ROOT
    monkeypatch.setattr(a, 'ROOT', tmp_path)
    for rel in ('integrations/openmvs_depth/dmap.py', 'integrations/mvs_surface/scene.py',
                'integrations/viewer/ply_data.py', 'integrations/viewer/export.py'):
        target = tmp_path/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo/rel).read_bytes())
    source = tmp_path/'outputs/source'
    for child in ('surface/work', 'bridge', 'masks'):
        (source/child).mkdir(parents=True, exist_ok=True)
    v = np.array([[2., 2., 1.], [28., 2., 1.], [2., 28., 1.]])
    f = np.array([[0, 1, 2]])
    raw = source/'surface/work/raw.ply'
    mesh = a.write_partition(raw, v, f, None, [0])
    a._save(source/'surface/result.json', dict(status='SURFACE_GENERATED_NOT_ACCEPTED',
        stages=[dict(outputs=dict(geometry=[dict(**mesh, role='pre_cleanup_surface')]))]))
    scene_path = source/'bridge/scene.mvs'; scene_path.write_bytes(b'synthetic parser fixture')
    image, d = camera_pair()
    refs = []
    for label in ('mask', 'validity'):
        p = source/'masks'/(label+'.png'); Image.new('L', (32, 32), 255).save(p)
        refs.append(dict(**a.bind(p), width=32, height=32))
    a._save(source/'masks/RESULT.json', dict(coverage_status='COMPLETE', adapter_source_chain='VERIFIED',
        scene_binding=a.bind(scene_path), frames=[dict(exported_name='a.png', input_index=0,
        target_image_id=10, status='OK', readback_verified=True, mask=refs[0], validity=refs[1])]))
    monkeypatch.setattr(a, 'read_scene', lambda p: dict(transform=np.eye(4).tolist(), images=[image]))
    depth_path = source/'surface/work/depth0010.dmap'; depth_path.write_bytes(b'synthetic depth fixture')
    bound = a.bind(depth_path); d.depth = np.ones((32, 32), dtype=np.float32)
    d.sha256 = bound['sha256']; d.file_bytes = bound['bytes']
    monkeypatch.setattr(a, 'read_dmap', lambda p: d)
    out = tmp_path/'outputs/derived'
    result = a.run(source, out)
    assert result['status'] == 'CANDIDATE_SURFACE_NOT_ACCEPTED', result
    assert result['geometry_accepted'] is False and result['source_unchanged'] is True
    assert result['counts']['quarantined_faces'] == 0
    assert result['counts']['retained_faces'] == 1
    assert result['schema'] == 'face-mask-quarantine-v3'
    assert result['counts']['v1_quarantined'] == result['counts']['gap_quarantined'] == 0
    assert result['counts']['gap_only_quarantined'] == 0
    assert (out/'per_view/view_10.npz').is_file()
    assert (out/'gap_per_view/view_10.npz').is_file()
    assert (out/'gap_per_view/view_10_gap.png').is_file()
    assert (out/'edge_per_view/view_10.npz').is_file()
    with np.load(out/'edge_per_view/view_10.npz') as edge:
        assert edge['edge_eligible'].dtype==np.bool_
        assert edge['core_protected'].shape==(1,)
    assert 'gap_policy' in a._read(out/'PLAN.json')
    assert 'edge_policy' in a._read(out/'PLAN.json')
    assert result['counts']['previous_v2_quarantined']==result['counts']['restored_from_v2']==0
    assert result['subset_of_previous_v2_assert'] is True
    assert result['non_edge_faces_unchanged_assert'] is True
    assert result['core_faces_retained_assert'] is True
    assert 'gap_counts' in a._read(out/'PROGRESS.json')['last_view']
    assert (out/'comparison_viewer.html').is_file()
    assert (out/'retained_viewer.html').is_file()
    retained_spec=a._read(out/'retained_scenes.json')
    assert len(retained_spec['scenes'])==1
    assert retained_spec['scenes'][0]['ply']==str(out/'retained.ply')
    assert (out/'FACE_LEDGER.csv').read_text().count('RETAINED_NOT_ACCEPTED') == 1
    assert (out/'original.ply').read_bytes() == raw.read_bytes()
    assert a._read(out/'EVIDENCE.json')['quality_status'] == 'NOT_EVALUATED'


def test_output_cannot_overwrite_protected_mesh_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'ROOT', tmp_path)
    source = tmp_path/'outputs/source'; (source/'surface').mkdir(parents=True)
    with pytest.raises(ValueError, match='protected'):
        a.run(source, source/'surface/new')
    assert not (source/'surface/new').exists()


def test_independent_rule_quorums_never_mix_one_vote_each():
    vertices=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    faces=np.array([[0,1,2]])
    cameras={'left':[-10.,0.,0.],'right':[10.,0.,0.]}
    def row(key,status):
        return dict(view_id=key,status=np.array([status],np.uint8),face_indices=np.array([0]))
    depth=[row('left',a.CONFLICT),row('right',a.UNKNOWN)]
    gap=[row('left',a.UNKNOWN),row('right',a.CONFLICT)]
    v1,v2,combined=a.aggregate_rules(vertices,faces,depth,gap,cameras,a.Policy(),a.GapPolicy())
    assert v1['conflict_views'].tolist()==v2['conflict_views'].tolist()==[1]
    assert combined.tolist()==[False]
    gap[0]=row('left',a.CONFLICT)
    v1,v2,combined=a.aggregate_rules(vertices,faces,depth,gap,cameras,a.Policy(),a.GapPolicy())
    assert v1['quarantine'].tolist()==[False]
    assert v2['quarantine'].tolist()==combined.tolist()==[True]
    assert v2['witness_view_ids'][0]==['left','right']


def test_same_camera_direction_cannot_supply_gap_angular_quorum():
    vertices=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    faces=np.array([[0,1,2]])
    rows=[dict(view_id=key,status=np.array([a.CONFLICT],np.uint8)) for key in ('a','b')]
    _,gap,combined=a.aggregate_rules(vertices,faces,rows,rows,{'a':[0.,0.,0.],'b':[0.,0.,0.]},a.Policy(),a.GapPolicy())
    assert not combined.any() and not gap['quarantine'].any()


def test_edge_quorum_uses_same_view_flags_and_core_veto_without_changing_original_votes():
    v=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    f=np.array([[0,1,2]]*4)
    depth=[dict(view_id=key,status=np.full(4,a.CONFLICT,np.uint8)) for key in ('a','b')]
    gap=[dict(view_id=key,status=np.full(4,a.UNKNOWN,np.uint8)) for key in ('a','b')]
    edges=[dict(view_id='a',edge_eligible=np.array([True,False,True,False]),
                core_protected=np.array([False,False,True,False])),
           dict(view_id='b',edge_eligible=np.array([True,True,True,False]),
                core_protected=np.zeros(4,bool))]
    result=a.aggregate_edge_rules(v,f,depth,gap,edges,{'a':[-10.,0.,0.],'b':[10.,0.,0.]},a.Policy(),a.GapPolicy())
    assert result['previous_v2_quarantine'].tolist()==[True]*4
    assert result['edge_depth']['quarantine'].tolist()==[True,False,True,False]
    assert result['quarantine'].tolist()==[True,False,False,False]
    assert result['edge_eligible_any'].tolist()==[True,True,True,False]
    assert result['protected_any'].tolist()==[False,False,True,False]
    assert all(row['status'].tolist()==[a.CONFLICT]*4 for row in depth)
    assert result['subset_assert'] is True


def test_edge_rows_cannot_join_a_foreign_view_or_invalid_face_shape():
    v=np.eye(3);f=np.array([[0,1,2]])
    rows=[dict(view_id='a',status=np.array([a.CONFLICT]))]
    edges=[dict(view_id='foreign',edge_eligible=np.ones(1,bool),core_protected=np.zeros(1,bool))]
    with pytest.raises(ValueError,match='identities'):
        a.aggregate_edge_rules(v,f,rows,rows,edges,{'a':[0.,0.,0.]},a.Policy(),a.GapPolicy())
    edges[0]['view_id']='a';edges[0]['core_protected']=np.zeros(2,bool)
    with pytest.raises(ValueError,match='original face'):
        a.aggregate_edge_rules(v,f,rows,rows,edges,{'a':[0.,0.,0.]},a.Policy(),a.GapPolicy())
