import numpy as np
import pytest
from integrations.video_masks.boundary_geometry import select_registered,combine


def view(i,angle,depth=True):
    return dict(frame=dict(decode_index=i),depth_path='bound.dmap' if depth else None,
                camera_scene=dict(C=[10*np.cos(angle),10*np.sin(angle),1]))


def test_registered_selection_never_uses_missing_depth_and_is_stable():
    views=[view(i,i*.2,i not in (0,4)) for i in range(12)]
    risks={i:float(i%3) for i in range(12)}
    a=select_registered(views,risks,[0,0,0],6)
    assert a==select_registered(list(reversed(views)),risks,[0,0,0],6)
    assert len(a['indices'])==6 and not set(a['indices'])&{0,4}
    assert a['eligible_count']==10 and a['registered_count']==12
    assert a['unselected_status']=='NOT_EVALUATED'


def test_selected_centers_spread_when_temporal_anchors_cluster():
    views=[view(i,0) for i in range(8)]+[view(8,np.pi)]
    selected=select_registered(views,{i:10. if i in (0,3,5,7) else 0. for i in range(9)},[0,0,0],5)
    assert 8 in selected['indices']


def test_no_depth_empty_instead_of_camera_interpolation():
    r=select_registered([view(7,0,False),view(8,1,False)],{},[0,0,0])
    assert r['indices']==[] and len(r['excluded'])==2


def test_unproved_derivation_excluded_even_with_a_camera():
    a=view(7,0);a['frame_identity']={'status':'UNVERIFIED_FRAME_IDENTITY'}
    r=select_registered([a,view(8,1)],{},[0,0,0])
    assert r['indices']==[8] and r['excluded'][0]['reason']=='UNVERIFIED_FRAME_IDENTITY'


@pytest.mark.parametrize('limit',[0,33,True])
def test_selection_budget_is_bounded(limit):
    with pytest.raises(ValueError):select_registered([view(1,0)],{},[0,0,0],limit)


def test_duplicate_frame_cannot_supply_additional_view():
    with pytest.raises(ValueError):select_registered([view(1,0),view(1,1)],{},[0,0,0])


def test_invalid_risk_rejected():
    with pytest.raises(ValueError):select_registered([view(1,0)],{1:float('nan')},[0,0,0])


def mesh():return np.array([[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]]),np.array([[0,1,2]])
def row(i,conflict=True,protected=False):
    return dict(view_id=i,status=np.array([2 if conflict else 0],np.uint8),core_protected=np.array([protected]))


def test_two_directions_required_and_protection_vetoes_conflict():
    v,f=mesh();cams={'a':[0,0,0],'b':[4,0,0]}
    assert not combine(v,f,[row('a')],cams)['candidate'][0]
    assert combine(v,f,[row('a'),row('b')],cams)['candidate'][0]
    assert not combine(v,f,[row('a'),row('b',protected=True)],cams)['candidate'][0]
    assert not combine(v,f,[row('a'),row('b')],cams,np.array([True]))['candidate'][0]


def test_same_camera_different_labels_not_independent_directions():
    v,f=mesh()
    assert not combine(v,f,[row('a'),row('b')],{'a':[0,0,0],'b':[0,0,0]})['candidate'][0]


def test_duplicate_view_identity_rejected():
    v,f=mesh()
    with pytest.raises(ValueError):combine(v,f,[row('a'),row('a')],{'a':[0,0,0]})


def test_viewer_referenced_evidence_exists_before_export(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from integrations.video_masks import boundary_geometry as b
    from integrations.viewer import export
    mesh=tmp_path/'mesh.ply';mesh.write_bytes(b'fixture')
    def check(spec,target,profile):
        evidence=Path(json.loads(Path(spec).read_text(encoding='utf8'))['scenes'][0]['evidence'][0])
        assert evidence.name=='EVIDENCE.json' and evidence.is_file()
        assert json.loads(evidence.read_text())['geometry_changed'] is False
        assert not (tmp_path/'RESULT.json').exists()
        Path(target).write_text('fixture')
    monkeypatch.setattr(export,'export_html',check)
    b.publish_mesh_viewer(tmp_path,mesh,dict(geometry_changed=False))
    assert (tmp_path/'mesh_viewer.html').is_file()
