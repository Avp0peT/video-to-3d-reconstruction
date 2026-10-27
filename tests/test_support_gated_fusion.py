import copy
import numpy as np
import pytest
from integrations.mvs_surface.support_gated_fusion import assemble_supported as assemble
from integrations.mvs_surface.support_gated_fusion import assemble_with_policy


def data():
    return dict(global_points=np.array([[0.,0.,0.],[99.,99.,99.]]),global_colors=np.array([[20,30,40],[50,60,70]],np.uint8),global_supported=np.array([True,False]),
        local_vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1e-8]]),local_colors=np.full((4,3),150,np.uint8),local_supported=np.ones(4,bool),
        local_faces=np.array([[0,1,2],[3,2,1]],np.int64),local_face_supported=np.ones(2,bool),local_vertex_source=np.array([7,7,7,8],np.int64))


def test_supported_completion_excludes_bad_global_retains_local_and_origins():
    args=data();before=copy.deepcopy(args);r=assemble(**args)
    assert r['global_to_output'].tolist()==[0,-1]
    assert r['local_to_output'].tolist()==[0,1,2,3]
    assert len(r['vertices'])==4 and len(r['faces'])==2
    assert r['counts']['local_new_unique_vertices']==3
    assert r['counts']['local_origins_exact_duplicate_global']==1
    assert r['origin_offsets'].tolist()==[0,2,3,4,5]
    assert r['origin_kind'][:2].tolist()==[0,1]
    np.testing.assert_array_equal(r['origin_colors'][:2],np.array([[20,30,40],[150,150,150]],np.uint8))
    assert r['output_global_member'].tolist()==[True,False,False,False]
    assert r['output_local_member'].all()
    assert r['face_original_ids'].tolist()==[0,1]
    assert r['face_source_id'].tolist()==[7,-1]
    np.testing.assert_array_equal(r['face_vertex_source_ids'],[[7,7,7],[8,7,7]])
    assert not r['geometry_accepted']
    for key in before:np.testing.assert_array_equal(args[key],before[key])


def test_close_opposite_layers_are_never_snapped():
    args=data();r=assemble(**args)
    assert r['local_to_output'][0]!=r['local_to_output'][3]
    assert r['vertices'][r['local_to_output'][3],2]==1e-8


def test_vertex_exclusion_never_retriangulates_or_bridges():
    args=data();args['local_supported'][1]=False;r=assemble(**args)
    assert not len(r['faces']) and (r['local_face_to_output']==-1).all()
    assert len(r['vertices'])==3
    assert r['counts']['rejected_vertex_unsupported_faces']==2
    assert r['counts']['isolated_output_vertices']==3


def test_unsupported_face_does_not_remove_supported_isolated_vertices():
    args=data();args['local_face_supported'][:]=False;r=assemble(**args)
    assert not len(r['faces']) and len(r['vertices'])==4
    assert r['counts']['rejected_unsupported_faces']==2


def test_exact_local_duplicates_collapse_but_no_invalid_faces_added():
    args=data();args['local_vertices'][3]=args['local_vertices'][0]
    args['local_faces']=np.array([[0,3,1],[0,0,2],[0,1,2]])
    args['local_face_supported']=np.ones(3,bool)
    r=assemble(**args)
    assert r['local_to_output'][0]==r['local_to_output'][3]
    assert r['face_original_ids'].tolist()==[2]
    assert r['face_reason'].tolist()==[3,3,5]
    assert r['origin_offsets'][1]==3


def test_collinear_zero_area_rejected():
    args=data();args['local_vertices'][2]=[2,0,0]
    r=assemble(**args)
    assert r['face_reason'][0]==4
    assert r['face_original_ids'].tolist()==[1]


def test_duplicate_global_and_local_only_origins_all_reversible():
    args=data();args['global_points'][1]=args['global_points'][0];args['global_supported'][:]=True
    args['local_vertices'][3]=args['local_vertices'][1]
    r=assemble(**args)
    assert r['counts']['global_origins_exact_duplicate_global']==1
    assert r['counts']['local_origins_exact_duplicate_local']==1
    assert len(r['origin_index'])==6
    for kind,points,mapping in ((0,args['global_points'],r['global_to_output']),(1,args['local_vertices'],r['local_to_output'])):
        for out in range(len(r['vertices'])):
            sl=slice(r['origin_offsets'][out],r['origin_offsets'][out+1])
            ids=r['origin_index'][sl][r['origin_kind'][sl]==kind]
            for index in ids:
                np.testing.assert_array_equal(points[index],r['vertices'][out]);assert mapping[index]==out


def test_empty_and_all_excluded_are_well_shaped():
    args=data();args['global_supported'][:]=False;args['local_supported'][:]=False
    r=assemble(**args)
    assert r['vertices'].shape==r['colors'].shape==r['faces'].shape==(0,3)
    assert r['origin_offsets'].tolist()==[0]
    assert (r['global_to_output']==-1).all()


@pytest.mark.parametrize('field,value',[
    ('global_supported',np.array([1,0])),('local_supported',np.ones(4,np.uint8)),
    ('local_faces',np.array([[0.,1.,2.]])),('local_faces',np.array([[0,1,10]])),
    ('local_face_supported',np.array([1,1])),('local_vertex_source',np.array([True]*4)),
    ('global_points',np.array([[0.,0,np.nan],[1,2,3]])),('global_colors',np.zeros((2,3),float)),
    ('local_vertex_source',np.array([0,0,-1,0]))])
def test_strict_input_validation(field,value):
    args=data();args[field]=value
    with pytest.raises(ValueError):assemble(**args)


def test_cancellation():
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):assemble(**data(),check=stop)


def test_policy_default_preserves_old_geometry_mappings_and_face_reasons():
    args=data();args['local_supported'][1]=False
    old=assemble(**args);new=assemble_with_policy(**args)
    for key in old:
        if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],new[key])
    assert old['status']==new['status']
    assert 'supported_local_origins' not in new['counts']
    assert new['counts']['admitted_local_origins']==new['counts']['evidence_supported_local_origins']==3


def test_all_local_admits_points_global_stays_filtered_faces_not_relaxed():
    args=data();args['local_supported'][1]=False;before=copy.deepcopy(args)
    r=assemble_with_policy(**args,local_policy='all-points')
    assert r['global_admitted'].tolist()==[True,False]
    assert r['global_to_output'].tolist()==[0,-1]
    assert r['local_admitted'].all()
    np.testing.assert_array_equal(r['local_evidence_supported'],args['local_supported'])
    assert (r['local_to_output']>=0).all() and len(r['vertices'])==4
    assert len(r['faces'])==0 and not r['local_face_admitted'].any()
    assert r['counts']['admitted_local_origins']==4
    assert r['counts']['evidence_supported_local_origins']==3
    assert r['counts']['admitted_evidence_unsupported_local_origins']==1
    assert 'supported_local_origins' not in r['counts']
    assert r['status']=='ALL_LOCAL_COMPARISON_NOT_GEOMETRY_ACCEPTED'
    assert not r['geometry_accepted']
    for key in before:np.testing.assert_array_equal(args[key],before[key])


def test_all_local_preserves_face_filter_and_original_evidence_ids():
    args=data();args['local_supported'][3]=False;args['local_face_supported'][0]=False
    r=assemble_with_policy(**args,local_policy='all-points')
    assert not len(r['faces'])
    assert r['local_face_input_supported'].tolist()==[False,True]
    assert r['local_face_vertices_evidence_supported'].tolist()==[True,False]
    args['local_face_supported'][0]=True
    r=assemble_with_policy(**args,local_policy='all-points')
    assert r['face_original_ids'].tolist()==[0]
    np.testing.assert_array_equal(r['face_vertex_source_ids'],[[7,7,7]])
    assert r['local_face_admitted'].tolist()==[True,False]


@pytest.mark.parametrize('policy',['all','',None,True,0,['all-points']])
def test_bad_local_admission_policy(policy):
    with pytest.raises(ValueError):assemble_with_policy(**data(),local_policy=policy)


def test_all_original_keeps_all_origins_and_restores_only_existing_faces():
    args=data();args['global_supported'][:]=False;args['local_supported'][:]=False;args['local_face_supported'][:]=False
    before=copy.deepcopy(args)
    r=assemble_with_policy(**args,global_policy='all-points',local_policy='all-points',face_policy='original')
    assert r['global_admitted'].all() and r['local_admitted'].all()
    assert (r['global_to_output']>=0).all() and (r['local_to_output']>=0).all()
    assert not r['global_evidence_supported'].any() and not r['local_evidence_supported'].any()
    assert not r['local_face_input_supported'].any()
    assert len(r['faces'])==len(args['local_faces'])
    np.testing.assert_array_equal(r['vertices'][r['faces']],args['local_vertices'][args['local_faces']])
    np.testing.assert_array_equal(r['vertices'][r['global_to_output']],args['global_points'])
    np.testing.assert_array_equal(r['vertices'][r['local_to_output']],args['local_vertices'])
    assert r['face_original_ids'].tolist()==[0,1]
    assert r['counts']['supported_global_origins']==r['counts']['evidence_supported_global_origins']==0
    assert r['counts']['admitted_global_origins']==2 and r['counts']['admitted_local_origins']==4
    assert r['counts']['restored_failed_face_audit_faces']==2
    assert r['counts']['restored_faces_without_full_evidence']==2
    assert r['status']=='ALL_ORIGINAL_COMPARISON_NOT_GEOMETRY_ACCEPTED'
    for key in before:np.testing.assert_array_equal(args[key],before[key])


def test_original_faces_still_reject_collapsed_and_zero_area():
    args=data();args['local_vertices'][2]=[2,0,0]
    args['local_faces']=np.array([[0,0,1],[0,1,2],[0,1,3]])
    args['local_face_supported']=np.zeros(3,bool);args['local_supported'][:]=False
    r=assemble_with_policy(**args,local_policy='all-points',face_policy='original')
    assert r['face_reason'].tolist()==[3,4,5]
    assert r['face_original_ids'].tolist()==[2]
    assert r['counts']['restored_failed_face_audit_faces']==1
    assert r['counts']['rejected_collapsed_faces']==r['counts']['rejected_zero_area_faces']==1
    assert r['status']=='RELAXED_ADMISSION_COMPARISON_NOT_GEOMETRY_ACCEPTED'


def test_all_global_does_not_relax_local_or_face_evidence():
    args=data();args['local_supported'][1]=False
    r=assemble_with_policy(**args,global_policy='all-points')
    assert r['global_admitted'].all() and r['global_to_output'][1]>=0
    assert r['local_to_output'][1]==-1 and not len(r['faces'])
    assert r['counts']['supported_global_origins']==1
    assert r['counts']['admitted_global_origins']==2
    assert r['status']=='RELAXED_ADMISSION_COMPARISON_NOT_GEOMETRY_ACCEPTED'


@pytest.mark.parametrize('kwargs',[
    {'face_policy':'original'}, {'face_policy':'original','local_policy':'supported-only'},
    {'global_policy':'all'}, {'global_policy':True}, {'face_policy':'all'}, {'face_policy':None}])
def test_invalid_relaxation_policies_rejected(kwargs):
    with pytest.raises(ValueError):assemble_with_policy(**data(),**kwargs)
