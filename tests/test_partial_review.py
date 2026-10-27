"""End-to-end bounded subimage orchestration with measured synthetic inputs."""
import copy
import json
import numpy as np
import pytest
from integrations.demo_pipeline import partial_review as review
from test_temporal_review import chain
from integrations.demo_pipeline.temporal_review import review_video


def fixture(count=6):
    xy=np.array([[20,20],[100,20],[30,100],[95,110],[60,55],[150,60]],float)[:count]
    descriptors=np.zeros((count,128),np.uint8)
    for i in range(count):descriptors[i,i]=255
    records=[]
    for j in range(2):
        records.append(dict(frame_index=j,input_index=10+j*10,points=[dict(id=f'{j}:{i}',xy=(p+[j*4,j*3]).tolist()) for i,p in enumerate(xy)],
                            descriptors=descriptors.copy(),selection={}))
    return dict(frames=records,selection={'selected_rows':count*2})


def test_descriptors_are_not_sqrt_or_offset_and_inputs_unchanged():
    f=fixture();before=copy.deepcopy(f)
    a,b,edges,receipt=review.descriptor_proposals(*f['frames'])
    assert len(edges)==6 and all(e['descriptor_distance']==0 for e in edges)
    assert a==before['frames'][0]['points'] and b==before['frames'][1]['points']
    assert all(np.array_equal(x['descriptors'],y['descriptors']) for x,y in zip(f['frames'],before['frames']))
    assert 'no repeat sqrt' in receipt['normalization']


def test_tie_cutoff_and_duplicate_coordinate_aliases_not_false_unique():
    f=fixture()
    for frame in f['frames']:frame['descriptors'][:]=1
    a,b,edges,r=review.descriptor_proposals(*f['frames'])
    assert not edges and r['cutoff_tie_rows']==[list(range(6)),list(range(6))]
    f=fixture();f['frames'][0]['points'][1]['xy']=f['frames'][0]['points'][0]['xy'][:]
    a,b,edges,r=review.descriptor_proposals(*f['frames'])
    assert {p['id'] for p in r['source_removed']}=={'0:0','0:1'} and len(a)==4


def test_real_kernel_report_path_preserves_uncertainty(monkeypatch,tmp_path):
    f=fixture();monkeypatch.setattr(review,'load_feature_frames',lambda *args:f)
    frames=[dict(input_index=10),dict(input_index=20)]
    grays=[np.zeros((180,180),np.uint8)]*2
    result=review.review_partial({'stages':{'sfm':{'run_json':'fixture'}}},frames,[None]*2,grays,{'pairs':[]},tmp_path/'out',[])
    assert result['status']=='CANDIDATE_SEARCH_COMPLETE_NOT_VERIFIED'
    assert result['candidate_subsets']==1 and result['unique_candidate_edges']==6
    assert result['unique_edge_geometry_counts']=={'NO_INDEPENDENT_GEOMETRY':6}
    assert result['descriptor_geometry_counts']=={'NO_INDEPENDENT_GEOMETRY':6}
    assert result['descriptor_geometry_gated_by_affine'] is False
    assert not result['native_injected'] and result['third_view_validation']=='NOT_EVALUATED'
    assert result['multiview_cycles']['status']=='CANDIDATE_DIAGNOSTIC_COMPLETE'
    assert result['multiview_cycles']['counts']['seed_edges']==0
    assert (tmp_path/'out/multiview_cycles/MULTIVIEW.json').is_file()
    assert (tmp_path/'out/candidate_00.png').is_file()
    pair=json.loads((tmp_path/'out/PAIRS.json').read_text(encoding='utf-8'))[0]
    assert pair['search']['candidates'][0]['area_code']['comparisons']==380
    assert pair['search']['candidates'][0]['extra_point_count']==3
    with pytest.raises(ValueError,match='fresh'):
        review.review_partial({},[],[],[],{},tmp_path/'out',[])


def test_missing_source_is_not_mathematical_no_match(tmp_path):
    result=review.review_partial({},[],[],[],{},tmp_path/'out',[])
    assert result['status']=='NOT_AVAILABLE' and 'SIFT' in result['reason']


def test_input_failure_not_success_and_report_retained(monkeypatch,tmp_path):
    def fail(*args):raise ValueError('binding mismatch')
    monkeypatch.setattr(review,'load_feature_frames',fail)
    result=review.review_partial({'stages':{'sfm':{'run_json':'fixture'}}},[],[],[],{},tmp_path/'out',[])
    assert result['status']=='FAILED' and 'binding mismatch' in result['reason']
    assert (tmp_path/'out/index.html').is_file()


def test_automatic_video_path_exposes_partial_stage_without_changing_primary(chain,tmp_path):
    result=review_video(chain['pipeline'],tmp_path/'out',chain['discovery'])
    assert result['status']=='DIAGNOSTIC_COMPLETE'
    # This existing fixture claims an SfM run but has no native feature database.
    # The subreport must refuse it while the primary temporal diagnostic remains intact.
    assert result['partial_subimages']['status']=='FAILED'
    assert (tmp_path/'out/partial_subimages/index.html').is_file()
    assert not result['geometry_modified']


def test_frame_pair_choice_is_bounded_and_deterministic():
    f=fixture()['frames'][0]
    pairs,receipt=review.choose_pairs([f]*72)
    assert len(pairs)==24 and len(set(pairs))==24
    assert pairs[0]==(0,1) and pairs[-1]==(70,71)
    assert not receipt['all_frame_pairs_searched']


def test_known_nonplanar_correspondence_not_vetoed_by_2d_area_code(monkeypatch,tmp_path):
    # Same labelled body, two orthographic directions. Book pp42-43 separates
    # this case from changing the projection plane at a fixed view direction.
    xyz=np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,3],[2,1,4],[1,2,5]],float)
    left=xyz[:,[0,1]]*100+20
    right=xyz[:,[0,2]]*100+20
    affine=np.linalg.solve(np.column_stack([left[:3],np.ones(3)]),right[:3])
    assert not np.allclose(np.column_stack([left,np.ones(6)])@affine,right)
    f=fixture()
    for frame,xy in zip(f['frames'],(left,right)):
        for point,coord in zip(frame['points'],xy):point['xy']=coord.tolist()
    monkeypatch.setattr(review,'load_feature_frames',lambda *args:f)
    geometry={'pairs':[dict(frames=[0,1],status='CANDIDATE_SUPPORT_NOT_VERIFIED',
        F=[[0,0,-1],[0,0,0],[1,0,0]],train_ids=['independent_training'],holdout_ids=[],
        candidates=[dict(stable_id='independent_training',xy_first=[-100,-100],xy_second=[-100,300])],
        planar_or_low_parallax_ambiguity=False)]}
    before=copy.deepcopy(geometry)
    result=review.review_partial({'stages':{'sfm':{'run_json':'synthetic'}}},
        [dict(input_index=10),dict(input_index=20)],[None]*2,[np.zeros((600,300),np.uint8)]*2,
        geometry,tmp_path/'out',[])
    assert result['candidate_subsets']==0
    assert result['descriptor_proposal_edges']==6
    assert result['descriptor_geometry_counts']=={'EXISTING_F_SUPPORT_CANDIDATE':6}
    assert not result['native_injected'] and not result['identity_verified']
    assert geometry==before
    row=json.loads((tmp_path/'out/PAIRS.json').read_text(encoding='utf-8'))[0]
    assert row['proposal_geometry']['model_refit'] is False
    assert all(e['training_endpoint_independent'] for e in row['proposal_geometry']['links'])
