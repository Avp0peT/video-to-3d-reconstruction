from copy import deepcopy
import json
import cv2
import numpy as np
import pytest
from integrations.video_masks.motion_part_coherence import refine_motion_parts


def fixture(n=3, drift=0, collinear=False):
    images=[np.zeros((64,100,3),np.uint8) for _ in range(n)]
    whole=[np.ones((64,100),bool) for _ in images]
    candidates=[]; positions=[]
    for f in range(n):
        mask=np.zeros((64,100),bool); mask[10:20,10:20]=True; mask[35:45,70:80]=True
        candidates.append(mask)
        xy=[]
        for x,y in [(12,12),(72,37)]:
            xy.extend([[x+f*drift,y],[x+2+f*drift,y],[x+f*drift,y+(0 if collinear else 2)],[x+2+f*drift,y+(0 if collinear else 2)]])
        positions.append(xy)
    pairs=[dict(pair=[f,f+1],track_ids=list(range(8)),source_points=positions[f],destination_points=positions[f+1]) for f in range(n-1)]
    raw=dict(status='raw',global_masks=whole,part_masks={'raw_1':candidates},fallback_masks=[m&~c for m,c in zip(whole,candidates)],unknown_masks=[np.zeros_like(m) for m in whole],evidence_domain_masks=whole,ledger=dict(pairs=pairs,parts=[dict(id='raw_1',track_ids=list(range(8)))]))
    return images,whole,raw


def test_disconnected_domains_are_separate_and_inputs_unchanged():
    images,masks,raw=fixture(); original=deepcopy(raw)
    r=refine_motion_parts(images,masks,raw)
    assert len(r['part_masks'])==2
    assert raw['ledger']==original['ledger']
    json.dumps(r['refinement_ledger']); json.dumps(r['ledger'])
    for f in range(3):
        union=np.zeros_like(masks[f])
        for parts in r['part_masks'].values():
            assert cv2.connectedComponents(parts[f].astype(np.uint8),connectivity=4)[0]<=2
            assert not (parts[f]&~raw['part_masks']['raw_1'][f]).any()
            union|=parts[f]
        assert np.array_equal(union|r['fallback_masks'][f],masks[f])
        assert np.array_equal(raw['global_masks'][f],original['global_masks'][f])
        assert np.array_equal(raw['part_masks']['raw_1'][f],original['part_masks']['raw_1'][f])
        assert np.array_equal(r['global_masks'][f],masks[f])
    assert not r['geometry_accepted']


def test_missing_frame_no_bridge():
    images,masks,raw=fixture(5)
    raw['part_masks']['raw_1'][2][:]=False
    r=refine_motion_parts(images,masks,raw)
    assert not r['part_masks']
    assert r['unknown_masks'][0][12,12]


def test_noncontinuous_track_identity_is_not_linked():
    images,masks,raw=fixture()
    raw['ledger']['pairs'][1]['track_ids']=list(range(100,108))
    r=refine_motion_parts(images,masks,raw)
    assert not r['part_masks']


def test_collinear_tracks_rejected():
    r=refine_motion_parts(*fixture(collinear=True))
    assert not r['part_masks']
    assert any(e['reason']=='DEGENERATE_TRACK_GEOMETRY' for e in r['refinement_ledger']['edges'])


def test_fixed_anchor_blocks_transitive_overlap_drift():
    r=refine_motion_parts(*fixture(drift=2),policy={'min_warp_iou':0.5})
    assert not r['part_masks']
    assert any(e.get('check')=='FIXED_ANCHOR_COMMON_IDS' and e['reason']=='WARP_OVERLAP' for e in r['refinement_ledger']['edges'])


def test_holes_remain_and_bridge_not_invented():
    images,masks,raw=fixture()
    for mask in raw['part_masks']['raw_1']: mask[15:17,15:17]=False
    r=refine_motion_parts(images,masks,raw)
    assert len(r['part_masks'])==2
    for parts in r['part_masks'].values():
        assert not parts[0][15:17,15:17].any()
        assert not parts[0][25,50]


def test_component_and_part_limits_report_loss():
    r=refine_motion_parts(*fixture(),policy={'max_components':1,'max_parts':1})
    assert len(r['part_masks'])==1
    assert any(e['reason']=='MAX_COMPONENTS_LIMIT' for e in r['refinement_ledger']['rejections'])
    assert all(m['removed_candidate_pixels']>0 for m in r['metrics'])
    r=refine_motion_parts(*fixture(),policy={'max_parts':1})
    assert any(e['reason']=='MAX_PARTS_LIMIT' for e in r['refinement_ledger']['rejections'])


@pytest.mark.parametrize('policy',[{'min_shared_tracks':3},{'max_parts':33},{'max_components':257},{'min_frames':True},{'min_warp_iou':float('nan')}])
def test_invalid_policy(policy):
    with pytest.raises(ValueError): refine_motion_parts(*fixture(),policy=policy)


def test_insufficient_frames_all_fallback():
    images,masks,raw=fixture(2)
    r=refine_motion_parts(images,masks,raw)
    assert not r['part_masks']
    for f in range(2): assert np.array_equal(r['fallback_masks'][f],masks[f])


def test_split_is_ambiguous_not_best_match():
    images,masks,raw=fixture()
    xy=[[12,12],[14,12],[12,14],[14,14],[24,12],[26,12],[24,14],[26,14]]
    for f,mask in enumerate(raw['part_masks']['raw_1']):
        mask[:]=False; mask[10:20,10:30]=True
        if f: mask[:,19:21]=False
    for pair in raw['ledger']['pairs']:
        pair['source_points']=xy; pair['destination_points']=xy
    r=refine_motion_parts(images,masks,raw)
    assert not r['part_masks']
    assert any(e['reason']=='AMBIGUOUS_ASSOCIATION' for e in r['refinement_ledger']['edges'])


def test_alien_track_ids_cannot_supply_candidate_identity():
    images,masks,raw=fixture()
    raw['ledger']['parts'][0]['track_ids']=list(range(100,108))
    r=refine_motion_parts(images,masks,raw)
    assert not r['part_masks']

def test_report_metrics_remain_compatible():
    images,masks,raw=fixture()
    raw['metrics']=[dict(tracked_seed_count=27,custom='kept') for _ in images]
    r=refine_motion_parts(images,masks,raw)
    for m in r['metrics']:
        assert m['global_pixels']==6400 and m['part_pixels']==200
        assert m['tracked_seed_count']==27 and m['custom']=='kept'

@pytest.mark.parametrize('mutation',['missing_part','duplicate_part','bool_frame','duplicate_pair','overlap'])
def test_malformed_raw_identity_rejected(mutation):
    images,masks,raw=fixture()
    if mutation=='missing_part': raw['ledger']['parts']=[]
    elif mutation=='duplicate_part': raw['ledger']['parts']*=2
    elif mutation=='bool_frame': raw['ledger']['pairs'][0]['pair']=[False,1]
    elif mutation=='duplicate_pair': raw['ledger']['pairs'].append(deepcopy(raw['ledger']['pairs'][0]))
    else:
        raw['part_masks']['raw_2']=[m.copy() for m in raw['part_masks']['raw_1']]
        raw['ledger']['parts'].append(dict(id='raw_2',track_ids=list(range(8))))
    with pytest.raises(ValueError): refine_motion_parts(images,masks,raw)


def test_refined_evidence_unknown_partition_and_raw_copy():
    images,masks,raw=fixture(2)
    old=[e.copy() for e in raw['evidence_domain_masks']]
    r=refine_motion_parts(images,masks,raw)
    for f,w in enumerate(masks):
        e=r['evidence_domain_masks'][f]; u=r['unknown_masks'][f]
        assert not (e&u).any()
        assert np.array_equal(e|u,w)
        assert np.array_equal(r['raw_evidence_domain_masks'][f],old[f])
        assert np.array_equal(raw['evidence_domain_masks'][f],old[f])
        assert not np.shares_memory(r['raw_evidence_domain_masks'][f],raw['evidence_domain_masks'][f])
        assert e[0,0] and r['fallback_masks'][f][0,0]
        assert u[12,12] and not e[12,12]

@pytest.mark.parametrize('bad',['overlap','gap','outside'])
def test_raw_evidence_unknown_contract_rejected(bad):
    images,masks,raw=fixture()
    raw['evidence_domain_masks']=[e.copy() for e in raw['evidence_domain_masks']]
    if bad=='overlap': raw['unknown_masks'][0][0,0]=True
    elif bad=='gap': raw['evidence_domain_masks'][0][0,0]=False
    else:
        masks[0][0,0]=False
    with pytest.raises(ValueError): refine_motion_parts(images,masks,raw)

def robust_fixture(inconsistent=False):
    images,masks,raw=fixture()
    grid=[[15+5*x,15+5*y] for y in range(4) for x in range(5)]
    positions=[]
    rng=np.random.default_rng(39)
    for f in range(3):
        points=np.asarray(grid,float)
        if inconsistent and f:
            points+=rng.uniform(-9,9,points.shape)
        elif f:
            points[-1]+=[f*7,0]
        positions.append(points.tolist())
        raw['part_masks']['raw_1'][f][:]=False
        raw['part_masks']['raw_1'][f][5:50,5:55]=True
    raw['ledger']['parts'][0]['track_ids']=list(range(20))
    raw['ledger']['pairs']=[dict(pair=[f,f+1],track_ids=list(range(20)),source_points=positions[f],destination_points=positions[f+1]) for f in range(2)]
    return images,masks,raw


def test_robust_minority_bad_tracks_do_not_discard_consistent_domain():
    r=refine_motion_parts(*robust_fixture())
    assert len(r['part_masks'])==1
    assert r['ledger']['parts'][0]['track_ids']==list(range(19))
    for edge in r['refinement_ledger']['edges']:
        if edge['reason']=='PASS':
            assert 19 not in edge['accepted_track_ids']
            assert 19 in edge['outlier_track_ids'] or 19 not in edge['shared_track_ids']
    assert r['refinement_ledger']['policy_version']=='motion-coherence-robust-v2'


def test_robust_majority_inconsistent_is_rejected():
    r=refine_motion_parts(*robust_fixture(inconsistent=True))
    assert not r['part_masks']
    assert any(e['reason']=='INSUFFICIENT_INLIER_RATIO' for e in r['refinement_ledger']['edges'])

def test_four_tracks_with_bad_track_cannot_accept_three_point_core():
    images,masks,raw=fixture()
    raw['ledger']['pairs'][0]['destination_points'][3]=[18,18]
    raw['ledger']['pairs'][1]['source_points'][3]=[18,18]
    r=refine_motion_parts(images,masks,raw)
    # The other remote island can survive, but the corrupted four-track island cannot.
    assert len(r['part_masks'])==1
    assert r['ledger']['parts'][0]['track_ids']==[4,5,6,7]
    assert any(e['reason'] in ('INSUFFICIENT_COMMON_INLIERS','INSUFFICIENT_INLIER_RATIO','AFFINE_RESIDUAL') for e in r['refinement_ledger']['edges'])


def test_robust_consensus_is_deterministic_and_does_not_mutate_ledger():
    args=robust_fixture(); before=deepcopy(args[2]['ledger'])
    a=refine_motion_parts(*args); b=refine_motion_parts(*args)
    assert a['refinement_ledger']==b['refinement_ledger']
    assert args[2]['ledger']==before


@pytest.mark.parametrize('ratio',[True,0,1.01,float('nan')])
def test_invalid_robust_ratio(ratio):
    with pytest.raises(ValueError): refine_motion_parts(*fixture(),policy={'min_inlier_ratio':ratio})


def test_edge_scope_and_initial_support_retention_are_explicit():
    r=refine_motion_parts(*robust_fixture())
    rp=r['refinement_ledger']['ransac_policy']
    assert rp['ratio_scope']=='per_edge_shared_tracks'
    assert rp['whole_window_ratio_gate'] is False
    for edge in r['refinement_ledger']['edges']:
        if edge.get('check')=='FIXED_ANCHOR_COMMON_IDS':
            assert edge['initial_support_count']==20
            assert edge['accepted_core_count']==len(edge['accepted_track_ids'])
            assert edge['start_support_retention']==edge['accepted_core_count']/20
    part=r['ledger']['parts'][0]
    assert part['initial_track_ids']==list(range(20))
    assert part['initial_support_count']==20
    assert part['accepted_core_count']==19
    assert part['start_support_retention']==19/20
