from copy import deepcopy
from itertools import combinations
import json
import pytest
from kozlov_lab.stereo.multiview_cycles import analyze_candidate_graph


def node(frame,feature='point'):return dict(frame=frame,feature_id=feature)
def edge(ident,a,b):return dict(edge_id=ident,a=a,b=b)


def oracle(edges):
    graph={}
    for e in edges:
        a=(e['a']['frame'],e['a']['feature_id']);b=(e['b']['frame'],e['b']['feature_id'])
        graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    paths=[];triangles=set()
    for center in sorted(graph):
        for a,b in combinations(sorted(graph[center]),2):
            if len({center[0],a[0],b[0]})<3:continue
            paths.append((center,a,b))
            if b in graph[a]:triangles.add(tuple(sorted((center,a,b))))
    return paths,triangles


@pytest.mark.parametrize('case',range(11))
def test_eleven_forest_cases_match_full_path_oracle(case):
    edges=[]
    for i in range(1,case+2):
        parent=(i-1)//2 if case%2 else i-1
        edges.append(edge('e'+str(i),node(parent,'v'+str(parent)),node(i,'v'+str(i))))
    expected,triangles=oracle(edges);r=analyze_candidate_graph(edges)
    actual=[((w['center']['frame'],w['center']['feature_id']),
             (w['endpoint_a']['frame'],w['endpoint_a']['feature_id']),
             (w['endpoint_b']['frame'],w['endpoint_b']['feature_id'])) for w in r['wedges']]
    assert sorted(actual)==sorted(expected) and r['total_wedges']==len(expected)
    assert r['unique_triangle_count']==len(triangles)==0
    assert all(w['direct_existing_edge_id'] is None for w in r['wedges'])


def test_triangle_three_paths_one_triangle_wrong_identity_not_certified():
    edges=[edge('a',node(0,'wrong0'),node(1,'wrong1')),edge('b',node(1,'wrong1'),node(2,'wrong2')),
           edge('c',node(2,'wrong2'),node(0,'wrong0'))]
    before=deepcopy(edges);r=analyze_candidate_graph(edges)
    assert len(r['wedges'])==3 and r['unique_triangle_count']==1
    assert r['counts']['closed_wedges']==3 and all(w['direct_existing_edge_id'] for w in r['wedges'])
    assert edges==before and not r['identity_verified'] and not r['geometry_acceptance'] and not r['native_injected']
    assert r['uniqueness']=='NOT_EVALUATED'
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('center',[0,5,10])
def test_star_center_earliest_middle_latest(center):
    others=[f for f in [1,3,7,9] if f!=center]
    edges=[edge(str(f),node(center,'center'),node(f)) for f in others]
    r=analyze_candidate_graph(edges)
    assert r['total_wedges']==6 and all(w['center']['frame']==center for w in r['wedges'])


def test_same_frame_conflicts_and_reused_strings_not_merged():
    edges=[edge('a',node(0,'same'),node(1,'same')),edge('b',node(1,'same'),node(2,'same')),
           edge('c',node(2,'same'),node(0,'different'))]
    r=analyze_candidate_graph(edges);c=r['components'][0]
    assert r['counts']['nodes']==4 and len(c['same_frame_conflicts'])==1
    assert c['same_frame_conflicts'][0]['frame']==0
    assert all(w['component_conflict'] for w in r['wedges'])
    assert all(len({w['endpoint_a']['frame'],w['center']['frame'],w['endpoint_b']['frame']})==3 for w in r['wedges'])


def test_sampling_deterministic_global_domain_and_group_unranking():
    edges=[edge('e'+str(i),node(0),node(1+i%3,'f'+str(i))) for i in range(45)]
    full=oracle(edges)[0];r=analyze_candidate_graph(edges,max_wedges=7)
    assert r['total_wedges']==675 and r['omitted_count']==668 and r['status']=='PARTIAL_WEDGE_SAMPLE'
    assert r['sampled_wedge_indices']==[i*674//6 for i in range(7)]
    # Domain order is neighbor-frame groups, then row-major group products.
    grouped=[]
    for fa,fb in combinations([1,2,3],2):
        left=sorted([(e['b']['frame'],e['b']['feature_id']) for e in edges if e['b']['frame']==fa])
        right=sorted([(e['b']['frame'],e['b']['feature_id']) for e in edges if e['b']['frame']==fb])
        grouped.extend((a,b) for a in left for b in right)
    for w,index in zip(r['wedges'],r['sampled_wedge_indices']):
        assert ((w['endpoint_a']['frame'],w['endpoint_a']['feature_id']),(w['endpoint_b']['frame'],w['endpoint_b']['feature_id']))==grouped[index]
    assert r==analyze_candidate_graph(list(reversed(edges)),max_wedges=7)


def test_disconnected_components_deterministic():
    edges=[edge('z',node(8),node(9)),edge('a',node(0),node(1))]
    r=analyze_candidate_graph(edges)
    assert [c['views'] for c in r['components']]==[[0,1],[8,9]]
    assert not r['wedges']
    assert analyze_candidate_graph([])['counts']['components']==0


@pytest.mark.parametrize('case',['duplicate_id','duplicate_pair','same_frame','bool_frame','negative_frame','empty_id','empty_feature','too_many'])
def test_invalid_inputs_rejected(case):
    e=[edge('a',node(0),node(1))]
    if case=='duplicate_id':e.append(edge('a',node(2),node(3)))
    elif case=='duplicate_pair':e.append(edge('b',node(1),node(0)))
    elif case=='same_frame':e[0]['b']=node(0,'other')
    elif case=='bool_frame':e[0]['b']['frame']=True
    elif case=='negative_frame':e[0]['b']['frame']=-1
    elif case=='empty_id':e[0]['edge_id']=''
    elif case=='empty_feature':e[0]['a']['feature_id']=''
    else:e*=4097
    with pytest.raises(ValueError):analyze_candidate_graph(e)


@pytest.mark.parametrize('budget',[0,257,True,1.5])
def test_bad_budget(budget):
    with pytest.raises(ValueError):analyze_candidate_graph([],max_wedges=budget)
