"""Saved observation graph census, without importing a geometry solver."""
from collections import Counter
from pathlib import Path
from .balanced_point_run import read, save, binding, check_bindings


def components(nodes,edges):
    adjacent={i:set() for i in nodes}
    if len(adjacent)!=len(nodes): raise ValueError('duplicate graph node')
    for a,b in edges:
        if a==b or a not in adjacent or b not in adjacent: raise ValueError('invalid graph edge')
        adjacent[a].add(b); adjacent[b].add(a)
    remaining=set(nodes); result=[]
    while remaining:
        queue=[min(remaining)]; seen=set(queue)
        while queue:
            for i in adjacent[queue.pop()]-seen: seen.add(i); queue.append(i)
        result.append(sorted(seen)); remaining-=seen
    return sorted(result,key=lambda c:(-len(c),c))


def census(tracks,native,plan):
    frames=tracks['frames']; nodes=[f['input_index'] for f in frames]
    ids=[]
    for frame in frames:
        observations=frame['observations']; values={o['track_id'] for o in observations}
        if len(values)!=len(observations): raise ValueError('duplicate track in frame')
        if any(type(o['track_id']) is not int or o.get('measured') is not True or o.get('predicted',False) is not False for o in observations):
            raise ValueError('only real measured candidate identities')
        ids.append(values)
    frequency=Counter(t for row in ids for t in row)
    eligible={t for t,n in frequency.items() if n>=2}
    ids=[row&eligible for row in ids]
    pairs=[dict(a=nodes[a],b=nodes[b],sequence_distance=b-a,common_tracks=len(ids[a]&ids[b]),outside_window=b-a>3)
        for a in range(len(nodes)) for b in range(a+1,len(nodes)) if len(ids[a]&ids[b])>=15]
    all_edges=[(p['a'],p['b']) for p in pairs]
    local=[(p['a'],p['b']) for p in pairs if not p['outside_window']]
    outside=[p for p in pairs if p['outside_window']]
    ledger={f['database_image_id']:f['input_index'] for f in plan['frames']}
    if set(ledger.values())!=set(nodes): raise ValueError('source/native frame identity differs')
    verified=[(ledger[p['image_a']],ledger[p['image_b']]) for p in native['pairs'] if p['inliers']>=15]
    comps=components(nodes,verified); which={i:k for k,c in enumerate(comps) for i in c}
    memberships={i:set() for i in nodes}
    for model in native['models']:
        for i in model['registered_input_indices']: memberships[i].add(model['model_id'])
    outside_info=[dict(**p,models_a=sorted(memberships[p['a']]),models_b=sorted(memberships[p['b']]),
        connects_distinct_models=bool(memberships[p['a']] and memberships[p['b']] and not memberships[p['a']]&memberships[p['b']]),
        cross_verified_components=which[p['a']]!=which[p['b']]) for p in outside]
    registered=set(native['unique_registered_input_indices'])
    unregistered=[]
    for i in nodes:
        if i not in registered:
            neighbors=sorted(b if a==i else a for a,b in all_edges if i in (a,b))
            new_neighbors=sorted(p['b'] if p['a']==i else p['a'] for p in outside if i in (p['a'],p['b']))
            unregistered.append(dict(input_index=i,allpair_raw_sufficient_neighbors=neighbors,
                outside_raw_sufficient_neighbors=new_neighbors,outside_edges_to_registered=sum(j in registered for j in new_neighbors)))
    return dict(schema='saved-track-connectivity-audit-v1',sample_frames=len(nodes),all_possible_pairs=len(nodes)*(len(nodes)-1)//2,
        eligible_ids=len(eligible),candidate_pairs=len(pairs),local_candidate_pairs=len(local),outside_candidate_pairs=len(outside),
        raw_local_components=components(nodes,local),raw_allpair_components=components(nodes,all_edges),outside_pairs=outside,
        arms={'geometry_evidence':dict(verified_edge_threshold=15,verified_edges=len(verified),verified_components=comps,
            outside_cross_verified_component_pairs=sum(p['cross_verified_components'] for p in outside_info),
            outside_model_connections=outside_info,outside_connects_distinct_models=sum(p['connects_distinct_models'] for p in outside_info),
            registered_unique=len(registered),unregistered_count=len(unregistered),unregistered=unregistered)},
        meaning='Existing measured ID intersections only; candidates are not validated identity or new inter-component evidence.')


def audit(tracks_path,native_dir,output_path):
    output_path=Path(output_path)
    if output_path.exists(): raise ValueError('fresh graph audit required')
    paths=[Path(tracks_path),Path(native_dir)/'RESULT.json',Path(native_dir)/'PLAN.json']
    bound=[binding(p) for p in paths]
    result=census(*(read(p) for p in paths))
    result.update(source_bindings=bound,source_drift=check_bindings(bound))
    if result['source_drift']: raise ValueError('source graph drift')
    save(output_path,result)
    return result
