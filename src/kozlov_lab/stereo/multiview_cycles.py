"""Bounded candidate-identity graph diagnostics; cycles do not prove truth.

This is an engineering consistency check, not a book correspondence algorithm.
Nodes are immutable (frame, feature_id) identities; no observations are invented.
"""
from bisect import bisect_right
from collections import defaultdict


def _node(value):
    if type(value) is not dict:raise ValueError('node dict required')
    frame=value.get('frame');feature=value.get('feature_id')
    if type(frame) is not int or frame<0 or not isinstance(feature,str) or not feature:
        raise ValueError('nonnegative non-bool frame and nonempty feature ID required')
    return frame,feature


def _record(node):return dict(frame=node[0],feature_id=node[1])


def _sample(total,limit):
    if total<=limit:return list(range(total))
    return [i*(total-1)//(limit-1) for i in range(limit)] if limit>1 else [0]


def analyze_candidate_graph(edges,*,max_wedges=256):
    """Enumerate a bounded sample of all distinct-frame two-edge paths.

    The virtual order is center node, then ordered neighbor-frame group pair,
    then row-major Cartesian product of sorted neighbors. It includes centers
    earlier or later than both endpoints. Counts cover the full graph; path
    output can be partial. A closed triangle yields three paths, one triangle.
    """
    if type(edges) is not list or len(edges)>4096:raise ValueError('at most 4096 edges required')
    if type(max_wedges) is not int or not 1<=max_wedges<=256:raise ValueError('max_wedges must be 1..256')
    graph=defaultdict(dict);edge_lookup={};edge_ids=set()
    for edge in edges:
        if type(edge) is not dict:raise ValueError('edge dict required')
        ident=edge.get('edge_id')
        if not isinstance(ident,str) or not ident or ident in edge_ids:raise ValueError('unique nonempty edge ID required')
        a=_node(edge.get('a'));b=_node(edge.get('b'))
        if a[0]==b[0]:raise ValueError('edges require distinct frames')
        a,b=sorted((a,b));pair=(a,b)
        if pair in edge_lookup:raise ValueError('duplicate undirected endpoint pair')
        edge_ids.add(ident);edge_lookup[pair]=ident;graph[a][b]=ident;graph[b][a]=ident
    components=[];node_component={};remaining=set(graph)
    while remaining:
        start=min(remaining);todo=[start];members=set()
        while todo:
            node=todo.pop()
            if node in members:continue
            members.add(node);todo.extend(n for n in graph[node] if n not in members)
        remaining.difference_update(members)
        cid='component_'+str(len(components));frames=defaultdict(list)
        for node in sorted(members):node_component[node]=cid;frames[node[0]].append(node)
        conflicts=[dict(frame=frame,nodes=[_record(n) for n in nodes]) for frame,nodes in sorted(frames.items()) if len(nodes)>1]
        c=dict(component_id=cid,nodes=[_record(n) for n in sorted(members)],
            edge_ids=sorted({eid for n in members for eid in graph[n].values()}),views=sorted(frames),
            same_frame_conflicts=conflicts,component_conflict=bool(conflicts),
            identity_verified=False,geometry_acceptance=False,native_injected=False,uniqueness='NOT_EVALUATED')
        components.append(c)
    component_by_id={c['component_id']:c for c in components}
    # A center contributes choose(degree,2) minus same-neighbor-frame pairs.
    # Store O(E) groups and O(V) prefix sums, never all group pairs or wedges.
    domains=[];prefix=[];total=0
    for center in sorted(graph):
        groups=defaultdict(list)
        for neighbor in sorted(graph[center]):groups[neighbor[0]].append(neighbor)
        groups=[nodes for _,nodes in sorted(groups.items())]
        degree=sum(map(len,groups));count=(degree*degree-sum(len(g)**2 for g in groups))//2
        if count:
            suffix=[0]*(len(groups)+1)
            for i in range(len(groups)-1,-1,-1):suffix[i]=suffix[i+1]+len(groups[i])
            domains.append((center,groups,suffix));total+=count;prefix.append(total)
    # Count actual graph triangles independently of sampled wedge output.
    # Bitset intersection keeps this O(E*V/word) rather than materializing paths.
    ordered=sorted(graph);indices={n:i for i,n in enumerate(ordered)}
    bits={n:sum(1<<indices[v] for v in graph[n]) for n in ordered}
    triangles=0
    for a,b in sorted(edge_lookup):
        # Edges only connect distinct frames; a shared neighbor therefore also
        # has a distinct frame from both endpoints.
        common=bits[a]&bits[b]
        triangles+=(common >> (indices[b]+1)).bit_count()
    sampled=_sample(total,max_wedges);wedges=[]
    for global_index in sampled:
        domain_index=bisect_right(prefix,global_index);center,groups,suffix=domains[domain_index]
        local=global_index-(prefix[domain_index-1] if domain_index else 0)
        # Unrank a frame-group Cartesian product without enumerating its pairs.
        left_group=None;right_group=None
        for i,g in enumerate(groups):
            block=len(g)*suffix[i+1]
            if local>=block:local-=block;continue
            left_group=g
            for h in groups[i+1:]:
                size=len(g)*len(h)
                if local>=size:local-=size;continue
                right_group=h;break
            break
        a=left_group[local//len(right_group)];b=right_group[local%len(right_group)]
        a,b=sorted((a,b));cid=node_component[center]
        wedges.append(dict(wedge_index=global_index,endpoint_a=_record(a),endpoint_b=_record(b),center=_record(center),
            two_edge_ids=[graph[center][a],graph[center][b]],direct_existing_edge_id=edge_lookup.get((a,b)),
            component_id=cid,component_conflict=component_by_id[cid]['component_conflict'],
            identity_verified=False,geometry_acceptance=False,native_injected=False,uniqueness='NOT_EVALUATED'))
    partial=len(sampled)<total
    return dict(schema='candidate-multiview-cycles-v1',status='PARTIAL_WEDGE_SAMPLE' if partial else 'CANDIDATE_GRAPH_ANALYSIS_COMPLETE',
        components=components,wedges=wedges,total_wedges=total,sampled_wedge_indices=sampled,omitted_count=total-len(sampled),
        max_wedges=max_wedges,unique_triangle_count=triangles,
        counts=dict(nodes=len(graph),edges=len(edges),components=len(components),
            conflicting_components=sum(c['component_conflict'] for c in components),
            total_wedges=total,returned_wedges=len(wedges),omitted_wedges=total-len(wedges),
            unique_triangles=triangles,closed_wedges=3*triangles,open_wedges=total-3*triangles),
        identity_verified=False,geometry_acceptance=False,native_injected=False,uniqueness='NOT_EVALUATED',
        observations_synthesized=False,feature_ids_merged=False,
        limitations=['Cycle closure is internal graph consistency, not physical identity proof.',
            'Same-frame conflicts are preserved, not resolved by merging feature IDs.',
            'No edges or observations are synthesized; closing pairs are proposals only.'])
