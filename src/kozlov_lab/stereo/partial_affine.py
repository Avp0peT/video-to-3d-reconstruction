"""Book-guided local affine common-subset proposals, engineering float domain.

Printed 61-62: three anchors propose; extra real points and the complete proposed
subimage code must agree. This is not exact certification or arbitrary perspective.
"""
from collections import Counter
from bisect import bisect_right
from itertools import combinations
import math
import numpy as np
from .area_code import compare_area_codes

PROFILE=dict(max_points_per_side=256,max_proposal_edges=48,max_hypotheses=512,
    max_candidates=128,min_mapping_points=6,
    min_anchor_area_pixels2=32.,max_condition=50.,max_transfer_pixels=2.,
    anchor_neighborhood_pixels=160.,quad_epsilon=.08,min_quad_denominator_area_pixels2=32.,
    coordinate_abs_bound=1000000.,projection_domain='engineering local-affine approximation only')
PROFILE.update(code_definition='book-full-unsigned-area-v1',code_scope='all_distinct_triangle_pairs',mapping_points=6,max_subset_checks=512,subset_domain='exactly three anchors plus three additional degree-one geometric edges')


def _sample_indices(total,limit):
    if total<=limit:return list(range(total))
    return [i*(total-1)//(limit-1) for i in range(limit)] if limit>1 else [0]


def _unrank_triple(n,rank):
    """Lexicographic combination identity without materializing its universe."""
    chosen=[];start=0
    for remaining in (2,1,0):
        for value in range(start,n-remaining):
            block=math.comb(n-value-1,remaining)
            if rank<block:
                chosen.append(value);start=value+1;break
            rank-=block
    return chosen


def _points(points):
    if type(points) is not list or len(points)>256:raise ValueError('at most 256 measured points per side required')
    result={};seen=set()
    for row in points:
        if type(row) is not dict:raise ValueError('point records must be dicts')
        ident=row.get('id');xy=row.get('xy')
        if not isinstance(ident,str) or not ident or ident in result:raise ValueError('unique nonempty string point IDs required')
        if not isinstance(xy,(list,tuple)) or len(xy)!=2 or any(isinstance(v,(bool,np.bool_)) or
            not isinstance(v,(int,float,np.integer,np.floating)) or not math.isfinite(float(v)) or abs(v)>1000000 for v in xy):
            raise ValueError('bounded finite real xy required')
        key=tuple(map(float,xy))
        if key in seen:raise ValueError('duplicate coordinates rejected; no identity merging')
        seen.add(key);result[ident]=np.asarray(key,dtype=float)
    return result


def _area(points):
    a,b,c=points
    return float(((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2.)


def _area_code_check(source, target):
    """Book full unsigned code; float stability threshold remains engineering."""
    return compare_area_codes(source, target, scope='full', epsilon=.08,
                              min_denominator_area=32.)


def search_partial_affine(source_points,target_points,proposal_edges,*,max_hypotheses=512):
    source=_points(source_points);target=_points(target_points)
    if type(max_hypotheses) is not int or not 1<=max_hypotheses<=512:raise ValueError('max_hypotheses must be 1..512')
    if type(proposal_edges) is not list or len(proposal_edges)>48:raise ValueError('at most 48 proposal edges required')
    edges=[];seen=set()
    for row in proposal_edges:
        if type(row) is not dict:raise ValueError('edge records must be dicts')
        s=row.get('source_id');t=row.get('target_id');d=row.get('descriptor_distance')
        if not isinstance(s,str) or not isinstance(t,str) or s not in source or t not in target:raise ValueError('edge references unknown point ID')
        if isinstance(d,(bool,np.bool_)) or not isinstance(d,(int,float,np.integer,np.floating)) or not math.isfinite(float(d)) or d<0:
            raise ValueError('finite nonnegative descriptor distance required')
        if (s,t) in seen:raise ValueError('duplicate proposal edge rejected')
        seen.add((s,t));edges.append(dict(source_id=s,target_id=t,descriptor_distance=float(d)))
    universe=list(combinations(range(len(edges)),3));total=len(universe)
    sampled=_sample_indices(total,max_hypotheses)
    counts=Counter();ledger=[];candidates=[];by_mapping={};overflow=0;plans=[];subset_ledger=[];ends=[];subset_total=0
    all_source=np.asarray([source[e['source_id']] for e in edges],dtype=float).reshape(-1,2)
    all_target=np.asarray([target[e['target_id']] for e in edges],dtype=float).reshape(-1,2)
    for combination_index in sampled:
        anchors=universe[combination_index];chosen=[edges[i] for i in anchors]
        row=dict(combination_index=combination_index,anchor_edge_indices=list(anchors),total_subsets=0,sampled_subsets=[])
        def reject(status,**details):
            counts[status]+=1;row.update(status=status,**details);ledger.append(row)
        if len({e['source_id'] for e in chosen})<3 or len({e['target_id'] for e in chosen})<3:
            reject('ANCHOR_ID_COLLISION');continue
        a=all_source[list(anchors)];b=all_target[list(anchors)]
        sa=abs(_area(a));ta=abs(_area(b))
        if min(sa,ta)<32:reject('ANCHOR_AREA_REJECTED');continue
        sc=float(np.linalg.cond((a[1:]-a[0]).T));tc=float(np.linalg.cond((b[1:]-b[0]).T))
        if not math.isfinite(sc) or not math.isfinite(tc) or max(sc,tc)>50:reject('ANCHOR_CONDITION_REJECTED');continue
        coefficients=np.linalg.solve(np.column_stack([a,np.ones(3)]),b)
        matrix=coefficients[:2].T;translation=coefficients[2]
        if abs(float(np.linalg.det(matrix)))<1e-8 or float(np.linalg.cond(matrix))>50:
            reject('SINGULAR_OR_ILL_CONDITIONED_MAP');continue
        inverse=np.linalg.inv(matrix)
        forward=np.linalg.norm(all_source@matrix.T+translation-all_target,axis=1)
        backward=np.linalg.norm((all_target-translation)@inverse.T-all_source,axis=1)
        source_near=np.min(np.linalg.norm(all_source[:,None,:]-a[None,:,:],axis=2),axis=1)<=160
        target_near=np.min(np.linalg.norm(all_target[:,None,:]-b[None,:,:],axis=2),axis=1)<=160
        eligible=[i for i in range(len(edges)) if forward[i]<=2 and backward[i]<=2 and source_near[i] and target_near[i]]
        sd=Counter(edges[i]['source_id'] for i in eligible);td=Counter(edges[i]['target_id'] for i in eligible)
        unique=[i for i in eligible if sd[edges[i]['source_id']]==1 and td[edges[i]['target_id']]==1]
        ambiguous=[i for i in eligible if i not in unique]
        if any(i not in unique for i in anchors):reject('AMBIGUOUS_ANCHOR_MAPPING',ambiguous_edge_indices=ambiguous);continue
        if len(unique)<6:reject('INSUFFICIENT_EXTRA_POINTS',mapping_count=len(unique),extra_count=len(unique)-3,ambiguous_edge_indices=ambiguous);continue
        extras=[i for i in unique if i not in anchors]
        subset_count=math.comb(len(extras),3)
        row.update(status='GEOMETRIC_SUBSET_DOMAIN',eligible_edge_indices=eligible,unique_edge_indices=unique,
            ambiguous_edge_indices=ambiguous,total_subsets=subset_count,subset_global_offset=subset_total,sampled_subsets=[])
        counts['GEOMETRIC_SUBSET_DOMAIN']+=1;ledger.append(row)
        subset_total+=subset_count;ends.append(subset_total)
        plans.append(dict(row=row,anchors=anchors,extras=extras,unique=unique,ambiguous=ambiguous,
            matrix=matrix,translation=translation,forward=forward,backward=backward,sa=sa,ta=ta,sc=sc,tc=tc))
    sampled_subset_indices=_sample_indices(subset_total,512)
    for global_index in sampled_subset_indices:
        plan_index=bisect_right(ends,global_index);p=plans[plan_index];hypothesis=p['row']
        local_index=global_index-hypothesis['subset_global_offset']
        local_combo=_unrank_triple(len(p['extras']),local_index)
        extra_edges=[p['extras'][i] for i in local_combo]
        selected=sorted(list(p['anchors'])+extra_edges)
        subset_row=dict(global_subset_index=global_index,hypothesis_combination_index=hypothesis['combination_index'],
            local_subset_index=local_index,extra_local_indices=local_combo,extra_edge_indices=extra_edges,selected_edge_indices=selected)
        hypothesis['sampled_subsets'].append(dict(subset_row))
        def subset_record(status,**details):
            counts[status]+=1;subset_row.update(status=status,**details);subset_ledger.append(subset_row)
        anchors=p['anchors'];ambiguous=p['ambiguous'];matrix=p['matrix'];translation=p['translation']
        forward=p['forward'];backward=p['backward'];sa=p['sa'];ta=p['ta'];sc=p['sc'];tc=p['tc']
        mapping=tuple(sorted((edges[i]['source_id'],edges[i]['target_id']) for i in selected))
        quad=_area_code_check(np.asarray([source[s] for s,t in mapping]),np.asarray([target[t] for s,t in mapping]))
        if quad['status']!='PASSED':subset_record('AREA_'+quad['status'],area_code=quad);continue
        if mapping in by_mapping:
            candidate=by_mapping[mapping];candidate['supporting_anchor_sets'].append(list(anchors));candidate['supporting_hypotheses']+=1
            candidate['supporting_subset_indices'].append(global_index)
            subset_record('SUPPORTS_EXISTING_MAPPING',candidate_id=candidate['candidate_id']);continue
        if len(candidates)>=128:overflow+=1;subset_record('OUTPUT_CAP_PARTIAL',mapping=[dict(source_id=s,target_id=t) for s,t in mapping]);continue
        cid='candidate_'+str(len(candidates))
        candidate=dict(candidate_id=cid,status='CANDIDATE_NOT_IDENTITY_VERIFIED',
            mapping=[dict(source_id=s,target_id=t) for s,t in mapping],mapping_count=len(mapping),extra_point_count=len(mapping)-3,
            matrix=matrix.tolist(),translation=translation.tolist(),area_code=quad,
            max_forward_transfer_pixels=float(max(forward[i] for i in selected)),max_backward_transfer_pixels=float(max(backward[i] for i in selected)),
            anchor_source_area=sa,anchor_target_area=ta,source_anchor_condition=sc,target_anchor_condition=tc,
            supporting_anchor_sets=[list(anchors)],supporting_hypotheses=1,ambiguous_edge_indices=ambiguous,
            supporting_subset_indices=[global_index],selected_edge_indices=selected,
            unselected_geometric_edge_indices=[i for i in p['unique'] if i not in selected],
            unmatched_source_ids=sorted(set(source)-{s for s,t in mapping}),unmatched_target_ids=sorted(set(target)-{t for s,t in mapping}),
            identity_verified=False,geometry_acceptance=False,native_injected=False)
        candidate.update(uniqueness='NOT_EVALUATED',ambiguity_scope='retained bounded proposal hypotheses only')
        candidates.append(candidate);by_mapping[mapping]=candidate;subset_record('ACCEPTED_MAPPING',candidate_id=cid)
    associations_s={};associations_t={}
    for c in candidates:
        for pair in c['mapping']:
            associations_s.setdefault(pair['source_id'],set()).add(pair['target_id'])
            associations_t.setdefault(pair['target_id'],set()).add(pair['source_id'])
    for c in candidates:
        c['competing_mapping_pairs']=[p for p in c['mapping'] if len(associations_s[p['source_id']])>1 or len(associations_t[p['target_id']])>1]
        c['ambiguous']=bool(c['ambiguous_edge_indices'] or c['competing_mapping_pairs'])
    partial=len(sampled)<total or len(sampled_subset_indices)<subset_total or overflow>0
    return dict(schema='engineering-partial-affine-v2',status='PARTIAL_SEARCH_NOT_EXHAUSTIVE' if partial else 'BOUNDED_PROPOSAL_SEARCH_COMPLETE',
        profile=dict(PROFILE),candidates=candidates,candidate_count=len(candidates),rejection_counts=dict(counts),hypothesis_ledger=ledger,
        source_point_ids=list(source),target_point_ids=list(target),proposal_edges=edges,total_anchor_combinations=total,
        max_hypotheses=max_hypotheses,sampled_combination_indices=sampled,
        sampled_combinations=[list(universe[i]) for i in sampled],hypotheses_evaluated=len(sampled),output_overflow_count=overflow,
        total_subsets=subset_total,checked_subsets=len(sampled_subset_indices),sampled_subset_indices=sampled_subset_indices,
        subset_ledger=subset_ledger,subset_domain='fixed six-point mappings only; not all subset cardinalities',
        proposal_universe_exhausted=not partial,absence_claim=False,identity_verified=False,geometry_acceptance=False,native_injected=False,
        uniqueness='NOT_EVALUATED',ambiguity_scope='retained bounded proposal hypotheses only',
        limitations=['Float local-affine approximation; arbitrary perspective is outside the model.',
            'Three anchors fit only; at least three additional real points and all ordered triangle-area ratios are checked; a nonplanar 3D match need not pass.',
            'Unmatched points do not prove occlusion or newly visible surface; descriptor proposals are not identity truth.',
            'Completeness is only over supplied proposal-edge triples and fixed six-point subsets, never all possible correspondences.'])
