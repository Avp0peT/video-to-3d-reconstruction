"""Frozen-F checks of already enumerated competing edges, never identity fusion."""
from collections import Counter, defaultdict
import copy
import json

from kozlov_lab.stereo.multiview_cycles import analyze_candidate_graph

SUPPORTED = 'EXISTING_F_SUPPORT_CANDIDATE'
REJECTED = 'GEOMETRY_DISAGREES'


def _edge_key(frames, source, target):
    if len(frames) != 2 or any(type(f) is not int or f < 0 for f in frames) or frames[0] >= frames[1]:
        raise ValueError('canonical distinct frame identities required')
    if not all(isinstance(s, str) and s for s in (source, target)):
        raise ValueError('nonempty native feature identities required')
    return frames[0], frames[1], source, target


def _as_graph(key):
    a, b, source, target = key
    return dict(edge_id=json.dumps([[a, source], [b, target]], separators=(',', ':')),
                a=dict(frame=a, feature_id=source), b=dict(frame=b, feature_id=target))


def _new_checks(row, edges, geometry, geometry_check):
    """Bind callback results exactly to requested endpoint identities."""
    checks = geometry_check(dict(mapping=edges), *row['frames'], row['source_points'],
                            row['target_points'], geometry,
                            kind='ENUMERATED_COMPETITOR_NOT_IDENTITY_VERIFIED')
    found = {}
    for check in checks['links']:
        key = (check['from_track_id'], check['to_track_id'])
        if key in found:
            raise ValueError('duplicate competitor geometry result')
        found[key] = check
    expected = {(e['source_id'], e['target_id']) for e in edges}
    if set(found) != expected:
        raise ValueError('competitor geometry identity coverage differs')
    return found


def assess_competitors(closures, pair_rows, supported_edges, geometry, *, geometry_check,
                       max_new_edges=256):
    """Evaluate omitted competitors only; reuse retained checks without refitting.

    Every occurrence and every unique edge is counted separately. Exhausting a
    budget, missing F, dependent endpoints or a failed query can never exclude a
    competitor. A negative F result is conditional disagreement, not truth.
    """
    if type(max_new_edges) is not int or not 1 <= max_new_edges <= 256:
        raise ValueError('max_new_edges must be 1..256')
    pair_map = {}
    for row in pair_rows:
        key = tuple(row['frames'])
        if key in pair_map:
            raise ValueError('duplicate competitor source pair')
        pair_map[key] = row
    occurrences = []; records = {}; case_edges = defaultdict(set)
    for case in closures:
        for group in case['competition']:
            frames = group['frames']; row = pair_map[tuple(frames)]
            retained = {(e['source_id'], e['target_id']): e for e in row['proposal_edges']}
            omitted = {(e['source_id'], e['target_id']): e for e in row['proposal_receipt']['omitted_edges']}
            if set(retained) & set(omitted):
                raise ValueError('retained and omitted competitor identities overlap')
            old_checks = {(c['from_track_id'], c['to_track_id']): c for c in row['proposal_geometry']['links']}
            for alternative in group['alternatives']:
                key = _edge_key(frames, alternative['source_id'], alternative['target_id'])
                local = key[2:]
                if local not in retained and local not in omitted:
                    raise ValueError('competitor is absent from bound descriptor proposals')
                if alternative != (retained.get(local) or omitted[local]):
                    raise ValueError('competitor descriptor record differs from source')
                case_edges[case['wedge_index']].add(key)
                occurrences.append(dict(wedge_index=case['wedge_index'], key=list(key)))
                if key in records:
                    continue
                old = old_checks.get(local) if local in retained else None
                records[key] = dict(key=list(key), edge_id=_as_graph(key)['edge_id'],
                    descriptor=copy.deepcopy(alternative), origin='retained' if local in retained else 'omitted',
                    status=old['status'] if old is not None else 'NOT_CHECKED_BUDGET',
                    geometry=copy.deepcopy(old), check_source='reused_retained' if old is not None else 'pending',
                    identity_verified=False, eligible_for_triangulation=False)
    pending = sorted(k for k, r in records.items() if r['geometry'] is None)
    selected = (pending if len(pending) <= max_new_edges else
                [pending[i*(len(pending)-1)//(max_new_edges-1)] for i in range(max_new_edges)]
                if max_new_edges > 1 else pending[:1])
    by_pair = defaultdict(list)
    for key in selected:
        by_pair[key[:2]].append(key)
    for frames, keys in sorted(by_pair.items()):
        row = pair_map[frames]
        try:
            checks = _new_checks(row, [records[k]['descriptor'] for k in keys], geometry, geometry_check)
            for key in keys:
                check = checks[key[2:]]
                records[key].update(status=check['status'], geometry=copy.deepcopy(check), check_source='new_original_F_check')
        except Exception as exc:
            for key in keys:
                records[key].update(status='GEOMETRY_EVALUATION_FAILED', check_source='failed',
                    reason=f'{type(exc).__name__}: {exc}')
    augmented = {e['edge_id']: copy.deepcopy(e) for e in supported_edges}
    for key, row in records.items():
        if row['status'] == SUPPORTED:
            edge = _as_graph(key); augmented[edge['edge_id']] = edge
    graph = analyze_candidate_graph(list(augmented.values()))
    conflicts = {(n['frame'], n['feature_id']): c['component_conflict']
                 for c in graph['components'] for n in c['nodes']}
    cases = []
    for case in closures:
        keys = sorted(case_edges[case['wedge_index']])
        statuses = Counter(records[k]['status'] for k in keys)
        nodes = [case['endpoint_a'], case['center'], case['endpoint_b']]
        conflict = any(conflicts.get((n['frame'], n['feature_id']), False) for n in nodes)
        unresolved = sum(v for k, v in statuses.items() if k not in (SUPPORTED, REJECTED))
        if not case['descriptor_cycle_closed']:
            resolution = 'NOT_APPLICABLE_NO_CLOSED_CYCLE'
        elif case['direct_geometry']['status'] != SUPPORTED:
            resolution = 'CYCLE_LACKS_THIRD_EDGE_GEOMETRY_SUPPORT'
        elif conflict:
            resolution = 'SUPPORTED_GRAPH_COMPONENT_CONFLICT'
        elif statuses[SUPPORTED]:
            resolution = 'GEOMETRY_SUPPORTED_COMPETITOR'
        elif unresolved:
            resolution = 'COMPETITION_UNRESOLVED'
        else:
            resolution = 'NO_SURVIVING_COMPETITOR_IN_TESTED_SCOPE'
        cases.append(dict(wedge_index=case['wedge_index'], status=resolution,
            unique_competitor_keys=[list(k) for k in keys], geometry_counts=dict(statuses),
            unresolved_competitors=unresolved, component_conflict=conflict,
            identity_verified=False, uniqueness='NOT_PROVEN', eligible_for_triangulation=False))
    return dict(schema='enumerated-competitor-geometry-v1', status='COMPETITOR_DIAGNOSTIC_COMPLETE',
        max_new_edges=max_new_edges, unique_competitors=[records[k] for k in sorted(records)],
        occurrences=occurrences, cases=cases, supported_graph=graph,
        counts=dict(occurrences=len(occurrences), unique_competitors=len(records),
            reused_checks=sum(r['check_source']=='reused_retained' for r in records.values()),
            requested_new_checks=len(pending), attempted_new_checks=len(selected),
            completed_new_checks=sum(r['check_source']=='new_original_F_check' for r in records.values()),
            budget_omitted=len(pending)-len(selected),
            geometry=dict(Counter(r['status'] for r in records.values())),
            case_resolution=dict(Counter(c['status'] for c in cases))),
        new_check_selection=dict(selected=[list(k) for k in selected], omitted=[list(k) for k in pending if k not in selected]),
        exhaustive_correspondence_search=False, geometry_refit=False, identity_verified=False,
        native_injected=False, observations_synthesized=False, eligible_for_triangulation=False,
        interpretation='F disagreement excludes only under the frozen model and threshold; unsupported is not physically disproved. Feature/pair sampling outside this scope remains unknown.')
