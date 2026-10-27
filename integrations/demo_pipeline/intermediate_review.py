"""Automatically diagnose failed sampled edges using original intermediate frames."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from . import temporal_review as inputs

PROFILE = dict(id='automatic-intermediate-review-v2', max_edges=2, max_points_per_edge=64,
               max_decode_gap=8, minimum_failed_fraction=.5, max_extracted_frames=18,
               selection='first measured route edges; unique groups; failure count descending',
               geometry_accepted=False, identity_verified=False, new_segmentation=False)


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def plan_intervals(frames, extension):
    """Freeze failure-conditioned edges before decoding or either comparison arm.

    Only original source observations are used. Later route steps would use an
    already drifted point and therefore cannot define the matched source cohort.
    Repeated use of a source group by different query routes counts once.
    """
    if extension.get('status') != 'VIEW_EXTENSION_LEDGER_COMPLETE':
        raise ValueError('completed view extension required')
    by_index = {f['input_index']: f for f in frames}
    if len(by_index) != len(frames):
        raise ValueError('duplicate frame identity')
    bins = {}
    for query in extension['query_rows']:
        for point in query['points']:
            for route in point.get('routes', []):
                steps = route.get('steps', [])
                if not steps:
                    continue
                step = steps[0]
                a, b = step['source_input_index'], step['target_input_index']
                if a != route['source_input_index'] or route['path_input_indices'][:2] != [a, b]:
                    raise ValueError('route source identity differs')
                if step['source_raw_xy'] != route['source_raw_xy']:
                    raise ValueError('route source coordinates differ')
                key = (query['model_key'], a, b)
                row = bins.setdefault(key, dict(model_key=key[0], source_input_index=a,
                    target_input_index=b, points={}))
                sample = dict(group_id=point['group_id'], source_raw_xy=deepcopy(route['source_raw_xy']),
                    prior_step_status=step['status'], prior_reason=step.get('reason'))
                previous = row['points'].setdefault(point['group_id'], sample)
                if previous != sample:
                    raise ValueError('inconsistent repeated source measurement')
    ledger = []
    for key, row in sorted(bins.items()):
        a, b = by_index[row['source_input_index']], by_index[row['target_input_index']]
        row['points'] = sorted(row['points'].values(), key=lambda p: p['group_id'])
        row.update(source_decode_index=a['decode_index'], target_decode_index=b['decode_index'],
                   source_frame_id=a['frame_id'], target_frame_id=b['frame_id'])
        gap = abs(a['decode_index']-b['decode_index'])
        failures = sum(p['prior_step_status'] != 'PASSED' for p in row['points'])
        row.update(prior_unique_points=len(row['points']), prior_failed_points=failures,
                   failure_fraction=failures/len(row['points']), status='ELIGIBLE')
        if not 2 <= gap <= PROFILE['max_decode_gap']:
            row['status'] = 'NO_SUPPORTED_ORIGINAL_GAP'
        elif a['original_mask_status'] != 'OK' or b['original_mask_status'] != 'OK':
            row['status'] = 'NO_ORIGINAL_ENDPOINT_MASKS'
        elif row['failure_fraction'] < PROFILE['minimum_failed_fraction']:
            row['status'] = 'NO_MASS_FAILURE'
        direction = 1 if b['decode_index'] > a['decode_index'] else -1
        row['decode_path'] = list(range(a['decode_index'], b['decode_index']+direction, direction))
        ledger.append(row)
    eligible = sorted((r for r in ledger if r['status'] == 'ELIGIBLE'),
                      key=lambda r: (-r['prior_failed_points'], r['model_key'], r['source_input_index'], r['target_input_index']))
    edges = []
    for i, row in enumerate(eligible):
        row['status'] = 'SELECTED' if i < PROFILE['max_edges'] else 'NOT_EVALUATED_EDGE_BUDGET'
        if row['status'] == 'SELECTED':
            edge = deepcopy(row)
            edge['not_evaluated_point_ids'] = [p['group_id'] for p in edge['points'][PROFILE['max_points_per_edge']:]]
            edge['points'] = edge['points'][:PROFILE['max_points_per_edge']]
            edges.append(edge)
    indices = sorted({i for edge in edges for i in edge['decode_path']})
    if len(indices) > PROFILE['max_extracted_frames']:
        raise ValueError('extraction count outside frozen bound')
    return dict(profile=deepcopy(PROFILE), edges=edges, edge_ledger=ledger, decode_indices=indices,
                selection_condition='retrospective failure-conditioned diagnostic; not population accuracy',
                source_points_changed=False, thresholds_changed=False, geometry_accepted=False)


def review_intermediates(discovery, frames, masks, extension, output, refs, cv, *, deadline, budget=lambda: None, tracking=None, camera_result=None, classification=None):
    from .intermediate_decode import decode_intermediates
    from .intermediate_tracking import compare_edge
    from .intermediate_report import render_report
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = plan_intervals(frames, extension)
    save(output/'PLAN.json', plan)
    population_plan = None
    anchor_plan = None
    extraction_indices = list(plan['decode_indices'])
    if tracking is not None:
        from .intermediate_population import build_population_plan
        population_plan = build_population_plan(frames,masks,tracking,plan)
        save(output/'POPULATION_PLAN.json',population_plan)
        refs.append(inputs.fingerprint(output/'POPULATION_PLAN.json'))
        from .anchor_route_tracking import plan_anchor_routes
        anchor_plan = plan_anchor_routes(population_plan, frames, camera_result or {})
        save(output/'ANCHOR_ROUTE_PLAN.json', anchor_plan)
        refs.append(inputs.fingerprint(output/'ANCHOR_ROUTE_PLAN.json'))
        extraction_indices = sorted(set(extraction_indices) | set(anchor_plan['decode_indices']))
        if len(extraction_indices) > 36:
            raise ValueError('combined original and anchor-route extraction exceeds 36 frames')
    result = dict(status='RUNNING', profile=deepcopy(PROFILE), plan=plan, edges=[], extraction=None,
                  geometry_accepted=False, identity_verified=False, new_segmentation=False,
                  new_cameras=0, new_model_points=0, source_observations_modified=False,
                  flow_calls=0, counts={}, intermediate_masks_verified=False)
    event_path = output/'EVENTS.jsonl'
    event_digest = hashlib.sha256()
    sequence = 0
    decoded = {}
    def event(value):
        nonlocal sequence
        data = (json.dumps(dict(seq=sequence, monotonic=time.monotonic(), payload=value),
                ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
        with event_path.open('ab') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        event_digest.update(data); sequence += 1
    def verify():
        for ref in refs:
            if inputs.fingerprint(ref['path']) != ref:
                raise ValueError('intermediate review source drift: '+ref['path'])
    event(dict(event='PLAN', plan=plan))
    try:
        if plan['edges']:
            verify(); budget()
            decoded, result['extraction'] = decode_intermediates(discovery, frames, extraction_indices,
                output/'extract', refs, deadline=deadline, budget=budget)
            save(output/'EXTRACTION.json', result['extraction'])
            originals = {f['decode_index']: (f,m) for f,m in zip(frames,masks)}
            for edge in plan['edges']:
                # Per-edge copies: an endpoint of another selected interval may
                # not become an asserted original mask in this interval's interior.
                domain = {i: dict(decoded[i]) for i in edge['decode_path']}
                for i in (edge['source_decode_index'],edge['target_decode_index']):
                    f,mask = originals[i]
                    if not domain[i]['endpoint_verified'] or mask is None:
                        raise ValueError('original endpoint was not verified')
                    domain[i].update(mask=mask, original_mask_status=f['original_mask_status'])
                event(dict(event='EDGE_BEGIN', edge=edge))
                try:
                    measured = compare_edge(edge, domain, cv, budget=budget,
                        progress=lambda value: event(dict(event='POINT_PROGRESS', measurement=value)))
                except Exception as exc:
                    if hasattr(exc,'partial_result'):
                        result['edges'].append(exc.partial_result)
                    raise
                result['edges'].append(measured)
                result['flow_calls'] += measured['flow_calls']
                event(dict(event='EDGE_END', measurement=measured))
            verify(); budget()
            result['status'] = 'INTERMEDIATE_COMPARISON_COMPLETE'
        else:
            result['status'] = 'NOT_APPLICABLE_NO_ELIGIBLE_INTERVALS'
        result['counts'] = dict(Counter({key:sum(e['counts'].get(key,0) for e in result['edges'])
                           for key in ('RECOVERED','LOST','BOTH_PASS','BOTH_FAIL')}))
        event(dict(event='COMPLETE', status=result['status'], counts=result['counts']))
        for edge in plan['edges']:
            for i in (edge['source_decode_index'],edge['target_decode_index']):
                f,mask = originals[i]
                decoded[i].update(mask=mask, original_mask_status=f['original_mask_status'])
        if population_plan is not None:
            from .intermediate_population_review import review_population
            result['population'] = review_population(population_plan,frames,masks,decoded,
                output/'population',refs,cv,budget=budget,camera_result=camera_result,
                classification=classification,anchor_plan=anchor_plan)
            refs.extend(result['population']['source_bindings'])
        artifacts = render_report(output, result, decoded, budget=budget)
        verify(); budget()
        if hashlib.sha256(event_path.read_bytes()).hexdigest() != event_digest.hexdigest():
            raise ValueError('intermediate event evidence drift')
        result['source_unchanged'] = True
        save(output/'RESULT.json', result)
        paths = [output/'PLAN.json',event_path,output/'RESULT.json'] + list(artifacts)
        if (output/'EXTRACTION.json').is_file(): paths.append(output/'EXTRACTION.json')
        if decoded: paths += [Path(item['path']) for item in decoded.values()]
        bindings = [inputs.fingerprint(p) for p in dict.fromkeys(paths)]
        return dict(status=result['status'], counts=result['counts'], flow_calls=result['flow_calls'],
                    edges=len(plan['edges']), original_frames_decoded=len(decoded),
                    new_intermediate_frames=sum(i not in {f['decode_index'] for f in frames} for i in decoded),
                    population=result.get('population'),
                    geometry_accepted=False, intermediate_masks_verified=False,
                    report='intermediate_frames/index.html', source_bindings=bindings)
    except Exception as exc:
        result['flow_calls'] = sum(e.get('flow_calls',0) for e in result['edges'])
        result['counts'] = dict(Counter(r.get('outcome','INCOMPLETE')
            for e in result['edges'] for r in e.get('rows',[])))
        result.update(status='RESOURCE_LIMIT' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED',
                      reason=f'{type(exc).__name__}: {exc}', source_unchanged=None)
        try:
            save(output/'FAILURE.json', result)
        except Exception as delivery_error:
            if hasattr(exc,'add_note'):
                exc.add_note('Failure evidence could not be written: '+str(delivery_error))
        raise
