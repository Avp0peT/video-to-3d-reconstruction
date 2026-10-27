"""Bounded measured region growth as part of the default point-review run."""
from collections import Counter
from pathlib import Path
import json
import time

from . import temporal_review as inputs
from .region_tracking import build_region_plan, measure_regions
from .region_growth import next_region_plan
from .region_tracking_report import render_regions
from .area_review import review_area_sequence
from .overlap_review import review_overlap_sequence


def _write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def review_regions(ledger, frames, masks, decoded, output, refs, cv, *, budget=lambda: None):
    budget()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rounds = []
    artifacts = []
    result = dict(status='RUNNING', regions=[], counts={}, flow_calls=0, rounds=[],
        geometry_accepted=False, identity_verified=False, new_cameras=0, new_model_points=0,
        segmentation_calls=0, coverage_scope='source image sampling only, not 3D surface coverage')

    def verify():
        budget()
        for row in refs:
            if inputs.fingerprint(row['path']) != row:
                raise ValueError('region growth source drift: '+row['path'])

    try:
        verify()
        plan = build_region_plan(ledger, frames, masks, decoded, budget=budget)
        source_plan = plan
        for level in range(3):
            budget()
            plan_path = output/f'PLAN_{level:02d}.json'
            _write(plan_path, plan); artifacts.append(plan_path)
            if not plan['regions']:
                break
            measured = measure_regions(plan, decoded, cv, budget=budget)
            record_path = output/f'MEASURED_{level:02d}.json'
            _write(record_path, measured); artifacts.append(record_path)
            rounds.append(measured)
            result['regions'].extend(measured['regions'])
            result['flow_calls'] += measured['flow_calls']
            result['rounds'].append(dict(level=level, regions=len(measured['regions']),
                points=sum(len(r['points']) for r in measured['regions']),
                counts=measured['counts'], flow_calls=measured['flow_calls'], record=record_path.name))
            if len(result['regions']) > 24:
                raise ValueError('GLOBAL_REGION_BOUND_24')
            plan = next_region_plan(plan, rounds, budget=budget)
        # This frontier is produced from measured parent regions, never fake success.
        result['growth'] = {k: v for k, v in plan.items() if k not in ('frame_domains', 'regions')}
        result['unexecuted_frontier_count'] = len(plan['regions'])
        result['counts'] = dict(Counter(p['status'] for r in result['regions'] for p in r['points']))
        result['area_alignment'] = review_area_sequence(source_plan, decoded, output/'whole_area', refs, cv,
            point_baseline=rounds[0]['regions'] if rounds else (), budget=budget)
        artifacts += [Path(b['path']) for b in result['area_alignment']['source_bindings']]
        result['overlap_regions'] = review_overlap_sequence(source_plan, decoded, output/'overlap', refs, cv,
            baseline_path=output/'whole_area/RESULT.json', budget=budget)
        artifacts += [Path(b['path']) for b in result['overlap_regions']['source_bindings']]
        result['status'] = 'REGION_GROWTH_REVIEW_COMPLETE'
        result['source_unchanged'] = True
        result['measurement_wall_seconds'] = time.monotonic()-started
        verify()
        artifacts += render_regions(output, result, decoded, budget=budget)
        result['wall_seconds'] = time.monotonic()-started
        _write(output/'RESULT.json', result); artifacts.append(output/'RESULT.json')
        verify()
        bindings = [inputs.fingerprint(p) for p in artifacts]
        return dict(status=result['status'], regions=len(result['regions']),
            counts=result['counts'], flow_calls=result['flow_calls'], rounds=result['rounds'],
            area_alignment=result['area_alignment'],
            overlap_regions=result['overlap_regions'],
            wall_seconds=result['wall_seconds'], geometry_accepted=False, new_model_points=0,
            report='intermediate_frames/population/regions/index.html', source_bindings=bindings)
    except Exception as exc:
        result.update(status='RESOURCE_LIMIT' if isinstance(exc, (TimeoutError, MemoryError)) else 'FAILED',
            reason=f'{type(exc).__name__}: {exc}', source_unchanged=None,
            wall_seconds=time.monotonic()-started)
        if getattr(exc, 'partial_result', None) is not None:
            result['interrupted_measurement'] = exc.partial_result
        try:
            _write(output/'FAILURE.json', result)
        except Exception as delivery:
            if hasattr(exc, 'add_note'):
                exc.add_note('Region failure evidence delivery: '+str(delivery))
        raise
