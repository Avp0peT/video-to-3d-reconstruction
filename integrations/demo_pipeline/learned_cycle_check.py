"""Prepare a frozen reverse TAPNext++ run; assess heuristic cycle consistency.

Preparation never imports torch. Model execution belongs to the existing bounded
supervisor and learned_point_run.worker, without learned_point_report.render.
"""
from copy import deepcopy
from pathlib import Path
from collections import Counter
import argparse
import math

from .balanced_point_run import read, save, binding, check_bindings
from . import learned_point_run

VISIBLE = 'VISIBLE_CANDIDATE'
FIRST, ENDPOINT = 555, 600


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate ' + key)
    return result


def prepare(source, output):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError('fresh output directory required')
    plan = deepcopy(learned_point_run.check(source))
    inputs, masks, tracks, queries = [read(source / name) for name in
                                    ('INPUTS.json', 'MASKS.json', 'TRACKS.json', 'QUERIES.json')]
    if plan['dimensions'] != [576, 1024]:
        raise ValueError('native 576x1024 domain required')
    frames = indexed(inputs['frames'], 'decode_index')
    mask_rows = indexed(masks['frames'], 'decode_index')
    forward = indexed(tracks['frames'], 'decode_index')
    original = indexed(queries['points'], 'track_id')
    ids = list(range(ENDPOINT, FIRST - 1, -1))
    if not all(i in frames and i in mask_rows and i in forward for i in ids):
        raise ValueError('actual complete raw555..600 source interval required')
    candidates = [p for p in forward[ENDPOINT]['points'] if p['status'] == VISIBLE]
    indexed(candidates, 'track_id')
    if not candidates or len(candidates) > 512:
        raise ValueError('bounded nonempty endpoint candidates required')
    points = []
    for point in candidates:
        if point['track_id'] not in original or not all(math.isfinite(x) for x in point['xy']):
            raise ValueError('invalid endpoint query')
        points.append(dict(track_id=point['track_id'], xy=point['xy'],
                           identity=original[point['track_id']].get('identity'),
                           source_uncertainty='forward predicted candidate; physical identity unverified'))
    refs = [binding(source / n) for n in ('PLAN.json', 'SOURCE_BINDINGS.json',
            'INPUTS.json', 'MASKS.json', 'TRACKS.json', 'QUERIES.json')]
    for i in ids:
        f, m = frames[i], mask_rows[i]
        ref = binding(f['path'])
        if ref['sha256'] != f['sha256'] or m['raw_sha256'] != f['sha256'] or Path(m['raw_path']).resolve() != Path(f['path']).resolve():
            raise ValueError('frame/mask binding mismatch')
        refs.append(ref)
        if m['current_observed']:
            ref = binding(m['mask_path'])
            if ref['sha256'] != m['mask_sha256']:
                raise ValueError('mask drift')
            refs.append(ref)
    if not mask_rows[ENDPOINT]['current_observed']:
        raise ValueError('endpoint actual mask required')
    new_inputs, new_masks = deepcopy(inputs), deepcopy(masks)
    new_inputs['frames'] = [frames[i] for i in ids]
    new_inputs['seed'] = dict(points[0], decode_index=ENDPOINT,
                             time_seconds=frames[ENDPOINT]['time_seconds'])
    new_masks['frames'] = [mask_rows[i] for i in ids]
    plan.update(query_count=len(points), frames=len(ids))
    plan['implementation'] += [binding(__file__)]
    provenance = dict(schema='learned-cycle-v1', source=str(source), first_raw=FIRST,
        endpoint_raw=ENDPOINT, endpoint_selection='raw600 fixed before reverse inference',
        selection='all forward VISIBLE_CANDIDATE at raw600, original IDs retained',
        coordinate_domain=[576, 1024], physical_identity_verified=False,
        threshold_pixels=4.0, threshold_status='DECLARED_HEURISTIC_NOT_GROUND_TRUTH',
        interpolation=False, frame_order=ids)
    if check_bindings(refs):
        raise ValueError('source drift during preparation')
    output.mkdir(parents=True)
    for name, value in [('INPUTS.json', new_inputs), ('MASKS.json', new_masks),
                        ('QUERIES.json', dict(points=points)), ('PLAN.json', plan),
                        ('CYCLE_PLAN.json', provenance)]:
        save(output / name, value)
    refs += [binding(output / n) for n in ('INPUTS.json', 'MASKS.json', 'QUERIES.json', 'CYCLE_PLAN.json')]
    save(output / 'SOURCE_BINDINGS.json', refs)
    save(output / 'RESULT.json', dict(status='PREPARED', quality_status='NOT_EVALUATED', geometry_accepted=False))
    learned_point_run.check(output)
    return provenance


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def statistics(values):
    values = sorted(values)
    def percentile(p):
        if not values:
            return None
        pos = (len(values) - 1) * p
        low, high = math.floor(pos), math.ceil(pos)
        return values[low] + (values[high] - values[low]) * (pos - low)
    return dict(count=len(values), p50=percentile(.5), p90=percentile(.9),
                p95=percentile(.95), maximum=max(values) if values else None)


def assess_data(forward, reverse, queries, selected_ids, first=FIRST, endpoint=ENDPOINT):
    frows, rrows = indexed(forward, 'decode_index'), indexed(reverse, 'decode_index')
    qmap = indexed(queries, 'track_id')
    per_track, all_values = [], []
    for track_id in selected_ids:
        rows, values = [], []
        for raw in range(first, endpoint + 1):
            f = indexed(frows.get(raw, {}).get('points', []), 'track_id').get(track_id)
            r = indexed(rrows.get(raw, {}).get('points', []), 'track_id').get(track_id)
            error = None
            if f and r and f['status'] == r['status'] == VISIBLE:
                error = distance(f['xy'], r['xy'])
                if not math.isfinite(error):
                    raise ValueError('nonfinite visible coordinate')
                values.append(error)
            rows.append(dict(decode_index=raw, forward_status=f.get('status') if f else 'MISSING',
                reverse_status=r.get('status') if r else 'MISSING', discrepancy_pixels=error,
                status='UNKNOWN' if error is None else ('PASS_HEURISTIC' if error <= 4 else 'FAIL_HEURISTIC')))
        start = rows[0]
        r = indexed(rrows.get(first, {}).get('points', []), 'track_id').get(track_id)
        # Original query differs from the forward model's prediction at raw555.
        query_error = distance(r['xy'], qmap[track_id]['xy']) if start['discrepancy_pixels'] is not None else None
        per_track.append(dict(track_id=track_id, rows=rows, statistics=statistics(values),
            return_error_vs_forward_first_pixels=start['discrepancy_pixels'],
            return_error_vs_original_query_pixels=query_error,
            endpoint_discrepancy_pixels=rows[-1]['discrepancy_pixels'],
            status='UNKNOWN' if any(row['status'] == 'UNKNOWN' for row in rows) else
                   ('PASS_HEURISTIC' if max(values) <= 4 else 'FAIL_HEURISTIC')))
        all_values.extend(values)
    counts = dict(Counter(row['status'] for track in per_track for row in track['rows']))
    return dict(schema='learned-cycle-assessment-v1', threshold_pixels=4,
        threshold_status='DECLARED_HEURISTIC_NOT_GROUND_TRUTH', native_dimensions=[576,1024],
        physical_identity_verified=False, geometry_accepted=False, interpolation=False,
        all_queries=statistics(all_values), comparison_status_counts=counts,
        track_status_counts=dict(Counter(track['status'] for track in per_track)),
        selected_query_count=len(selected_ids), first_raw=first, endpoint_raw=endpoint, tracks=per_track,
        track_id_0=next((p for p in per_track if p['track_id'] == 0),
                        dict(track_id=0, status='NOT_SELECTED_AT_ENDPOINT')))


def assess(output):
    output = Path(output).resolve(strict=True)
    learned_point_run.check(output)
    cycle = read(output / 'CYCLE_PLAN.json')
    source = Path(cycle['source'])
    reverse = read(output/'TRACKS.json')['frames']
    expected = read(output/'INPUTS.json')['frames']
    selected = [p['track_id'] for p in read(output/'QUERIES.json')['points']]
    if [row['decode_index'] for row in reverse] != [row['decode_index'] for row in expected[:len(reverse)]]:
        raise ValueError('reverse output frame order/identity mismatch')
    for row, frame in zip(reverse, expected):
        if row['time_seconds'] != frame['time_seconds'] or set(indexed(row['points'], 'track_id')) != set(selected):
            raise ValueError('reverse output timestamp/query identity mismatch')
    result = assess_data(read(source/'TRACKS.json')['frames'], reverse,
                         read(source/'QUERIES.json')['points'],
                         selected)
    result['bindings'] = [binding(p) for p in (source/'TRACKS.json', output/'TRACKS.json', output/'CYCLE_PLAN.json')]
    save(output/'CYCLE_ASSESSMENT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'assess'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.source is None:
            parser.error('prepare requires --source')
        prepare(args.source, args.output)
    else:
        assess(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
