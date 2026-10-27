"""Freeze complete native-frame propagation coverage without running a model."""
from pathlib import Path

from .planning import _validated_evidence, _read, _hash, _require, _integer


def _reuse(run, bindings):
    """Revalidate the sealed batch, including pixels and every committed output."""
    if run is None:
        return {}, {}
    from PIL import Image
    run = Path(run).resolve()
    inputs = {}

    def read(path):
        path = Path(path).resolve()
        value = _read(path)
        inputs[str(path)] = _hash(path)
        return value

    plan, result = read(run / 'PLAN.json'), read(run / 'RESULT.json')
    _require(result.get('status') == 'REVIEW_READY' and not result.get('input_drift')
             and not result.get('code_drift'), 'reuse batch not review ready or drifted')
    for key, value in bindings.items():
        _require(plan.get(key) == value, f'reuse identity mismatch: {key}')
    for row in plan['coverage_ledger']:
        with Image.open(row['source_image']) as image:
            image.load()
            _require(image.format == 'PNG' and image.size == (row['width'], row['height']),
                     'reuse sampled PNG metadata mismatch')
        if row['original_status'] == 'OK':
            with Image.open(row['mask_path']) as image:
                image.load()
                _require(image.format == 'PNG' and image.mode == 'L'
                         and image.size == (row['width'], row['height'])
                         and sum(image.histogram()[1:255]) == 0, 'reuse original mask metadata mismatch')
    frozen = read(run / 'START_BINDINGS.json')['input_files']
    timeline = _read(Path(bindings['timeline_path']))
    for path, expected in frozen.items():
        _require(_hash(Path(path)) == expected, f'reuse input hash mismatch: {path}')
        inputs[path] = expected
    worker = read(run / 'worker/result.json')
    _require(worker['status'] == 'PROPAGATED_NOT_SEMANTICALLY_EVALUATED', 'reuse worker incomplete')
    clips = {r['clip_id']: r for r in worker['clips']}
    reused = {}
    for interval in plan['intervals']:
        clip_id = interval['interval_id']
        base = run / 'clips' / clip_id
        clip_plan = read(base / 'PLAN.json')
        _require(clip_plan == interval, 'reuse clip PLAN mismatch')
        clip = read(base / 'worker/result.json')
        _require(clip == clips[clip_id], 'reuse worker clip mismatch')
        count = interval['frame_count']
        _require(clip['status'] == worker['status'] and clip['frame_count'] == count,
                 'reuse clip incomplete')
        decoded = read(base / 'DECODE.json')['frames']
        _require(len(decoded) == count, 'reuse decoded coverage mismatch')
        for i, row in enumerate(decoded):
            _require(row['local_frame_index'] == i and row['decode_index'] == interval['start_decode_index'] + i,
                     'reuse decoded index mismatch')
            path = Path(row['path'])
            _require(_hash(path) == row['sha256'], 'reuse frame hash mismatch')
            inputs[str(path)] = row['sha256']
            with Image.open(path) as image:
                image.load()
                _require(image.size == (interval['width'], interval['height']), 'reuse frame dimensions mismatch')
            native = base / 'decoded' / f'{i:06d}.png'
            _require(_hash(native) == row['decoded_sha256'], 'reuse decoded PNG hash mismatch')
            inputs[str(native)] = row['decoded_sha256']
            with Image.open(native) as image:
                image.load()
                _require(image.format == 'PNG' and image.size == (timeline['width'], timeline['height']),
                         'reuse decoded dimensions mismatch')
        frames = []
        for direction in ('forward', 'reverse'):
            direct = read(base / 'worker' / direction / 'result.json')
            _require(direct['status'] == worker['status'] and direct['frame_count'] == count,
                     'reuse direction incomplete')
            rows = [r for r in clip['frames'] if r['direction'] == direction]
            _require({r['frame_id'] for r in rows} == set(range(count)) and len(rows) == count,
                     'reuse committed coverage mismatch')
            _require(direct['frames'] == rows, 'reuse direction commits mismatch')
            for row in rows:
                path = Path(row['mask_path'])
                _require(row['status'] == 'propagated' and _hash(path) == row['mask_sha256'],
                         'reuse mask hash mismatch')
                inputs[str(path)] = row['mask_sha256']
                with Image.open(path) as image:
                    image.load()
                    _require(image.format == 'PNG' and image.mode == 'L'
                             and image.size == (row['width'], row['height'])
                             == (interval['width'], interval['height']), 'reuse mask metadata mismatch')
                    histogram = image.histogram()
                    _require(sum(histogram[1:255]) == 0 and histogram[255] == row['foreground_pixels'],
                             'reuse mask pixels mismatch')
                frames.append(dict(row, decode_index=interval['start_decode_index'] + row['frame_id']))
        bounds = (interval['start_decode_index'], interval['end_decode_index'])
        _require(bounds not in reused, 'duplicate reuse interval')
        reused[bounds] = dict(run=str(run), interval_id=clip_id, frames=frames,
                              plan_path=str(base / 'PLAN.json'), worker_result_path=str(base / 'worker/result.json'))
    return reused, inputs


def make_full_plan(pipeline, reuse_run=None, max_frames=96, overlap=16):
    _integer(max_frames, 'max_frames', 2)
    _require(max_frames <= 96, 'max_frames exceeds 96')
    _integer(overlap, 'overlap', 1)
    _require(overlap < max_frames, 'overlap must be less than max_frames')
    ordered, bindings = _validated_evidence(pipeline)
    timeline = _read(Path(bindings['timeline_path']))
    indexes = [r['decode_index'] for r in timeline['frames']]
    count = len(indexes)
    _require(1 <= count <= 4096 and indexes == list(range(count)), 'timeline must be contiguous 0..N-1, N<=4096')
    anchors = [r for r in ordered if r['original_status'] == 'OK']
    _require(bool(anchors), 'complete plan requires at least one original OK anchor')
    _require(all(0 <= r['decode_index'] < count for r in ordered), 'sample outside full timeline')
    _require(len({(r['width'], r['height']) for r in ordered}) == 1, 'source dimensions differ')
    reused, reuse_inputs = _reuse(reuse_run, bindings)
    specs = []
    if anchors[0]['decode_index'] > 0:
        specs.append((0, anchors[0]['decode_index'], 'head', [('reverse', anchors[0])]))
    specs.extend((a['decode_index'], b['decode_index'], 'between_anchors', [('forward', a), ('reverse', b)])
                 for a, b in zip(anchors, anchors[1:]))
    if anchors[-1]['decode_index'] < count - 1:
        specs.append((anchors[-1]['decode_index'], count - 1, 'tail', [('forward', anchors[-1])]))
    intervals, tasks = [], []
    for number, (lo, hi, role, directions) in enumerate(specs):
        interval_id = f'interval_{number:03d}'
        interval = dict(interval_id=interval_id, start_decode_index=lo, end_decode_index=hi,
                        frame_count=hi-lo+1, role=role, directions=[d for d, _ in directions],
                        no_independent_opposite_anchor=role != 'between_anchors', task_ids=[])
        if role == 'between_anchors' and (lo, hi) in reused:
            interval.update(status='REUSED_VALIDATED', reuse=reused[(lo, hi)])
        else:
            interval['status'] = 'PLANNED'
            for direction, anchor in directions:
                previous = None
                edge = lo if direction == 'forward' else hi
                depth = 0
                while True:
                    start, end = ((edge, min(hi, edge+max_frames-1)) if direction == 'forward'
                                  else (max(lo, edge-max_frames+1), edge))
                    n = end-start+1
                    seed_decode = start if direction == 'forward' else end
                    seed_index = 0 if direction == 'forward' else n-1
                    task = dict(task_id=f'task_{len(tasks):03d}', interval_id=interval_id,
                                start_decode_index=start, end_decode_index=end, frame_count=n,
                                direction=direction, root_seed_decode_index=anchor['decode_index'], chain_depth=depth)
                    if previous is None:
                        task['seed'] = dict(anchor, frame_index=seed_index)
                    else:
                        task['seed_from'] = dict(task_id=previous['task_id'], decode_index=seed_decode,
                                                 frame_id=seed_decode-previous['start_decode_index'], frame_index=seed_index)
                    tasks.append(task)
                    interval['task_ids'].append(task['task_id'])
                    if (direction == 'forward' and end == hi) or (direction == 'reverse' and start == lo):
                        break
                    previous, depth = task, depth+1
                    edge = end-overlap+1 if direction == 'forward' else start+overlap-1
        intervals.append(interval)
    total = sum(t['frame_count'] for t in tasks)
    _require(len(tasks) <= 64 and total <= 2600, 'complete plan exceeds task or new direction-frame budget')
    return dict(bindings, schema='video-mask-full-plan-v1', status='PLANNED', frame_count=count,
                width=ordered[0]['width'], height=ordered[0]['height'], max_frames=max_frames, overlap=overlap,
                expected_decode_indexes=indexes, original_anchors=anchors, coverage_ledger=ordered,
                intervals=intervals, tasks=tasks, task_count=len(tasks), total_new_direction_frames=total,
                reuse_inputs=reuse_inputs, seed_quality='NOT_EVALUATED', candidate_quality='NOT_EVALUATED',
                propagation_status='NOT_RUN', original_masks_preserved=True, output_classification='review_candidate_only')
