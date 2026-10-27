"""Publication guard v2; historical learned-point runtime stays immutable."""
import argparse
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
from . import learned_point_run as legacy
from .balanced_point_run import read, save, binding, check_bindings

ROOT = legacy.ROOT
OUTPUTS = ('motion.gif', 'contact_sheet.png', 'overview.png', 'index.html',
           'learned_frame_panels', 'LEARNED_POINT_REPORT.json')
RECEIPT = 'GUARDED_RUN_RECEIPT.json'


def controllers():
    return [binding(ROOT / p) for p in (
        'integrations/demo_pipeline/learned_point_guarded.py',
        'scripts/track_learned_object_guarded.py')]


def prepare(args):
    out = args.output.resolve()
    if (out / 'PLAN.json').exists() or (out / 'SOURCE_BINDINGS.json').exists():
        raise FileExistsError('existing preparations cannot be extended in place')
    original_save = legacy.save
    def guarded_save(path, value):
        if Path(path).name == 'PLAN.json':
            value = dict(value, schema='learned-point-plan-v2',
                         publication_guard='guarded-controller-v2',
                         implementation=value['implementation'] + controllers())
        elif Path(path).name == 'SOURCE_BINDINGS.json':
            # The augmented plan is already final before source bindings freeze.
            value = value + [binding(out / 'PLAN.json')]
        original_save(path, value)
    with patch.object(legacy, 'save', guarded_save):
        legacy.prepare(args)


def check(out):
    plan = legacy.check(out)
    if plan.get('publication_guard') != 'guarded-controller-v2':
        raise ValueError('requires a newly prepared guarded v2 run')
    required = {r['path'] for r in controllers()}
    if not required <= {r['path'] for r in plan['implementation']}:
        raise ValueError('guarded controller bindings missing')
    if str((out / 'PLAN.json').resolve()) not in {
            r['path'] for r in read(out / 'SOURCE_BINDINGS.json')}:
        raise ValueError('final plan is not source-bound')
    return plan


def completed(entry):
    return (entry.get('status') == 'completed' and entry.get('errors') == []
            and entry.get('log_errors') == [] and entry.get('root_exit_code') == 0
            and entry.get('reclaim_confirmed') is True)


def validate_tracks(out, plan):
    tracks = read(out / 'TRACKS.json')
    inputs, queries = read(out / 'INPUTS.json'), read(out / 'QUERIES.json')
    expected = [r['decode_index'] for r in inputs['frames']]
    ids = [p['track_id'] for p in queries['points']]
    if (tracks.get('status') != 'TRACKS_GENERATED_NOT_IDENTITY_VERIFIED'
            or tracks.get('error') or len(expected) != plan['frames']
            or len(ids) != plan['query_count'] or len(set(ids)) != len(ids)
            or len(set(expected)) != len(expected)
            or [r['decode_index'] for r in tracks['frames']] != expected):
        raise ValueError('incomplete or failed tracking records')
    for row in tracks['frames']:
        if ([p['track_id'] for p in row['points']] != ids
                or any(p.get('status') == 'FAILED' for p in row['points'])):
            raise ValueError('tracking IDs/status mismatch')
    return tracks


def snapshot(out, plan):
    return (read(out / 'SOURCE_BINDINGS.json') + plan['implementation']
            + plan['upstream_bindings'] + [plan['checkpoint']]
            + [binding(out / n) for n in ('TRACKS.json', 'INPUTS.json', 'QUERIES.json', 'PLAN.json')])


def worker(out):
    check(out)
    return legacy.worker(out)


def validate_sources(out):
    refs = {str(Path(r['path']).resolve()): r for r in read(out / 'SOURCE_BINDINGS.json')}
    for frame in read(out / 'INPUTS.json')['frames']:
        path = Path(frame['path'])
        if not path.is_absolute():
            path = out / path
        actual = binding(path)
        if actual['sha256'] != frame.get('sha256') or refs.get(actual['path']) != actual:
            raise ValueError('source frame SHA or binding identity mismatch')


def report(out):
    # All admission checks precede even temporary output creation.
    out = Path(out).resolve(strict=True)
    plan = check(out)
    validate_sources(out)
    receipt = read(out / RECEIPT)
    if receipt.get('schema') != 'guarded-run-receipt-v2' or not completed(receipt['inference']):
        raise ValueError('completed supervised run receipt required')
    if receipt['bindings'] != snapshot(out, plan) or check_bindings(receipt['bindings']):
        raise ValueError('receipt identity drift')
    entry = read(out / 'processes/inference/ENTRY.json')
    if entry != receipt['inference']:
        raise ValueError('supervision receipt mismatch')
    tracks = validate_tracks(out, plan)
    refs = snapshot(out, plan) + [binding(out / RECEIPT), binding(out / 'processes/inference/ENTRY.json')]
    if any((out / n).exists() for n in OUTPUTS):
        raise FileExistsError('report outputs already exist')
    from .learned_point_report import render
    with tempfile.TemporaryDirectory(prefix='learned-report-', dir=out) as tmp:
        stage = Path(tmp)
        for name in ('INPUTS.json', 'TRACKS.json', 'QUERIES.json'):
            shutil.copyfile(out / name, stage / name)
        returned = render(stage)
        check(out)
        validate_sources(out)
        if check_bindings(refs) or read(stage / 'TRACKS.json') != tracks:
            raise ValueError('identity changed during rendering')
        if (returned != read(stage / 'LEARNED_POINT_REPORT.json')
                or [(r['decode_index'], r['points']) for r in returned['frames_rendered']]
                != [(r['decode_index'], r['points']) for r in tracks['frames']]):
            raise ValueError('returned report identity mismatch')
        if any((out / n).exists() for n in OUTPUTS):
            raise FileExistsError('report destination appeared')
        for name in OUTPUTS:
            # copy with exclusive creation; directories also reject existing paths.
            src, dst = stage / name, out / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                with src.open('rb') as source, dst.open('xb') as target:
                    shutil.copyfileobj(source, target)
    return returned


def run(out):
    from .masked_trace_run import invoke
    out = Path(out).resolve(strict=True)
    plan = check(out)
    if (out / RECEIPT).exists() or (out / 'processes/inference').exists():
        raise FileExistsError('guarded runs are single execution')
    argv = [plan['python'], '-B', '-X', 'utf8', str(ROOT / 'scripts/track_learned_object_guarded.py'),
            'worker', '--output', str(out)]
    entry = invoke(out, 'inference', argv, plan['wall_seconds'], plan['job_commit_bytes'])
    result = dict(status='FAILED', inference=entry, quality_status='NOT_EVALUATED', geometry_accepted=False)
    try:
        if not completed(entry):
            raise ValueError('supervisor did not complete cleanly')
        check(out)
        validate_tracks(out, plan)
        save(out / RECEIPT, dict(schema='guarded-run-receipt-v2', inference=entry,
                                bindings=snapshot(out, plan)))
        report(out)
        result['status'] = 'EXPERIMENT_COMPLETE_NOT_IDENTITY_VERIFIED'
        save(out / 'OUTPUT_BINDINGS.json', [binding(p) for n in OUTPUTS
             for p in ([out / n] if (out / n).is_file() else sorted((out / n).rglob('*')))
             if p.is_file()])
    except Exception as exc:
        result['publication_error'] = type(exc).__name__ + ': ' + str(exc)
    save(out / 'RESULT.json', result)
    return 0 if result['status'].startswith('EXPERIMENT_COMPLETE') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    for name in ('source-run', 'mask-run', 'upstream', 'checkpoint'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--points', type=int, default=1)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        if not all((args.source_run, args.mask_run, args.upstream, args.checkpoint)):
            parser.error('prepare requires source-run, mask-run, upstream and checkpoint')
        prepare(args)
        return 0
    out = args.output.resolve(strict=True)
    if args.action == 'report':
        report(out)
        return 0
    return worker(out) if args.action == 'worker' else run(out)
