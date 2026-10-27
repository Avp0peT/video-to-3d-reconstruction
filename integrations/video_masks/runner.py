"""Automatically plan, decode, propagate and review a short video interval."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import signal
import threading
import time

from integrations.demo_pipeline.pipeline import read, save, sha, _invoke, entry_code


def complete_records(path: Path) -> list[dict]:
    """Ignore an unfinished writer tail; complete malformed records still fail."""
    content = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in content.splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


class ResourceWatch:
    """Independent OS RAM and device samplers; cancellation only targets our Job."""
    def __init__(self, run: Path, device: str):
        from integrations.sam3.sam3_adapter.observer import resolve_device_identity
        identity = resolve_device_identity(device)
        if not identity['identity_provable']:
            raise ValueError(identity['reason'])
        self.index = identity['physical_index']
        self.run = run
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.reason = None
        self.threads = []
        self.stream = (run/'resources.jsonl').open('x', encoding='utf-8')

    def record(self, kind: str, values: dict, reason=None):
        with self.lock:
            if reason and self.reason is None:
                self.reason = reason
            self.stream.write(json.dumps(dict(kind=kind, monotonic=time.monotonic(), **values))+'\n')
            self.stream.flush()

    def device_sample(self):
        from integrations.sam3.sam3_adapter.observer import driver_device_query
        row = driver_device_query(self.index, timeout=5.0, freshness=5.0)
        used = row.get('device_used_bytes')
        reason = None
        if not row.get('fresh') or type(used) is not int or used < 0:
            reason = 'device observation unavailable: '+str(row.get('error'))
        elif used > 7 << 30:
            reason = 'device used exceeds 7 GiB'
        self.record('device', row, reason)

    def device_loop(self):
        try:
            while not self.stop.wait(0.5):
                self.device_sample()
        except Exception as exc:
            self.record('monitor_error', {'error': str(exc)}, 'device monitor failed')

    def ram_loop(self):
        from integrations.sam3.sam3_adapter.observer import (
            open_identity_handle, handle_alive, handle_creation_time, close_handle,
            enumerate_processes, descendants_of, process_creation_time)
        from integrations.sam3.sam3_adapter.resources import read_process_memory_for_pid
        handle = None
        pid = None
        try:
            while not self.stop.wait(0.25):
                if handle is None:
                    events = self.run/'worker/events.jsonl'
                    log = self.run/'logs/propagate/supervision.jsonl'
                    if not events.is_file() or not log.is_file():
                        continue
                    lines = complete_records(events)
                    if not lines:
                        continue
                    ready = lines[0]
                    created = next((s for s in complete_records(log) if s.get('kind') == 'created'), None)
                    if not created:
                        continue
                    pid = ready['pid']
                    table = enumerate_processes()
                    members = descendants_of(created['pid'], table) if table is not None else None
                    if members is None or pid not in {created['pid'], *(m['pid'] for m in members)}:
                        self.record('identity', {'pid': pid}, 'worker OS ancestry could not be verified')
                        return
                    handle = open_identity_handle(pid)
                    if not handle:
                        self.record('identity', {'pid': pid}, 'cannot open worker identity handle')
                        return
                    identity = handle_creation_time(handle)
                    self.record('identity', {'pid': pid, 'creation_time': identity,
                                            'launcher_pid': created['pid']})
                alive = handle_alive(handle)
                if alive is False:
                    return
                if alive is not True or identity is None or process_creation_time(pid) != identity:
                    self.record('identity', {'pid': pid}, 'worker identity became unknown')
                    return
                started = time.monotonic()
                row = read_process_memory_for_pid(pid)
                row.update(read_started=started, read_finished=time.monotonic())
                value = row.get('commit_bytes')
                # A process that exited during the read has no live missing channel.
                reason = None
                if type(value) is not int and handle_alive(handle) is not False:
                    reason = 'worker RAM observation unavailable'
                elif type(value) is int and value > 12 << 30:
                    reason = 'worker RAM commit exceeds 12 GiB'
                self.record('worker_ram', row, reason)
        except Exception as exc:
            self.record('monitor_error', {'error': str(exc)}, 'RAM monitor failed')
        finally:
            if handle:
                close_handle(handle)

    def start(self):
        self.device_sample()
        if self.reason:
            return
        for function in (self.device_loop, self.ram_loop):
            thread = threading.Thread(target=function, daemon=True)
            self.threads.append(thread)
            thread.start()

    def close(self):
        self.stop.set()
        deadline = time.monotonic()+7
        for thread in self.threads:
            thread.join(max(0, deadline-time.monotonic()))
        complete = all(not t.is_alive() for t in self.threads)
        if complete:
            self.stream.close()
        return {'joined': complete, 'stop_reason': self.reason}


def decode(root: Path, run: Path, plan: dict, ffmpeg: str) -> dict:
    """Decode the exact contiguous original frame interval; declare JPEG derivation."""
    from PIL import Image, ImageChops, ImageStat
    raw = run/'decoded'; raw.mkdir()
    frames = run/'frames'; frames.mkdir()
    start, end = plan['start_decode_index'], plan['end_decode_index']
    argv = [ffmpeg, '-hide_banner', '-nostdin', '-v', 'error', '-threads', '2',
            '-i', plan['source_video_path'], '-vf', f'select=between(n\\,{start}\\,{end})',
            '-fps_mode', 'passthrough', '-start_number', '0', '-threads', '2', str(raw/'%06d.png')]
    entry = _invoke(root, run, 'decode', argv, seconds=120, memory=1<<30)
    if entry_code(entry) != 0:
        raise RuntimeError('bounded video decode failed')
    paths = sorted(raw.glob('*.png'))
    if len(paths) != plan['frame_count']:
        raise ValueError('decoded count differs from frozen interval')
    bindings = []
    for local, path in enumerate(paths):
        with Image.open(path) as image:
            canonical = image.convert('RGB').resize((plan['width'], plan['height']), Image.Resampling.LANCZOS)
        target = frames/f'{local:06d}.jpg'
        canonical.save(target, quality=98, subsampling=0)
        bindings.append({'local_frame_index': local, 'decode_index': start+local,
                         'path': str(target), 'sha256': sha(target), 'decoded_sha256': sha(path)})
    seeds = []
    for seed in plan['seeds']:
        path = frames/f"{seed['local_frame_index']:06d}.jpg"
        with Image.open(path) as a, Image.open(seed['source_image']) as b:
            delta = ImageStat.Stat(ImageChops.difference(a.convert('RGB'), b.convert('RGB')))
        seeds.append({'input_index': seed['input_index'], 'mean_absolute_rgb_delta': delta.mean,
                      'interpretation': 'decode_resize_JPEG_difference_not_segmentation_accuracy'})
    record = {'frames': bindings, 'seed_pixel_comparison': seeds,
              'derivation': 'exact source decode indexes; RGB LANCZOS to canonical dimensions; JPEG quality98 4:4:4',
              'same_coordinates': True, 'byte_identical_to_photo_io': False}
    save(run/'DECODE.json', record)
    return record


def review(run: Path, plan: dict) -> dict:
    """Produce inspectable per-frame overlays; candidates never replace source masks."""
    import numpy as np
    from PIL import Image, ImageDraw
    from .policy import choose_propagation
    worker_path = run/'worker/result.json'
    worker = read(worker_path) if worker_path.is_file() else {'status': 'NOT_COMPLETED', 'frames': []}
    committed = {(f['direction'], f['frame_id']): f for f in worker.get('frames', [])}
    views = run/'review'; views.mkdir(exist_ok=True)
    targets = {t['local_frame_index']: t for t in plan['targets']}
    rows, gif = [], []
    for local in range(plan['frame_count']):
        image_path = run/'frames'/f'{local:06d}.jpg'
        if not image_path.is_file():
            continue
        masks = {}
        sources = {}
        for direction in ('forward', 'reverse'):
            item = committed.get((direction, local))
            if item:
                path = Path(item['mask_path'])
                if not path.resolve().is_relative_to((run/'worker').resolve()) or sha(path) != item['mask_sha256']:
                    raise ValueError('committed mask identity drift')
                with Image.open(path) as mask:
                    if mask.mode != 'L' or mask.size != (plan['width'], plan['height']):
                        raise ValueError('committed mask format mismatch')
                    masks[direction] = np.array(mask)
                sources[direction] = item
        target = targets.get(local)
        original = None
        if target and target.get('mask_path'):
            if sha(Path(target['mask_path'])) != target['mask_sha256']:
                raise ValueError('original mask drift')
            with Image.open(target['mask_path']) as mask:
                original = np.array(mask)
        decision = choose_propagation(masks.get('forward'), masks.get('reverse'), original)
        row = dict(local_frame_index=local, decode_index=plan['start_decode_index']+local,
                   input_index=target['input_index'] if target else None,
                   original_status=target['original_status'] if target else 'NOT_SAMPLED',
                   decision=decision, sources=sources)
        rows.append(row)
        with Image.open(image_path) as image:
            base = np.array(image.convert('RGB'))
        panels = []
        for name, mask, color in [('original', original, [245,190,35]),
                                  ('forward', masks.get('forward'), [35,210,100]),
                                  ('reverse', masks.get('reverse'), [60,150,250])]:
            canvas = base.copy()
            if mask is not None:
                active = mask != 0
                canvas[active] = (0.6*canvas[active]+0.4*np.array(color)).astype(np.uint8)
            panel = Image.fromarray(canvas)
            panel.thumbnail((240,360))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0,0,panel.width,36), fill='black')
            draw.text((4,4), f'{name} decode={row["decode_index"]}', fill='white')
            draw.text((4,18), 'no mask' if mask is None else 'mask overlay', fill='white')
            panels.append(panel)
        sheet = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)), 'white')
        x = 0
        for panel in panels:
            sheet.paste(panel,(x,0)); x += panel.width
        sheet.save(views/f'{local:06d}.jpg', quality=90)
        gif.append(sheet)
    if gif:
        gif[0].save(run/'motion.gif', save_all=True, append_images=gif[1:], duration=120, loop=0)
    result = {'schema': 'video-mask-review-v1', 'worker_status': worker.get('status'), 'rows': rows,
              'candidate_count': sum(r['decision']['status']=='PROPAGATED_CANDIDATE' for r in rows),
              'source_masks_modified': False, 'published_for_reconstruction': False,
              'quality_status': 'NOT_EVALUATED', 'seed_quality': 'NOT_EVALUATED',
              'scope': 'short contiguous interval; agreement is not semantic accuracy'}
    save(run/'REVIEW.json',result)
    cards = []
    for r in rows:
        i = r['local_frame_index']
        cards.append(f'<figure><figcaption>原视频帧 {r["decode_index"]} / 采样 {r["input_index"]} / '
                     f'{r["original_status"]} → {html.escape(r["decision"]["status"])}</figcaption>'
                     f'<img loading="lazy" src="review/{i:06d}.jpg"></figure>')
    page = '<!doctype html><meta charset="utf-8"><title>视频掩膜双向传播复核</title>'
    page += '<style>body{background:#14212c;color:#eef4f6;font:16px sans-serif;margin:32px}img{max-width:100%}figure{margin:24px 0}a{color:#8cd5ff}pre{white-space:pre-wrap}</style>'
    page += '<h1>手套视频掩膜：连续帧双向传播</h1><p>左：原 SAM3；中：正向传播；右：反向传播。端点种子为自动掩膜，不是人工真值。双向一致不代表轮廓正确。</p>'
    page += f'<p>原视频 {plan["start_decode_index"]}–{plan["end_decode_index"]}，{plan["frame_count"]} 帧。模型状态：{html.escape(str(worker.get("status")))}；原掩膜保持，未自动用于重建。</p>'
    if worker.get('error'):
        page += '<pre>'+html.escape(worker['error'])+'</pre>'
    page += '<p><a href="REVIEW.json">逐帧记录</a> · <a href="RESULT.json">执行结果</a> · <a href="PLAN.json">输入计划</a></p>'
    if gif:
        page += '<img src="motion.gif">'
    page += ''.join(cards)
    (run/'index.html').write_text(page,encoding='utf-8')
    return result


def execute(pipeline: Path, output: Path, max_frames: int = 48, execute_model: bool = False) -> dict:
    from .planning import make_plan
    root = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('output must be new; previous attempts are immutable')
    plan = make_plan(Path(pipeline), max_frames=max_frames)
    output.mkdir(parents=True)
    save(output/'PLAN.json', plan)
    if plan.get('status') != 'PLANNED':
        return plan
    source_plan = read(Path(pipeline)/'PLAN.json')
    if not execute_model:
        return dict(status='PREPARED', plan=str(output/'PLAN.json'), model_executions=0)
    source = Path(plan['source_video_path'])
    checkpoint = Path(source_plan['checkpoint'])
    before = {'video': sha(source), 'manifest': sha(Path(plan['source_manifest_path'])),
              'checkpoint': sha(checkpoint)}
    if (before['checkpoint'] != source_plan['checkpoint_sha256'] or before['video'] != plan['source_video_sha256']
            or before['manifest'] != plan['source_manifest_sha256']):
        raise ValueError('input/checkpoint binding changed')
    input_bindings = {}
    for name in ('source_video','source_manifest','video_result','pipeline_plan','io_run','io_frames','timeline'):
        input_bindings[plan[name+'_path']] = plan[name+'_sha256']
    for target in plan['targets']:
        input_bindings[target['source_image']] = target['source_image_sha256']
        if target.get('mask_path'):
            input_bindings[target['mask_path']] = target['mask_sha256']
    for path, digest in input_bindings.items():
        if sha(Path(path)) != digest:
            raise ValueError('planned input drift: '+path)
    weight_stat = [checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]
    code_files = (list((root/'integrations/video_masks').glob('*.py'))
                  + list((root/'integrations/sam3/src/sam3-repo/sam3').rglob('*.py'))
                  + [root/p for p in ('scripts/run_photo_demo.py','scripts/propagate_video_masks.py',
                       'integrations/demo_pipeline/pipeline.py','integrations/demo_pipeline/supervision.py',
                       'integrations/sam3/sam3_adapter/observer.py','integrations/sam3/sam3_adapter/resources.py')])
    bindings = {str(p): sha(p) for p in code_files}
    save(output/'START_BINDINGS.json', {'inputs': before, 'input_files': input_bindings,
                                      'checkpoint_stat':weight_stat,'code': bindings})
    result = dict(status='RUNNING', model_entries=0, automatic_retry=False)
    save(output/'RESULT.json', result)
    try:
        decode(root, output, plan, source_plan['video_tools']['ffmpeg'])
        repo = root/'integrations/sam3/src/sam3-repo'
        cfg = dict(repo_root=str(repo), checkpoint_path=str(checkpoint),
                   bpe_path=str(repo/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
                   frames_dir=str(output/'frames'), output_dir=str(output/'worker'),
                   frame_count=plan['frame_count'], max_frames=max_frames, device='cuda:0', threads=2,
                   direction='both', seeds=[dict(frame_index=s['local_frame_index'], mask_path=s['mask_path'],
                                                mask_sha256=s['mask_sha256']) for s in plan['seeds']])
        save(output/'CONFIG.json', cfg)
        watch = ResourceWatch(output, cfg['device'])
        from integrations.demo_pipeline.supervision import supervise
        logs = output/'logs/propagate'; logs.mkdir(parents=True)
        temp = output/'tmp'; temp.mkdir()
        cancelled = threading.Event()
        previous = signal.signal(signal.SIGINT, lambda *_: cancelled.set())
        try:
            watch.start()
            if watch.reason:
                raise RuntimeError(watch.reason)
            result['model_entries'] = 1
            save(output/'RESULT.json', result)
            entry = supervise([source_plan['sam3_python'], '-B', '-X', 'utf8',
                str(root/'integrations/video_masks/worker.py'), '--config', str(output/'CONFIG.json')],
                cwd=root, env_overrides={**{k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                    'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','TEMP':str(temp),'TMP':str(temp)},
                log_path=logs/'supervision.jsonl', stdout_path=logs/'stdout.txt',stderr_path=logs/'stderr.txt',
                wall_clock_seconds=600,job_memory_limit_bytes=14<<30,run_dir=output,
                run_dir_threshold_bytes=1<<30,total_roots=[output],total_threshold_bytes=1<<30,
                poll_seconds=0.5,reclamation_seconds=15,label='temporal-mask-sam3',script_sha256=sha(Path(__file__)),
                cancel_requested=lambda: cancelled.is_set() or watch.reason is not None)
            save(logs/'ENTRY.json', entry)
            result.update(exit_code=entry_code(entry), reclaim_confirmed=entry.get('reclaim_confirmed'))
        finally:
            result['monitor'] = watch.close()
            signal.signal(signal.SIGINT, previous)
        result.update(status='REVIEW_READY' if result['exit_code']==0 and not watch.reason and result['monitor']['joined'] else 'FAILED',
                      review=review(output,plan))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        if (output/'frames').is_dir():
            try:
                result['review'] = review(output,plan)
            except Exception as report_error:
                result['review_error'] = str(report_error)
    finally:
        drift = [p for p,digest in bindings.items() if not Path(p).is_file() or sha(Path(p))!=digest]
        result['code_drift'] = drift
        result['input_drift'] = [p for p,digest in input_bindings.items() if not Path(p).is_file() or sha(Path(p))!=digest]
        result['checkpoint_stat_unchanged'] = [checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]==weight_stat
        result['source_unchanged'] = not result['input_drift'] and result['checkpoint_stat_unchanged']
        if drift or not result['source_unchanged']:
            result['status'] = 'FAILED'
        save(output/'RESULT.json',result)
    return result
