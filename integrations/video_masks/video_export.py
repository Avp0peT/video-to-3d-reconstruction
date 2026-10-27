"""CPU-only, supervised full-timeline mask publication and pixel readback."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
from integrations.demo_pipeline import pipeline


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _binding(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': _hash(path)}


def _times(timeline, count):
    frames = timeline['frames']
    if len(frames) != count or [r['decode_index'] for r in frames] != list(range(count)):
        raise ValueError('timeline must cover every decode index exactly')
    def rational(value):
        return Fraction(*value) if isinstance(value, list) else Fraction(value)
    times = [Fraction(r['pts']) * rational(r.get('time_base', timeline.get('time_base'))) for r in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('timeline PTS must be strictly increasing')
    origin = times[0]
    times = [t - origin for t in times]
    tick = rational(frames[0].get('time_base', timeline.get('time_base')))
    cfr = all(abs(t - Fraction(i, 28)) <= tick / 2 for i, t in enumerate(times))
    return times, origin, cfr


def overlay_annotation(index, status, missing, independent):
    """Original evidence is not a single-direction propagation claim."""
    if missing:
        color=(255,0,255)
    elif status in ('FUSED_UNION_CANDIDATE','SUPPORTED_HOLE_CANDIDATE'):
        color=(190,130,255)
    elif status=='ORIGINAL_RETAINED':
        color=(40,210,255)
    elif not independent or status in ('DISAGREEMENT_CANDIDATE','PREDICTED_EMPTY'):
        color=(255,160,0)
    else:
        color=(0,255,0)
    return f'{index:06d} {status}',color


def export_video(root: Path, run: Path, plan: dict, rows: list,
                 ffmpeg: str, ffprobe: str) -> dict:
    """Publish every timestamp, keeping absent masks explicit and unaccepted."""
    root, run = Path(root).resolve(), Path(run).resolve()
    count = int(plan.get('total_frames', plan.get('frame_count', 0)))
    width, height = int(plan['width']), int(plan['height'])
    if count < 1 or [r['decode_index'] for r in rows] != list(range(count)):
        raise ValueError('rows must cover every decode index exactly in order')
    timeline_path = Path(plan['timeline_path'])
    if plan.get('timeline_sha256') and _hash(timeline_path) != plan['timeline_sha256']:
        raise ValueError('timeline hash differs from plan')
    timeline = json.loads(timeline_path.read_text(encoding='utf-8'))
    times, origin, cfr = _times(timeline, count)
    final = run / 'final_masks'
    overlay = run / 'export' / 'overlay_frames'
    final.mkdir(parents=True, exist_ok=False)
    overlay.mkdir(parents=True, exist_ok=False)
    manifest = []
    for index, row in enumerate(rows):
        target = final / f'{index:06d}.png'
        source = Path(row['mask_path']) if row.get('mask_path') else None
        missing = source is None
        if missing:
            mask = Image.new('L', (width, height), 0)
            mask.save(target)
        else:
            if not row.get('mask_sha256') or _hash(source) != row['mask_sha256']:
                raise ValueError(f'mask hash differs at {index}')
            with Image.open(source) as image:
                if image.format != 'PNG' or image.mode != 'L' or image.size != (width, height):
                    raise ValueError(f'mask format or size differs at {index}')
                mask = image.copy()
            if any(mask.histogram()[1:255]):
                raise ValueError(f'mask is not binary at {index}')
            shutil.copyfile(source, target)
        status = 'UNKNOWN' if missing else str(row['status'])
        pixels = mask.tobytes()
        record = dict(row, **_binding(target), status=status,
                      placeholder=missing, source_mask_path=str(source) if source else None,
                      pixel_sha256=hashlib.sha256(pixels).hexdigest())
        manifest.append(record)
        with Image.open(run / 'frames' / f'{index:06d}.jpg') as image:
            if image.size != (width, height):
                raise ValueError(f'canonical JPEG dimensions differ at {index}')
            rgb = image.convert('RGB')
        tinted = Image.blend(rgb, Image.new('RGB', rgb.size, (0, 255, 0)), 0.35)
        rgb.paste(tinted, mask=mask)
        label,color=overlay_annotation(index,status,missing,row.get('independent_bidirectional',False))
        draw = ImageDraw.Draw(rgb)
        draw.rectangle((2, 2, width - 3, height - 3), outline=color, width=5)
        draw.rectangle((8, 8, width - 8, 40), fill=(0, 0, 0))
        draw.text((12, 16), label, fill=color)
        rgb.save(overlay / f'{index:06d}.jpg', quality=90)
    pipeline.save(run / 'MANIFEST.json', {'schema': 'full-video-mask-manifest-v1', 'rows': manifest})

    def command(stage, argv):
        entry = pipeline._invoke(root, run, 'export_' + stage, [str(x) for x in argv],
                                 seconds=120, memory=1 << 30)
        if pipeline.entry_code(entry) != 0:
            raise RuntimeError(f'export stage {stage} failed; inspect logs/export_{stage}')
        return run / 'logs' / ('export_' + stage) / 'stdout.txt'

    def inputs(folder, suffix, name):
        if cfr:
            return ['-framerate', '28', '-start_number', '0', '-i', str(folder / ('%06d.' + suffix))]
        concat = run / 'export' / (name + '.ffconcat')
        lines = ['ffconcat version 1.0']
        for i in range(count):
            escaped = (folder / f'{i:06d}.{suffix}').as_posix().replace("'", "'\\''")
            lines += [f"file '{escaped}'", 'option framerate 1000']
            if i + 1 < count:
                lines.append(f'duration {float(times[i+1] - times[i]):.9f}')
        concat.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return ['-f', 'concat', '-safe', '0', '-i', str(concat)]

    outputs = {}
    for name, folder, suffix, codec in [
        ('masks_lossless.mkv', final, 'png', ['-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gray']),
        ('masks_preview.mp4', final, 'png', ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']),
        ('overlay.mp4', overlay, 'jpg', ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])]:
        path = run / name
        command(name.replace('.', '_'), [ffmpeg, '-hide_banner', '-nostdin', '-v', 'error', '-y',
                 '-threads', '2', *inputs(folder, suffix, name), '-frames:v', count,
                 '-fps_mode', 'passthrough', *codec, '-threads', '2', '-fs', 1 << 30, path])
        probe_log = command('probe_' + name.replace('.', '_'), [ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-show_frames', '-show_entries',
            'stream=width,height,nb_frames:frame=best_effort_timestamp_time', '-of', 'json', path])
        data = json.loads(probe_log.read_text(encoding='utf-8'))
        frames = data['frames']
        stream = data['streams'][0]
        pts = [Fraction(f['best_effort_timestamp_time']) for f in frames]
        if len(pts) != count or (stream['width'], stream['height']) != (width, height):
            raise ValueError(f'output count or dimensions differ: {name}')
        if any(b <= a for a, b in zip(pts, pts[1:])):
            raise ValueError(f'output PTS are not monotonic: {name}')
        error = max(abs(a-b) for a, b in zip(pts, times))
        bound = Fraction(5, 1000) if cfr else max((b-a for a,b in zip(times,times[1:])), default=Fraction(1,28))
        if error > bound:
            raise ValueError(f'output PTS error exceeds bound: {name}')
        outputs[name] = dict(_binding(path), decoded_frames=len(pts), width=width, height=height,
                             max_source_relative_pts_error_seconds=float(error), monotonic_pts=True)

    raw = run / 'export' / 'mask_readback.gray'
    command('pixel_readback', [ffmpeg, '-hide_banner', '-nostdin', '-v', 'error', '-y', '-threads', '2',
        '-i', run / 'masks_lossless.mkv', '-map', '0:v:0', '-fps_mode', 'passthrough',
        '-pix_fmt', 'gray', '-f', 'rawvideo', '-threads', '2', '-fs', 1 << 30, raw])
    with raw.open('rb') as stream:
        for i, record in enumerate(manifest):
            pixels = stream.read(width * height)
            if len(pixels) != width * height or hashlib.sha256(pixels).hexdigest() != record['pixel_sha256']:
                raise ValueError(f'lossless pixel readback differs at {i}')
        if stream.read(1):
            raise ValueError('lossless readback has additional frames')
    raw_binding = _binding(raw)
    raw.unlink()
    result = {'schema': 'full-video-mask-export-v1', 'status': 'EXPORTED_PIXEL_VERIFIED',
        'source_video_path': plan.get('source_video_path'), 'timeline': _binding(timeline_path),
        'source_pts_origin_seconds': str(origin), 'timeline_mode': 'CFR_28_ROUNDED_SOURCE_TICKS' if cfr else 'VFR_MILLISECOND_QUANTIZATION',
        'total_frames': count, 'counts': dict(Counter(r['status'] for r in manifest)),
        'placeholder_count': sum(r['placeholder'] for r in manifest),
        'pixel_readback_verified': True, 'pixel_readback': dict(raw_binding, retained=False),
        'outputs': outputs, 'manifest': _binding(run / 'MANIFEST.json'),
        'overlay_source': 'derived canonical RGB JPEG; not source-exact pixels',
        'model_or_gpu_used_by_exporter': False, 'geometry_acceptance': False}
    pipeline.save(run / 'EXPORT.json', result)
    return result
