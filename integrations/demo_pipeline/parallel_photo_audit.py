"""Read-only bridge from native photo observations to exact parallel-domain tests.

This is a project diagnostic, not a noisy/perspective extension of S-1996.
It never projects points3D to manufacture observations or modifies a model.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import html
from itertools import combinations
import json
import math
from pathlib import Path
import struct

MAX_BYTES = 64 << 20
SELECTION = 'max tracks with exactly one observation in each selected view, tie ascending image IDs; six lowest eligible point3D IDs; no geometry-based reselection'


def binding(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('bounded input exceeds 64 MiB')
    data = path.read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError('bounded input exceeds 64 MiB')
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError('duplicate JSON key')
        out[key] = value
    return out


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('bounded JSON required')
    value = loads(path.read_text(encoding='utf-8-sig'))
    if type(value) is not dict:
        raise ValueError('JSON object required')
    return value


def under(root, relative):
    if type(relative) is not str or not relative:
        raise ValueError('nonempty relative path required')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path == root:
        raise ValueError('input escapes its run')
    return path


class Reader:
    def __init__(self, path):
        if Path(path).stat().st_size > MAX_BYTES:
            raise ValueError('native file exceeds byte budget')
        self.data = Path(path).read_bytes()
        self.offset = 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        if self.offset + size > len(self.data):
            raise ValueError('truncated native file')
        result = struct.unpack_from(fmt, self.data, self.offset)
        self.offset += size
        return result

    def name(self):
        end = self.data.find(b'\0', self.offset, self.offset + 4097)
        if end < 0:
            raise ValueError('invalid bounded native image name')
        name = self.data[self.offset:end].decode('utf-8')
        self.offset = end + 1
        if not name:
            raise ValueError('empty native image name')
        return name

    def finish(self):
        if self.offset != len(self.data):
            raise ValueError('trailing native bytes')


def read_native(images_path, points_path):
    """COLMAP images.bin/points3D.bin; tracks checked reciprocally in full."""
    r = Reader(images_path)
    count, = r.take('<Q')
    if not 2 <= count <= 256:
        raise ValueError('requires 2..256 registered native images')
    images = {}
    names = set()
    for _ in range(count):
        record = r.take('<idddddddi')
        ident, camera = record[0], record[-1]
        name = r.name()
        if ident <= 0 or ident in images or camera <= 0 or name in names or not all(math.isfinite(v) for v in record[1:-1]):
            raise ValueError('invalid native image identity/pose')
        names.add(name)
        n, = r.take('<Q')
        if n > 100000 or n > (len(r.data) - r.offset) // 24:
            raise ValueError('native observation budget/truncation')
        observations = [r.take('<ddq') for _ in range(n)]
        if any(not math.isfinite(x) or not math.isfinite(y) or pid < -1 or pid == 0 for x, y, pid in observations):
            raise ValueError('invalid native observation')
        images[ident] = dict(image_id=ident, camera_id=camera, name=name, observations=observations)
    r.finish()
    r = Reader(points_path)
    count, = r.take('<Q')
    if count > 100000:
        raise ValueError('native point budget')
    points = {}
    for _ in range(count):
        record = r.take('<QdddBBBd')
        ident = record[0]
        if not 0 < ident < 2**63 or ident in points or not all(math.isfinite(v) for v in (*record[1:4], record[-1])):
            raise ValueError('invalid native point identity/coordinates')
        n, = r.take('<Q')
        if not 2 <= n <= 100000 or n > (len(r.data) - r.offset) // 8:
            raise ValueError('native track cardinality')
        track = [r.take('<ii') for _ in range(n)]
        if len(set(track)) != n:
            raise ValueError('duplicate native track element')
        for image_id, index in track:
            if image_id not in images or not 0 <= index < len(images[image_id]['observations']):
                raise ValueError('invalid track reference')
            if images[image_id]['observations'][index][2] != ident:
                raise ValueError('track/image identity mismatch')
        by_image = {}
        for image_id, index in track:
            by_image.setdefault(image_id, []).append(index)
        points[ident] = dict(track={i:indices[0] for i,indices in by_image.items() if len(indices)==1},
                            track_elements=set(track), image_ids=sorted(by_image),
                            ambiguous_observations={str(i):indices for i,indices in by_image.items() if len(indices)>1})
    r.finish()
    for image_id, image in images.items():
        for index, (_, _, pid) in enumerate(image['observations']):
            if pid != -1 and (pid not in points or (image_id,index) not in points[pid]['track_elements']):
                raise ValueError('image/track reciprocal identity mismatch')
    return images, points


def choose_pair(images, points):
    counts = Counter()
    raw_counts = Counter()
    for point in points.values():
        counts.update(combinations(sorted(point['track']), 2))
        raw_counts.update(combinations(point.get('image_ids',sorted(point['track'])), 2))
    eligible = [(pair, count) for pair, count in counts.items() if count >= 6]
    if not eligible:
        raise ValueError('no pair with six common native tracks')
    pair, count = min(eligible, key=lambda item: (-item[1], item[0]))
    shared = sorted(pid for pid, p in points.items() if all(i in p['track'] for i in pair))
    return pair, shared, [dict(image_ids=list(p), shared_tracks=counts[p], raw_shared_tracks=n,
                              ambiguous_tracks_excluded=n-counts[p]) for p,n in sorted(raw_counts.items())]


def views_for(pair, point_ids, images, points):
    views = []
    for image_id in pair:
        view = {}
        for pid in point_ids:
            index = points[pid]['track'][image_id]
            xy = images[image_id]['observations'][index][:2]
            view[pid] = tuple(Fraction.from_float(value) for value in xy)
        views.append(view)
    return views


def numerical_residual(first, second):
    """Unweighted TLS distance to a 3D affine flat in raw four-pixel space.

    No threshold, denoising or fitted coordinates are passed to exact AlgT'.
    """
    import numpy as np
    labels = sorted(first)
    if set(labels) != set(second) or len(labels) < 5:
        raise ValueError('at least five explicitly paired points required')
    joint = np.asarray([[*first[i], *second[i]] for i in labels], dtype=np.float64)
    if joint.shape != (len(labels), 4) or not np.isfinite(joint).all():
        raise ValueError('finite paired image coordinates required')
    centered = joint - joint.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    distances = np.abs(centered @ vt[-1])
    return dict(method='centered_unweighted_TLS_affine_rank3_raw_pixel_4D', n=len(labels),
                singular_values=singular.tolist(), rms_joint_distance_px=float(np.sqrt(np.mean(distances**2))),
                median_joint_distance_px=float(np.median(distances)), max_joint_distance_px=float(distances.max()),
                acceptance_threshold=None, model_accepted=None,
                meaning='4D joint-coordinate orthogonal residual; not per-image reprojection or physical accuracy')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def _run_exact(first, second):
    from kozlov_lab.stereo.official.given_parallel_correspondence import assess_given_parallel_input, assess_given_parallel_certificate
    from kozlov_lab.kozlov2025.exact_center import encode_exact, decode_exact
    mapping = {i: i for i in sorted(first)}
    result = assess_given_parallel_input(first, second, mapping, max_work=2000000, seconds=10)
    restored = decode_exact(loads(json.dumps(encode_exact(result), allow_nan=False)))
    certificate = restored.get('certificate')
    check = (assess_given_parallel_certificate(first, second, mapping, certificate, max_work=2000000, seconds=10)
             if certificate is not None else dict(status='NOT_RUN', reason='no exact certificate'))
    return result, encode_exact(restored), check


def audit(sfm_run, model_id, output):
    root = Path(sfm_run).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('output must be separate from the immutable SfM input')
    if output.exists():
        raise ValueError('new output required; no overwrite')
    run_path = root / 'run.json'
    refs = [binding(run_path)]
    run = read_json(run_path)
    if run.get('contract_version') != 'sfm-native-object-v1' or run.get('status') not in ('SUCCESS', 'PARTIAL'):
        raise ValueError('requires accepted native object SfM run')
    models = [m for m in run['models'] if str(m['model_id']) == str(model_id)]
    if len(models) != 1 or models[0].get('usable') is not True or models[0].get('roundtrip_verified') is not True:
        raise ValueError('explicit usable roundtrip-verified model required')
    model = models[0]
    if (type(model.get('registered_image_ids')) is not list
            or any(type(i) is not int for i in model['registered_image_ids'])
            or len(set(model['registered_image_ids'])) != len(model['registered_image_ids'])
            or type(model.get('num_points3D')) is not int):
        raise ValueError('typed unique native model identities required')
    binary = under(root, model['binary_path'])
    paths = [binary / 'images.bin', binary / 'points3D.bin', binary / 'cameras.bin', root / 'frames_map.jsonl']
    refs.extend(binding(p) for p in paths)
    images, points = read_native(paths[0], paths[1])
    if sorted(images) != sorted(model['registered_image_ids']) or len(points) != model['num_points3D']:
        raise ValueError('run/native identity and count mismatch')
    pair, common, pair_counts = choose_pair(images, points)
    rows = [loads(line) for line in paths[3].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    frames = []
    for image_id in pair:
        selected = [row for row in rows if any(str(reg['model_id']) == str(model_id) and reg['image_id'] == image_id for reg in row['registrations'])]
        if len(selected) != 1:
            raise ValueError('one frame identity per selected image required')
        row = selected[0]
        if row['staged_image_name'] != images[image_id]['name'] or row['upstream_io_run_id'] != run['upstream_io_run_id']:
            raise ValueError('frame/name/upstream identity mismatch')
        matching_registrations = [reg for reg in row['registrations'] if str(reg['model_id']) == str(model_id) and reg['image_id'] == image_id]
        if (len(matching_registrations) != 1 or matching_registrations[0]['camera_id'] != images[image_id]['camera_id']
                or type(row['input_index']) is not int or row['input_index'] < 0 or type(row['frame_id']) is not str
                or not row['frame_id'] or row['status'] != 'REGISTERED'):
            raise ValueError('frame camera/index/identity/status mismatch')
        photo = under(root, row['staged_image_name'])
        photo_ref = binding(photo)
        if photo_ref['sha256'] != row['canonical_image_sha256']:
            raise ValueError('canonical image byte mismatch')
        refs.append(photo_ref)
        frames.append(dict(image_id=image_id, input_index=row['input_index'], frame_id=row['frame_id'],
                           camera_id=images[image_id]['camera_id'], name=row['staged_image_name'], image=photo_ref))
    selected_ids = common[:6]
    first, second = views_for(pair, selected_ids, images, points)
    raw_rows = [dict(point3D_id=pid, observations=[dict(image_id=i, point2D_index=points[pid]['track'][i],
                   xy=list(images[i]['observations'][points[pid]['track'][i]][:2]),
                   binary64_hex=[v.hex() for v in images[i]['observations'][points[pid]['track'][i]][:2]]) for i in pair]) for pid in selected_ids]
    output.mkdir(parents=True, exist_ok=False)
    frozen = dict(schema='photo-parallel-selected-observations-v1', selection_rule=SELECTION, image_pair=list(pair),
                  common_tracks=len(common), point_ids=selected_ids, frames=frames, observations=raw_rows,
                  source_bindings=refs, pair_counts=pair_counts, scope='six-track scene subset, not object-wide or foreground-certified')
    rationals = [v for view in (first,second) for xy in view.values() for v in xy]
    frozen['rational_bounds'] = dict(max_numerator=max(abs(v.numerator) for v in rationals),
                                    max_denominator=max(v.denominator for v in rationals), limit=10**12,
                                    conversion='Fraction.from_float; exact binary64, no rounding')
    frozen['native_multiple_observation_tracks'] = [dict(point3D_id=pid, ambiguous_observations=p['ambiguous_observations'])
                                                   for pid,p in points.items() if p['ambiguous_observations']]
    frozen['selected_pair_excluded_tracks'] = [pid for pid,p in points.items()
                                              if all(i in p['image_ids'] for i in pair) and not all(i in p['track'] for i in pair)]
    write_json(output / 'INPUT.json', frozen)  # before invoking the research constructor
    result, exact, independent = _run_exact(first, second)
    write_json(output / 'EXACT_RESULT.json', exact)
    all_first, all_second = views_for(pair, common, images, points)
    diagnostic = numerical_residual(all_first, all_second)
    body = [(0,0,0), (1,0,0), (0,1,0), (1,1,1), (2,3,2), (3,1,-1)]
    control_first = {i:(p[0],p[1]) for i,p in enumerate(body)}
    control_second = {i:(p[0]+p[2],p[1]+2*p[2]) for i,p in enumerate(body)}
    control, control_exact, control_check = _run_exact(control_first, control_second)
    write_json(output / 'SYNTHETIC_CONTROL.json', control_exact)
    unchanged = all(binding(ref['path']) == ref for ref in refs)
    research_evidence_ok = independent.get('status') == 'VERIFIED' or (result['status'] in ('UNKNOWN','INVALID_INPUT') and result.get('certificate') is None)
    if not unchanged or not research_evidence_ok or control_check.get('status') != 'VERIFIED' or control['status'] != 'CONSTRUCTED':
        raise ValueError('input stability or independent verification failed; results not accepted')
    summary = dict(schema='photo-parallel-applicability-v1', status='DIAGNOSTIC_COMPLETE', input_scope=frozen['scope'],
                   exact_status=result['status'], exact_reason=result['reason'], independent_check=independent, source_unchanged=unchanged,
                   image_pair=list(pair), input_indices=[f['input_index'] for f in frames], selected_point_ids=selected_ids,
                   shared_track_count=len(common), numerical_diagnostic=diagnostic,
                   raw_shared_track_count=next(r['raw_shared_tracks'] for r in pair_counts if r['image_ids']==list(pair)),
                   ambiguous_tracks_excluded=frozen['selected_pair_excluded_tracks'],
                   synthetic_control=dict(status=control['status'], independent_check=control_check, data_kind='synthetic'),
                   reconstruction_source='stored native 2D observations; tracks supplied by existing COLMAP',
                   photo_mesh_created=False, photo_reconstruction_gain=None, scale_status='unknown',
                   limitations=['Exact rank rejection does not identify perspective, distortion, noise, quantisation or bad tracks as the cause.',
                                'No point reselection, coordinate rounding, un-distortion, denoising, refitting or model fallback.',
                                'The six-track subset is scene sparse; object segmentation is not certified.',
                                'TLS has no acceptance threshold and is not a new reconstruction.',
                                'A positive six-point result would not certify all common points or an entire photograph.'])
    write_json(output / 'RESULT.json', summary)
    report = '<!doctype html><meta charset="utf-8"><title>真实照片：Kozlov 适用性检验</title>'
    report += '<style>body{font:17px/1.6 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#243142}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    report += '<h1>真实照片与精确平行投影假设</h1><p>本页检验已冻结对应是否满足算法输入条件；未生成照片新网格。</p>'
    report += f'<p>照片 index {summary["input_indices"]}；共同 track {len(common)}；固定六点 {selected_ids}。</p>'
    report += '<table><tr><th>检验</th><th>结果</th></tr>'
    for label, value in [('真实六点精确条件', result['status']), ('独立证书回读', independent['status']),
                         ('合成平行投影：实际 AlgT′', control['status']), ('全部共同点四维正交残差 RMS（像素）', diagnostic['rms_joint_distance_px'])]:
        report += f'<tr><td>{html.escape(label)}</td><td>{html.escape(str(value))}</td></tr>'
    report += '</table><p>非零残差不独自说明是哪一种成像/匹配误差；没有阈值就不判近似平行可用。此处不是毫米误差或每图重投影误差。</p>'
    report += '<p>原始像素未改动；未重跑 SAM3/SfM/MVS；未改旧模型。精确拒绝的含义限于固定坐标和对应的精确模型。</p>'
    report += '<p><a href="INPUT.json">输入与身份</a> · <a href="EXACT_RESULT.json">精确证书</a> · <a href="RESULT.json">完整结果</a> · <a href="SYNTHETIC_CONTROL.json">合成正控制</a></p>'
    (output / 'report.html').write_text(report, encoding='utf-8')
    return summary
