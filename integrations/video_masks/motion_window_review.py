"""Associate overlapping window candidates without altering any source masks."""
from pathlib import Path
import hashlib
import html
import json

import numpy as np
from PIL import Image, ImageDraw

from .motion_window_linking import associate_pair


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _binding(path):
    p = Path(path).resolve(strict=True)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=_sha(p))


def _save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def _remember(refs, current):
    key = current['path']
    if key in refs and refs[key] != current:
        raise ValueError('input drift between reads: ' + key)
    refs.setdefault(key, current)


def _read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def reject(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=pairs, parse_constant=reject)


def _verified(ref, refs, *, root=None):
    path = Path(ref['path']).resolve(strict=True)
    if root is not None and not path.is_relative_to(root):
        raise ValueError('window artifact escapes review root')
    now = _binding(path)
    if any(now[k] != ref[k] for k in ('sha256', 'bytes') if k in ref):
        raise ValueError('window input hash drift: ' + str(path))
    _remember(refs, now)
    return path


def _load_window(record, rows, root, refs, check):
    """Load at most one six-frame window; IDs remain scoped to its ledger."""
    check()
    number = record['window_id']
    if type(number) is not int or number < 0:
        raise ValueError('invalid window ID')
    path = root / ('window_%03d' % number) / 'MOTION_EVIDENCE.json'
    before = _binding(path)
    ledger = _read_json(path)
    if _binding(path) != before:
        raise ValueError('ledger drift during read')
    _remember(refs, before)
    cores = {}
    for part in ledger['parts']:
        pid = part['id']; ids = part['track_ids']
        if pid in cores or not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids) or len(set(ids)) != len(ids):
            raise ValueError('invalid local core identity')
        cores[pid] = ids
    if set(cores) != set(record['part_ids']):
        raise ValueError('part ledger/record mismatch')
    frames = record['frames']
    if not 1 <= len(frames) <= 6:
        raise ValueError('window must have one to six frames')
    positions = [{} for _ in frames]
    seen_pairs = set()
    for edge in ledger['pairs']:
        pair = edge['pair']; tids = edge['track_ids']
        if len(pair) != 2 or any(type(i) is not int for i in pair) or not 0 <= pair[0] < pair[1] < len(frames) or pair[1] != pair[0]+1 or tuple(pair) in seen_pairs:
            raise ValueError('invalid local track edge')
        seen_pairs.add(tuple(pair))
        if any(type(t) is not int or t < 0 for t in tids) or len(set(tids)) != len(tids):
            raise ValueError('invalid track IDs')
        for f, key in zip(pair, ('source_points', 'destination_points')):
            xy = edge[key]
            if len(xy) != len(tids):
                raise ValueError('coordinate count mismatch')
            for tid, point in zip(tids, xy):
                if len(point) != 2 or not np.isfinite(point).all():
                    raise ValueError('invalid track point')
                if tid in positions[f] and not np.allclose(positions[f][tid], point, atol=1e-4, rtol=0):
                    raise ValueError('inconsistent track coordinates')
                positions[f][tid] = point
    normalized = []; pixels_loaded = 0
    for f, frame in enumerate(frames):
        check()
        pos = frame['input_position']
        if type(pos) is not int or not 0 <= pos < len(rows):
            raise ValueError('invalid source position')
        source = rows[pos]
        if frame['frame_id'] != source['frame_id'] or frame['time_seconds'] != source['time_seconds']:
            raise ValueError('source frame identity mismatch')
        _verified(source['image'], refs)
        _verified(source['mask'], refs)
        whole_path = _verified(frame['whole'], refs, root=root)
        if frame['whole']['sha256'] != source['mask']['sha256']:
            raise ValueError('whole mask no longer matches source')
        with Image.open(whole_path) as image:
            whole = np.asarray(image).copy()
        if whole.ndim != 2 or whole.dtype != np.uint8 or not np.isin(whole, [0,255]).all():
            raise ValueError('invalid whole binary mask')
        if set(frame['parts']) != set(cores):
            raise ValueError('frame part identity mismatch')
        parts = {}; tracks = {}; occupied = np.zeros(whole.shape, bool)
        for pid, ref in frame['parts'].items():
            # Array cap is an admission check, not an OS hard-memory guarantee.
            pixels_loaded += whole.size
            if pixels_loaded > 128 << 20:
                raise ValueError('window mask arrays exceed 128 MiB admission cap')
            p = _verified(ref, refs, root=root)
            with Image.open(p) as image:
                raw = np.asarray(image).copy()
            if raw.shape != whole.shape or raw.dtype != np.uint8 or not np.isin(raw,[0,255]).all():
                raise ValueError('invalid part binary mask')
            mask = raw != 0
            if (mask & ((whole == 0) | occupied)).any():
                raise ValueError('part outside whole or overlapping another part')
            occupied |= mask; parts[pid] = mask; tracks[pid] = {}
            if mask.any():
                for tid in cores[pid]:
                    if tid not in positions[f]:
                        raise ValueError('active core coordinate missing')
                    tracks[pid][tid] = positions[f][tid]
        normalized.append(dict(frame_id=source['frame_id'], time_seconds=source['time_seconds'],
            image_sha256=source['image']['sha256'], whole_sha256=source['mask']['sha256'],
            parts=parts, tracks=tracks))
    return dict(window_id=number, frames=normalized)


def build_review(result, rows, root, output, *, check=lambda: None):
    """Automatic evidence-only identity stage. Whole/part PNG bytes are untouched."""
    root = Path(root).resolve(strict=True); out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(root):
        raise ValueError('new child review directory required')
    records = result['windows']; ids = [r['window_id'] for r in records]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError('unique ordered windows required')
    out.mkdir(parents=True)
    refs = {}; pair_reports = []; previews = []; groups = []; assignments = {}
    by_id = {r['window_id']:r for r in records}
    source_by_id = {r['frame_id']:r for r in rows}
    decision_cache = {}

    def compare(a, b):
        key = (a, b)
        if key not in decision_cache:
            left = _load_window(by_id[a], rows, root, refs, check)
            right = _load_window(by_id[b], rows, root, refs, check)
            decision = associate_pair(left, right)
            decision_cache[key] = decision
        return decision_cache[key]

    def has_link(decision, lp, rp):
        return any(x['left_part'] == lp and x['right_part'] == rp for x in decision['accepted_links'])

    for i, record in enumerate(records):
        check(); wid = record['window_id']; accepted = {}
        if i:
            previous = records[i-1]['window_id']
            decision = compare(previous, wid)
            report = dict(left_window=previous, right_window=wid, evidence=decision,
                          group_consistency=[])
            pair_reports.append(report)
            for edge in decision['accepted_links']:
                group = assignments[(previous, edge['left_part'])]
                conflicts = []
                for member in groups[group]['members']:
                    old = member['window_id']
                    if old == previous:
                        continue
                    overlap = set(f['frame_id'] for f in by_id[old]['frames']) & set(f['frame_id'] for f in record['frames'])
                    if len(overlap) >= 2 and not has_link(compare(old,wid), member['local_part'], edge['right_part']):
                        conflicts.append(member)
                report['group_consistency'].append(dict(left_part=edge['left_part'],right_part=edge['right_part'], conflicts=conflicts))
                if not conflicts:
                    accepted[edge['right_part']] = group
            # Show every candidate comparison in the exact same two RGB frames.
            left = _load_window(by_id[previous], rows, root, refs, check)
            right = _load_window(record, rows, root, refs, check)
            common = decision['shared_frame_ids']
            left_frames = {f['frame_id']:f for f in left['frames']}
            right_frames = {f['frame_id']:f for f in right['frames']}
            for j, pair in enumerate(decision['pairs']):
                for k, fid in enumerate(common[:6]):
                    check(); lf=left_frames[fid]; rf=right_frames[fid]
                    lp=pair['left_part']; rp=pair['right_part']
                    with Image.open(source_by_id[fid]['image']['path']) as image:
                        rgb=np.asarray(image.convert('RGB')).copy()
                    masks=[lf['parts'][lp],rf['parts'][rp]]
                    panels=[]
                    for mask,color in zip(masks,[(255,145,35),(40,190,255)]):
                        panel=rgb.copy(); panel[mask]=(.4*panel[mask]+.6*np.array(color)).astype(np.uint8)
                        panels.append(panel)
                    both=rgb.copy(); both[masks[0]]=(255,145,35); both[masks[1]]=(40,190,255)
                    both[masks[0]&masks[1]]=(255,255,255)
                    height,width=rgb.shape[:2]; canvas=Image.new('RGB',(width*3,height+42),(20,30,40))
                    for col, panel in enumerate([*panels,both]):
                        canvas.paste(Image.fromarray(panel),(col*width,42))
                    ImageDraw.Draw(canvas).text((10,10),f'{fid}  window {previous}/{lp} | window {wid}/{rp} | white = overlap',fill='white')
                    name=f'pair_{i:03d}_{j:03d}_{k:02d}.jpg';canvas.save(out/name,quality=88)
                    previews.append(dict(left_window=previous,right_window=wid,left_part=lp,right_part=rp,frame_id=fid,path=name))
        for pid in record['part_ids']:
            g=accepted.get(pid)
            if g is None:
                g=len(groups);groups.append(dict(candidate_sequence_id='sequence_%04d'%g,members=[]))
            if any(m['window_id']==wid for m in groups[g]['members']):
                raise ValueError('same-window identity collision')
            assignments[(wid,pid)]=g
            groups[g]['members'].append(dict(window_id=wid,local_part=pid))
    changed=[p for p,b in refs.items() if _binding(p)!=b]
    if changed:
        raise ValueError('inputs changed during linking')
    linked=sum(len(g['members'])-1 for g in groups)
    output_result=dict(schema='motion-window-association-v1',object_id=result['object_id'],
        status='CANDIDATE_LINKS_ESTABLISHED' if linked else 'NO_CROSS_WINDOW_LINK',
        accepted_link_count=linked,window_count=len(records),candidate_sequence_count=len(groups),
        groups=groups,comparisons=pair_reports,previews=previews,
        masks_modified=False,geometry_accepted=False,physical_identity='NOT_ESTABLISHED',
        long_range_identity='TRANSITIVE_CANDIDATE_ONLY_NO_INDEPENDENT_PROOF',
        new_model_executions=0,source_mask_quality='NOT_EVALUATED')
    _save(out/'RESULT.json',output_result);_save(out/'INPUT_BINDINGS.json',list(refs.values()))
    body=['<!doctype html><meta charset="utf-8"><title>跨窗口运动区域对应</title>',
        '<style>body{max-width:1250px;margin:30px auto;background:#14202d;color:#edf5ff;font:17px system-ui;line-height:1.6}img{max-width:100%}a{color:#8cd0ff}td,th{padding:8px;border:1px solid #617180}table{border-collapse:collapse}</style>',
        '<h1>跨窗口运动区域对应</h1><p>状态：'+output_result['status']+'；已关联 '+str(linked)+' 对。</p>',
        '<p>只关联候选身份，原图和全部掩膜保持原字节。窗口内编号相同不代表同一部件。没有补洞、连接手指或生成新三维模型。</p>',
        '<p>左右两列取自同一个真实画面：橙色为前窗候选，蓝色为后窗候选；第三列白色是重合区域。低重合或缺少持续对应时保留未关联状态。</p>',
        '<p><a href="RESULT.json">逐项证据与拒绝原因</a> · <a href="INPUT_BINDINGS.json">绑定</a> · <a href="../index.html">所有帧</a></p>']
    for report in pair_reports:
        body.append('<h2>窗口 '+str(report['left_window'])+' → '+str(report['right_window'])+'</h2><pre>'+html.escape(json.dumps(report['evidence'],ensure_ascii=False,indent=2))+'</pre>')
    for item in previews:
        body.append('<details><summary>'+html.escape(item['frame_id']+' · '+str(item['left_window'])+'/'+item['left_part']+' → '+str(item['right_window'])+'/'+item['right_part'])+'</summary><img src="'+item['path']+'"></details>')
    body.append('<p>完整物理部件轮廓、遮挡后再识别、长期身份、几何改善仍未验证。短窗重新选点和分区可能导致同一物体上出现不同候选域；不能用降低门槛把不同区域冒认为同一区域。</p>')
    (out/'index.html').write_text(''.join(body),encoding='utf-8');check()
    return output_result
