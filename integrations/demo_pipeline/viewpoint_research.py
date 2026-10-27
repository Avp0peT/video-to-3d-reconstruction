"""Bounded, original-observation viewpoint research; never edits native tracks.

The exact S-1996 admission/construction and fitted engineering scores remain
separate. Neither a supplied COLMAP identity nor a small residual is truth.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import html
import time

from . import parallel_photo_audit as native
from .approximate_photo_bridge import verify_binding
from kozlov_lab.stereo.viewpoint_consistency import analyze


def group_pixels(ids, rows):
    """Keep any exact left OR right coordinate collisions in the same split."""
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError('one observation row per unique track required')
    parent = {i:i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = [{}, {}]
    for pid, row in zip(ids, rows):
        if len(row) != 4:
            raise ValueError('four pixel coordinates required')
        for side in (0, 1):
            xy = tuple(row[2*side:2*side+2])
            if xy in seen[side]:
                a, b = find(pid), find(seen[side][xy])
                parent[max(a,b)] = min(a,b)
            else:
                seen[side][xy] = pid
    groups = {}
    for pid in ids:
        groups.setdefault(find(pid), []).append(pid)
    return sorted((sorted(g) for g in groups.values()), key=lambda g:g[0])


def select_injective_six(ids, rows):
    """Declared input-domain selection, not an error-based correspondence repair."""
    selected, excluded, seen = [], [], [set(), set()]
    for pid, row in zip(ids, rows):
        left, right = tuple(row[:2]), tuple(row[2:])
        sides = [side for side, xy in enumerate((left, right)) if xy in seen[side]]
        if sides:
            excluded.append(dict(point_id=pid, reason='exact_coordinate_collision', sides=sides,
                                 collision_pixels=[list((left,right)[side]) for side in sides]))
            continue
        selected.append(pid); seen[0].add(left); seen[1].add(right)
        if len(selected) == 6:
            break
    return selected, excluded


def make_plan(pair, images, points):
    ids = sorted(pid for pid, p in points.items() if all(i in p['track'] for i in pair))
    if len(ids) > 4096:
        raise ValueError('pair track budget exceeded')
    rows = [[v for i in pair for v in images[i]['observations'][points[pid]['track'][i]][:2]] for pid in ids]
    groups = group_pixels(ids, rows)
    train = {pid for group in groups[::2] for pid in group}
    selected, excluded = select_injective_six(ids, rows)
    return dict(image_ids=list(pair), point_ids=ids, raw_pixels=rows,
        binary64_hex=[[float(v).hex() for v in row] for row in rows],
        point2D_indices=[[points[pid]['track'][i] for i in pair] for pid in ids],
        coordinate_groups=groups, duplicate_groups=[g for g in groups if len(g)>1],
        train_indices=[j for j,pid in enumerate(ids) if pid in train],
        holdout_indices=[j for j,pid in enumerate(ids) if pid not in train],
        nominal_six=ids[:6], injective_six=selected, injective_scan_exclusions=excluded,
        ambiguous_tracks_excluded=[pid for pid,p in points.items()
            if all(i in p['image_ids'] for i in pair) and not all(i in p['track'] for i in pair)])


def exact(first, second, path):
    result, encoded, independent = native._run_exact(first, second)
    native.write_json(path, encoded)
    if result.get('certificate') is not None and independent.get('status') != 'VERIFIED':
        raise ValueError('independent exact certificate rejected')
    return dict(status=result['status'], reason=result['reason'],
        algorithm_called=result.get('algorithm_called'), independent=independent,
        evidence=path.name)


def load_input(prior_input):
    frozen = native.read_json(prior_input)
    if frozen.get('schema') != 'photo-parallel-selected-observations-v1':
        raise ValueError('prior original-pixel audit INPUT required')
    refs = list(frozen['source_bindings']) + [native.binding(prior_input)]
    for ref in refs:
        verify_binding(ref)
    paths = {}
    for ref in refs:
        p = Path(ref['path'])
        if p.name in ('run.json', 'images.bin', 'points3D.bin', 'cameras.bin', 'frames_map.jsonl'):
            if p.name in paths:
                raise ValueError('duplicate source binding')
            paths[p.name] = p
    if len(paths) != 5:
        raise ValueError('complete native source bindings required')
    root = paths['run.json'].parent
    run = native.read_json(paths['run.json'])
    if run.get('contract_version') not in ('sfm-native-object-v1', 'sfm-native-object-pinhole-v1') or run.get('status') not in ('SUCCESS','PARTIAL'):
        raise ValueError('native object run required')
    model = [m for m in run['models'] if native.under(root,m['binary_path']) == paths['images.bin'].parent]
    if len(model) != 1 or not model[0].get('usable') or not model[0].get('roundtrip_verified'):
        raise ValueError('one usable verified native model required')
    model = model[0]
    if any(paths[n].parent != paths['images.bin'].parent for n in ('points3D.bin','cameras.bin')):
        raise ValueError('native model files must share a bound model directory')
    images, points = native.read_native(paths['images.bin'],paths['points3D.bin'])
    if (not 2 <= len(images) <= 12 or sorted(images) != sorted(model['registered_image_ids'])
            or len(points) != model['num_points3D']):
        raise ValueError('model identity/count or twelve-view bound failed')
    frames = []
    frame_rows = [native.loads(s) for s in paths['frames_map.jsonl'].read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    for ident in sorted(images):
        matches = [(row,reg) for row in frame_rows for reg in row['registrations']
                   if str(reg['model_id']) == str(model['model_id']) and reg['image_id'] == ident]
        if len(matches) != 1:
            raise ValueError('one registered frame per native identity required')
        row, reg = matches[0]
        if (row['status'] != 'REGISTERED' or reg['camera_id'] != images[ident]['camera_id']
                or row['staged_image_name'] != images[ident]['name']
                or row['upstream_io_run_id'] != run['upstream_io_run_id']
                or type(row['input_index']) is not int or row['input_index'] < 0):
            raise ValueError('frame identity differs')
        ref = native.binding(native.under(root,row['staged_image_name']))
        if ref['sha256'] != row['canonical_image_sha256']:
            raise ValueError('canonical source byte drift')
        refs.append(ref)
        frames.append(dict(image_id=ident,input_index=row['input_index'],frame_id=row['frame_id'],image=ref))
    refs = list({r['path']:r for r in refs}.values())
    return images, points, frames, refs


def run(prior_input, output):
    prior_input, output = Path(prior_input).resolve(strict=True), Path(output).resolve()
    images, points, frames, refs = load_input(prior_input)
    if output.exists() or any(output == Path(r['path']) or output in Path(r['path']).parents for r in refs):
        raise ValueError('fresh output outside bound inputs required')
    sfm_root = next(Path(r['path']).parent for r in refs if Path(r['path']).name == 'run.json')
    if output.is_relative_to(sfm_root) or output.is_relative_to(prior_input.parent):
        raise ValueError('output must be outside both immutable source runs')
    output.mkdir(parents=True, exist_ok=False)
    pairs = [make_plan(pair,images,points) for pair in combinations(sorted(images),2)]
    implementation = [native.binding(p) for p in (Path(__file__),
        Path(__file__).parents[2]/'src/kozlov_lab/stereo/viewpoint_consistency.py',
        Path(native.__file__),Path(__file__).parents[2]/'src/kozlov_lab/stereo/official/given_parallel_correspondence.py',
        Path(__file__).parents[2]/'src/kozlov_lab/stereo/official/bounded_parallel_correspondence.py',
        Path(__file__).parents[2]/'src/kozlov_lab/stereo/official/parallel_construction.py')]
    native.write_json(output/'PLAN.json',dict(schema='original-viewpoint-research-plan-v1',frames=frames,
        source_bindings=refs,implementation=implementation,pairs=pairs,soft_seconds=120,
        split='connected exact pixel-collision groups ordered by minimum track ID; alternating groups train/holdout',
        exact_subset='ascending first six IDs distinct in each view; prior nominal-six refusal retained separately',
        negative_controls='right holdout cyclic one and reverse; both frozen before fitting; no refit',
        undistortion=False,calibration_used=False,xyz_read_for_fitting=False,
        correspondence_source='existing COLMAP tracks, not independent truth',
        quality_threshold=None,physical_accuracy=None))
    # All membership and coordinate decisions have landed before model fitting.
    body = [(0,0,0),(1,0,0),(0,1,0),(1,1,1),(2,3,2),(3,1,-1)]
    first = {i:(p[0],p[1]) for i,p in enumerate(body)}
    second = {i:(p[0]+p[2],p[1]+2*p[2]) for i,p in enumerate(body)}
    positive = exact(first,second,output/'SYNTHETIC_POSITIVE.json')
    wrong = {i:second[(i+1)%6] for i in first}
    negative = exact(first,wrong,output/'SYNTHETIC_WRONG.json')
    if positive['status'] != 'CONSTRUCTED' or negative['status'] != 'INCOMPATIBLE_EXACT_PARALLEL':
        raise ValueError('synthetic exact discrimination control failed')
    deadline, results = time.monotonic()+120, []
    for number, p in enumerate(pairs):
        if time.monotonic() >= deadline:
            results.append(dict(image_ids=p['image_ids'],status='NOT_RUN_BUDGET')); continue
        directory = output/f'pair_{p["image_ids"][0]}_{p["image_ids"][1]}'
        directory.mkdir()
        row = dict(image_ids=p['image_ids'],common_count=len(p['point_ids']),
            train_count=len(p['train_indices']),holdout_count=len(p['holdout_indices']),
            duplicate_groups=p['duplicate_groups'],injective_six=p['injective_six'],
            status='DIAGNOSTIC_COMPLETE')
        if len(p['nominal_six']) == 6:
            a,b = native.views_for(p['image_ids'],p['nominal_six'],images,points)
            row['nominal_exact'] = exact(a,b,directory/'EXACT_NOMINAL.json')
        if len(p['injective_six']) == 6:
            a,b = native.views_for(p['image_ids'],p['injective_six'],images,points)
            row['exact_raw'] = exact(a,b,directory/'EXACT_RAW.json')
            ids = p['injective_six']
            row['exact_wrong'] = exact(a,{pid:b[ids[(j+1)%6]] for j,pid in enumerate(ids)},directory/'EXACT_WRONG.json')
            row['exact_wrong_right_ids'] = ids[1:]+ids[:1]
        else:
            row['exact_raw'] = row['exact_wrong'] = dict(status='NOT_EVALUATED',reason='fewer than six injective observations')
        if len(p['point_ids']) >= 12 and len(p['train_indices']) >= 8 and len(p['holdout_indices']) >= 2:
            scoring = analyze(p['raw_pixels'],p['train_indices'],p['holdout_indices'])
            for kind in ('affine_flat','fundamental'):
                if scoring[kind]['status'] == 'EVALUATED':
                    for ctrl in scoring[kind]['negative_controls'].values():
                        ctrl['right_source_point_ids'] = [p['point_ids'][i] for i in ctrl['right_source_indices']]
                        ctrl['changed_identity_count'] = sum(a != b for a,b in zip(p['holdout_indices'],ctrl['right_source_indices']))
                        ctrl['changed_coordinate_count'] = sum(p['raw_pixels'][a][2:] != p['raw_pixels'][b][2:]
                            for a,b in zip(p['holdout_indices'],ctrl['right_source_indices']))
                        ctrl['fixed_identity_count'] = len(p['holdout_indices'])-ctrl['changed_identity_count']
                        ctrl['changed_identity_same_pixels_count'] = ctrl['changed_identity_count']-ctrl['changed_coordinate_count']
                        ctrl['right_source_binary64_hex'] = [p['binary64_hex'][i][2:] for i in ctrl['right_source_indices']]
                        ctrl['right_displacement_pixels'] = [sum((p['raw_pixels'][a][k]-p['raw_pixels'][b][k])**2 for k in (2,3))**.5
                            for a,b in zip(p['holdout_indices'],ctrl['right_source_indices'])]
                        changed = [j for j,d in enumerate(ctrl['right_displacement_pixels']) if d != 0]
                        correct = scoring[kind]['correct']['residual_pixels']
                        signs = [0 if abs(ctrl['residual_difference_pixels'][j]) <= 1e-10*max(1.,correct[j],ctrl['residual_pixels'][j])
                            else (1 if ctrl['residual_difference_pixels'][j]>0 else -1) for j in changed]
                        ctrl['all_slot_counts_scope'] = 'includes fixed identities and unchanged-coordinate aliases'
                        ctrl['changed_coordinate_paired_counts'] = dict(count=len(changed),
                            increase=signs.count(1),decrease=signs.count(-1),equal=signs.count(0))
            native.write_json(directory/'SCORES.json',scoring)
            row['scores'] = {kind:scoring[kind] for kind in ('affine_flat','fundamental')}
        else:
            row['scores'] = {kind:dict(status='NOT_EVALUATED',reason='insufficient frozen split') for kind in ('affine_flat','fundamental')}
        results.append(row)
        native.write_json(directory/'RESULT.json',row)
    for ref in refs+implementation:
        verify_binding(ref)
    result = dict(schema='original-viewpoint-research-result-v1',status='DIAGNOSTICS_NOT_GEOMETRY_ACCEPTED',
        view_count=len(images),track_count=len(points),pairs=results,
        source_unchanged=True,synthetic_control=dict(positive=positive,wrong=negative),
        exact_status_counts=dict(Counter(r.get('exact_raw',{}).get('status',r['status']) for r in results)),
        raw_pixels_modified=False,original_tracks_removed=0,model_created=False,
        correspondence_verified=False,quality_status='NOT_EVALUATED',physical_accuracy=None,
        limitations=['Raw exact rejection does not distinguish noise, perspective, lens distortion, or mismatches.',
            'Injective subset selection is disclosed input-domain selection, not point correction.',
            'Both engineering models use raw distorted pixels; their residual definitions and freedoms differ.',
            'Negative controls are specified identity alternatives, not representative real matching-error statistics.',
            'Reverse controls may have fixed points; changed-coordinate counts are recorded.',
            'Shared tracks and views make pair results dependent; no pooled independent accuracy claim.',
            'Training and holdout avoid exact coordinate duplication but remain COLMAP-selected observations.',
            'No new correspondence search, repair, surface generation, or source-pipeline acceptance.'])
    native.write_json(output/'RESULT.json',result)
    report(output,result,frames)
    return result


def report(output,result,frames):
    frame_index = {f['image_id']:f['input_index'] for f in frames}
    def metric(r, kind, control=None):
        score = r.get('scores',{}).get(kind,{})
        if score.get('status') != 'EVALUATED': return '未评估'
        item = score['correct'] if control is None else score['negative_controls'][control]
        return f'{item["rms"]:.3f}'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PPP 跨视角研究检验</title>'
    page += '<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#223146}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd5df;text-align:left}aside{padding:16px;background:#fff0cc}a{color:#075f9c}</style>'
    page += '<h1>PPP：跨视角对应能否由研究约束检验？</h1><aside>原始像素、对应和模型均未改动。论文精确检验与工程近似评分分开；此页没有生成新的合格模型。</aside>'
    page += f'<p>{result["view_count"]} 个已注册视角，{result["track_count"]} 条既有轨迹，{len(result["pairs"])} 组视角。先冻结分组，再训练一次；右图留出点循环错配、反序错配均不重新拟合。</p>'
    page += '<p>名义最前六点的结果单独记录，遇到同坐标不同 ID 时保留拒绝。下表另选升序首六个在两图各自互异的点，仅用于输入域检查，不删除原轨迹。重复坐标连通组整体进入训练或留出，避免精确重复穿过划分。</p>'
    page += '<h2>原始像素的精确条件与冻结留出评分</h2><p>数值单位是各自定义的像素残差，不是毫米精度；两模型不可按残差数值直接排名。错配数值变大只说明能区分该负控制，不证明原对应正确。</p>'
    page += '<table><tr><th>输入帧</th><th>共同/训练/留出</th><th>精确六点</th><th>仿射 原/循环/反序</th><th>F 原/循环/反序</th></tr>'
    for r in result['pairs']:
        ids=r['image_ids']; key=f'pair_{ids[0]}_{ids[1]}'
        label='/'.join(str(frame_index[i]) for i in ids)
        size='/'.join(str(r.get(k,'—')) for k in ('common_count','train_count','holdout_count'))
        status=r.get('exact_raw',{}).get('status',r['status'])
        scores=[' / '.join(metric(r,k,c) for c in (None,'cyclic_one','reverse')) for k in ('affine_flat','fundamental')]
        page += f'<tr><td><a href="{key}/RESULT.json">{label}</a></td><td>{size}</td><td>{html.escape(status)}</td><td>{scores[0]}</td><td>{scores[1]}</td></tr>'
    page += '</table><h2>可证与不可证</h2><p>合成平行投影：实际 AlgT′ 构造并独立核验通过；预设循环错配：精确不相容证书通过。真实点没有被投影到代理平面上，所以不会把人为施加的相容性当作照片通过。</p>'
    page += '<p>真实照片若被拒绝，原因可能包含透视、畸变、像素噪声或错误对应，本实验不能单独归因。当前诊断不自动删点，也不自动补面。</p>'
    page += '<p><a href="PLAN.json">预冻结像素、身份与分组</a> · <a href="RESULT.json">完整结果和局限</a> · <a href="SYNTHETIC_POSITIVE.json">AlgT′ 正控制</a> · <a href="SYNTHETIC_WRONG.json">错误对应反例</a></p></html>'
    (output/'report.html').write_text(page,encoding='utf-8')
