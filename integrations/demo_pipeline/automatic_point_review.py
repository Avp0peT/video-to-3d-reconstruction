"""Current-run point evidence automation; missing evidence never becomes geometry."""
from collections import Counter
from copy import deepcopy
import html
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image

from . import temporal_review as inputs
from . import parallel_photo_audit as io
from kozlov_lab.stereo.coverage_tracks import track_coverage, profile_for
from kozlov_lab.stereo.mask_association import associate_candidates
from kozlov_lab.stereo.local_motion import evaluate_local_motion
from kozlov_lab.stereo.exterior_witness import track_reference_exterior
from kozlov_lab.stereo.boundary_evidence import assess_candidate_boundary
from kozlov_lab.stereo.candidate_point_policy import decide_candidate_points
from kozlov_lab.stereo.candidate_point_consumer import consume_point_decisions
from kozlov_lab.stereo.candidate_geometry_inputs import build_anchor_groups
from kozlov_lab.stereo.candidate_anchor_geometry import evaluate_anchor_group
from scripts.review_mask_association import trajectories

ROOT = Path(__file__).resolve().parents[2]
PROFILE = dict(id='automatic-point-review-v17', max_frames=192, max_candidates_per_frame=8,
    reference_distance_positions=6, exterior_max_sequence_frames=7,
    tracking_profile='whole-video-temporal-support-v2',
    threads=2, wall_seconds=180, memory_bytes=2 << 30,
    geometry_accepted=False, full_mask_accepted=False, new_point_seeding_allowed=False,
    pose_estimation_enabled=True, native_pose_registration_enabled=False, projective_camera_diagnostics=True,
    dense_saved_camera_anchor_routes=True, conditional_query_camera_extension=True,
    measured_region_seeding=True, bounded_region_growth=True, whole_area_image_alignment=True,
    foreground_overlap_regions=True, spatially_connected_region_selection=True, neighborhood_candidate_review=True, retry_count=0)


def save(path, data):
    # Mutable task-local checkpoints only; callers require a fresh output root.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    os.replace(temporary, path)


class CountCV:
    def __init__(self, cv):
        self.cv = cv; self.flow_calls = 0
    def __getattr__(self, name):
        return getattr(self.cv, name)
    def calcOpticalFlowPyrLK(self, *args, **kwargs):
        self.flow_calls += 1
        return self.cv.calcOpticalFlowPyrLK(*args, **kwargs)


def reference_pair(frames, position):
    """Position distance is distinct from input_index/decode_index or time."""
    before = [i for i in range(max(0, position-6), position) if frames[i]['original_mask_status'] == 'OK']
    after = [i for i in range(position+1, min(len(frames), position+7)) if frames[i]['original_mask_status'] == 'OK']
    return (before[-1], after[0]) if before and after else None


def canonical_tracks(tracking, frames):
    """Explicit local-position -> original-input identity map; never range(N) IDs."""
    indices = [f['input_index'] for f in frames]
    ids = [f['frame_id'] for f in frames]
    if (any(type(i) is not int or i < 0 for i in indices) or indices != sorted(set(indices))
            or len(ids) != len(set(ids))):
        raise ValueError('frame identity map must be unique and increasing')
    out = deepcopy(tracking)
    for track in out['tracks']:
        for obs in track['observations']:
            position = obs['frame_index']
            if type(position) is not int or not 0 <= position < len(frames):
                raise ValueError('observation outside explicit frame position map')
            obs['sequence_position'] = position
            obs['frame_index'] = indices[position]
    return out


def query_witnesses(ledger, pair, position):
    out = []
    for witness in ledger['accepted_witnesses']:
        rows = witness['observations']
        if [r['frame_index'] for r in rows] != list(range(pair[0], pair[1]+1)):
            raise ValueError('exterior witness missing actual intermediate observations')
        q = rows[position-pair[0]]
        out.append(dict(witness_id=f'{pair[0]}_{pair[1]}:{witness["witness_id"]}',
            query_point_xy=q['xy'], reference_exterior_witness=True, verified_background=False,
            selection_condition='RETROSPECTIVE_SURVIVOR_CONDITIONED'))
    return out


def candidate_bindings(discovery, frames, refs):
    path = Path(discovery['stages']['segmentation']['run_dir']).resolve(strict=True) / 'manifest.json'
    manifest = inputs.read_bound(path, refs)
    rows = inputs.unique(manifest['frames'], 'input_index')
    bound = {}
    for frame in frames:
        row = rows[frame['input_index']]
        if row['frame_id'] != frame['frame_id'] or row['status'] != frame['original_mask_status']:
            raise ValueError('candidate/source frame differs')
        candidates = row['candidates']
        if not isinstance(candidates, list) or len(candidates) > PROFILE['max_candidates_per_frame']:
            raise ValueError('candidate count outside frozen bound')
        seen = set(); verified = []
        for candidate in candidates:
            cid = candidate['candidate_id']
            if type(cid) is not str or not cid or cid in seen:
                raise ValueError('invalid or duplicate candidate identity')
            seen.add(cid)
            p = io.under(path.parent, candidate['mask_path'])
            ref = inputs.bind(p, refs, dict(sha256=candidate['mask_sha256']))
            with Image.open(p) as image:
                if image.mode != 'L' or image.size != (frame['width'], frame['height']):
                    raise ValueError('candidate mask dimensions or mode differ')
                mask = np.asarray(image)
                if not np.isin(mask, [0, 255]).all():
                    raise ValueError('candidate labels differ')
            verified.append(dict(candidate_id=cid, binding=ref))
        if row['status'] == 'NO_MATCH' and candidates:
            raise ValueError('NO_MATCH cannot contain candidate masks')
        if row['status'] == 'NEEDS_SELECTION' and not candidates:
            raise ValueError('NEEDS_SELECTION without any candidate is malformed source evidence')
        bound[frame['input_index']] = verified
    return bound


def query_geometry(group, cameras):
    rows = []
    for query in group['queries']:
        c = cameras.get(query['input_index'])
        xyz = group['geometry']['xyz']
        row = dict(input_index=query['input_index'], frame_id=query['frame_id'],
                   geometry_accepted=False, status='NOT_EVALUATED', error_px=None)
        if c is None:
            row['reason'] = 'NO_SAVED_CAMERA'
        elif xyz is None:
            row['reason'] = 'NO_SUPPORTED_ANCHOR_POINT'
        elif c.get('camera_model') != 'SIMPLE_RADIAL' or len(c.get('params', [])) != 4 or c['params'][3] != 0:
            row['reason'] = 'UNSUPPORTED_CAMERA_MODEL'
        else:
            local = np.asarray(c['R']) @ np.asarray(xyz) + np.asarray(c['t'])
            if not np.isfinite(local).all() or local[2] <= 0:
                row['reason'] = 'NON_POSITIVE_DEPTH'
            else:
                pixel = c['params'][0] * local[:2] / local[2] + c['params'][1:3]
                error = float(np.linalg.norm(pixel-np.asarray(query['source_xy'])))
                if not np.isfinite(error):
                    raise ValueError('nonfinite query reprojection')
                row.update(status='CONDITIONAL_REPROJECTION_CHECK', reason=None, error_px=error,
                           within_2px=error <= 2., independent_quality_truth=False)
        rows.append(row)
    return rows


def analyse_sequence(frames, grays, masks, candidates, tracking, models, cv, output, budget, *, engines=None):
    """General current-sequence adapter. Optional engines allow isolated contract tests."""
    kernels = dict(trajectories=trajectories, associate=associate_candidates, local=evaluate_local_motion,
                   exterior=track_reference_exterior, boundary=assess_candidate_boundary,
                   policy=decide_candidate_points)
    if engines:
        kernels.update(engines)
    if not (len(frames) == len(grays) == len(masks) and 2 <= len(frames) <= PROFILE['max_frames']):
        raise ValueError('bounded complete frame arrays required')
    mapped = canonical_tracks(tracking, frames)
    decisions = []; point_decisions = []; ledger = []; pairs = {}
    registry = {f['input_index']: deepcopy(f) for f in frames}
    for position, frame in enumerate(frames):
        budget()
        index = frame['input_index']; candidate_rows = candidates[index]
        pair = reference_pair(frames, position)
        row = dict(input_index=index, sequence_position=position, frame_id=frame['frame_id'],
            original_status=frame['original_mask_status'], before_index=frames[pair[0]]['input_index'] if pair else None,
            after_index=frames[pair[1]]['input_index'] if pair else None, reference_positions=pair,
            candidates=candidate_rows, positive_evidence=[], exterior_evidence=[],
            assessment=None, status='NO_CANDIDATE', selected_candidate_id=None, publishable_for_geometry=False,
            association=None, local_motion=None, reason=None)
        if frame['original_mask_status'] not in ('OK', 'NO_MATCH', 'NEEDS_SELECTION'):
            row.update(status='SOURCE_FRAME_UNAVAILABLE', reason='original frame error or NOT_RUN preserved')
            decisions.append(row); save(output / 'DECISIONS.json', decisions)
            continue
        if candidate_rows:
            loaded = []
            for c in candidate_rows:
                with Image.open(c['binding']['path']) as image:
                    loaded.append(dict(candidate_id=c['candidate_id'], mask=np.asarray(image).copy()))
            if pair:
                before, after = pair
                points, ids, segments = kernels['trajectories'](tracking, before, position, after, len(frames),
                                                               masks[before], masks[after], cv)
                row['geometry_segments'] = segments
                if ids:
                    row['association'] = kernels['associate'](masks[before], masks[after], *points, ids, loaded, cv)
                    row['local_motion'] = kernels['local'](masks[before], masks[after], *points, ids, cv)
                    row['positive_evidence'] = row['local_motion']['profiles']['local_affine']['measured_query_evidence']
                if (frame['original_mask_status'] == 'NEEDS_SELECTION'
                        and after-before+1 > PROFILE['exterior_max_sequence_frames']):
                    row['exterior_status'] = 'EXTERIOR_REFERENCE_SPAN_UNSUPPORTED'
                    row['reason'] = 'Original references span more than 7 frames; no exterior extrapolation'
                elif frame['original_mask_status'] == 'NEEDS_SELECTION':
                    key = f'{before}_{after}'
                    if key not in pairs:
                        budget()
                        # Use the same RGB->OpenCV gray convention as the reviewed exterior path.
                        exterior_gray = []
                        for j in range(before, after+1):
                            with Image.open(frames[j]['path']) as image:
                                exterior_gray.append(cv.cvtColor(np.asarray(image.convert('RGB')), cv.COLOR_RGB2GRAY))
                        pairs[key] = kernels['exterior'](exterior_gray, masks[before], masks[after],
                                                        list(range(before, after+1)), cv)
                        save(output / 'EXTERIOR_PAIRS.json', pairs)
                    row['exterior_evidence'] = query_witnesses(pairs[key], pair, position)
            else:
                row['reason'] = 'NO_TWO_SIDED_ORIGINAL_REFERENCE'
            with Image.open(frame['path']) as image:
                rgb = np.asarray(image.convert('RGB'))
            row['assessment'] = kernels['boundary'](rgb, loaded, row['positive_evidence'], row['exterior_evidence'], cv)
            row['status'] = 'EVIDENCE_ONLY'
        decision = kernels['policy'](row)
        if decision['status'] == 'INVALID_EVIDENCE':
            raise ValueError('point policy rejected generated evidence: ' + decision.get('reason', ''))
        decision.update(input_index=index, frame_id=frame['frame_id'])
        row['point_decision'] = decision
        decisions.append(row); point_decisions.append(decision)
        if decision['status'] == 'PROVISIONAL_POINT_SUPPORT_ONLY':
            chosen = next(c for c in candidate_rows if c['candidate_id'] == decision['provisional_candidate_id'])
            ledger.append(dict(decision, candidate_binding=chosen['binding']))
        save(output / 'DECISIONS.json', decisions)
        save(output / 'POINT_SUPPORT_LEDGER.json', ledger)
    annotations = consume_point_decisions(mapped, point_decisions, 0)
    save(output / 'POINT_ANNOTATIONS.json', annotations)
    # Pixel checks and camera geometry apply only to tracks actually in the provisional ledger.
    selected = {o['track_id'] for d in ledger for o in d['supported_observations']}
    for track in mapped['tracks']:
        if track['track_id'] not in selected:
            continue
        for obs in track['observations']:
            f = registry[obs['frame_index']]; position = obs['sequence_position']; mask = masks[position]
            if f['original_mask_status'] != 'OK' or mask is None:
                continue
            x, y = np.floor(obs['xy']).astype(int)
            inside = bool(0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y,x] == 255)
            f.setdefault('measured_point_mask_checks', []).append(dict(track_id=track['track_id'],
                frame_index=obs['frame_index'], source_xy=deepcopy(obs['xy']), inside_original_mask=inside))
    prepared = build_anchor_groups(mapped, ledger, registry, models, 0)
    camera_models = {m['model_key']: m['cameras'] for m in models}
    for group in prepared['groups']:
        budget()
        if len(group['anchors']) > 32:
            group['geometry'] = dict(status='NOT_EVALUATED_ANCHOR_BUDGET', xyz=None,
                reason='More than 32 measured anchors retained; no implicit selection or truncation',
                geometry_accepted=False, publishable_for_geometry=False)
        else:
            group['geometry'] = evaluate_anchor_group(group['anchors'])
        group['query_checks'] = query_geometry(group, camera_models.get(group['model_key'], {}))
    save(output / 'GEOMETRY_GROUPS.json', prepared['groups'])
    save(output / 'MASK_SELECTION_LEDGER.json', [])
    return dict(decisions=decisions, geometry_groups=prepared['groups'],
        counts=dict(frames=len(frames), processed_frames=len(decisions),
            source_statuses=dict(Counter(f['original_mask_status'] for f in frames)),
            point_statuses=dict(Counter(d['status'] for d in point_decisions)),
            provisional_frames=len(ledger), provisional_observations=sum(len(d['supported_observations']) for d in ledger),
            geometry_statuses=dict(Counter(g['geometry']['status'] for g in prepared['groups'])),
            original_observations=len(annotations['rows'])),
        frame_map=[dict(sequence_position=i, input_index=f['input_index'], frame_id=f['frame_id'],
                        decode_index=f.get('decode_index'), pts=f.get('pts')) for i,f in enumerate(frames)])


def render(output, result):
    esc = html.escape
    rows = result.get('decisions', [])
    table = ''.join(f'<tr><td>{r["input_index"]}</td><td>{esc(r["original_status"])}</td><td>{esc(r.get("point_decision",{}).get("status",r["status"]))}</td><td>{len(r["positive_evidence"])}</td><td>{len(r["exterior_evidence"])}</td><td>{esc(str(r["before_index"]))} / {esc(str(r["after_index"]))}</td></tr>' for r in rows)
    original = result.get('original_anchor_review', {})
    original_html = ''
    if original:
        original_html = ('<h2>原始前景与径向相机：自动检查</h2><p>'
            + esc(original.get('status', 'NOT_EVALUATED'))
            + '。固定相机与原始畸变像素；起始点仅在单独双向复核通过后纳入，原strong标记和空误差保留。'
            + '负径向系数的中心逆解是明确假设；结果不写回原生模型。</p><pre>'
            + esc(json.dumps(original.get('counts', {}), ensure_ascii=False, indent=2))
            + '</pre><p><a href="ORIGINAL_ANCHOR_DIAGNOSTICS.json">逐组准入、径向分支、拟合与逐视图留一结果</a></p>')
    seed = result.get('seed_pair_validation')
    if seed:
        original_html += ('<h2>自动起始点复核</h2><p>重算同一相邻采样帧对的双向光流，核对原始物体mask与保存坐标。'
            + '这是重复的图像对一致性检查，不是独立身份真值；留一核查也受准入条件影响。'
            + '通过后优先检查三视图组，几何与像素门限保持不变。</p><pre>'
            + esc(json.dumps(seed['counts'], ensure_ascii=False, indent=2))
            + '</pre><p><a href="SEED_PAIR_VALIDATION.json">实际调用、原始输出及排除原因</a></p>')
    batches=result.get('batch_execution',{})
    if batches:
        batch_links=' · '.join(f'<a href="batches/{k}/BATCH_STATUS.json">{label}批次状态</a>'
            for k,label in [('seed','起始点'),('geometry','几何')] if k in batches and batches[k]['status']!='NOT_PLANNED')
        original_html += ('<h2>自动分批进度</h2><p>每批最多32组、每阶段最多3批；所有阶段共享原180秒截止和2GiB Job预算。'
            + '失败不补位，不重复成功批次；部分证据保留，未完成不发布为成功候选。</p><pre>'
            + esc(json.dumps({k:{a:b for a,b in v.items() if a not in ('batches','events')} for k,v in batches.items()},ensure_ascii=False,indent=2))
            + '</pre><p>'+batch_links+'</p>')
    clouds = result.get('candidate_clouds')
    if clouds and result.get('source_unchanged') is True and result['status'] == 'AUTO_POINT_REVIEW_COMPLETE':
        original_html += ('<h2>自动分级候选点云</h2><p>' + esc(clouds['status'])
            + '。按原模型、证据等级与径向分支分文件，尚未接受为原生模型点。</p><pre>'
            + esc(json.dumps(clouds['counts'], ensure_ascii=False, indent=2))
            + '</pre><p><a href="candidate_clouds/index.html">查看三维候选点层、PLY与拒绝原因</a></p>')
    extension = result.get('view_extension')
    if extension and result.get('source_unchanged') is True and result['status'] == 'AUTO_POINT_REVIEW_COMPLETE':
        original_html += ('<h2>自动相邻视角对应扩展</h2><p>从两个已有源视图沿相邻帧补测对应，'
            '每一步都核对双向光流与原始物体mask。通过仅表示条件一致性；查询帧内参未建立，不自动借用邻帧焦距。</p><pre>'
            + esc(json.dumps(extension['counts'], ensure_ascii=False, indent=2))
            + '</pre><p><a href="view_extension/index.html">查看新增对应、叠图与注册阻断原因</a></p>')
    elif result.get('view_extension_progress', {}).get('events'):
        original_html += '<p>视角扩展未完整完成；<a href="view_extension/EVENTS.jsonl">已写测量与中断证据</a>保留。</p>'
    intermediate = result.get('intermediate_frames')
    if intermediate and result.get('source_unchanged') is True:
        original_html += ('<h2>自动补入原视频中间帧</h2><p>按失败区间自动补帧，固定同一批起点、终点与LK阈值，比较直接跟踪和逐帧跟踪。'
            '新增帧未独立分割，诊断通过不代表物体身份或三维几何通过。</p><pre>'
            + esc(json.dumps(intermediate, ensure_ascii=False, indent=2, default=str)[:2500])
            + '</pre><p><a href="intermediate_frames/index.html">查看原画面、逐点对照和失败原因</a></p>')
        if intermediate.get('population'):
            original_html += ('<h2>物体内全部候选的补帧复核</h2><p>按原始物体掩膜保留新点与前序跟踪点，分批对照直接与逐帧路径；二维覆盖和持续步数不代表三维质量。</p>'
                + '<p><a href="intermediate_frames/population/index.html">查看数百个候选点、覆盖率、生命周期和逐点交互图</a></p>')
            if intermediate['population'].get('regions'):
                original_html += '<h3>单点扩成区域并逐层生长</h3><p><a href="intermediate_frames/population/regions/index.html">查看批量实测、区域扩展和停止原因</a>；保留未知掩膜，点的二维连续性不等于三维完成。</p>'
                if intermediate['population']['regions'].get('area_alignment'):
                    original_html += '<h3>整块区域直接配准</h3><p><a href="intermediate_frames/population/regions/whole_area/index.html">查看区域整体平移和形变、拉回图与残差</a>；此分支由区域纹理联合估计变换，不把派生网格当独立对应点。</p>'
                if intermediate['population']['regions'].get('overlap_regions'):
                    original_html += '<h3>物体前景重叠区域网络</h3><p><a href="intermediate_frames/population/regions/overlap/index.html">查看前景覆盖、区域间运动一致性及未测区</a>；重叠只增加相互核对，不重复计独立几何证据。</p>'
            if intermediate['population'].get('short_tracks'):
                original_html += ('<h2>保留连续至少三帧的短轨迹</h2><p>后续失败保留此前真实前缀；分别检查同模型相机与物体证据，缺位姿不借邻帧补齐。</p>'
                    + '<p><a href="intermediate_frames/population/short_tracks/index.html">查看短轨迹、原图与几何阻塞</a></p>')
            if intermediate['population'].get('camera_diagnostics'):
                original_html += ('<h2>短轨迹投影与极线验证</h2><p>自动连接条件三维锚点，冻结训练与留出点，比较实际观测与预测；不直接升级为相机或三维模型。</p>'
                    + '<p><a href="intermediate_frames/population/camera_diagnostics/index.html">查看原图、投影误差与错配对照</a></p>')
            if intermediate['population'].get('anchor_routes'):
                original_html += ('<h2>向已知视角自动补跟踪</h2><p>短轨迹源点沿另一时间方向连接两个已存相机；记录三视图观测、条件几何与未通过原因。</p>'
                    + '<p><a href="intermediate_frames/population/anchor_routes/index.html">查看对应图像、点云与补充投影检验</a> · '
                    + '<a href="intermediate_frames/population/anchor_routes/query_cameras/index.html">查询相机与新增条件点</a></p>')
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动点证据复核</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:16px;color:#24334a}}aside{{background:#fff2cc;padding:18px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap}}</style>
<h1>本次建模 · 自动点证据复核</h1><aside>{esc(result['status'])}<br>诊断完成不等于完整建模。原始NO_MATCH/NEEDS_SELECTION不覆盖，未确认区域继续留空。</aside>
<p>自动按本次视频的实际帧顺序执行时序跟踪 → 候选关联 → 局部运动与边界证据 → 点级支持 → 已存相机几何。没有历史帧号或手工挑选区间。</p>
<p>缺相机标为待注册；本流程不猜测相机、不自动重跑重建，也不填补未知区域。物理身份、真实尺度与质量验收仍独立。</p>
<p>{esc(result.get('reason') or '')}</p><pre>{esc(__import__('json').dumps(result.get('counts',{}),ensure_ascii=False,indent=2))}</pre>
<table><tr><th>原始帧index</th><th>原始SAM状态</th><th>点级结论</th><th>正点证据</th><th>参考外侧证据</th><th>原始参考帧</th></tr>{table}</table>
{original_html}
<p><a href="RESULT.json">结果</a> · <a href="PLAN.json">冻结计划</a></p></html>'''
    (output / 'index.html').write_text(body, encoding='utf-8')


def review(pipeline, output):
    start = time.monotonic()
    pipeline = Path(pipeline).resolve(strict=True); output = Path(output).resolve()
    if (output.exists() or output == pipeline or output in pipeline.parents
            or (output.is_relative_to(pipeline) and output != pipeline/'point_review')):
        raise ValueError('fresh point_review directory or nonoverlapping sibling required')
    refs = []; counted = None; original = None; recorders={};extension_evidence=None;output.mkdir(parents=True)
    result = dict(status='INSUFFICIENT_DATA', reason=None, geometry_accepted=False,
        publishable_for_geometry=False, full_mask_accepted=False, new_point_seeding_allowed=False,
        new_model_points=0, pose_fits=0, neural_model_loads=0, native_saved_model_reads=0, segmentation_calls=0,
        gpu_calls=0, reconstruction_retries=0, tracking_calls=0, optical_flow_calls=0,
        source_unchanged=None, decisions=[], geometry_groups=[])
    def budget():
        if time.monotonic()-start > PROFILE['wall_seconds']:
            raise TimeoutError('automatic point review elapsed budget')
    try:
        discovery = inputs.read_bound(pipeline/'pipeline.json', refs)
        save(output/'DISCOVERY.json', discovery)
        if discovery.get('source_kind') != 'video':
            result['reason'] = 'NOT_APPLICABLE_NON_VIDEO'
        elif not discovery.get('stages',{}).get('segmentation',{}).get('run_dir'):
            result['reason'] = 'NO_SEGMENTATION_INPUT'
        else:
            cv = inputs.load_cv(refs)
            if cv is None:
                result['reason'] = 'PINNED_OPENCV_NOT_AVAILABLE'
            else:
                cv.setNumThreads(2); cv.setRNGSeed(0); cv.ocl.setUseOpenCL(False)
                counted = CountCV(cv)
                frames, grays, masks, meta = inputs.load_inputs(pipeline, discovery, refs,
                                                               max_frame_profile='whole-video-incremental-v1')
                bound = candidate_bindings(discovery, frames, refs)
                implementation = []
                files = ['integrations/demo_pipeline/automatic_point_review.py',
                    'integrations/demo_pipeline/point_review_cameras.py', 'integrations/demo_pipeline/temporal_review.py',
                    'integrations/demo_pipeline/original_anchor_review.py',
                    'integrations/demo_pipeline/seed_pair_validation.py',
                    'integrations/demo_pipeline/batch_evidence.py',
                    'integrations/demo_pipeline/view_extension.py',
                    'integrations/demo_pipeline/view_extension_report.py',
                    'integrations/demo_pipeline/intermediate_review.py',
                    'integrations/demo_pipeline/intermediate_tracking.py',
                    'integrations/demo_pipeline/intermediate_decode.py',
                    'integrations/demo_pipeline/intermediate_report.py',
                    'integrations/demo_pipeline/intermediate_population.py',
                    'integrations/demo_pipeline/intermediate_population_review.py',
                    'integrations/demo_pipeline/intermediate_population_report.py',
                    'integrations/demo_pipeline/region_tracking.py',
                    'integrations/demo_pipeline/region_growth.py',
                    'integrations/demo_pipeline/region_tracking_report.py',
                    'integrations/demo_pipeline/area_alignment.py',
                    'integrations/demo_pipeline/area_tracking.py',
                    'integrations/demo_pipeline/area_tracking_report.py',
                    'integrations/demo_pipeline/area_review.py',
                    'integrations/demo_pipeline/overlap_regions.py',
                    'integrations/demo_pipeline/connected_regions.py',
                    'integrations/demo_pipeline/overlap_review.py',
                    'integrations/demo_pipeline/overlap_report.py',
                    'integrations/demo_pipeline/neighbor_consensus.py',
                    'integrations/demo_pipeline/neighborhood_review.py',
                    'integrations/demo_pipeline/neighborhood_report.py',
                    'integrations/demo_pipeline/region_review.py',
                    'integrations/demo_pipeline/short_tracklets.py',
                    'integrations/demo_pipeline/short_tracklet_geometry.py',
                    'integrations/demo_pipeline/short_tracklet_report.py',
                    'integrations/demo_pipeline/tracklet_camera_math.py',
                    'integrations/demo_pipeline/tracklet_camera_review.py',
                    'integrations/demo_pipeline/tracklet_camera_report.py',
                    'integrations/demo_pipeline/anchor_route_tracking.py',
                    'integrations/demo_pipeline/anchor_route_geometry.py',
                    'integrations/demo_pipeline/anchor_route_review.py',
                    'integrations/demo_pipeline/query_camera_math.py',
                    'integrations/demo_pipeline/query_camera_extension.py',
                    'integrations/demo_pipeline/query_camera_review.py',
                    'integrations/demo_pipeline/tracklet_epipolar.py',
                    'src/kozlov_lab/stereo/viewpoint_consistency.py',
                    'scripts/prepare_video.py', 'integrations/video_io/timeline.py',
                    'integrations/demo_pipeline/anchor_candidate_gate.py',
                    'integrations/demo_pipeline/anchor_candidate_export.py',
                    'integrations/viewer/ply_data.py', 'integrations/viewer/export.py',
                    'integrations/viewer/assets/viewer.html', 'integrations/viewer/assets/viewer.js',
                    'integrations/viewer/assets/viewer.css',
                    'src/kozlov_lab/stereo/baseline/radial_branch.py',
                    'src/kozlov_lab/stereo/baseline/radial_diagnostics.py',
                    'src/kozlov_lab/stereo/baseline/multiview_fixed.py',
                    'src/kozlov_lab/stereo/baseline/observability.py',
                    'scripts/review_point_sequence.py', 'scripts/review_mask_association.py',
                    'src/kozlov_lab/stereo/coverage_tracks.py', 'src/kozlov_lab/stereo/temporal_support.py',
                    'src/kozlov_lab/stereo/mask_association.py', 'src/kozlov_lab/stereo/local_motion.py',
                    'src/kozlov_lab/stereo/exterior_witness.py', 'src/kozlov_lab/stereo/boundary_evidence.py',
                    'src/kozlov_lab/stereo/candidate_point_policy.py', 'src/kozlov_lab/stereo/candidate_point_consumer.py',
                    'src/kozlov_lab/stereo/candidate_geometry_inputs.py', 'src/kozlov_lab/stereo/candidate_anchor_geometry.py',
                    'integrations/demo_pipeline/coverage_triangulation.py', 'integrations/demo_pipeline/partial_point_gate.py']
                for relative in files:
                    inputs.bind(ROOT/relative, implementation)
                refs.extend(implementation)
                from .original_anchor_review import review_original_anchors, PROFILE as ORIGINAL_PROFILE
                from .seed_pair_validation import validate_seed_pairs, PROFILE as SEED_PROFILE
                from .anchor_candidate_gate import PROFILE as CANDIDATE_PROFILE
                from .view_extension import PROFILE as EXTENSION_PROFILE
                from .intermediate_review import PROFILE as INTERMEDIATE_PROFILE
                save(output/'PLAN.json', dict(status='FROZEN_BEFORE_POINT_REVIEW', profile=PROFILE,
                     original_anchor_profile=ORIGINAL_PROFILE,
                     seed_pair_profile=SEED_PROFILE,
                     candidate_profile=CANDIDATE_PROFILE,
                     view_extension_profile=EXTENSION_PROFILE,
                     intermediate_profile=INTERMEDIATE_PROFILE,
                     tracking_profile=profile_for(profile_id=PROFILE['tracking_profile']), frames=frames,
                     candidate_bindings=bound, source=meta, source_bindings=refs))
                budget()
                result['tracking_calls'] = 1
                tracking = track_coverage(grays, masks, counted, profile_id=PROFILE['tracking_profile'])
                save(output/'TRACKS.json', tracking)
                from .point_review_cameras import load_registered_cameras
                result['native_saved_model_reads'] = None
                result['native_saved_model_reads_reason'] = 'Unknown if camera adapter does not return'
                camera_result = load_registered_cameras(discovery, frames, refs)
                result['native_saved_model_reads'] = camera_result['native_reads']
                result['native_saved_model_reads_reason'] = None
                save(output/'CAMERAS.json', camera_result)
                budget()
                value = analyse_sequence(frames, grays, masks, bound, tracking, camera_result['models'],
                                         counted, output, budget)
                budget()
                mapped = canonical_tracks(tracking, frames)
                admission = review_original_anchors(frames, masks, mapped, camera_result, budget=budget, evaluate=False)
                from .batch_evidence import BatchEvidence
                recorders['seed']=BatchEvidence(output/'batches/seed')
                seed = validate_seed_pairs(frames, grays, masks, mapped, admission, counted,
                    budget=budget, camera_result=camera_result,progress=recorders['seed'])
                save(output/'SEED_PAIR_VALIDATION.json', seed)
                refs.append(inputs.fingerprint(output/'SEED_PAIR_VALIDATION.json'))
                refs.extend(recorders['seed'].bindings())
                recorders['geometry']=BatchEvidence(output/'batches/geometry')
                original = review_original_anchors(frames, masks, mapped, camera_result,
                    budget=budget, seed_validation=seed,progress=recorders['geometry'])
                refs.extend(recorders['geometry'].bindings())
                save(output/'ORIGINAL_ANCHOR_DIAGNOSTICS.json', original)
                result.update(value, status='AUTO_POINT_REVIEW_COMPLETE', reason=None,
                    original_anchor_review=dict(status=original['status'], counts=original['counts'],
                        camera_count=original['camera_count'], geometry_acceptance=False,
                        native_injected=False, record='ORIGINAL_ANCHOR_DIAGNOSTICS.json'),
                    seed_pair_validation=dict(status=seed['status'],counts=seed['counts'],flow_calls=seed['flow_calls'],
                        unique_pair_measurements=seed['unique_pair_measurements'],record='SEED_PAIR_VALIDATION.json',
                        independent_identity_truth=False,original_observations_modified=False),
                    cameras={k:v for k,v in camera_result.items() if k != 'models'},
                    source=meta, tracking_calls=1, optical_flow_calls=counted.flow_calls,
                    numeric_anchor_attempts=sum(2 <= len(g['anchors']) <= 32 for g in value['geometry_groups']))
        for ref in refs:
            if inputs.fingerprint(ref['path']) != ref:
                raise ValueError('automatic source or implementation drift: '+ref['path'])
        if original is not None:
            from .anchor_candidate_export import export_candidate_clouds
            def verify_candidate_sources():
                for ref in refs:
                    if inputs.fingerprint(ref['path']) != ref:
                        raise ValueError('automatic source or implementation drift during export: '+ref['path'])
            clouds = export_candidate_clouds(original, output/'candidate_clouds',
                inputs.fingerprint(output/'ORIGINAL_ANCHOR_DIAGNOSTICS.json'), budget=budget,
                verify_sources=verify_candidate_sources)
            result['candidate_clouds'] = clouds
            # The classifier was independently recomputed by the exporter. Bind
            # those actual bytes before extending measured query correspondences.
            classification = inputs.read_bound(output/'candidate_clouds/CLASSIFICATION.json', refs)
            refs.append(inputs.fingerprint(output/'ORIGINAL_ANCHOR_DIAGNOSTICS.json'))
            refs.append(inputs.fingerprint(output/'CAMERAS.json'))
            refs.append(inputs.fingerprint(output/'TRACKS.json'))
            from .view_extension import extend_views
            from .view_extension_report import ExtensionEvidence
            extension_evidence = ExtensionEvidence(output/'view_extension')
            extension = extend_views(frames, grays, masks, classification, camera_result, counted,
                budget=budget, progress=extension_evidence)
            verify_candidate_sources()
            result['view_extension'] = extension_evidence.finish(extension, frames, budget)
            refs.extend(result['view_extension']['source_bindings'])
            verify_candidate_sources()
            from .intermediate_review import review_intermediates
            result['intermediate_frames'] = review_intermediates(discovery, frames, masks, extension,
                output/'intermediate_frames', refs, counted,
                deadline=start+PROFILE['wall_seconds'], budget=budget, tracking=tracking,
                camera_result=camera_result, classification=classification)
            refs.extend(result['intermediate_frames']['source_bindings'])
            verify_candidate_sources()
        result['source_unchanged'] = True
    except Exception as exc:
        result.update(status='RESOURCE_LIMIT' if isinstance(exc, (TimeoutError, MemoryError)) else 'FAILED',
                      reason=f'{type(exc).__name__}: {exc}', source_unchanged=None)
        if (output/'DECISIONS.json').is_file():
            result['decisions'] = __import__('json').loads((output/'DECISIONS.json').read_text(encoding='utf-8'))
        result['partial_evidence_retained'] = True
    if not (output/'PLAN.json').is_file():
        save(output/'PLAN.json', dict(profile=PROFILE, source_bindings=refs, execution_started=False))
    if counted is not None:
        result['optical_flow_calls'] = counted.flow_calls
    result['batch_execution']={k:v.summary() for k,v in recorders.items()}
    if extension_evidence is not None:
        result['view_extension_progress'] = extension_evidence.snapshot()
    result.update(wall_seconds=time.monotonic()-start, result_path=str(output/'RESULT.json'),
                  report_path=str(output/'index.html'))
    save(output/'FINAL_BINDINGS.json', refs)
    save(output/'RESULT.json', result); render(output, result)
    return result
