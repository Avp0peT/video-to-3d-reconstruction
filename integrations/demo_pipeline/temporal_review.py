"""Identity-bound CPU video track diagnostics. Never changes reconstruction."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import html
import importlib
import math
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from . import parallel_photo_audit as io
from .temporal_geometry import analyse_track_pairs, PROFILE as GEOMETRY_PROFILE
from kozlov_lab.stereo.temporal_tracks import track_sequence, PROFILE
from kozlov_lab.stereo.temporal_reassociation import reassociate_tracklets
from .reassociation_geometry import assess_reassociated_links
from .reassociation_report import render_reassociation
from .partial_review import review_partial

ROOT = Path(__file__).resolve().parents[2]
CV_ROOT = ROOT/'outputs/toolchains/opencv-4.13.0.92'


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer required: '+label)
    return value


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError('duplicate identity: '+key)
        result[value] = row
    return result


def same_path(a, b):
    return Path(a).resolve() == Path(b).resolve()


def fingerprint(path):
    path=Path(path).resolve(strict=True)
    before=path.stat()
    if before.st_size > 512 << 20:
        raise ValueError('diagnostic input file exceeds 512 MiB binding budget')
    digest=hashlib.sha256();count=0
    with path.open('rb') as stream:
        while block := stream.read(1 << 20):
            count+=len(block);digest.update(block)
            if count > 512 << 20:
                raise ValueError('diagnostic input file grew beyond budget')
    after=path.stat()
    if count != before.st_size or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError('input changed while hashing')
    return dict(path=str(path),bytes=count,sha256=digest.hexdigest())


def bind(path, refs, declared=None):
    current = fingerprint(path)
    if declared is not None:
        for field in ('bytes', 'sha256'):
            if field in declared and current[field] != declared[field]:
                raise ValueError('binding differs: '+str(path))
        if 'path' in declared and not same_path(declared['path'], path):
            raise ValueError('bound path differs')
    refs.append(current)
    return current


def read_bound(path, refs, declared=None):
    bind(path, refs, declared)
    return io.read_json(path)


def receipt_path(pipeline, discovery):
    argv = discovery.get('stages', {}).get('input', {}).get('argv', [])
    positions = [i for i, value in enumerate(argv) if value == '--output']
    if len(positions) != 1 or positions[0]+1 >= len(argv):
        raise ValueError('one recorded object-input output path required')
    path = Path(argv[positions[0]+1])
    if not path.is_absolute():
        raise ValueError('absolute recorded object-input path required')
    return path.resolve(strict=True)/'RESULT.json'


def validate_manifest_counts(manifest):
    states=Counter(row['status'] for row in manifest['frames'])
    allowed={'OK','NO_MATCH','NEEDS_SELECTION','INPUT_ERROR','BACKEND_ERROR','RESOURCE_LIMIT','NOT_RUN'}
    if not set(states).issubset(allowed):
        raise ValueError('invalid mask frame status')
    expected=dict(requested=len(manifest['frames']),success=states['OK'],no_match=states['NO_MATCH'],
                  needs_selection=states['NEEDS_SELECTION'],not_run=states['NOT_RUN'],
                  error=sum(states[s] for s in ('INPUT_ERROR','BACKEND_ERROR','RESOURCE_LIMIT')))
    for key,value in expected.items():
        if integer(manifest['counts'][key], 'mask counts') != value:
            raise ValueError('mask counts disagree with frame states')
    for row in manifest['frames']:
        if row['status'] != 'OK' and (row.get('selected_mask') is not None or row.get('selected_candidate_id') is not None):
            raise ValueError('non-OK mask frame has a selected candidate')


def load_inputs(pipeline, discovery, refs, *, max_frame_profile=None, sfm_run=None):
    if max_frame_profile not in (None, "whole-video-incremental-v1"):
        raise ValueError("unknown max-frame profile")
    max_frames = 192 if max_frame_profile else PROFILE["max_frames"]
    receipt = read_bound(receipt_path(pipeline, discovery), refs)
    if receipt.get('contract_version') != 'object-input-v1' or receipt.get('status') != 'INPUTS_PREPARED' or receipt.get('input_kind') != 'video':
        raise ValueError('prepared video object-input receipt required')
    video_root = Path(receipt['reused_video_run']).resolve(strict=True)
    declared = {str(Path(r['path']).resolve()): r for r in receipt['source_bindings']}
    result_path = video_root/'RESULT.json'
    if str(result_path) not in declared:
        raise ValueError('video receipt is not bound by input receipt')
    video = read_bound(result_path, refs, declared[str(result_path)])
    if video.get('contract_version') != 'video-photo-v1' or video.get('status') != 'SUCCESS':
        raise ValueError('successful video-photo receipt required')
    if not same_path(discovery['source'], video['source_video']):
        raise ValueError('pipeline/video source differs')
    source_ref = bind(video['source_video'], refs)
    if source_ref['sha256'] != video['source_sha256'] or source_ref['sha256'] != video['source_sha256_after']:
        raise ValueError('source video changed')
    photo = receipt['photo_io']; t01_path = Path(photo['run']['path']).resolve(strict=True)
    t01 = read_bound(t01_path, refs, photo['run']); t01_root = t01_path.parent
    if t01.get('contract_version') != 'photo-io-v1' or t01.get('status') not in ('SUCCESS','PARTIAL'):
        raise ValueError('committed photo-io-v1 required')
    if not same_path(video['photo_io']['run'], t01_path) or video['photo_io']['sha256'] != refs[-1]['sha256']:
        raise ValueError('video/T01 run binding differs')
    frames_ref = bind(photo['frames']['path'], refs, photo['frames'])
    if not same_path(frames_ref['path'], t01_root/'frames.jsonl'):
        raise ValueError('T01 frame file must belong to bound run')
    if video['photo_io']['frames_sha256'] != frames_ref['sha256']:
        raise ValueError('video/T01 frame binding differs')
    trows = unique([io.loads(s) for s in Path(frames_ref['path']).read_text(encoding='utf-8-sig').splitlines() if s.strip()], 'input_index')
    ok=sum(row['status']=='OK' for row in trows.values())
    expected=dict(input_count=len(trows),ok_count=ok,error_count=len(trows)-ok)
    for key,value in expected.items():
        if integer(t01[key], 'T01 counts') != value:
            raise ValueError('T01 counts disagree with frame states')
    identities = unique(photo['identities'], 'input_index')
    canonical = {str(Path(r['path']).resolve()): r for r in photo['canonical_images']}
    manifest_path = Path(discovery['stages']['segmentation']['run_dir']).resolve(strict=True)/'manifest.json'
    manifest = read_bound(manifest_path, refs); ms = manifest['source_io']
    validate_manifest_counts(manifest)
    if (manifest.get('contract_version') != 'object-mask-v1' or not same_path(ms['path'], t01_root)
            or ms['run_id'] != t01['run_id'] or ms['run_json_sha256'] != photo['run']['sha256']
            or ms['frames_jsonl_sha256'] != frames_ref['sha256']):
        raise ValueError('mask/T01 source binding differs')
    mrows = unique(manifest['frames'], 'input_index')
    for rows in (trows, identities, mrows):
        for key in rows:
            integer(key, 'frame identity index')
    all_rows = video['frames']; selected = [r for r in all_rows if r['status'] == 'PHOTO_IO_OK']
    if not 2 <= len(selected) <= max_frames:
        raise ValueError(f'video tracking requires 2..{max_frames} existing accepted frames; no silent truncation')
    if len(selected) != photo['count'] or len(trows) != len(selected) or len(identities) != len(selected):
        raise ValueError('accepted video/T01 identity counts differ')
    frames, grays, masks, prior = [], [], [], None
    for v in selected:
        index = integer(v['photo_input_index'], 'input index'); out_index = integer(v['output_index'], 'output index')
        decode = integer(v['decode_index'], 'decode index'); pts = integer(v['pts'], 'PTS')
        base = v['time_base']
        if len(base) != 2 or any(type(n) is not int or n <= 0 for n in base):
            raise ValueError('positive rational timebase required')
        exact = Fraction(pts)*Fraction(*base)
        if type(v['time_seconds']) not in (float, int) or not math.isfinite(v['time_seconds']) or abs(float(exact)-v['time_seconds']) > 1e-8:
            raise ValueError('video time identity differs')
        if prior is not None and not (index > prior[0] and out_index > prior[1] and decode > prior[2] and exact > prior[3]):
            raise ValueError('video frames must have increasing presentation and canonical order')
        prior = (index, out_index, decode, exact)
        row, identity, mr = trows[index], identities[index], mrows[index]
        if row['status'] != 'OK' or not (row['frame_id'] == identity['frame_id'] == v['photo_frame_id'] == mr['frame_id']):
            raise ValueError('video/T01/mask frame identity differs')
        path = io.under(t01_root, row['image_output']); ref = bind(path, refs, canonical[str(path)])
        if (not same_path(mr['source_image'], path) or mr['source_image_sha256'] != ref['sha256']
                or mr['photo_io_image_sha256'] != row['image_sha256'] or mr['image_domain'] != 'photo_io_canonical'):
            raise ValueError('canonical/raw image identity differs')
        with Image.open(path) as image:
            width, height = image.size
            if max(width, height) > PROFILE['max_long_side']:
                raise ValueError('canonical image long side exceeds 1024')
            rgb = image.convert('RGB')
            digest = hashlib.sha256(f'{width}x{height}:RGB:'.encode('ascii')+rgb.tobytes()).hexdigest()
            gray = np.array(rgb.convert('L'))
        if digest != identity['decoded_rgb_sha256'] or digest != v['decoded_rgb_sha256']:
            raise ValueError('decoded pixel identity differs')
        if [width, height] != identity['prepared_dimensions'] or (width, height) != (row['canonical_width'], row['canonical_height']) or (width, height) != (mr['width'], mr['height']):
            raise ValueError('canonical dimensions differ')
        mask = None; mref = None
        if mr['status'] == 'OK':
            chosen = mr['selected_mask']; candidates = [c for c in mr['candidates'] if c['candidate_id'] == mr['selected_candidate_id']]
            if len(candidates) != 1 or candidates[0]['mask_path'] != chosen['path'] or candidates[0]['mask_sha256'] != chosen['sha256']:
                raise ValueError('selected mask/candidate differs')
            mp = io.under(manifest_path.parent, chosen['path']); mref = bind(mp, refs, {'sha256': chosen['sha256']})
            with Image.open(mp) as image:
                if image.mode != 'L':
                    raise ValueError('mask requires L mode')
                mask = np.array(image)
            if mask.shape != gray.shape or not np.isin(mask, [0, 255]).all() or not np.any(mask):
                raise ValueError('mask labels/shape/foreground invalid')
        frames.append(dict(input_index=index, frame_id=row['frame_id'], path=str(path), sha256=ref['sha256'],
                           width=width, height=height, time_seconds=float(exact), pts=pts, time_base=base, decode_index=decode,
                           original_mask_status=mr['status'], mask_pixels=int(np.count_nonzero(mask)) if mask is not None else 0,
                           mask_binding=mref, model_ids=[]))
        grays.append(gray); masks.append(mask)
    if len({g.shape for g in grays}) != 1:
        raise ValueError('same-size canonical frames required')
    original_sfm_path = discovery.get('stages', {}).get('sfm', {}).get('run_json')
    sfm_path = str(Path(sfm_run).resolve(strict=True)) if sfm_run else original_sfm_path
    override_binding = None
    if sfm_run:
        if max_frame_profile != 'whole-video-incremental-v1':raise ValueError('SfM override requires explicit incremental profile')
        if original_sfm_path:bind(original_sfm_path, refs)
        override_binding = bind(sfm_path, refs)
    if sfm_path:
        sfm = read_bound(sfm_path, refs)
        if sfm['upstream_io_run_id'] != t01['run_id']:
            raise ValueError('SfM upstream differs')
        fm = Path(sfm_path).parent/'frames_map.jsonl'; bind(fm, refs)
        fmap = unique([io.loads(s) for s in fm.read_text(encoding='utf-8-sig').splitlines() if s.strip()], 'input_index')
        models = {str(m['model_id']): m for m in sfm.get('models', [])}
        if sfm_run:
            foreground=sfm.get('effective_options',{}).get('foreground',{})
            if foreground.get('requested') is not True:raise ValueError('override SfM must be object masked')
            old_masks=unique(foreground.get('mask_bindings',[]),'input_index')
            expected_masks={f['input_index']: f for f in frames if f['original_mask_status']=='OK'}
            if set(old_masks)!=set(expected_masks):raise ValueError('override SfM selected mask scope differs')
            for index,f in expected_masks.items():
                if old_masks[index]['sha256']!=f['mask_binding']['sha256']:
                    raise ValueError('override SfM mask identity differs')
        for f in frames:
            m = fmap[f['input_index']]
            if m['frame_id'] != f['frame_id'] or m['canonical_image_sha256'] != f['sha256'] or m['upstream_io_run_id'] != t01['run_id']:
                raise ValueError('SfM frame identity differs')
            for reg in m['registrations']:
                ident = str(reg['model_id'])
                if ident not in models or reg['image_id'] not in models[ident]['registered_image_ids']:
                    raise ValueError('SfM registration not bound to recorded model')
                f['model_ids'].append(ident)
    meta = dict(source_video=source_ref, decoded_count=video['decoded_count'], processed_existing_frames=len(frames),
                selection=video['selection'], max_frame_profile=max_frame_profile, source_sfm_override=override_binding,
                pipeline_sfm_run=original_sfm_path, effective_sfm_run=sfm_path, object=manifest['object'], source_mask_counts=dict(Counter(f['original_mask_status'] for f in frames)),
                rejected_video_rows=[dict(decode_index=r.get('decode_index'), status=r['status']) for r in all_rows if r['status'] != 'PHOTO_IO_OK'],
                new_decoding=False, new_segmentation=False)
    return frames, grays, masks, meta


def load_cv(refs):
    if not (CV_ROOT/'cv2/__init__.py').is_file():
        return None
    sys.path.insert(0, str(CV_ROOT))
    try:
        cv = importlib.import_module('cv2')
    finally:
        sys.path.remove(str(CV_ROOT))
    if not Path(cv.__file__).resolve().is_relative_to(CV_ROOT.resolve()) or cv.__version__ != '4.13.0':
        raise ValueError('unexpected OpenCV origin/version')
    for p in (CV_ROOT/'cv2/__init__.py', CV_ROOT/'cv2/cv2.pyd', CV_ROOT/'opencv_python_headless-4.13.0.92.dist-info/METADATA'):
        bind(p, refs)
    return cv


def contact_board(frames, grays, track_rows, selected, path):
    # Show actual coordinate witnesses. No inferred point or synthesized view.
    common = []
    for track in track_rows:
        obs = {o['frame']: o for o in track['observations'] if o['target_verified']}
        if all(i in obs for i in selected):
            xy = [obs[i]['xy'] for i in selected]
            if not any(all(np.linalg.norm(np.subtract(a,b)) <= 2 for a,b in zip(xy,old[1])) for old in common):
                common.append((track['track_id'], xy))
    common = common[:40]
    canvas = Image.new('RGB', (360*len(selected), 680), '#eef2f5'); draw = ImageDraw.Draw(canvas)
    for col, index in enumerate(selected):
        tile = Image.fromarray(grays[index]).convert('RGB'); tile.thumbnail((350, 610))
        sx = tile.width/grays[index].shape[1]; sy = tile.height/grays[index].shape[0]
        ox, oy = col*360+5, 45; canvas.paste(tile,(ox,oy))
        draw.text((ox,5), f"frame {frames[index]['input_index']}  {frames[index]['time_seconds']:.3f}s", fill='black')
        draw.text((ox,22), f"source model {','.join(frames[index]['model_ids']) or '-'} / witnesses {len(common)}", fill='black')
        for n, (_, xy) in enumerate(common):
            x,y=xy[col]; px,py=ox+(x-.5)*sx,oy+(y-.5)*sy
            color=(60+(n*79)%190,60+(n*47)%190,60+(n*113)%190)
            draw.ellipse((px-3,py-3,px+3,py+3), fill=color, outline='black')
            draw.text((px+4,py), str(n+1), fill=color, stroke_width=1, stroke_fill='black')
    canvas.save(path)
    return dict(frames=selected, track_ids=[r[0] for r in common], coordinates=[r[1] for r in common], image=path.name,
                selection='first 40 stable source tracks after <=2px all-endpoint dedup; numbers are display labels only')


def render(output, result, frames, geometry, contacts):
    esc = html.escape
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>视频点对应复核</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:16px;color:#203246}aside{padding:15px;background:#fff1cf}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    page+='<h1>视频点对应复核</h1><aside>'+esc(result['status'])+'：真实角点轨迹候选；未自动合并模型、未补面、未确认完整物体。</aside>'
    page+='<p>从已有目标 mask 内的真实角点出发，用书稿启发的 r-cover 分散种子，逐帧前后向光流检查。未识别帧不改状态；轨迹在其中仍属目标未验证。两端进入 mask 也不证明为同一物理点。</p>'
    counts=result.get('counts',{})
    if counts:
        page+=f'<p><b>{counts["seed_tracks"]} 条双向候选种子中，{counts["seed_only_tracks"]} 条没有通过下一帧检查，{counts["tracks_with_two_or_more_observations"]} 条至少有2帧观测，最长{counts["max_track_observations"]}帧。</b>这些数含双向重复候选，不是独立三维点。跨原建模分量的候选图对：{result["cross_component_candidate_pairs"]}。</p>'
    if result.get('gap_recovery'):
        detail=result['gap_recovery']
        page+='<h2>失去观测后的候选重识别</h2><p>原轨迹保持不变，另行检验缺帧两端的候选身份。中间空缺不填预测坐标，也不自动判断遮挡。</p>'
        page+='<p><a href="reassociation/index.html">查看断点两端、拒绝原因与独立几何检查</a></p><pre>'+esc(__import__('json').dumps(detail,ensure_ascii=False,indent=2))+'</pre>'
    if result.get('partial_subimages'):
        page+='<h2>书中共同子图思路：独立候选来源</h2><p><a href="partial_subimages/index.html">查看局部仿射子图、额外点与原F检查</a>。该阶段不要求光流跨缺失段成功；仍未认证身份或写入模型。</p>'
    page+='<details><summary>配置、计数及停止原因</summary><pre>'+esc(__import__('json').dumps({k:result.get(k) for k in ('reason','source','counts','termination_reasons','pair_status_counts','cross_component_candidate_pairs')},ensure_ascii=False,indent=2))+'</pre></details>'
    if geometry:
        page+='<h2>跨原分量候选</h2><p>仅以两端目标区域、固定训练/留出点、F 残差作筛查。H 支持意味着平面或低视差歧义；没有测得真实相机夹角，不能据此三角化或拼接。</p><table><tr><th>原帧</th><th>原分量</th><th>去重候选</th><th>留出 F 中位 / P90 px</th><th>H 歧义</th></tr>'
        by_pair={tuple(p['frames']):p for p in geometry['pairs']}
        for b in geometry['proposed_bridges']:
            p=by_pair[tuple(b['frames'])]; m=p['holdout_sampson_px']
            page+=f'<tr><td>{p["input_indices"]}</td><td>{b["model_ids"]}</td><td>{p["unique_candidates"]}</td><td>{m["median"]:.3f} / {m["p90"]:.3f}</td><td>{p.get("planar_or_low_parallax_ambiguity")}</td></tr>'
        page+='</table><h2>对应见证图</h2><p>相同数字来自同一候选轨迹，最多显示40条；不是人工标注，也不等于正交三视图。</p>'
        for c in contacts:
            page+='<p>原帧 '+str([frames[i]['input_index'] for i in c['frames']])+'</p><img src="'+c['image']+'">'
    page+='<h2>完整时间轴</h2><table><tr><th>原 index / decode index</th><th>秒</th><th>原分割状态</th><th>已注册分量</th></tr>'
    for f in frames:
        page+=f'<tr><td>{f["input_index"]} / {f["decode_index"]}</td><td>{f["time_seconds"]:.4f}</td><td>{esc(f["original_mask_status"])}</td><td>{f["model_ids"]}</td></tr>'
    page+='</table><p>当前为已有抽帧，不是视频所有解码帧。LK 依赖局部亮度与运动连续，遮挡、手部、重复纹理、形变可能造成错误。F 在未去畸变像素上只作近似诊断；训练/留出及双向轨迹仍相关，没有独立真值。不能把轨迹数量当三维点数量或精度。</p><p><a href="RESULT.json">结果</a> · <a href="PLAN.json">输入/代码/版本绑定</a>'
    for name,label in [('TRACKS.json','每点逐帧轨迹与断点'),('GEOMETRY.json','F/H、留出点与三帧候选')]:
        if (output/name).is_file():
            page+=' · <a href="'+name+'">'+label+'</a>'
    page+='</p></html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def review_video(pipeline, output, discovery=None):
    pipeline=Path(pipeline).resolve(strict=True);output=Path(output).resolve()
    if output.exists() or output == pipeline or output in pipeline.parents:
        raise ValueError('fresh nonoverlapping diagnostic output required')
    discovery=io.read_json(pipeline/'pipeline.json') if discovery is None else discovery
    from .post_review import check_output_scope
    # Default post-review owns this one reserved nested child.
    scope=output.parent if output == pipeline/'post_review'/'temporal_tracks' else output
    check_output_scope(pipeline,scope,discovery)
    refs=[];frames=[];contacts=[];geometry=None;started=time.monotonic()
    result=dict(status='NOT_AVAILABLE',reason=None,source_unchanged=None,geometry_modified=False,
                correspondence_verified=False,quality_status='NOT_EVALUATED',model_executions=0,
                components_merged=False,stitch_authorized=False)
    output.mkdir(parents=True,exist_ok=False)
    try:
        bind(pipeline/'pipeline.json',refs)
        for p in (Path(__file__),ROOT/'integrations/demo_pipeline/temporal_geometry.py',ROOT/'src/kozlov_lab/stereo/temporal_tracks.py',ROOT/'src/kozlov_lab/stereo/point_coverage.py',ROOT/'integrations/demo_pipeline/parallel_photo_audit.py',
                  ROOT/'src/kozlov_lab/stereo/temporal_reassociation.py',ROOT/'integrations/demo_pipeline/reassociation_geometry.py',ROOT/'integrations/demo_pipeline/reassociation_report.py'):
            bind(p,refs)
        for p in (ROOT/'src/kozlov_lab/stereo/partial_affine.py',ROOT/'src/kozlov_lab/stereo/area_code.py',ROOT/'integrations/demo_pipeline/partial_feature_source.py',ROOT/'integrations/demo_pipeline/partial_review.py'):
            bind(p,refs)
        for p in (ROOT/'src/kozlov_lab/stereo/multiview_cycles.py',ROOT/'integrations/demo_pipeline/multiview_review.py',ROOT/'integrations/demo_pipeline/candidate_competition.py'):
            bind(p,refs)
        for name in ('integrations/demo_pipeline/candidate_observability.py','integrations/demo_pipeline/registration_audit.py','integrations/demo_pipeline/registration_gate.py','integrations/demo_pipeline/approximate_photo_bridge.py',
                     'src/kozlov_lab/stereo/baseline/multiview_fixed.py','src/kozlov_lab/stereo/baseline/observability.py',
                     'src/kozlov_lab/stereo/baseline/radial_diagnostics.py','src/kozlov_lab/stereo/baseline/radial_branch.py'):
            bind(ROOT/name,refs)
        if discovery.get('source_kind') != 'video':
            result['reason']='Video source required; photo pipeline has no temporal identity'
        elif not discovery.get('stages',{}).get('segmentation',{}).get('run_dir'):
            result['reason']='No bound object segmentation manifest'
        else:
            cv=load_cv(refs)
            if cv is None:
                result['reason']='Fixed isolated OpenCV 4.13.0.92 toolchain unavailable'
            else:
                frames,grays,masks,meta=load_inputs(pipeline,discovery,refs)
                result['source']=meta
                io.write_json(output/'PLAN.json',dict(schema='temporal-video-plan-v1',frames=frames,source=meta,bindings=refs,tracking_profile=PROFILE,geometry_profile=GEOMETRY_PROFILE))
                tracks=track_sequence(grays,masks,cv)
                io.write_json(output/'TRACKS.json',tracks)
                geometry=analyse_track_pairs(tracks['tracks'],frames,cv)
                io.write_json(output/'GEOMETRY.json',geometry)
                partial=review_partial(discovery,frames,masks,grays,geometry,output/'partial_subimages',refs)
                result['partial_subimages']={k:partial.get(k) for k in ('status','reason','pairs_evaluated','candidate_subsets','unique_candidate_edges','unique_edge_geometry_counts','unique_edges_absent_from_original_lk','multiview_cycles','candidate_observability')}
                recovery=reassociate_tracklets(tracks['tracks'],grays,masks,cv)
                recovery['baseline_continuity']=dict(tracks['counts'],
                    tracks_with_two_or_more_observations=sum(len(t['observations'])>=2 for t in tracks['tracks']),
                    max_track_observations=max((len(t['observations']) for t in tracks['tracks']),default=0))
                io.write_json(output/'RECOVERY.json',recovery)
                assessment=assess_reassociated_links(recovery,geometry)
                io.write_json(output/'RECOVERY_ASSESSMENT.json',assessment)
                render_reassociation(output/'reassociation',recovery,assessment,frames,grays)
                result['gap_recovery']=dict(status='CANDIDATE_REASSOCIATION_EVALUATED',
                    counts=recovery['counts'],independent_geometry_counts=assessment['counts'],
                    report='reassociation/index.html',source_tracklets_unchanged=True,
                    observations_filled_in_gaps=0,geometry_modified=False,stitch_authorized=False)
                chosen=[b['frames'] for b in geometry['proposed_bridges'][:3]]
                chosen += [t['frames'] for t in geometry['top_three_frame_candidates'][:3]]
                for n,indices in enumerate(chosen):
                    contacts.append(contact_board(frames,grays,tracks['tracks'],indices,output/f'contact_{n:02d}.png'))
                io.write_json(output/'CONTACTS.json',contacts)
                observations=[o for t in tracks['tracks'] for o in t['observations']]
                result.update(status='DIAGNOSTIC_COMPLETE',backend=tracks['backend'],counts=dict(tracks['counts'],
                    observations=len(observations),mask_supported_observations=sum(o['target_verified'] for o in observations),
                    target_unverified_observations=sum(not o['target_verified'] for o in observations),
                    seed_only_tracks=sum(len(t['observations'])==1 for t in tracks['tracks']),
                    tracks_with_two_or_more_observations=sum(len(t['observations'])>=2 for t in tracks['tracks']),
                    max_track_observations=max((len(t['observations']) for t in tracks['tracks']),default=0),
                    track_length_histogram=dict(Counter(len(t['observations']) for t in tracks['tracks']))),
                    termination_reasons=dict(Counter(t['termination']['reason'] for t in tracks['tracks'])),
                    pair_status_counts=dict(Counter(p['status'] for p in geometry['pairs'])),
                    cross_component_candidate_pairs=len(geometry['proposed_bridges']),
                    candidate_three_frame_sets=len(geometry['top_three_frame_candidates']))
        for ref in refs:
            if fingerprint(ref['path']) != ref:
                raise ValueError('diagnostic input changed: '+ref['path'])
        result['source_unchanged']=True
    except Exception as exc:
        result.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}',source_unchanged=None)
    if not (output/'PLAN.json').exists():
        io.write_json(output/'PLAN.json',dict(bindings=refs,tracking_profile=PROFILE,geometry_profile=GEOMETRY_PROFILE))
    result.update(wall_seconds=time.monotonic()-started,report_path=str(output/'index.html'),result_path=str(output/'RESULT.json'))
    io.write_json(output/'RESULT.json',result);render(output,result,frames,geometry,contacts)
    return result
