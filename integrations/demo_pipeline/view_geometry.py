"""Automatic view choice and all-observation checks on immutable native tracks.

This engineering side branch changes no SfM model and executes no Kozlov theorem.
Camera calibration and track filtering already used the diagnostic observations.
"""
from collections import Counter
import html
from pathlib import Path
import time

from . import parallel_photo_audit as native
from . import approximate_photo_bridge as bridge
from .heldout_multiview import _observation
from kozlov_lab.stereo.baseline import multiview_fixed as fixed
from kozlov_lab.stereo.baseline import observability, view_selection

MAX_TRACKS = 256
MAX_VIEWS = 64
SOFT_SECONDS = 60
PIXEL_LINE = 2.0
LIMITATIONS = [
    '普通照片/视频的两视图选择，不要求双目相机；视图编号不是物理设备编号。',
    '只读取原生特征坐标及估计位姿；未重新匹配、估计相机、分割或修改源模型。',
    '几何分离只用于排序，不能证明对应正确；大视角错配也可能被选中。',
    '其余视图参与了选视图和旧标定，不是独立留出验证。',
    '2 px是已有诊断线，不是物理精度阈值；低残差不能作为匹配或模型验收。',
    '只分析按原生point_id排序的前256条轨迹；其余明确列为未分析，不静默删点。',
    'SIMPLE_RADIAL与SIMPLE_PINHOLE为当前支持域；尺度未知，未运行Kozlov构造、MVS或CAD。',
    '选图使用相对退化判据；既有拟合器另有绝对数值界限，FIT_REFUSED不能直接解释为物理退化。',
]


def check_candidate(observations, pair_ids):
    """Only selected views fit XYZ; every supplied view is then checked."""
    pair_set = set(pair_ids)
    pair = [o for o in observations if o['image_id'] in pair_set]
    extra = [o for o in observations if o['image_id'] not in pair_set]
    fitted = fixed.fit(pair)
    result = dict(pair_image_ids=list(pair_ids), fit=bridge.json_value(fitted),
                  pair=None, additional=None, all_observations=None,
                  additional_image_ids=[o['image_id'] for o in extra],
                  observability=None, geometry_acceptance=False)
    if fitted['xyz'] is None:
        result['status'] = 'FIT_REFUSED'
        return result
    xyz = fitted['xyz']
    result['pair'] = fixed.evaluate(xyz, pair)
    result['additional'] = fixed.evaluate(xyz, extra) if extra else None
    result['all_observations'] = fixed.evaluate(xyz, observations)
    result['per_view'] = [dict(image_id=o['image_id'],diagnostic=fixed.evaluate(xyz,[o]))
                          for o in observations]
    result['observability'] = observability.analyze(xyz, pair)
    full = result['all_observations']
    if full['status'] != 'EVALUATED':
        result['status'] = 'PROJECTION_REFUSED'
    elif full['max'] > PIXEL_LINE:
        result['status'] = 'OBSERVATION_INCONSISTENCY_DETECTED'
    elif not extra:
        result['status'] = 'PAIR_ONLY_NOT_VERIFIED'
    else:
        result['status'] = 'WITHIN_DIAGNOSTIC_LINE_NOT_VERIFIED'
    return result


def audit_track(point_id, observations, ambiguous=None):
    row = dict(point_id=point_id, observations=bridge.json_value(observations),
               ambiguous_views_excluded=ambiguous or {}, selected=None,
               ordered_pair_control=None, geometry_acceptance=False, all_native_views_checked=False)
    # Ambiguous native image observations stay explicit; no arbitrary pick.
    if not 2 <= len(observations) <= MAX_VIEWS:
        row['status'] = 'NOT_EVALUATED_VIEW_COUNT'
        row['reason'] = 'requires 2..64 singleton observations'
        return row
    selection = view_selection.select_pair(observations)
    row['selection'] = selection
    if selection.get('selected_image_ids') is None:
        row['status'] = 'NO_USABLE_VIEW_PAIR'
        return row
    row['selected'] = check_candidate(observations, selection['selected_image_ids'])
    first = sorted(o['image_id'] for o in observations)[:2]
    row['ordered_pair_control'] = (row['selected'] if first == selection['selected_image_ids']
                                    else check_candidate(observations, first))
    row['status'] = row['selected']['status']
    row['all_native_views_checked'] = not bool(ambiguous) and row['selected']['status'] not in ('FIT_REFUSED','PROJECTION_REFUSED')
    if ambiguous and row['status'] in ('WITHIN_DIAGNOSTIC_LINE_NOT_VERIFIED','PAIR_ONLY_NOT_VERIFIED'):
        row['status'] = 'INCOMPLETE_AMBIGUOUS_VIEWS'
    return row


def audit(sfm_run, model_id, output):
    source = Path(sfm_run).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh output must not overlap the native source run')
    if not isinstance(model_id, str) or not model_id:
        raise ValueError('explicit nonempty model_id required')
    run_path = source/'run.json'
    refs = [native.binding(run_path)]
    run = native.read_json(run_path)
    if run.get('contract_version') not in ('sfm-native-object-v1', 'sfm-native-object-pinhole-v1') or run.get('status') not in ('SUCCESS','PARTIAL'):
        raise ValueError('usable native object SfM run required')
    models = [m for m in run.get('models', []) if str(m.get('model_id')) == model_id]
    if len(models) != 1 or models[0].get('usable') is not True or models[0].get('roundtrip_verified') is not True:
        raise ValueError('explicit model must have usable verified native roundtrip')
    model = models[0]
    binary = native.under(source, model['binary_path'])
    paths = {name:native.under(source, str((binary/name).relative_to(source)))
             for name in ('images.bin','points3D.bin','cameras.bin')}
    refs.extend(native.binding(path) for path in paths.values())
    images, points = native.read_native(paths['images.bin'], paths['points3D.bin'])
    cameras, poses = bridge.read_calibration(paths['cameras.bin'], paths['images.bin'])
    registered = model.get('registered_image_ids')
    if (model.get('num_points3D') != len(points) or not isinstance(registered,list)
            or any(type(i) is not int for i in registered)
            or sorted(registered) != sorted(images)):
        raise ValueError('model counts or registered image identities disagree')
    if 'num_registered_images' in model and model['num_registered_images'] != len(images):
        raise ValueError('registered view count differs')
    ids = sorted(points)
    selected_ids = ids[:MAX_TRACKS]
    for ref in refs: bridge.verify_binding(ref)
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='native-view-geometry-plan-v1', source_model_id=model_id,
        source_bindings=refs, source_sfm_status=run['status'],
        source_cameras={str(cid):dict(native_model_id=c['native_model_id'], native_params=list(c['native_params']),
            width=c['width'], height=c['height']) for cid,c in cameras.items()},
        selected_point_ids=selected_ids, not_sampled_point_ids=ids[MAX_TRACKS:],
        point_selection='first 256 ascending native point_id; no reselection',
        pair_selection='maximum sine of measured world-ray angle; exclude numerical degeneracies; tie ascending image IDs',
        max_views=MAX_VIEWS, soft_seconds=SOFT_SECONDS, outer_seconds=90,
        diagnostic_line_pixels=PIXEL_LINE, threshold_is_accuracy_acceptance=False,
        old_xyz_used=False, camera_estimation=False, paper_algorithm_called=False,
        geometry_mutation=False, independent_validation=False)
    bridge.write(output/'PLAN.json', plan)
    deadline = time.monotonic()+SOFT_SECONDS
    tracks = []
    for pid in selected_ids:
        if time.monotonic() >= deadline:
            tracks.append(dict(point_id=pid,status='NOT_EVALUATED_BUDGET',geometry_acceptance=False))
            continue
        track = points[pid]
        obs = [_observation(i,pid,images,points,cameras,poses) for i in sorted(track['track'])]
        tracks.append(audit_track(pid,obs,track['ambiguous_observations']))
    bridge.write(output/'TRACKS.json',dict(schema='native-view-geometry-tracks-v1',tracks=tracks))
    for ref in refs: bridge.verify_binding(ref)
    paired = [r for r in tracks if r.get('selected') and r.get('ordered_pair_control')
              and all((r[m].get('all_observations') or {}).get('status') == 'EVALUATED'
                      for m in ('selected','ordered_pair_control'))]
    counts = dict(native_tracks=len(ids), selected_tracks=len(selected_ids),
                  not_sampled_tracks=len(ids)-len(selected_ids), registered_views=len(images),
                  track_statuses=dict(Counter(r['status'] for r in tracks)),
                  with_ambiguous_views=sum(bool(r.get('ambiguous_views_excluded')) for r in tracks),
                  pair_changed=sum(r.get('selected') is not None and r['selected']['pair_image_ids'] !=
                                   r['ordered_pair_control']['pair_image_ids'] for r in tracks))
    result = dict(schema='native-view-geometry-v1',
        status='PARTIAL_DIAGNOSTIC' if len(ids)>len(selected_ids) or any(not r.get('all_native_views_checked',False) for r in tracks)
               else 'DIAGNOSTIC_COMPLETE', source_model_id=model_id, source_sfm_status=run['status'],
        source_bindings=refs, source_unchanged=True, counts=counts, source_cameras=plan['source_cameras'],
        ordered_pair_comparison=dict(eligible_tracks=len(paired),
            selected_lower_full_rms=sum(r['selected']['all_observations']['rms'] < r['ordered_pair_control']['all_observations']['rms'] for r in paired),
            selected_higher_full_rms=sum(r['selected']['all_observations']['rms'] > r['ordered_pair_control']['all_observations']['rms'] for r in paired),
            selected_equal_full_rms=sum(r['selected']['all_observations']['rms'] == r['ordered_pair_control']['all_observations']['rms'] for r in paired),
            interpretation='same all-view diagnostic set; selected pair optimized differently, not independent validation or proven accuracy gain'),
        geometry_acceptance=False, geometry_modified=False, scale_status='unknown',
        quality_status='NOT_EVALUATED', algorithm='standard fixed-camera pair DLT/LM plus multiview diagnostics',
        kozlov_algorithm_called=False, limitations=LIMITATIONS, artifacts=[])
    _report(output,result,tracks)
    result['artifacts']=[native.binding(output/name) for name in ('PLAN.json','TRACKS.json','report.html')]
    bridge.write(output/'RESULT.json',result)
    return result


def _report(output,result,tracks):
    esc=lambda v:html.escape(str(v),quote=True)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动选视图与几何诊断</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#213547}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}aside{background:#fff1d6;padding:14px}</style><h1>自动选视图与几何诊断</h1><aside>这是普通照片/视频的多视图诊断，不要求双目相机。结果不会自动修改、删除或验收三维点。</aside>'
    page+=f'<p>状态：{esc(result["status"])}；范围：{esc(result["counts"])}</p><p>编号最小两图对照（编号不保证时间相邻）：{esc(result["ordered_pair_comparison"])}</p>'
    page+='<table><tr><th>轨迹</th><th>所选视图</th><th>编号最小两图</th><th>所有观测最大残差/px</th><th>状态</th></tr>'
    for row in tracks:
        s=row.get('selected') or {}; c=row.get('ordered_pair_control') or {}
        cells=[row['point_id'],s.get('pair_image_ids'),c.get('pair_image_ids'),(s.get('all_observations') or {}).get('max'),row['status']]
        page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in cells)+'</tr>'
    page+='</table><ul>'+''.join('<li>'+esc(v)+'</li>' for v in LIMITATIONS)+'</ul><p><a href="PLAN.json">冻结范围</a> · <a href="TRACKS.json">逐轨迹诊断</a> · <a href="RESULT.json">结果与文件绑定</a></p></html>'
    (output/'report.html').write_text(page,encoding='utf-8')
