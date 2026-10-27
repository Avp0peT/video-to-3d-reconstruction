"""Direct third-pair checks on a bounded candidate graph, never track fusion.

Book compatibility conditions motivate this diagnostic. Cycle consistency is
an engineering supplement; no MatchLift/SDP or physical identity proof is used.
"""
from collections import Counter
from pathlib import Path
import html
import json

from PIL import Image, ImageDraw
from kozlov_lab.stereo.multiview_cycles import analyze_candidate_graph
from . import parallel_photo_audit as io
from .candidate_competition import assess_competitors

PROFILE = dict(max_seed_edges=4096, max_wedges=256, max_closing_pairs=8,
               max_competitor_checks=256, closure_passes=1, refit_geometry=False, descriptor_search_scope='all selected native rows')
SUPPORTED = 'EXISTING_F_SUPPORT_CANDIDATE'


def _node(frame, feature):
    return dict(frame=frame, feature_id=feature)


def _key(node):
    return node['frame'], node['feature_id']


def _edge(first, second, source, target):
    a, b = sorted(((first, source), (second, target)))
    return dict(edge_id=json.dumps([a, b], separators=(',', ':')),
                a=_node(*a), b=_node(*b))


def _eligible(row):
    """Preserve competition beyond the top-48 retained display/geometry set."""
    return row['proposal_edges'] + row['proposal_receipt']['omitted_edges']


def _check_rows(row):
    result = {}
    for check in row['proposal_geometry']['links']:
        key = (check['from_track_id'], check['to_track_id'])
        if key in result:
            raise ValueError('duplicate geometry endpoint identity')
        result[key] = check
    expected = {(e['source_id'], e['target_id']) for e in row['proposal_edges']}
    if len(expected) != len(row['proposal_edges']) or set(result) != expected:
        raise ValueError('descriptor and geometry endpoint sets differ')
    return result


def _competition(row, source, target):
    alternatives = [e for e in _eligible(row)
                    if (e['source_id'] == source or e['target_id'] == target)
                    and (e['source_id'], e['target_id']) != (source, target)]
    return dict(frames=row['frames'], source_id=source, target_id=target,
                alternatives=alternatives, scope='retained plus omitted eligible descriptor edges',
                uniqueness='NOT_EVALUATED')


def analyze_multiview(features, pairs, geometry, *, descriptor_proposals, geometry_check,
                      max_closing_pairs=8):
    """One frozen closure pass. A composed prediction is never a direct match.

    features and geometry are already identity-bound by the caller. The same
    descriptor and frozen-F functions are used for seed and missing pairs.
    Pair selection depends on seeds; this is conditional validation, not an
    independent statistical holdout or an exhaustive correspondence search.
    """
    if type(max_closing_pairs) is not int or not 1 <= max_closing_pairs <= 8:
        raise ValueError('max_closing_pairs must be 1..8')
    frame_map = {f['frame_index']: f for f in features}
    if len(frame_map) != len(features):
        raise ValueError('duplicate native frame identity')
    pair_map = {}
    seeds = []
    for row in pairs:
        first, second = row['frames']
        if type(first) is not int or type(second) is not int or first >= second:
            raise ValueError('canonical frame pair required')
        if (first, second) in pair_map:
            raise ValueError('duplicate original descriptor pair')
        pair_map[first, second] = row
        for (source, target), check in _check_rows(row).items():
            if check['status'] == SUPPORTED:
                seeds.append(_edge(first, second, source, target))
    seed_graph = analyze_candidate_graph(seeds)
    needed = sorted({(w['endpoint_a']['frame'], w['endpoint_b']['frame'])
                     for w in seed_graph['wedges']
                     if (w['endpoint_a']['frame'], w['endpoint_b']['frame']) not in pair_map})
    selected = (needed if len(needed) <= max_closing_pairs else
                [needed[i*(len(needed)-1)//(max_closing_pairs-1)] for i in range(max_closing_pairs)]
                if max_closing_pairs > 1 else needed[:1])
    new_pairs = []
    for first, second in selected:
        # Search all selected rows, NOT just the endpoints predicted by the path.
        a, b, edges, receipt = descriptor_proposals(frame_map[first], frame_map[second])
        row = dict(frames=[first, second], source_points=a, target_points=b,
                   proposal_edges=edges, proposal_receipt=receipt,
                   selection_reason='closing pair from frozen seed wedges; no recursive expansion')
        row['proposal_geometry'] = geometry_check(dict(mapping=edges), first, second, a, b,
                                                  geometry, kind='DIRECT_CYCLE_PROPOSAL_NOT_IDENTITY_VERIFIED')
        _check_rows(row)
        new_pairs.append(row)
        pair_map[first, second] = row
    augmented = {e['edge_id']: e for e in seeds}
    for row in new_pairs:
        for (source, target), check in _check_rows(row).items():
            if check['status'] == SUPPORTED:
                edge = _edge(*row['frames'], source, target)
                augmented[edge['edge_id']] = edge
    augmented_graph = analyze_candidate_graph(list(augmented.values()))
    components = { _key(n): c for c in augmented_graph['components'] for n in c['nodes'] }
    closures = []
    for wedge in seed_graph['wedges']:
        a, b, center = wedge['endpoint_a'], wedge['endpoint_b'], wedge['center']
        row = pair_map.get((a['frame'], b['frame']))
        case = dict(wedge, status='CLOSING_PAIR_NOT_TESTED_BUDGET',
                    direct_match=None, direct_geometry=None, descriptor_cycle_closed=False,
                    component_conflict_after_closing=components[_key(center)]['component_conflict'],
                    competition=[], eligible_for_triangulation=False,
                    third_view_validation='CONDITIONAL_CANDIDATE_CHECK',
                    physical_identity_verified=False)
        closures.append(case)
        if row is None:
            continue
        expected = (a['feature_id'], b['feature_id'])
        retained = {(e['source_id'], e['target_id']): e for e in row['proposal_edges']}
        omitted = {(e['source_id'], e['target_id']): e for e in row['proposal_receipt']['omitted_edges']}
        case['expected_edge_eligible_but_omitted'] = expected in omitted
        if expected not in retained:
            case['status'] = ('DIRECT_EDGE_OMITTED_BUDGET' if expected in omitted
                              else 'NO_RETAINED_DIRECT_DESCRIPTOR_EDGE')
            # A missing edge can mean visibility/descriptor/sampling limitations.
            # It is not a declaration that either seed edge is physically wrong.
            continue
        case['direct_match'] = retained[expected]
        case['direct_geometry'] = _check_rows(row)[expected]
        case['descriptor_cycle_closed'] = True
        nodes = [a, center, b]
        for left, right in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[0], nodes[2])):
            left, right = sorted((left, right), key=_key)
            case['competition'].append(_competition(pair_map[left['frame'], right['frame']],
                                                     left['feature_id'], right['feature_id']))
        case['observed_descriptor_competition'] = any(c['alternatives'] for c in case['competition'])
        if case['component_conflict_after_closing']:
            case['status'] = 'CLOSED_CYCLE_COMPONENT_CONFLICT'
        elif case['direct_geometry']['status'] != SUPPORTED:
            case['status'] = 'DESCRIPTOR_CYCLE_WITHOUT_THIRD_EDGE_GEOMETRY_SUPPORT'
        elif case['observed_descriptor_competition']:
            case['status'] = 'CYCLE_SUPPORTED_BUT_AMBIGUOUS'
        else:
            case['status'] = 'CYCLE_SUPPORTED_CANDIDATE'
    closed = {tuple(sorted((_key(c['endpoint_a']), _key(c['center']), _key(c['endpoint_b']))))
              for c in closures if c['descriptor_cycle_closed']}
    count = Counter(c['status'] for c in closures)
    geometric = Counter()
    for row in new_pairs:
        geometric.update(row['proposal_geometry']['counts'])
    competition = assess_competitors(closures, list(pair_map.values()), list(augmented.values()),
                                     geometry, geometry_check=geometry_check)
    return dict(schema='direct-multiview-review-v2', status='CANDIDATE_DIAGNOSTIC_COMPLETE',
                profile={**PROFILE, 'max_closing_pairs': max_closing_pairs},
                seed_graph=seed_graph, augmented_graph=augmented_graph,
                closing_pair_selection=dict(needed=[list(p) for p in needed],
                    selected=[list(p) for p in selected], omitted=[list(p) for p in needed if p not in selected],
                    all_frame_pairs_searched=False, rule='frozen wedges; sorted uniformly bounded pair selection'),
                direct_pairs=new_pairs, closures=closures, competition_geometry_review=competition,
                counts=dict(seed_edges=len(seeds), tested_paths=len(closures),
                    seed_wedges_omitted=seed_graph['omitted_count'], new_direct_pairs=len(new_pairs),
                    new_direct_edges=sum(len(r['proposal_edges']) for r in new_pairs),
                    direct_edge_geometry=dict(geometric), closure_status=dict(count),
                    closed_path_checks=sum(c['descriptor_cycle_closed'] for c in closures),
                    unique_closed_seed_triples=len(closed),
                    competitor_geometry=competition['counts'],
                    competition_case_resolution=competition['counts']['case_resolution']),
                identity_verified=False, geometry_acceptance=False, eligible_for_triangulation=False,
                native_injected=False, observations_synthesized=False, feature_ids_merged=False,
                geometry_refit=False, uniqueness='NOT_EVALUATED', quality_status='NOT_EVALUATED',
                limitations=['Cycle consistency is necessary for compatible maps, not physical identity proof.',
                    'Original F only, with training-endpoint exclusion; no new model fitting.',
                    'Enumerated competitors include omitted eligible edges; original-F checks are bounded and remaining sampling scope is unknown.',
                    'Missing closure does not identify which edge is wrong or prove occlusion.',
                    'One closure pass cannot certify full all-view consistency.'])


def _render(output, result, features, frames, grays):
    cards = []
    points = {(f['frame_index'], p['id']): p['xy'] for f in features for p in f['points']}
    # One board per view triple, plus at most two additional differently labelled paths.
    selected = []; seen = set()
    for case in result.get('closures', []):
        triple = tuple(sorted(n['frame'] for n in (case['endpoint_a'], case['center'], case['endpoint_b'])))
        label = (triple, case['status'])
        if label not in seen and len(selected) < 6:
            selected.append(case); seen.add(label)
    for i, case in enumerate(selected):
        board = Image.new('RGB', (990, 650), '#eef2f5'); draw = ImageDraw.Draw(board)
        nodes = sorted((case['endpoint_a'], case['center'], case['endpoint_b']), key=_key)
        for side, node in enumerate(nodes):
            frame = node['frame']; gray = grays[frame]
            tile = Image.fromarray(gray).convert('RGB'); tile.thumbnail((320, 570))
            ox, oy = side*330+5, 45; board.paste(tile, (ox, oy))
            draw.text((ox, 6), f'index {frames[frame]["input_index"]} / feature {node["feature_id"]}', fill='black')
            x, y = points[_key(node)]; px = ox+(x-.5)*tile.width/gray.shape[1]; py = oy+(y-.5)*tile.height/gray.shape[0]
            draw.ellipse((px-7, py-7, px+7, py+7), outline='#ff7d00', width=3)
            draw.line((px-12, py, px+12, py), fill='#ff7d00', width=2)
            draw.line((px, py-12, px, py+12), fill='#ff7d00', width=2)
        draw.text((8, 620), case['status'], fill='black')
        name = f'triple_{i:02d}.png'; board.save(output/name)
        with Image.open(output/name) as image:
            if image.convert('RGB').tobytes() != board.tobytes():
                raise ValueError('multiview board readback failed')
        cards.append('<h3>'+html.escape(case['status'])+'</h3><img src="'+name+'">')
    counts = result.get('counts', {})
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多视图候选闭环</title>
<style>body{font:17px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:18px;color:#203246}aside{background:#fff1d6;padding:16px}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>直接匹配第三对图像，检查候选是否闭环</h1>
<aside>只检查候选相容性。未确认物理点身份、未拼接轨迹、未生成新的三维点。</aside>
<p>书稿印刷41、53–54页要求对应能由同一物体及投影共同解释；61–62页讨论共同子图，116–117页讨论不同点数及集合接近。以下闭环是工程补充，不是书中算法或 MatchLift 的实现。</p>
<p>先冻结原F支持边的三视图路径，再对缺少的一对图像从全部已选原生特征中直接匹配。不会把前两条边的复合结果充当第三条匹配。原F检查不重新拟合，并排除训练端点依赖。</p>
<p>同一画面出现多个特征身份的分量保留为冲突；描述子竞争包括被前48名截断的合格边。缺边不等于物理点配错，闭环也不等于身份真值；采样范围之外仍未知。三角可产生三条路径，数量分别统计。</p>'''
    competition = result.get('competition_geometry_review', {})
    page += '<h2>竞争候选的冻结几何复核</h2><p>描述子歧义与几何支持分开：复用已有检查，对尚未检查的竞争边按同一原F补查。几何不符只是在冻结模型和阈值下不相容；缺F、训练端依赖或预算未查不能排除。即使当前列举范围无存活竞争，也不宣告物理唯一。</p>'
    page += '<pre>'+html.escape(json.dumps(competition.get('counts', {}), ensure_ascii=False, indent=2))+'</pre>'
    page += '<table><tr><th>原路径</th><th>竞争几何结论</th><th>各状态数量</th></tr>'+''.join(
        '<tr><td>'+str(c['wedge_index'])+'</td><td>'+html.escape(c['status'])+'</td><td>'+html.escape(json.dumps(c['geometry_counts']))+'</td></tr>'
        for c in competition.get('cases', []))+'</table>'
    page += '<pre>'+html.escape(json.dumps(dict(status=result['status'], counts=counts), ensure_ascii=False, indent=2))+'</pre>'
    page += ''.join(cards)
    page += '<p><a href="MULTIVIEW.json">全部路径、第三边、竞争项和冲突分量</a> · <a href="../index.html">返回子图候选</a></p></html>'
    (output/'index.html').write_text(page, encoding='utf-8')


def review_multiview(features, pairs, geometry, frames, grays, output, *, descriptor_proposals, geometry_check):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = analyze_multiview(features, pairs, geometry, descriptor_proposals=descriptor_proposals,
                                   geometry_check=geometry_check)
    except Exception as exc:
        result = dict(status='FAILED', reason=f'{type(exc).__name__}: {exc}', counts={},
                      identity_verified=False, native_injected=False, geometry_acceptance=False)
    io.write_json(output/'MULTIVIEW.json', result)
    _render(output, result, features, frames, grays)
    return result
