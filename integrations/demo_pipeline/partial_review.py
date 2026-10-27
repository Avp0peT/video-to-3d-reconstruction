"""Book-inspired common-subset hypotheses on bound native features.

Engineering local affine approximation; never a reconstruction acceptance gate.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import html
import itertools
import json
import numpy as np
from PIL import Image, ImageDraw

from . import parallel_photo_audit as io
from .partial_feature_source import load_feature_frames
from .reassociation_geometry import assess_reassociated_links
from .multiview_review import review_multiview
from .candidate_observability import review_candidate_observability
from kozlov_lab.stereo.partial_affine import search_partial_affine

PROFILE=dict(max_pairs=24,descriptor_neighbors=2,descriptor_l2_max=.75,
             descriptor_tie_tolerance=1e-12,max_proposal_edges=48,max_hypotheses=512)


def _prepared(frame):
    points=frame['points']; descriptors=np.asarray(frame['descriptors'])
    if descriptors.dtype!=np.uint8 or descriptors.shape!=(len(points),128):
        raise ValueError('aligned native uint8 SIFT rows required')
    counts=Counter(tuple(p['xy']) for p in points)
    norms=np.linalg.norm(descriptors.astype(float),axis=1)
    selected=[i for i,p in enumerate(points) if counts[tuple(p['xy'])]==1 and norms[i]>0]
    removed=[dict(id=p['id'],reason='DUPLICATE_COORDINATE_ALIAS' if counts[tuple(p['xy'])]>1 else 'ZERO_DESCRIPTOR')
             for i,p in enumerate(points) if i not in set(selected)]
    data=descriptors[selected].astype(float)
    if selected:data/=norms[selected,None]
    return [points[i] for i in selected],data,removed


def descriptor_proposals(first,second):
    """Mutual top-two L2 shortlist; tied cutoffs rejected, truncated scope explicit."""
    a,da,removed_a=_prepared(first); b,db,removed_b=_prepared(second)
    receipt=dict(source_removed=removed_a,target_removed=removed_b,source_count=len(a),target_count=len(b),
        normalization='native quantized SIFT -> float64 -> row L2; no repeat sqrt',
        complete_correspondence_search=False,cutoff_tie_rows=[[],[]])
    if not a or not b:return a,b,[],dict(receipt,eligible_edges=0,omitted_edges=[],selected_edges=0)
    distance=np.sqrt(np.maximum(0.,2.-2.*(da@db.T)))
    def nearest(matrix,side):
        sets=[]
        for index,row in enumerate(matrix):
            order=np.argsort(row,kind='stable');k=min(2,len(order))
            if len(order)>k and row[order[k-1]]<=.75 and abs(float(row[order[k]]-row[order[k-1]]))<=1e-12:
                receipt['cutoff_tie_rows'][side].append(index);sets.append(set());continue
            sets.append({int(i) for i in order[:k] if row[i]<=.75})
        return sets
    ab,ba=nearest(distance,0),nearest(distance.T,1)
    edges=[dict(source_id=a[i]['id'],target_id=b[j]['id'],descriptor_distance=float(distance[i,j]))
           for i,values in enumerate(ab) for j in sorted(values) if i in ba[j]]
    edges.sort(key=lambda e:(e['descriptor_distance'],e['source_id'],e['target_id']))
    return a,b,edges[:48],dict(receipt,eligible_edges=len(edges),omitted_edges=edges[48:],selected_edges=min(48,len(edges)))


def choose_pairs(features):
    eligible=[i for i,f in enumerate(features) if len(f['points'])>=6]
    adjacent=list(zip(eligible,eligible[1:]))
    wider=[(eligible[i],eligible[i+4]) for i in range(max(0,len(eligible)-4))]
    room=max(0,24-len(adjacent))
    additions=[wider[i] for i in np.linspace(0,len(wider)-1,min(room,len(wider)),dtype=int)] if wider and room else []
    all_pairs=adjacent+additions
    if len(all_pairs)>24:
        all_pairs=[all_pairs[i] for i in np.linspace(0,len(all_pairs)-1,24,dtype=int)]
    return all_pairs,dict(eligible_frames=eligible,rule='adjacent eligible frames, then uniformly spaced eligible+4 pairs, maximum24',
                         tested_pairs=[list(p) for p in all_pairs],all_frame_pairs_searched=False)


def _candidate_geometry(candidate,first,second,source,target,geometry,*,kind='ENGINEERING_AFFINE_CANDIDATE'):
    aa={p['id']:p for p in source};bb={p['id']:p for p in target}
    links=[dict(from_frame=first,to_frame=second,from_track_id=edge['source_id'],to_track_id=edge['target_id'],
        xy_from=aa[edge['source_id']]['xy'],xy_to=bb[edge['target_id']]['xy'],
        status=kind,direction='forward') for edge in candidate['mapping']]
    checked=assess_reassociated_links(dict(links=links),geometry)
    old=next((p for p in geometry.get('pairs',[]) if p.get('frames')==[first,second]),{})
    for link,check in zip(links,checked['links']):
        check['present_in_original_lk_candidates']=any(
            np.linalg.norm(np.subtract(link['xy_from'],p['xy_first']))<=2 and
            np.linalg.norm(np.subtract(link['xy_to'],p['xy_second']))<=2 for p in old.get('candidates',[]))
    return checked


def _render(out,result,pairs,frames,grays):
    esc=html.escape
    cards=[]
    available=[(p,c) for p in pairs for c in p.get('search',{}).get('candidates',[])]
    positions=np.linspace(0,len(available)-1,min(6,len(available)),dtype=int) if available else []
    for display,position in enumerate(positions):
        pair,candidate=available[position];first,second=pair['frames']
        board=Image.new('RGB',(720,680),'#eef2f5');draw=ImageDraw.Draw(board)
        anchors=set(candidate.get('supporting_anchor_sets',[[]])[0])
        anchor_pairs={(pair['proposal_edges'][i]['source_id'],pair['proposal_edges'][i]['target_id']) for i in anchors}
        for side,(idx,key,points) in enumerate(((first,'source_id',pair['source_points']),(second,'target_id',pair['target_points']))):
            tile=Image.fromarray(grays[idx]).convert('RGB');tile.thumbnail((350,610));sx=tile.width/grays[idx].shape[1];sy=tile.height/grays[idx].shape[0]
            ox=side*360+5;oy=40;board.paste(tile,(ox,oy));draw.text((ox,8),f'input {frames[idx]["input_index"]}',fill='black')
            by_id={p['id']:p for p in points}
            for n,edge in enumerate(candidate['mapping']):
                x,y=by_id[edge[key]]['xy'];px=ox+(x-.5)*sx;py=oy+(y-.5)*sy
                color='#ff9d00' if (edge['source_id'],edge['target_id']) in anchor_pairs else '#00bd65'
                draw.ellipse((px-4,py-4,px+4,py+4),fill=color,outline='black');draw.text((px+5,py),str(n+1),fill=color,stroke_width=1,stroke_fill='black')
        name=f'candidate_{display:02d}.png';board.save(out/name)
        with Image.open(out/name) as image:
            if image.convert('RGB').tobytes()!=board.tobytes():raise ValueError('candidate board readback failed')
        cards.append(f'<h3>原帧{frames[first]["input_index"]} → {frames[second]["input_index"]}</h3><p>橙色为一个已登记的拟合三锚，绿色为额外一致点；编号仅显示候选对应。</p><img src="{name}"><pre>'+esc(json.dumps(candidate['external_geometry']['counts'],ensure_ascii=False))+'</pre>')
    table=''.join('<tr><td>'+str([frames[i]['input_index'] for i in p['frames']])+'</td><td>'+str(p['proposal_receipt']['selected_edges'])+'</td><td>'+str(p['search']['candidate_count'])+'</td><td>'+esc(p['search']['status'])+'</td></tr>' for p in pairs)
    rejections=Counter()
    for pair in pairs:rejections.update(pair['search']['rejection_counts'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>共同可见子图匹配</title><style>body{font:17px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:18px;color:#203246}aside{background:#fff1d6;padding:16px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>按书中思路寻找共同可见子图</h1><aside>'+esc(result['status'])+'：工程局部仿射候选，未认证物理身份、未写入三维模型。</aside><p>印刷61–62页：三点假设、其余点与整组码检查。本实现以描述子缩小搜索、最多512三锚组合、双向像素误差及完整普通面积比复核（六点380槽）。epsilon=0.08为工程阈值；未穷举所有子集，零候选不代表不存在对应。</p><p>额外点参与搜索内检查，不冒称独立留出。补充几何只读原轨迹拟合的F，并排除训练端点依赖；三维/额外视角身份验证仍未完成。不同点数和未匹配点允许存在，不自动解释为遮挡。书稿42–43页说明不同视向的立体投影一般不满足二维仿射：面积码未通过不等于物理点配错。因此所有已保留描述子边另查原F，结果见PAIRS.json的proposal_geometry；该分支仍不是身份真值。</p><pre>'+esc(json.dumps(result,ensure_ascii=False,indent=2))+'</pre><table><tr><th>原始index</th><th>候选边</th><th>子图候选</th><th>搜索范围</th></tr>'+table+'</table>'+''.join(cards)+'<p><a href="PAIRS.json">全部候选、锚点、未匹配点和歧义</a> · <a href="FEATURES.json">原生特征身份和采样记录</a> · <a href="PLAN.json">输入与配置绑定</a></p></html>'
    diagnostics='<h2>假设处理统计</h2><p>这些计数属于假设/子集尝试，不是互斥物理点数量；支持同一映射也可能有多个锚点假设。</p><pre>'+esc(json.dumps(dict(rejections),ensure_ascii=False,indent=2))+'</pre>'
    page=page.replace('</table>','</table>'+diagnostics,1)
    if result.get('multiview_cycles'):
        page=page.replace('<pre>', '<h2>多视图闭环与身份冲突</h2><p><a href="multiview_cycles/index.html">查看第三对直接匹配、闭环与竞争候选</a>。该检查不补造观测，也不自动合并轨迹。</p><pre>', 1)
    if result.get('candidate_observability'):
        page=page.replace('<pre>', '<p><a href="candidate_observability/index.html">查看同模型相机与条件化三视图可观测性</a>；缺姿态或逆畸变域不成立时明确保留原因。</p><pre>', 1)
    (out/'index.html').write_text(page,encoding='utf-8')


def review_partial(pipeline,frames,masks,grays,geometry,output,refs):
    """Called with already-bound inputs; failure cannot mutate primary models."""
    output=Path(output)
    if output.exists():raise ValueError('fresh partial correspondence output required')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='NOT_AVAILABLE',reason=None,geometry_modified=False,identity_verified=False,
        native_injected=False,stitch_authorized=False,third_view_validation='NOT_EVALUATED',
        algorithm='book-inspired local-affine engineering approximation',profile=PROFILE)
    pairs=[]
    try:
        if not pipeline.get('stages',{}).get('sfm',{}).get('run_json'):
            result['reason']='No bound native SIFT source'
        else:
            features=load_feature_frames(pipeline,frames,masks,refs)
            serial=[]
            for frame in features['frames']:
                serial.append({**{k:v for k,v in frame.items() if k!='descriptors'},
                    'descriptor_bytes_sha256':hashlib.sha256(frame['descriptors'].tobytes()).hexdigest()})
            io.write_json(output/'FEATURES.json',dict(selection=features['selection'],frames=serial))
            selected,selection=choose_pairs(features['frames'])
            for first,second in selected:
                a,b,edges,receipt=descriptor_proposals(features['frames'][first],features['frames'][second])
                search=search_partial_affine(a,b,edges,max_hypotheses=512)
                row=dict(frames=[first,second],input_indices=[frames[first]['input_index'],frames[second]['input_index']],
                    source_points=a,target_points=b,proposal_edges=edges,proposal_receipt=receipt,search=search)
                # Book pp42-43: changing view direction of a nonplanar body need
                # not preserve 2D affine area ratios. Check descriptor proposals
                # against the original F independently of the planar subset gate.
                row['proposal_geometry']=_candidate_geometry(
                    dict(mapping=edges),first,second,a,b,geometry,kind='DESCRIPTOR_PROPOSAL_NOT_IDENTITY_VERIFIED')
                for candidate in search['candidates']:
                    candidate['external_geometry']=_candidate_geometry(candidate,first,second,a,b,geometry)
                pairs.append(row)
            counter=Counter()
            unique={}
            proposal_geometry_counts=Counter()
            for row in pairs:
                proposal_geometry_counts.update(row['proposal_geometry']['counts'])
                for candidate in row['search']['candidates']:
                    counter.update(candidate['external_geometry']['counts'])
                    for edge in candidate['external_geometry']['links']:
                        unique[(tuple(row['frames']),edge['from_track_id'],edge['to_track_id'])]=edge
            result.update(status='CANDIDATE_SEARCH_COMPLETE_NOT_VERIFIED',feature_selection=features['selection'],
                pair_selection=selection,pairs_evaluated=len(pairs),candidate_subsets=sum(p['search']['candidate_count'] for p in pairs),
                unique_candidate_edges=len(unique),unique_edge_geometry_counts=dict(Counter(e['status'] for e in unique.values())),
                repeated_candidate_edge_geometry_counts=dict(counter),
                unique_edges_absent_from_original_lk=sum(not e['present_in_original_lk_candidates'] for e in unique.values()),
                descriptor_proposal_edges=sum(len(p['proposal_edges']) for p in pairs),
                descriptor_geometry_counts=dict(proposal_geometry_counts),
                descriptor_geometry_gated_by_affine=False,
                exhaustive_search=False,quality_status='NOT_EVALUATED')
            cycles=review_multiview(features['frames'],pairs,geometry,frames,grays,
                output/'multiview_cycles',descriptor_proposals=descriptor_proposals,
                geometry_check=_candidate_geometry)
            result['multiview_cycles']={k:cycles.get(k) for k in ('status','reason','counts')}
            if cycles['status']=='CANDIDATE_DIAGNOSTIC_COMPLETE' and cycles['counts']['unique_closed_seed_triples']:
                candidate_observability=review_candidate_observability(pipeline,features,frames,cycles,
                    output/'candidate_observability',refs)
                result['candidate_observability']={k:candidate_observability.get(k) for k in ('status','reason','counts')}
    except Exception as exc:
        result.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}',partial_pairs_retained=len(pairs))
    io.write_json(output/'PAIRS.json',pairs)
    io.write_json(output/'PLAN.json',dict(profile=PROFILE,bindings=refs,book_printed_pages=[21,22,42,43,55,61,62],
        original_geometry_sha256=hashlib.sha256(json.dumps(geometry,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        approximate_not_exact_book_certificate=True))
    io.write_json(output/'RESULT.json',result)
    _render(output,result,pairs,frames,grays)
    return result
