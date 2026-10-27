"""Automatic bounded post-run evidence review; never modifies model or matches."""
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import html
import math
import sqlite3
import uuid
import numpy as np
from PIL import Image, ImageDraw

from . import parallel_photo_audit as native
from .viewpoint_research import make_plan
from .surface import choose_model, read
from .coverage_review import build_point_coverage
from .surface_support import discover_surface_sources, review_surface_sources
from kozlov_lab.stereo.multiview_candidate_evidence import freeze_cases, analyse_plan, _groups
from kozlov_lab.stereo.appearance_evidence import compare_patches, descriptor_evidence, _patch

ROOT=Path(__file__).resolve().parents[2]


def evenly_spaced(values, maximum=12):
    if len(values)<=maximum:return list(values)
    return [values[j*(len(values)-1)//(maximum-1)] for j in range(maximum)]


def aliases_preserved(full_tracks, selected_tracks):
    """Do not conceal alias links established only in omitted native views."""
    selected=set(selected_tracks)
    full=sorted(sorted(selected.intersection(g)) for g in _groups(full_tracks) if selected.intersection(g))
    return full==sorted(_groups(selected_tracks))


def evidence_coverage(cases):
    coverage={key:sum(c[key]['status']=='EVALUATED' for c in cases)
              for key in ('pair','extra_summary','pair_plus_extra_summary')}
    coverage['appearance']={key:sum(c['appearance_summary'][key]['evaluated_count']>0 for c in cases)
                            for key in ('descriptor','patch8','patch16')}
    available=any(coverage[k] for k in ('pair','extra_summary','pair_plus_extra_summary')) or any(coverage['appearance'].values())
    return coverage,available


def check_output_scope(pipeline, output, discovery):
    # The exact post_review child is reserved for automatic execution.
    if output.is_relative_to(pipeline) and output!=pipeline/'post_review':
        raise ValueError('review output overlaps pipeline input/artifact tree')
    stages=discovery.get('stages',{});protected=[pipeline/'input']
    if stages.get('sfm',{}).get('run_json'):protected.append(Path(stages['sfm']['run_json']).resolve().parent)
    if stages.get('segmentation',{}).get('run_dir'):protected.append(Path(stages['segmentation']['run_dir']).resolve())
    if discovery.get('source'):protected.append(Path(discovery['source']).resolve())
    for path in protected:
        if output==path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('review output overlaps protected source: '+str(path))


def closest_swap(plan):
    """Use the previous diagnostic's frozen holdout, not its residuals."""
    choices=[]; rows=plan['raw_pixels']; ids=plan['point_ids']
    for a,b in combinations(plan['holdout_indices'],2):
        distance=math.hypot(*(rows[a][k]-rows[b][k] for k in (2,3)))
        if distance>0:choices.append((distance,a,b))
    if not choices:return None
    distance,a,b=min(choices)
    return dict(image_ids=plan['image_ids'],source_point_ids=[ids[a],ids[b]]),dict(
        image_ids=plan['image_ids'],holdout_row_indices=[a,b],source_point_ids=[ids[a],ids[b]],
        displacement_pixels=distance,selection='minimum positive right-image distance; tie original row indices')


def descriptor_database(path, images):
    """Read only; bind descriptor row identity to every native image observation."""
    if not path.is_file():return {},dict(status='UNAVAILABLE',reason='database.db absent')
    result={}
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for iid,image in images.items():
            identity=db.execute('SELECT name,camera_id FROM images WHERE image_id=?',(iid,)).fetchone()
            if identity!=(image['name'],image['camera_id']):raise ValueError('database image identity differs')
            k=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(iid,)).fetchone()
            d=db.execute('SELECT type,rows,cols,data FROM descriptors WHERE image_id=?',(iid,)).fetchone()
            if not k or not d:continue
            if (k[1] not in (2,4,6) or k[0]!=len(image['observations']) or len(k[2])!=4*k[0]*k[1]
                    or d[0]!=0 or d[1]!=k[0] or d[2]!=128 or len(d[3])!=d[1]*128):
                raise ValueError('unsupported or inconsistent stored feature representation')
            keypoints=np.frombuffer(k[2],dtype='<f4').reshape(k[0],k[1])
            native_xy=np.array([row[:2] for row in image['observations']])
            if not np.array_equal(keypoints[:,:2],native_xy):raise ValueError('database/native keypoint order or pixels differ')
            result[iid]=np.frombuffer(d[3],dtype=np.uint8).reshape(d[1],128)
    return result,dict(status='AVAILABLE' if result else 'UNAVAILABLE',image_ids=sorted(result),
        representation='database type=0, uint8[128], row identity checked against all native pixels',
        used_in_original_matching=True,independent_truth=False)


def load_sources(pipeline, discovery, refs):
    stages=discovery.get('stages',{}); sfm_record=stages.get('sfm',{})
    if not sfm_record.get('run_json'):return None,'No committed native SfM run in pipeline'
    run_path=Path(sfm_record['run_json']).resolve(strict=True); refs.append(native.binding(run_path))
    run=read(run_path)
    if run.get('status') not in ('SUCCESS','PARTIAL') or run.get('contract_version') not in ('sfm-native-object-v1', 'sfm-native-object-pinhole-v1'):
        return None,'Native SfM has no eligible object model'
    try:selected,selection=choose_model(run.get('models',[]),stages.get('surface',{}).get('selected_model_id'))
    except ValueError as exc:return None,str(exc)
    sfm_root=run_path.parent; binary=native.under(sfm_root,selected['binary_path'])
    for name in ('images.bin','points3D.bin','cameras.bin'):refs.append(native.binding(binary/name))
    images,points=native.read_native(binary/'images.bin',binary/'points3D.bin')
    if sorted(images)!=sorted(selected['registered_image_ids']) or len(points)!=selected['num_points3D']:
        raise ValueError('native model identity/count differs')
    fmap=sfm_root/'frames_map.jsonl';refs.append(native.binding(fmap))
    records=[native.loads(line) for line in fmap.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    frames={}
    for iid,image in images.items():
        matches=[(row,r) for row in records for r in row['registrations']
                 if str(r['model_id'])==str(selected['model_id']) and r['image_id']==iid]
        if len(matches)!=1:raise ValueError('one native frame identity required')
        row,reg=matches[0]
        if (row['staged_image_name']!=image['name'] or reg['camera_id']!=image['camera_id']
                or row['upstream_io_run_id']!=run['upstream_io_run_id'] or row['status']!='REGISTERED'
                or type(row['input_index']) is not int):raise ValueError('frame identity differs')
        frames[iid]=dict(image_id=iid,input_index=row['input_index'],frame_id=row['frame_id'],
            path=native.under(sfm_root,row['staged_image_name']),sha256=row['canonical_image_sha256'])
    order=sorted(images,key=lambda i:(frames[i]['input_index'],i))
    chosen=evenly_spaced(order)
    full_tracks={pid:{i:list(images[i]['observations'][idx][:2]) for i,idx in p['track'].items()}
                 for pid,p in points.items()} if len(chosen)<len(order) else None
    images={i:images[i] for i in chosen}; frames={i:frames[i] for i in chosen}
    tracks={pid:{i:list(images[i]['observations'][idx][:2]) for i,idx in p['track'].items() if i in images}
            for pid,p in points.items()}
    tracks={pid:obs for pid,obs in tracks.items() if obs}
    if len(tracks)>4096:return None,'Diagnostic track bound 4096 exceeded; no silent point truncation'
    if full_tracks is not None and not aliases_preserved(full_tracks,tracks):
        return None,'Omitted views conceal alias connections; refusing candidate scoring'
    if any(points[pid]['ambiguous_observations'] for pid in tracks):
        return None,'Native same-image multiple observations need explicit review; no silent choice'
    gray={}; masks={};mask_info={}
    manifest_path=None; manifest=None
    if stages.get('segmentation',{}).get('run_dir'):
        manifest_path=Path(stages['segmentation']['run_dir']).resolve(strict=True)/'manifest.json'
        refs.append(native.binding(manifest_path));manifest=read(manifest_path)
        if manifest.get('contract_version')!='object-mask-v1':raise ValueError('object-mask-v1 required')
    for iid,f in frames.items():
        ref=native.binding(f['path'])
        if ref['sha256']!=f['sha256']:raise ValueError('canonical image changed')
        refs.append(ref);f['binding']=ref;f['path']=str(f['path'])
        with Image.open(f['path']) as im:gray[iid]=np.asarray(im.convert('L'))
        if manifest is None:
            mask_info[iid]=dict(status='UNAVAILABLE',reason='No segmentation manifest');continue
        rows=[r for r in manifest['frames'] if r['input_index']==f['input_index']]
        if len(rows)!=1:raise ValueError('one mask-manifest frame required')
        row=rows[0]
        if row['frame_id']!=f['frame_id'] or row['source_image_sha256']!=f['sha256']:
            raise ValueError('mask/frame/canonical identity differs')
        if row['status']!='OK' or not row.get('selected_mask'):
            mask_info[iid]=dict(status='UNAVAILABLE',reason='source mask frame '+row['status']);continue
        if row.get('image_domain')!='photo_io_canonical':raise ValueError('mask must be canonical domain')
        source=native.binding(Path(row['source_image']))
        if source['sha256']!=f['sha256']:raise ValueError('mask canonical source bytes differ')
        refs.append(source)
        mref=native.binding(native.under(manifest_path.parent,row['selected_mask']['path']))
        if mref['sha256']!=row['selected_mask']['sha256']:raise ValueError('selected mask bytes differ')
        candidates=[c for c in row['candidates'] if c['candidate_id']==row['selected_candidate_id']]
        if len(candidates)!=1 or candidates[0]['mask_sha256']!=mref['sha256']:raise ValueError('selected candidate identity differs')
        refs.append(mref)
        with Image.open(mref['path']) as im:
            if im.mode!='L':raise ValueError('single-channel source mask required')
            mask=np.array(im)
        if mask.shape!=gray[iid].shape or mask.shape!=(row['height'],row['width']) or not np.isin(mask,[0,255]).all():
            raise ValueError('mask shape/labels differ')
        masks[iid]=mask;mask_info[iid]=dict(status='AVAILABLE',binding=mref)
    db=sfm_root/'database.db'
    if db.exists():refs.append(native.binding(db))
    descriptors,descriptor_info=descriptor_database(db,images)
    return dict(images=images,points=points,tracks=tracks,frames=frames,gray=gray,masks=masks,
        mask_info=mask_info,descriptors=descriptors,descriptor_info=descriptor_info,selection=selection,
        source_registered_count=len(order),selected_image_ids=chosen,omitted_image_ids=[i for i in order if i not in chosen],
        sfm_counts=run['counts']),None


def compare_appearance(source, case):
    left,right=case['pair_ids'];a,b=case['original_id'],case['alternative_id']
    reference_ids=[left]+[e['image_id'] for e in case['extra']]
    rows=[]
    for ref in reference_ids:
        p=source['tracks'][a];q=source['tracks'][b]
        row=dict(reference_image_id=ref,reference_point_id=a,target_image_id=right,
            original_point_id=a,alternative_point_id=b,reference_pixels=p[ref],
            original_pixels=p[right],alternative_pixels=q[right],patches={})
        if ref in source['descriptors'] and right in source['descriptors']:
            index=lambda pid,i:source['points'][pid]['track'][i]
            vectors=[source['descriptors'][i][index(pid,i)] for pid,i in ((a,ref),(a,right),(b,right))]
            row['descriptor']=descriptor_evidence(*vectors)
            row['descriptor']['row_sha256']=[hashlib.sha256(v.tobytes()).hexdigest() for v in vectors]
        else:row['descriptor']=dict(status='UNAVAILABLE',support=None,reason='stored descriptor unavailable')
        for radius in (8,16):
            if ref in source['masks'] and right in source['masks']:
                row['patches'][str(radius)]=compare_patches(source['gray'][ref],source['masks'][ref],p[ref],
                    source['gray'][right],source['masks'][right],p[right],q[right],radius)
            else:row['patches'][str(radius)]=dict(status='UNAVAILABLE',support=None,reason='bound object mask unavailable')
        rows.append(row)
    return rows


def summarize_appearance(rows):
    result={}
    for kind in ('descriptor','patch8','patch16'):
        source=[r['descriptor'] if kind=='descriptor' else r['patches'][kind[5:]] for r in rows]
        good=[s for s in source if s['status']=='EVALUATED']
        result[kind]=dict(status='EVALUATED' if good else 'UNAVAILABLE',reference_count=len(source),
            evaluated_count=len(good),unavailable_count=len(source)-len(good),support='unavailable')
        if good:
            fields=('original_distance','alternative_distance') if kind=='descriptor' else ('original_zncc','alternative_zncc')
            a,b=(float(np.mean([s[f] for s in good])) for f in fields)
            support='tie' if abs(a-b)<=1e-10 else ('original' if ((a<b) if kind=='descriptor' else (a>b)) else 'alternative')
            result[kind].update(original_mean=a,alternative_mean=b,support=support,
                support_counts=dict(Counter(s['support'] for s in good)))
    return result


def contact(source,case,path):
    left,right=case['pair_ids'];a,b=case['original_id'],case['alternative_id']
    refs=[(left,a,'Reference'),(right,a,'Original'),(right,b,'Alternative')]
    refs += [(e['image_id'],a,'Extra') for e in case['extra']]
    canvas=Image.new('RGB',(124*len(refs),148),'#ebeff4');draw=ImageDraw.Draw(canvas)
    for n,(iid,pid,label) in enumerate(refs):
        xy=np.array(source['tracks'][pid][iid]);mask=source['masks'].get(iid)
        # Visual-only extraction when no mask; it is never used as foreground evidence.
        values,stats=_patch(source['gray'][iid],mask if mask is not None else np.full(source['gray'][iid].shape,255,np.uint8),xy,16)
        if values is not None:
            tile=Image.fromarray(np.clip(np.rint(values),0,255).astype(np.uint8)).resize((99,99),Image.Resampling.NEAREST)
            canvas.paste(tile,(n*124+10,34))
        draw.text((n*124+5,3),label,fill='#19324c')
        draw.text((n*124+5,17),f'f{source["frames"][iid]["input_index"]} p{pid}',fill='#19324c')
    canvas.save(path)


def render(output,result):
    labels={'original':'原对应','alternative':'替代','tie':'相等','unavailable':'不可评估'}
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动建模后复核</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#243246}aside{background:#fff0cb;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}details{margin:12px 0}a{color:#146b99}</style>'
    page+=f'<h1>自动建模后复核</h1><aside>{html.escape(result["status"])}：这是对应一致性与外观证据报告，未修改点、掩膜、网格，也不等于几何验收。</aside>'
    page+=f'<p>来源：{html.escape(result["pipeline"])}</p><p>{html.escape(result.get("reason") or "从已提交数据自动选择视角、冻结近邻交换、排除候选别名后拟合，再检查额外视角与真实纹理。")}</p>'
    page+='<p>几何与特征距离越低越接近，图块 ZNCC 越高越接近；“倾向原对应”仅是两候选之间的比较。SIFT 已参与原匹配，不是独立真值；图块没有尺度、旋转或透视补偿。跨掩膜/边界/零纹理项不评分。</p>'
    if result.get('point_coverage'):
        p=result['point_coverage'];page+=f'<h2>物体选点覆盖</h2><p>{html.escape(p["status"])} · <a href="point_coverage/index.html">覆盖缺口、代表点与多视图支持</a></p>'
    if result.get('temporal_tracks'):
        p=result['temporal_tracks'];page+=f'<h2>视频连续点对应</h2><p>{html.escape(p["status"])} · <a href="temporal_tracks/index.html">逐帧轨迹、断点与跨分量候选</a></p>'
    if result.get('intrinsics_stability'):
        p=result['intrinsics_stability'];page+=f'<h2>焦距与内参稳定性</h2><p>{html.escape(p["status"])}'
        if p.get('report'):page+=' · <a href="intrinsics_stability/index.html">预处理、逐图对条件曲线与标定边界</a>'
        if p.get('reason'):page+=' · '+html.escape(p['reason'])
        page+='</p>'
    if result.get('intrinsics_sensitivity'):
        p=result['intrinsics_sensitivity'];page+=f'<h2>内参假设与退化风险</h2><p>{html.escape(p["status"])}'
        if p.get('report'):page+=' · <a href="intrinsics_sensitivity/index.html">主点、径向畸变与单应留出检查</a>'
        if p.get('reason'):page+=' · '+html.escape(p['reason'])
        page+='</p>'
    if result.get('surface_support'):
        p=result['surface_support'];page+=f'<h2>表面观测支持</h2><p>{html.escape(p["status"])}'
        if p.get('report'):page+=' · <a href="surface_support/index.html">网格采样支持距离</a>'
        if p.get('reason'):page+=' · '+html.escape(p['reason'])
        page+='</p>'
    page+='<table><tr><th>图对 / 点</th><th>双视角</th><th>额外视角</th><th>合并几何</th><th>特征向量</th><th>17px / 33px 图块</th></tr>'
    for c in result.get('cases',[]):
        def label(s):return labels.get(s,'不可评估')
        a=c['appearance_summary']
        page+=f'<tr><td><a href="#{c["case_id"]}">{c["pair_input_indices"]} / {c["original_id"]}→{c["alternative_id"]}</a></td><td>{label(c["pair"]["support"])}</td><td>{label(c["extra_summary"]["support"])}</td><td>{label(c["pair_plus_extra_summary"]["support"])}</td><td>{label(a["descriptor"]["support"])}</td><td>{label(a["patch8"]["support"])} / {label(a["patch16"]["support"])}</td></tr>'
    page+='</table>'
    for c in result.get('cases',[]):
        page+=f'<details id="{c["case_id"]}"><summary>{c["pair_input_indices"]} 点 {c["original_id"]} 与替代 {c["alternative_id"]}：原图固定采样块与数值</summary><img src="{c["contact"]}" alt="固定图块，无旋转补偿"><p>图块为原始灰度双线性采样的放大展示；可见图块也可能因掩膜边界而未参与评分。</p><pre>{html.escape(__import__("json").dumps(dict(pair=c["pair"],extra=c["extra_summary"],appearance=c["appearance_summary"]),ensure_ascii=False,indent=2))}</pre><a href="cases/{c["case_id"]}.json">全部逐视角证据</a></details>'
    page+='<p>邻近交换是预设敏感性诊断，不能换算错误检测率。额外视角仍沿用原轨迹，彼此相关；两候选都可能错误。没有引入毫米尺度，也未宣布 Kozlov 原算法已适用于透视噪声。</p><p><a href="RESULT.json">完整结果</a> · <a href="PLAN.json">冻结输入与选择</a> · <a href="DISCOVERY.json">主流程发现状态快照</a></p></html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def review(pipeline, output=None):
    pipeline=Path(pipeline).resolve(strict=True)
    output=(Path(output) if output is not None else pipeline.parent/('post_review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])).resolve()
    if output.exists() or output==pipeline or output in pipeline.parents:raise ValueError('new review output required')
    discovery=read(pipeline/'pipeline.json')
    check_output_scope(pipeline,output,discovery)
    output.mkdir(parents=True,exist_ok=False)
    native.write_json(output/'DISCOVERY.json',discovery)
    refs=[];result=dict(status='FAILED',pipeline=str(pipeline),reason=None,cases=[],geometry_changed=False,
        correspondence_verified=False,quality_status='NOT_EVALUATED',native_model_executions=0)
    try:
        # Database-only diagnostic also applies when no model was registered.
        # It must not depend on successful temporal tracking or point scoring.
        sfm_path=discovery.get('stages',{}).get('sfm',{}).get('run_json')
        if sfm_path:
            try:
                from .intrinsics_stability import audit_intrinsics
                intrinsics=audit_intrinsics(sfm_path,output/'intrinsics_stability',pipeline=pipeline)
                result['intrinsics_stability']=dict(status=intrinsics['status'],report=True,
                    decision=intrinsics['decision'],counts=intrinsics['counts'])
                refs+=intrinsics['source_bindings']+intrinsics['implementation']
                try:
                    from .intrinsics_sensitivity import audit_sensitivity
                    sensitivity=audit_sensitivity(output/'intrinsics_stability/INTRINSICS.json',output/'intrinsics_sensitivity')
                    result['intrinsics_sensitivity']=dict(status=sensitivity['status'],report=True,
                        decision=sensitivity['decision'],summary=sensitivity['summary'])
                    refs+=sensitivity['source_bindings']+sensitivity['implementation']
                except Exception as exc:
                    result['intrinsics_sensitivity']=dict(status='NOT_EVALUATED',report=False,
                        reason=f'{type(exc).__name__}: {exc}',calibration_accepted=False)
            except Exception as exc:
                result['intrinsics_stability']=dict(status='NOT_EVALUATED',report=False,
                    reason=f'{type(exc).__name__}: {exc}',calibration_accepted=False)
        else:
            result['intrinsics_stability']=dict(status='NOT_AVAILABLE',report=False,
                reason='No saved SfM database run',calibration_accepted=False)
        if discovery.get('source_kind')=='video':
            from .temporal_review import review_video
            result['temporal_tracks']=review_video(pipeline,output/'temporal_tracks',discovery=discovery)
        source,reason=load_sources(pipeline,discovery,refs)
        if source is None:
            result.update(status='INSUFFICIENT_DATA',reason=reason)
            native.write_json(output/'PLAN.json',dict(discovery_snapshot=native.binding(output/'DISCOVERY.json'),source_bindings=refs))
        else:
            pairs=[make_plan(pair,source['images'],source['points']) for pair in combinations(sorted(source['images']),2)]
            pairs=sorted(sorted(pairs,key=lambda p:(-len(p['point_ids']),p['image_ids']))[:10],key=lambda p:p['image_ids'])
            swaps=[];selection=[]
            for p in pairs:
                value=closest_swap(p)
                if value is not None:swaps.append(value[0]);selection.append(value[1])
            frozen=freeze_cases(source['tracks'],swaps) if swaps else None
            surface_inputs=None
            try:
                surface_inputs=discover_surface_sources(discovery,refs)
                if surface_inputs is None:
                    result['surface_support']=dict(status='NOT_AVAILABLE',reason='No same-run bound dense cloud and selected raw mesh')
            except (ValueError,OSError,KeyError,TypeError) as exc:
                result['surface_support']=dict(status='NOT_EVALUATED',reason=f'{type(exc).__name__}: {exc}')
            implementation=[native.binding(p) for p in (Path(__file__),ROOT/'scripts/review_photo_model.py',
                ROOT/'src/kozlov_lab/stereo/appearance_evidence.py',ROOT/'src/kozlov_lab/stereo/multiview_candidate_evidence.py',
                ROOT/'src/kozlov_lab/stereo/viewpoint_consistency.py',
                ROOT/'src/kozlov_lab/stereo/point_coverage.py',ROOT/'integrations/demo_pipeline/coverage_review.py',
                ROOT/'integrations/demo_pipeline/surface_support.py',ROOT/'integrations/meshprep/ply_io.py',
                ROOT/'integrations/viewer/ply_data.py')]
            plan=dict(schema='automatic-post-review-plan-v1',source_bindings=refs,implementation=implementation,
                discovery_snapshot=native.binding(output/'DISCOVERY.json'),selection=source['selection'],frames=list(source['frames'].values()),
                source_registered_count=source['source_registered_count'],selected_image_ids=source['selected_image_ids'],
                omitted_image_ids=source['omitted_image_ids'],view_selection='input-index order; up to twelve evenly spaced registered views',
                pair_selection='up to ten pairs by descending shared singleton tracks, ID tie; no residual selection',
                pair_plans=pairs,local_swaps=selection,frozen_candidate_plan=frozen,mask_info=source['mask_info'],
                descriptor_info=source['descriptor_info'],patch_radii=[8,16],quality_threshold=None,
                surface_support_inputs=surface_inputs,
                sfm_counts=source['sfm_counts'],segmentation_counts=discovery.get('stages',{}).get('segmentation',{}).get('counts'))
            native.write_json(output/'PLAN.json',plan)
            result['point_coverage']=build_point_coverage(source,output/'point_coverage')
            if surface_inputs is not None:
                try:
                    result['surface_support']=review_surface_sources(surface_inputs['mesh_path'],surface_inputs['dense_path'],output/'surface_support')
                except (ValueError,OSError,KeyError,TypeError) as exc:
                    result['surface_support']=dict(status='NOT_EVALUATED',reason=f'{type(exc).__name__}: {exc}')
            if frozen is None:result.update(status='INSUFFICIENT_DATA',reason='No two distinct heldout right-image positions')
            else:
                geometry=analyse_plan(frozen);native.write_json(output/'GEOMETRY.json',geometry)
                (output/'cases').mkdir();(output/'contacts').mkdir()
                for case in geometry['cases']:
                    case['appearance']=compare_appearance(source,case)
                    case['appearance_summary']=summarize_appearance(case['appearance'])
                    case['pair_input_indices']=[source['frames'][i]['input_index'] for i in case['pair_ids']]
                    case['contact']='contacts/'+case['case_id']+'.png'
                    contact(source,case,output/case['contact'])
                    native.write_json(output/'cases'/(case['case_id']+'.json'),case)
                result.update(status='DIAGNOSTIC_COMPLETE',cases=geometry['cases'],model_count=len(geometry['models']),
                    view_count=len(source['images']),track_count=len(source['tracks']),pair_count=len(swaps),
                    selected_model_id=source['selection']['selected_model_id'],source_registered_count=source['source_registered_count'])
                result['summary']={kind:dict(Counter(c[kind]['support'] for c in result['cases']))
                    for kind in ('pair','extra_summary','pair_plus_extra_summary')}
                result['summary']['appearance']={kind:dict(Counter(c['appearance_summary'][kind]['support'] for c in result['cases']))
                    for kind in ('descriptor','patch8','patch16')}
                result['evidence_coverage'],available=evidence_coverage(result['cases'])
                if not available:
                    result.update(status='INSUFFICIENT_DATA',reason='No paired evaluable geometry or appearance evidence; see per-case reasons')
            refs+=implementation
        for ref in refs:
            if native.binding(ref['path'])!=ref:raise ValueError('post-review source drift: '+ref['path'])
        result['source_unchanged']=True
    except Exception as exc:
        result.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}',source_unchanged=None)
        if not (output/'PLAN.json').exists():native.write_json(output/'PLAN.json',dict(source_bindings=refs,preparation_failed=True))
    native.write_json(output/'RESULT.json',result)
    render(output,result)
    result['result_path']=str(output/'RESULT.json');result['report_path']=str(output/'index.html')
    return result
