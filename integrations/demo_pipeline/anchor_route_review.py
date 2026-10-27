"""Automatic measured-anchor supplement, kept separate from original geometry."""
from collections import defaultdict
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .temporal_review import fingerprint
from .intermediate_population_review import write
from .intermediate_population_report import _link


def _clouds(output, classification, budget):
    from integrations.viewer.ply_data import read_ply
    from integrations.viewer.export import export_html
    # Viewer evidence embeds only the binding; large full ledgers stay separate.
    write(output/'GEOMETRY_BINDING.json',dict(geometry=fingerprint(output/'GEOMETRY.json'),
        note='Full measured and numerical ledger retained in GEOMETRY.json; no quality acceptance.'))
    groups = defaultdict(list)
    for row in classification['rows']:
        if row['tier'] in ('STRICT_CONDITIONAL', 'RELAXED_CONDITIONAL'):
            groups[(row['model_key'], row['tier'])].append(row)
    artifacts, scenes, manifest = [output/'GEOMETRY_BINDING.json'], [], []
    for number, ((model, tier), rows) in enumerate(sorted(groups.items())):
        budget()
        color = [70, 190, 125] if tier == 'STRICT_CONDITIONAL' else [240, 170, 65]
        path = output / f'conditional_{number:03d}.ply'
        header = ['ply', 'format ascii 1.0', 'comment measured dense route conditional points',
            'comment arbitrary saved model scale; diagnostic color; no accepted surface',
            f'element vertex {len(rows)}', 'property double x', 'property double y', 'property double z',
            'property uchar red', 'property uchar green', 'property uchar blue', 'end_header']
        lines = [' '.join([*(format(v,'.17g') for v in r['xyz']), *(str(c) for c in color)]) for r in rows]
        with path.open('x', encoding='ascii', newline='\n') as stream:
            stream.write('\n'.join(header + lines) + '\n')
        readback = read_ply(path, max_bytes=1<<20, max_vertices=96, max_faces=1)
        if readback['vertices'] != [r['xyz'] for r in rows] or readback['faces'] or readback['colors'] != [color]*len(rows):
            raise ValueError('SUPPLEMENTAL_PLY_READBACK')
        ref = fingerprint(path)
        manifest.append(dict(model_key=model, tier=tier, vertices=len(rows), faces=0,
            binding=ref, readback=True, vertex_ledger=[dict(vertex_index=i, **r) for i,r in enumerate(rows)],
            scale_status='unknown', geometry_accepted=False, native_injected=False))
        artifacts.append(path)
        scenes.append(dict(ply=path.name, sha256=ref['sha256'], title=f'模型 {model} · {tier}',
            data_kind='real_photos', declared_unit=None, evidence=['GEOMETRY_BINDING.json'],
            note='同一视频实测补跟踪得到的条件点；不同模型独立，尚未通过物理身份、尺度或表面验收。'))
    write(output/'CLOUDS.json', manifest); artifacts.append(output/'CLOUDS.json')
    if scenes:
        write(output/'scenes.json',dict(schema='ply-viewer-scenes-v1',scenes=scenes))
        budget(); export_html(output/'scenes.json',output/'viewer.html')
        artifacts += [output/'scenes.json',output/'viewer.html']
    return artifacts, manifest


def _report(output, plan, measured, geometry, projection, original_projection, decoded, clouds, budget):
    escaped = lambda value: html.escape(str(value), quote=True)
    classifications = {r['group_id']: r for r in geometry['classification']['rows']}
    artifacts, sections = [], []
    for number, route in enumerate(plan['routes']):
        budget()
        indices = route['camera_decode_indices']
        rows = [r for r in measured['records'] if r['model_key']==route['model_key'] and r['source_input_index']==route['source_input_index']]
        sheet = Image.new('RGB',(1080,700),'#15202b'); draw=ImageDraw.Draw(sheet)
        for column,index in enumerate(indices):
            budget(); frame=decoded[index]
            with Image.open(frame['path']) as image: pic=image.convert('RGB')
            if pic.size != (frame['width'],frame['height']): raise ValueError('ANCHOR_REPORT_IMAGE_DOMAIN')
            pic.thumbnail((340,600)); x=column*360+(360-pic.width)//2; y=70
            sheet.paste(pic,(x,y))
            draw.text((column*360+10,12),f'model={route["model_key"]} decode={index}',fill='white')
            draw.text((column*360+10,32),'green=strict orange=relaxed red=other',fill='white')
            for row in rows:
                obs=next((o for o in row['observations'] if o['decode_index']==index),None)
                if obs is None: continue
                c=classifications.get(f'{row["model_key"]}:{row["track_id"]}:{row["identity_segment"]}',{})
                color={'STRICT_CONDITIONAL':'#46be7d','RELAXED_CONDITIONAL':'#f0aa41'}.get(c.get('tier'),'#eb7777')
                px=x+(obs['raw_xy'][0]-.5)*pic.width/frame['width']
                py=y+(obs['raw_xy'][1]-.5)*pic.height/frame['height']
                draw.ellipse((px-3,py-3,px+3,py+3),outline=color,width=2)
        name=f'anchors_{number:03d}.png'
        with (output/name).open('xb') as stream: sheet.save(stream,format='PNG')
        artifacts.append(output/name)
        links=' · '.join(f'<a href="{escaped(_link(decoded[i]["path"],output))}">原图 {i}</a>' for i in indices)
        measured_count=sum(r['status']=='MEASURED_THREE_VIEW_ANCHOR' for r in rows)
        sections.append(f'<h2>模型 {escaped(route["model_key"])} · 源 {route["source_input_index"]}</h2>'
            f'<p>本轮冻结补测 {len(rows)} 点，其中 {measured_count} 点到达三个相机端点；'
            '其余短轨迹并未删除。图中只画本轮补测点，绿色/橙色为条件几何等级，红色为其它结果。</p>'
            f'<p>{links}</p><img src="{name}">')
    old={r['group_id'] for r in original_projection['joins']}
    new={r['group_id'] for r in projection['joins']}
    comparison=dict(previous_exact_xyz_links=len(old),supplement_exact_xyz_links=len(new),
        newly_linked_group_ids=sorted(new-old),shared_group_ids=sorted(new&old),
        comparison_scope='different conditional XYZ cohorts; no matched-cohort accuracy or causal claim')
    write(output/'COMPARISON.json',comparison);artifacts.append(output/'COMPARISON.json')
    table=[]
    for row in geometry['classification']['rows']:
        metrics=row.get('metrics') or {}
        table.append('<tr>'+''.join('<td>'+escaped(v)+'</td>' for v in
            (row['group_id'],row['tier'],metrics.get('max_raw_error_px'),metrics.get('max_loo_error_px'),
             metrics.get('measurement_ray_angle_deg'),'; '.join(row.get('reasons',[]))))+'</tr>')
    cloudlink='<a href="viewer.html">查看条件三维点云</a>' if clouds else '没有通过本轮条件门槛的点；未生成占位模型。'
    text='''<!doctype html><meta charset="utf-8"><title>向已知视角自动补跟踪</title>
<style>body{font:16px/1.7 system-ui;margin:28px;background:#15202b;color:#eef3fa}a{color:#8ecaff}img{max-width:100%}pre{white-space:pre-wrap}td,th{border:1px solid #556;padding:6px}table{border-collapse:collapse}</style>
<h1>向已知视角自动补跟踪</h1><p>同一短轨迹的源点沿另一时间方向，逐帧连接两个已保存相机。路线与点群在新增测量前冻结；不按拟合成功补位。原始轨迹、相机、掩膜和原有XYZ保持不变。</p>
<p>只补真实测量：原始掩膜用于已知端点，中间帧未知掩膜不补造。两段双向检查与三视图固定相机/径向模型拟合分开记账，跨段全局闭环未额外验证。孔洞和未通过点保留。</p>
<p>点云是条件结果，模型之间未对齐、尺度未知；短轨迹连续不自动等于正确深度。论文平行投影定理不能直接证明这些透视视频点。</p>'''
    text+=f'<p>{cloudlink} · <a href="projection/index.html">补充锚点的留出投影检验</a> · <a href="query_cameras/index.html">查询相机与新增条件点</a> · <a href="MEASUREMENT.json">测量账本</a> · <a href="GEOMETRY.json">几何及全部拒绝原因</a> · <a href="COMPARISON.json">与原锚点联接数量比较</a></p>'
    text+='<pre>'+escaped(json.dumps(dict(measurement=measured['counts'],geometry=geometry['counts'],comparison=comparison),ensure_ascii=False,indent=2))+'</pre>'
    text+=''.join(sections)+'<h2>逐点几何结果</h2><table><tr><th>group</th><th>tier</th><th>raw max px</th><th>LOO max px</th><th>ray angle deg</th><th>reason</th></tr>'+''.join(table)+'</table>'
    with (output/'index.html').open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    return artifacts+[output/'index.html'],comparison


def review_anchor_routes(plan, ledger, frames, masks, decoded, output, refs, cv, camera_result,
                         original_projection, *, budget=lambda:None):
    from .anchor_route_tracking import measure_anchor_routes
    from .anchor_route_geometry import assess_anchor_routes
    from .tracklet_camera_review import assess_cameras
    from .tracklet_camera_report import render_camera_review
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',geometry_accepted=False,new_cameras=0,new_model_points=0)
    try:
        budget()
        measured=measure_anchor_routes(plan,ledger,frames,masks,decoded,output/'tracking',refs,cv,budget=budget)
        write(output/'MEASUREMENT.json',measured)
        geometry=assess_anchor_routes(measured,frames,masks,camera_result,budget=budget)
        write(output/'GEOMETRY.json',geometry)
        # Do not silently replace old XYZ for the same source group. This is a separate cohort.
        projection=assess_cameras(ledger,geometry['classification'],frames,camera_result,budget=budget)
        sub=output/'projection';sub.mkdir(exist_ok=False)
        write(sub/'PROJECTION.json',projection)
        epipolar=dict(status='NOT_REPEATED',pairs=[],reason='Original measured-prefix F/H remain in sibling camera_diagnostics.')
        write(sub/'EPIPOLAR.json',epipolar)
        artifacts=[output/'MEASUREMENT.json',output/'GEOMETRY.json',sub/'PROJECTION.json',sub/'EPIPOLAR.json']
        artifacts+=render_camera_review(sub,projection,epipolar,decoded,budget=budget)
        from .query_camera_review import review_query_cameras
        query=review_query_cameras(ledger,projection,frames,masks,camera_result,decoded,
            output/'query_cameras',cv,budget=budget)
        artifacts += [Path(r['path']) for r in query['source_bindings']]
        cloudpaths,clouds=_clouds(output,geometry['classification'],budget);artifacts+=cloudpaths
        pagepaths,comparison=_report(output,plan,measured,geometry,projection,original_projection,decoded,clouds,budget)
        artifacts+=pagepaths
        for ref in refs+measured['source_bindings']:
            budget()
            if fingerprint(ref['path'])!=ref:raise ValueError('ANCHOR_ROUTE_SOURCE_DRIFT')
        result.update(status='ANCHOR_ROUTE_REVIEW_COMPLETE',measurement_counts=measured['counts'],
            geometry_counts=geometry['counts'],numeric_groups=geometry['numeric_groups'],
            projection_counts=projection['counts'],comparison=comparison,flow_calls=measured['flow_calls'],
            exported_conditional_points=sum(c['vertices'] for c in clouds),source_unchanged=True,
            query_cameras={k:v for k,v in query.items() if k!='source_bindings'},
            report='intermediate_frames/population/anchor_routes/index.html')
        write(output/'RESULT.json',result);artifacts.append(output/'RESULT.json')
        result['source_bindings']=[fingerprint(p) for p in dict.fromkeys(artifacts)] + measured['source_bindings']
        return result
    except Exception as exc:
        result.update(status='RESOURCE_LIMIT' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED',reason=f'{type(exc).__name__}: {exc}')
        write(output/'FAILURE.json',result)
        raise
