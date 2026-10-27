"""Automatic query-camera hypotheses and separate conditional track triangulation."""
from collections import Counter
from copy import deepcopy
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw
from .intermediate_population_review import write
from .temporal_review import fingerprint

PROFILE = dict(id='query-camera-extension-v1', max_records=32, max_points=96,
    lenses=['SIMPLE_PINHOLE', 'SIMPLE_RADIAL'], principal_point='image_center_assumption',
    aspect_ratio=1.0, physical_calibration=False, native_injection=False)


def _render(output, result, extension, decoded, clouds, *, budget):
    escaped=lambda v:html.escape(str(v), quote=True)
    artifacts=[];sections=[];table=[]
    for ordinal,row in enumerate(result['records']):
        budget();frame=decoded[row['query_decode_index']]
        models=row['assessment'].get('models',[])
        for model in models:
            hold=model.get('holdout') or {};train=model.get('train') or {}
            table.append('<tr>'+''.join('<td>'+escaped(v)+'</td>' for v in
                (row['model_key'],row['query_decode_index'],row['cohort'],model['model'],len(row['points']),
                 model['status'],model.get('f'),model.get('k'),train.get('max_px'),hold.get('median_px'),
                 hold.get('max_px'),'; '.join(model.get('reasons',[]))))+'</tr>')
        # Show actual frames even when all hypotheses are blocked; no invented projections.
        if ordinal >= 12:continue
        sheet=Image.new('RGB',(720,700),'#132030');draw=ImageDraw.Draw(sheet)
        for column,model in enumerate(models[:2]):
            budget()
            with Image.open(frame['path']) as src:
                if src.size != (row['width'],row['height']):raise ValueError('QUERY_IMAGE_DOMAIN_CHANGED')
                pic=src.convert('RGB')
            pic.thumbnail((340,590));x=column*360+(360-pic.width)//2;y=85
            sheet.paste(pic,(x,y))
            draw.text((column*360+8,8),f'decode={row["query_decode_index"]} {row["cohort"]}',fill='white')
            draw.text((column*360+8,26),model['model'],fill='white')
            draw.text((column*360+8,44),model['status'].encode('ascii','replace').decode(),fill='#ffb96e')
            predictions={p['point_id']:p.get('predicted_raw_xy') for p in (model.get('holdout') or {}).get('rows',[])}
            def pos(raw):return x+(raw[0]-.5)*pic.width/row['width'],y+(raw[1]-.5)*pic.height/row['height']
            def inside(raw):return raw is not None and .5<=raw[0]<=row['width']-.5 and .5<=raw[1]<=row['height']-.5
            outside=0
            for point in row['points']:
                ox,oy=pos(point['raw_xy']);draw.ellipse((ox-3,oy-3,ox+3,oy+3),outline='#ffac54',width=2)
                predicted=predictions.get(point['point_id'])
                if predicted is not None and not inside(predicted):outside+=1
                if inside(predicted):
                    px,py=pos(predicted);draw.line((ox,oy,px,py),fill='#ffac54',width=2)
                    draw.line((px-4,py,px+4,py),fill='#58def9',width=2)
                    draw.line((px,py-4,px,py+4),fill='#58def9',width=2)
            draw.text((column*360+8,682),f'out-of-frame predictions: {outside}',fill='white')
        if models:
            name=f'camera_{ordinal:03d}.png'
            with (output/name).open('xb') as stream:sheet.save(stream,format='PNG')
            artifacts.append(output/name)
            sections.append(f'<h2>模型 {escaped(row["model_key"])} / decode {row["query_decode_index"]} / {escaped(row["cohort"])}</h2><img src="{name}">')
    links=''.join(f'<li>{escaped(c["model_key"])} / {escaped(c["cohort"])} / {escaped(c["lens_model"])}：'
        f'{c["vertices"]}条件点，<a href="{escaped(c["viewer"])}">查看</a></li>' for c in clouds)
    page='''<!doctype html><meta charset="utf-8"><title>中间帧相机与条件三角化</title>
<style>body{font:16px/1.7 system-ui;margin:26px;background:#142030;color:#edf4fa}a{color:#8ad3ff}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #52657a;padding:7px;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>
<h1>从三维锚点验证查询相机，再尝试新增三角化</h1>
<p>中心主点、正方像素是当前假设，绝非已知标定。每帧独立估计焦距；针孔与单径向两种模型都在预先冻结的训练点上拟合。留出点不参与初始化/拟合；错配对照分别在训练、留出集合内部置换，不把留出坐标泄入训练。</p>
<p>检查镜头整个图像域的中心分支、正深度、参数范围、训练雅可比秩/条件与留出误差。通过也只支持给定锚点和假设下的条件相机，不建立物理精度或原生注册。不同模型/镜头假设不混合。</p>
<p>新增三角化排除全部相机锚点（含训练与留出），使用其它实测短轨迹及源已保存相机；至少三个视图。查询帧未知掩膜保留UNKNOWN，不称已确认物体前景；新增点继续经过原重投影、留一、正深度和射线夹角检查。</p>
<p>橙圆：实际点；青十字：留出预测；表格保留全部记录，图像最多展示12条。低残差与连续性均不能单独证明正确物理结构。</p>'''
    page+='<pre>'+escaped(json.dumps(dict(cameras=result['counts'],extensions=extension.get('counts',{})),ensure_ascii=False,indent=2))+'</pre>'
    page+='<p><a href="CAMERA_HYPOTHESES.json">相机参数与逐点证据</a> · <a href="EXTENSION.json">新增三角化完整账本</a></p>'
    page+='<h2>镜头假设对照</h2><table><tr><th>模型</th><th>decode</th><th>点集</th><th>镜头</th><th>点数</th><th>结果</th><th>f(px)</th><th>k</th><th>train max</th><th>hold median</th><th>hold max</th><th>原因</th></tr>'+''.join(table)+'</table>'
    page+='<h2>新增点层</h2>'+('<ul>'+links+'</ul>' if clouds else '<p>没有通过条件门槛的新增点；没有生成占位点云或表面。</p>')
    page+=''.join(sections)
    with (output/'index.html').open('x',encoding='utf-8',newline='\n') as f:f.write(page)
    return artifacts+[output/'index.html']


def review_query_cameras(ledger, projection, frames, masks, camera_result, decoded, output, cv, *, budget=lambda:None):
    from .query_camera_math import assess_query_camera
    from .query_camera_extension import assess_query_extensions
    from .anchor_route_review import _clouds
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    refs=[];artifacts=[]
    result=dict(status='RUNNING',profile=deepcopy(PROFILE),records=[],counts={},
                geometry_accepted=False,camera_pose_established=False,new_cameras=0,new_model_points=0)
    try:
        budget()
        records=projection.get('records')
        if not isinstance(records,list) or len(records)>PROFILE['max_records']:raise ValueError('QUERY_RECORD_BOUND')
        seen=set()
        # Domain and membership validation is completed before the first fit.
        for row in records:
            budget();identity=(row['model_key'],row['query_decode_index'],row['cohort'])
            if identity in seen:raise ValueError('DUPLICATE_QUERY_RECORD')
            seen.add(identity)
            if row['cohort'] not in ('STRICT','STRICT_PLUS_RELAXED'):raise ValueError('QUERY_COHORT')
            dec=row['query_decode_index']
            if type(dec) is not int or dec<0 or dec not in decoded:raise ValueError('QUERY_DECODE_IDENTITY')
            frame=decoded[dec]
            if not isinstance(row['points'],list) or len(row['points'])>PROFILE['max_points']:raise ValueError('QUERY_POINTS_BOUND')
            if type(row['model_key']) is not str or not row['model_key']:raise ValueError('QUERY_MODEL_IDENTITY')
            for key in ('width','height'):
                if type(frame[key]) is not int or not 0<frame[key]<=4096:raise ValueError('QUERY_DIMENSION')
            refs.append(fingerprint(frame['path']))
            result['records'].append(dict(model_key=row['model_key'],query_decode_index=dec,cohort=row['cohort'],
                width=frame['width'],height=frame['height'],points=deepcopy(row['points']),
                original_mask_status=row.get('original_mask_status'),assessment=None))
        for row in result['records']:
            budget()
            row['assessment']=assess_query_camera(row['points'],row['width'],row['height'],cv,budget=budget)
        result['counts']=dict(Counter(m['status'] for r in result['records'] for m in r['assessment'].get('models',[])))
        result['status']='QUERY_CAMERA_REVIEW_COMPLETE'
        write(output/'CAMERA_HYPOTHESES.json',result);artifacts.append(output/'CAMERA_HYPOTHESES.json')
        extension=assess_query_extensions(ledger,frames,masks,camera_result,result,budget=budget)
        write(output/'EXTENSION.json',extension);artifacts.append(output/'EXTENSION.json')
        clouds=[]
        for number,group in enumerate(extension.get('classifications',[])):
            budget();folder=output/f'clouds_{number:03d}';folder.mkdir(exist_ok=False)
            write(folder/'GEOMETRY.json',group)
            paths,items=_clouds(folder,group['classification'],budget)
            artifacts += [folder/'GEOMETRY.json',*paths]
            if items:
                clouds.append(dict(model_key=group['model_key'],cohort=group['cohort'],lens_model=group['lens_model'],
                    vertices=sum(c['vertices'] for c in items),viewer=f'{folder.name}/viewer.html'))
        artifacts+=_render(output,result,extension,decoded,clouds,budget=budget)
        for ref in refs:
            budget()
            if fingerprint(ref['path'])!=ref:raise ValueError('QUERY_IMAGE_DRIFT')
        summary=dict(status='QUERY_CAMERA_EXTENSION_COMPLETE',camera_counts=result['counts'],
            extension_counts=extension.get('counts',{}),conditional_vertices=sum(c['vertices'] for c in clouds),
            geometry_accepted=False,camera_pose_established=False,new_cameras=0,new_model_points=0,
            report='intermediate_frames/population/anchor_routes/query_cameras/index.html',source_unchanged=True)
        write(output/'RESULT.json',summary);artifacts.append(output/'RESULT.json')
        summary['source_bindings']=[fingerprint(p) for p in dict.fromkeys(artifacts)]+refs
        return summary
    except Exception as exc:
        write(output/'FAILURE.json',dict(status='RESOURCE_LIMIT' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED',
             reason=f'{type(exc).__name__}: {exc}',records=result['records'],geometry_accepted=False))
        raise
