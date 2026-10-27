"""Assess independent bridge reconstructions in decoded-frame identity space."""
from pathlib import Path
import argparse
import html
import json
import os

from .region_improvement_assessment import inspect, read, save, sha


def _with_legacy_original_aliases(values):
    """Retain historical JSON field names without fixing the endpoint count."""
    result=dict(values)
    result.update({key.replace('original_endpoint','original64'):value
                   for key,value in values.items() if 'original_endpoint' in key})
    return result


def mapper_scope(frames, result):
    """Validate actual mapper request coverage; empty COLMAP list means all images."""
    names=[f['name'] for f in frames]
    requested=result.get('actual_options',{}).get('mapping',{}).get('image_names')
    explicit=isinstance(requested,list)
    effective=names if explicit and not requested else requested if explicit else []
    unique_plan=len(names)==len(set(names));unique_request=len(effective)==len(set(effective))
    missing=sorted(set(names)-set(effective));extra=sorted(set(effective)-set(names))
    ok=explicit and unique_plan and unique_request and not missing and not extra
    return dict(status='INPUT_SCOPE_MATCH' if ok else 'INPUT_SCOPE_MISMATCH',evaluatable_bridge_effect=ok,planned_count=len(names),requested_count=len(effective),empty_request_means_all=explicit and not requested,unique_plan_names=unique_plan,unique_requested_names=unique_request,excluded_names=missing,unexpected_names=extra,requested_names=effective,semantics='Mapper request scope, not registration success. Scope mismatch prevents evaluating added-frame efficacy.')


def remap_models(models, frames, original_decodes):
    """Never compare old endpoint indices with new decode-valued indices."""
    mapping={f['input_index']:f['decode_index'] for f in frames}
    if len(mapping)!=len(frames) or len(set(mapping.values()))!=len(frames):
        raise ValueError('unique native input/decode mapping required')
    result=[]
    for model in models:
        m=dict(model);indices=m['registered_indices']
        if not set(indices)<=set(mapping):raise ValueError('registered index missing from native plan')
        decodes=sorted(mapping[i] for i in indices)
        m.update(_with_legacy_original_aliases(dict(registered_decode_indices=decodes,original_endpoint_registered_decode_indices=sorted(set(decodes)&original_decodes),added_registered_decode_indices=sorted(set(decodes)-original_decodes),original_endpoint_registered_count=len(set(decodes)&original_decodes),added_registered_count=len(set(decodes)-original_decodes),total_registered_count=len(decodes))))
        result.append(m)
    return result


def summarize(models, original):
    valid=[m for m in models if m['camera_domain_status']=='CAMERA_DOMAIN_VALID']
    decodes=set(d for m in models for d in m['registered_decode_indices'])
    vd=set(d for m in valid for d in m['registered_decode_indices'])
    largest=max(valid,key=lambda m:(m['total_registered_count'],m['points']),default=None)
    return _with_legacy_original_aliases(dict(components=len(models),registered_decode_indices=sorted(decodes),valid_registered_decode_indices=sorted(vd),original_endpoint_registered_count=len(decodes&original),added_registered_count=len(decodes-original),valid_original_endpoint_registered_count=len(vd&original),valid_added_registered_count=len(vd-original),largest_valid_component_id=largest['model_id'] if largest else None,largest_valid_component_total_views=largest['total_registered_count'] if largest else 0,largest_valid_component_original_endpoint_views=len(set(largest['registered_decode_indices'])&original) if largest else 0,largest_valid_component_added_views=largest['added_registered_count'] if largest else 0,largest_valid_component_points=largest['points'] if largest else 0))


def assess(source_run,new_run):
    from PIL import Image,ImageDraw,ImageFont
    old=Path(source_run).resolve(strict=True);new=Path(new_run).resolve(strict=True);output=new/'assessment'
    if output.exists():raise FileExistsError('fresh assessment output required')
    oldplan=read(old/'geometry/PLAN.json');newplan=read(new/'geometry/PLAN.json');inputs=read(new/'input/PLAN.json')
    oldresult=read(old/'geometry/RESULT.json');newresult=read(new/'geometry/RESULT.json')
    scopes=dict(old=mapper_scope(oldplan['frames'],oldresult),new=mapper_scope(newplan['frames'],newresult))
    scope_ok=all(s['evaluatable_bridge_effect'] for s in scopes.values())
    original={f['decode_index'] for f in oldplan['frames']}
    if len(original)!=len(oldplan['frames']):raise ValueError('duplicate original decode identity')
    initial=[old/'geometry/PLAN.json',new/'geometry/PLAN.json',new/'input/PLAN.json',new/'tracking/REGION_TRACKS.json',Path(__file__)]
    refs={str(p):sha(p) for p in initial};output.mkdir()
    oldmodels,oldrefs=inspect(old,'geometry','old',set(),output)
    newmodels,newrefs=inspect(new,'geometry','bridge',set(),output);refs.update(oldrefs);refs.update(newrefs)
    a_models=remap_models(oldmodels,oldplan['frames'],original);b_models=remap_models(newmodels,newplan['frames'],original)
    a,b=summarize(a_models,original),summarize(b_models,original)
    oldreg=set(a['registered_decode_indices']);newreg=set(b['registered_decode_indices']);oldvalid=set(a['valid_registered_decode_indices']);newvalid=set(b['valid_registered_decode_indices'])
    differences=_with_legacy_original_aliases(dict(original_endpoint_registered_added=sorted((newreg-oldreg)&original),original_endpoint_registered_lost=sorted((oldreg-newreg)&original),original_endpoint_valid_registered_added=sorted((newvalid-oldvalid)&original),original_endpoint_valid_registered_lost=sorted((oldvalid-newvalid)&original),newly_sampled_registered_decode_indices=sorted(newreg-original),newly_sampled_valid_registered_decode_indices=sorted(newvalid-original)))
    endpoints=[r for r in inputs['timeline'] if r.get('input_index') is not None]
    added=[r for r in endpoints if r['decode_index'] not in original]
    declared=inputs.get('added_decode_indices')
    if declared is not None and set(declared)!={r['decode_index'] for r in added}:raise ValueError('declared endpoint expansion differs from native original-decode complement')
    planned={f['decode_index'] for f in newplan['frames']}
    if {r['decode_index'] for r in endpoints}!=planned:raise ValueError('input endpoints/native plan decode identity differs')
    fontfile=Path('C:/Windows/Fonts/msyh.ttc');font=ImageFont.truetype(str(fontfile),15) if fontfile.exists() else ImageFont.load_default()
    sheets=[];tiles=[];frameledger=[]
    for r in added:
        p=Path(r['image_path']);h=sha(p)
        if h!=r['image_sha256']:raise ValueError('added source frame drift')
        refs[str(p)]=h
        with Image.open(p) as im:thumb=im.convert('RGB');thumb.thumbnail((180,280))
        dcode=r['decode_index'];nf=next(f for f in newplan['frames'] if f['decode_index']==dcode)
        status='NOT_REQUESTED' if nf['name'] in scopes['new']['excluded_names'] else 'VALID_REGISTERED' if dcode in newvalid else 'REGISTERED_DOMAIN_REJECTED' if dcode in newreg else 'UNREGISTERED'
        color=(35,150,80) if status=='VALID_REGISTERED' else (190,130,20) if dcode in newreg else (170,45,45)
        tile=Image.new('RGB',(204,334),(20,25,35));tile.paste(thumb,((204-thumb.width)//2,36));draw=ImageDraw.Draw(tile);draw.rectangle((0,0,203,333),outline=color,width=4);draw.text((8,5),f'原帧 {dcode}',font=font,fill='white');draw.text((6,310),status,font=font,fill=color);tiles.append(tile)
        frameledger.append(dict(decode_index=dcode,status=status,image_path=str(p),image_sha256=h,endpoint_expansion=r.get('endpoint_expansion')))
    for start in range(0,len(tiles),24):
        batch=tiles[start:start+24];canvas=Image.new('RGB',(204*6,334*((len(batch)+5)//6)),(12,16,24))
        for j,t in enumerate(batch):canvas.paste(t,((j%6)*204,(j//6)*334))
        name=f'added_keyframes_{start//24:02d}.jpg';canvas.save(output/name,quality=91);sheets.append(name)
    expansion={k:inputs.get(k) for k in ('endpoint_expansion','endpoint_expansion_policy','endpoint_selection','added_decode_indices','original_endpoint_decode_indices') if k in inputs}
    doc=dict(schema='glove-region-bridge-assessment-v1',status='MEASURED_NOT_GEOMETRY_ACCEPTED' if scope_ok else 'INPUT_SCOPE_MISMATCH',mapper_scope=scopes,bridge_effect_evaluatable=scope_ok,source_run=str(old),new_run=str(new),comparison_identity='decode_index via explicit native PLAN frames; old input_index is never compared directly with new input_index',original_endpoint_count=len(original),new_endpoint_count=len(endpoints),added_endpoint_count=len(added),summary=dict(old=a,new=b),differences=differences,models=a_models+b_models,added_frames=frameledger,contact_sheets=sheets,endpoint_expansion=expansion,components_merged=False,physical_identity_verified=False,geometry_accepted=False,quality_status='NOT_EVALUATED',source_bindings_before=refs,source_bindings_after={p:sha(p) for p in refs})
    if doc['source_bindings_before']!=doc['source_bindings_after']:raise ValueError('bridge assessment source drift')
    doc['source_unchanged']=True;save(output/'BRIDGE_ASSESSMENT.json',doc)
    esc=lambda v:html.escape(str(v))
    original_count=doc['original_endpoint_count']
    body=['<!doctype html><meta charset="utf-8"><title>物体桥接视角实验</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;background:#111923;color:#eee}td,th{padding:9px;border:1px solid #789}table{border-collapse:collapse}a{color:#9cf}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>新增真实中间视角：独立稀疏重建对照</h1><p>按原视频decode_index比较：旧端点编号与新端点编号不直接比较。新增图像是实际解码帧，不是插值/预测图像。相机域检查通过不证明标定、形状或物理身份正确；各分量独立展示，未拼接或封洞。</p>',f'<table><tr><th>版本</th><th>原始端点注册（共{original_count}）</th><th>新增注册</th><th>相机域有效原始端点（共{original_count}）</th><th>有效新增</th><th>最大有效分量 总/原始端点/新增</th><th>该分量点数</th></tr>']
    if not scope_ok:
        notice='<p><strong>INPUT_SCOPE_MISMATCH：实际mapper请求集未完整覆盖计划输入，或请求名称不唯一。不可据此评价真实补帧有效性；下表只保留本次执行的数量记录。</strong></p><pre>'+esc(json.dumps(scopes,ensure_ascii=False,indent=2))+'</pre>'
        body[1]=notice+body[1]
    for label,s in (('old',a),('bridge',b)):
        body.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (label,f'{s["original_endpoint_registered_count"]}/{original_count}',s['added_registered_count'],f'{s["valid_original_endpoint_registered_count"]}/{original_count}',s['valid_added_registered_count'],f'{s["largest_valid_component_total_views"]}/{s["largest_valid_component_original_endpoint_views"]}/{s["largest_valid_component_added_views"]}',s['largest_valid_component_points']))+'</tr>')
    display_differences={key:value for key,value in differences.items() if 'original64' not in key}
    body+=['</table><h2>共同原始视角的增加与丢失</h2><pre>'+esc(json.dumps(display_differences,ensure_ascii=False,indent=2))+'</pre><h2>各版本独立分量</h2><ul>']
    for m in a_models+b_models:body.append('<li><a href="'+esc(m['viewer'])+'">'+esc(f'{m["label"]} component {m["model_id"]}: {m["total_registered_count"]} views (original {m["original_endpoint_registered_count"]}/{original_count}, added {m["added_registered_count"]}), {m["points"]} points, {m["camera_domain_status"]}')+'</a></li>')
    body+=['</ul><h2>新增实际关键帧</h2><p>绿框：相机域有效分量中注册；黄框：仅在相机域被拒分量中注册；红框：未注册。NOT_REQUESTED表示未进入mapper请求，不能作为补帧失败证据。完整逐帧来源与endpoint_expansion见JSON。</p>']
    body.extend('<a href="'+s+'"><img src="'+s+'"></a>' for s in sheets)
    body+=['<p>数量变化是单次实验观测，不代表因果改进或表面完整度验收。</p><a href="BRIDGE_ASSESSMENT.json">完整机器账本</a>']
    (output/'index.html').write_text('\n'.join(body),encoding='utf-8');return doc


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source-run',required=True);p.add_argument('--new-run',required=True);a=p.parse_args();r=assess(a.source_run,a.new_run);print(json.dumps(dict(status=r['status'],summary=r['summary'])))
