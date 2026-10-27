"""Read native components, compare domain-valid coverage, export diagnostic viewers."""
from collections import defaultdict
from pathlib import Path
import hashlib
import html
import json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reverse_ids(tracks):
    """Use declared provenance, never numeric ID ranges guessed by the report."""
    if 'reverse_track_ids' in tracks:
        ids=tracks['reverse_track_ids']
    else:
        ids=[t['track_id'] for t in tracks.get('tracks',[]) if t.get('origin')=='reverse' or t.get('direction') in ('reverse','backward')]
        if not ids:raise ValueError('explicit reverse track provenance required')
    if any(type(t) is not int or t<0 for t in ids) or len(ids)!=len(set(ids)):
        raise ValueError('invalid reverse IDs')
    known={o['track_id'] for f in tracks['frames'] for o in f['observations']}
    if not set(ids)<=known:raise ValueError('reverse IDs absent from observations')
    return set(ids)


def source_geometry(new_run):
    plan=Path(new_run)/'PLAN.json'
    doc=read(plan) if plan.is_file() else {}
    name=doc.get('source_geometry','geometry02')
    if 'source_run' in doc:
        from .region_experiment import resolve_source_geometry
        return resolve_source_geometry(doc['source_run'],name)
    # Historical assessment-only records may have no source-run binding. Keep
    # their single-directory fallback; modern plans use the shared resolver.
    if not isinstance(name,str) or not name or Path(name).name!=name or name in ('.','..'):
        raise ValueError('invalid source_geometry directory')
    return name


def summarize(components):
    allviews=set(i for m in components for i in m['registered_indices'])
    valid=[m for m in components if m['camera_domain_status']=='CAMERA_DOMAIN_VALID']
    validviews=set(i for m in valid for i in m['registered_indices'])
    largest=max(valid,key=lambda m:(len(m['registered_indices']),m['points']),default=None)
    return dict(components=len(components),registered_indices=sorted(allviews),unique_registered_views=len(allviews),
                valid_registered_indices=sorted(validviews),valid_unique_registered_views=len(validviews),
                largest_valid_component_id=largest['model_id'] if largest else None,
                largest_valid_component_views=len(largest['registered_indices']) if largest else 0,
                largest_valid_component_points=largest['points'] if largest else 0,
                largest_valid_component_split_ids=len(largest.get('split_region_ids',[])) if largest else 0,
                largest_valid_component_triangulated_region_ids=largest.get('triangulated_region_ids',0) if largest else 0,
                largest_valid_component_split_ratio=(len(largest.get('split_region_ids',[]))/largest['triangulated_region_ids']) if largest and largest.get('triangulated_region_ids',0) else None,
                rejected_component_ids=[m['model_id'] for m in components if m['camera_domain_status']!='CAMERA_DOMAIN_VALID'])


def differences(old,new):
    a,b=set(old['registered_indices']),set(new['registered_indices'])
    av,bv=set(old['valid_registered_indices']),set(new['valid_registered_indices'])
    return dict(registered_added=sorted(b-a),registered_lost=sorted(a-b),valid_registered_added=sorted(bv-av),valid_registered_lost=sorted(av-bv),
                largest_valid_component_views_delta=new['largest_valid_component_views']-old['largest_valid_component_views'],
                valid_unique_views_delta=new['valid_unique_registered_views']-old['valid_unique_registered_views'])


def inspect(run,geometry_name,label,reverse,output):
    import numpy as np
    import pycolmap as pc
    from integrations.demo_pipeline.camera_domain_gate import audit_camera_domain
    from integrations.viewer.export import export_html
    geo=run/geometry_name;result=read(geo/'RESULT.json');plan=read(geo/'PLAN.json')
    frames={f['image_id']:f for f in plan['frames']}
    rowtrack={iid:{int(row):int(tid) for tid,row in f['row_by_track'].items()} for iid,f in frames.items()}
    models=[];bindings={str(p):sha(p) for p in (geo/'RESULT.json',geo/'PLAN.json')}
    for m in result['models']:
        binary=Path(m['binary_path']);rec=pc.Reconstruction(binary)
        for path in binary.iterdir():
            if path.is_file():bindings[str(path)]=sha(path)
        lineage=defaultdict(set);reverse_points=set();reverse_observations=0;reverse_views=set();errors=[];positive=0;total=0;invalid_reprojections=0
        for pid,point in rec.points3D.items():
            for e in point.track.elements:
                im=rec.images[e.image_id];q=im.points2D[e.point2D_idx];pose=im.cam_from_world()
                local=pose.rotation.matrix()@point.xyz+pose.translation;positive+=int(local[2]>0);total+=1
                px=rec.cameras[im.camera_id].img_from_cam(local)
                error=float(np.linalg.norm(px-q.xy)) if px is not None else float('nan')
                if np.isfinite(error):errors.append(error)
                else:invalid_reprojections+=1
                tid=rowtrack.get(e.image_id,{}).get(e.point2D_idx)
                if tid is not None:lineage[tid].add(int(pid))
                if tid in reverse:
                    reverse_points.add(int(pid));reverse_observations+=1;reverse_views.add(frames[e.image_id]['input_index'])
        domain=audit_camera_domain(binary)
        folder=output/'viewers'/f'{label}_{m["model_id"]}';folder.mkdir(parents=True)
        ply=Path(m['ply_path']);bindings[str(ply)]=sha(ply)
        spec=folder/'scenes.json'
        save(spec,dict(schema='ply-viewer-scenes-v1',scenes=[dict(ply=str(ply),sha256=sha(ply),title=f'{label} component {m["model_id"]}',data_kind='real_photos',declared_unit=None,note='Independent component; shape and physical geometry not accepted.',evidence=[])]))
        receipt=export_html(spec,folder/'viewer.html');save(folder/'EXPORT.json',receipt)
        models.append(dict(label=label,model_id=m['model_id'],registered_indices=sorted(frames[i]['input_index'] for i in rec.reg_image_ids()),points=int(rec.num_points3D()),
                           camera_domain_status=domain['status'],camera_domain=domain,cameras=[dict(camera_id=int(cid),model=c.model_name,params=c.params.tolist()) for cid,c in rec.cameras.items()],
                           reprojection_error_px=dict(observations=len(errors),median=float(np.median(errors)) if errors else None,p95=float(np.quantile(errors,.95)) if errors else None,max=max(errors) if errors else None),
                           invalid_reprojections=invalid_reprojections,positive_depth_observations=positive,total_observations=total,triangulated_region_ids=len(lineage),split_region_ids=sorted(t for t,pids in lineage.items() if len(pids)>1),
                           reverse_triangulated_ids=sorted(t for t in lineage if t in reverse),points_with_reverse_observations=len(reverse_points),reverse_observations=reverse_observations,reverse_registered_indices=sorted(reverse_views),
                           binary_path=str(binary),ply_path=str(ply),viewer=str((folder/'viewer.html').relative_to(output)).replace('\\','/'),geometry_accepted=False))
    return models,bindings


def correspondence_contribution(run, reverse):
    """Count stored pair rows by declared source; not unique 3D points."""
    import sqlite3
    import numpy as np
    geo=run/'geometry';plan=read(geo/'PLAN.json')
    reverse_rows={f['image_id']:{int(row) for tid,row in f['row_by_track'].items() if int(tid) in reverse} for f in plan['frames']}
    ledger=[]
    with sqlite3.connect('file:'+str(geo/'database.db')+'?mode=ro',uri=True) as db:
        for table in ('matches','two_view_geometries'):
            for pid,rows,data in db.execute('SELECT pair_id,rows,data FROM '+table):
                if not rows:continue
                a,b=pid//2147483647,pid%2147483647
                matrix=np.frombuffer(data,np.uint32).reshape(rows,2)
                count=sum(int(x) in reverse_rows.get(a,set()) and int(y) in reverse_rows.get(b,set()) for x,y in matrix)
                if count:ledger.append(dict(table=table,image_a=a,image_b=b,reverse_pair_rows=count))
    return dict(raw_pair_rows=sum(p['reverse_pair_rows'] for p in ledger if p['table']=='matches'),verified_pair_rows=sum(p['reverse_pair_rows'] for p in ledger if p['table']=='two_view_geometries'),pairs=ledger,counts_are_pair_rows_not_unique_points=True)


def render(doc,path):
    rows=[]
    for label in ('old','new'):
        s=doc['summary'][label]
        rows.append('<tr><td>'+label+'</td>'+''.join('<td>'+str(s[k])+'</td>' for k in ('unique_registered_views','valid_unique_registered_views','largest_valid_component_views','largest_valid_component_points','largest_valid_component_split_ids','largest_valid_component_triangulated_region_ids','largest_valid_component_split_ratio'))+'</tr>')
    links=[]
    for m in doc['models']:
        links.append('<li><a href="'+html.escape(m['viewer'],quote=True)+'">'+html.escape(f'{m["label"]} component {m["model_id"]}: {len(m["registered_indices"])} views, {m["points"]} points, {m["camera_domain_status"]}')+'</a></li>')
    body='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Region improvement assessment</title><style>body{font:16px system-ui;max-width:1050px;margin:36px auto;padding:0 18px}td,th{padding:10px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>目标稀疏重建对照</h1><p>形状、物理身份、尺度及表面完整性均未验收。相机域有效仅表示所检查数值域通过；各分量独立，点数不可相加作为完整模型。</p><table><tr><th>版本</th><th>唯一注册视图</th><th>相机域有效唯一视图</th><th>最大有效分量视图</th><th>该分量点数</th><th>分裂region ID</th><th>三角化region ID</th><th>分裂比率</th></tr>'+''.join(rows)+'</table><p>有效视图新增：'+html.escape(str(doc['differences']['valid_registered_added']))+'；丢失：'+html.escape(str(doc['differences']['valid_registered_lost']))+'。</p><ul>'+''.join(links)+'</ul><p>reverse贡献表示使用该来源观测的点/视图，并非因果增益。split ID表示轨迹被重建为多个点，不能单独证明误匹配。单次RANSAC结果不是统计有效性证明。</p><a href="ASSESSMENT.json">完整逐分量指标</a></html>'
    Path(path).write_text(body,encoding='utf-8')


def assess(old_run,new_run):
    old=Path(old_run).resolve(strict=True);new=Path(new_run).resolve(strict=True);output=new/'assessment'
    if output.exists():raise FileExistsError('fresh assessment output required')
    tracksfile=new/'tracking/REGION_TRACKS.json';tracks=read(tracksfile);reverse=reverse_ids(tracks)
    oldcomparison=read(old/'COMPARISON.json')
    output.mkdir()
    old_geometry=source_geometry(new)
    oldmodels,refs=inspect(old,old_geometry,'old',set(),output)
    if (new/'PLAN.json').is_file():refs[str(new/'PLAN.json')]=sha(new/'PLAN.json')
    newmodels,newrefs=inspect(new,'geometry','new',reverse,output);refs.update(newrefs)
    refs.update({str(p):sha(p) for p in (tracksfile,old/'COMPARISON.json',new/'geometry/database.db',Path(__file__))})
    contribution=correspondence_contribution(new,reverse)
    a,b=summarize(oldmodels),summarize(newmodels)
    doc=dict(schema='glove-region-improvement-assessment-v1',old_geometry=old_geometry,summary=dict(old=a,new=b),differences=differences(a,b),models=oldmodels+newmodels,
             reverse_declared_ids=len(reverse),reverse_correspondence_contribution=contribution,old_comparison_globalmetrics=oldcomparison['globalmetrics'],components_merged=False,geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown',
             source_bindings_before=refs,source_bindings_after={p:sha(p) for p in refs})
    if doc['source_bindings_before']!=doc['source_bindings_after']:raise ValueError('assessment source drift')
    doc['source_unchanged']=True
    save(output/'ASSESSMENT.json',doc);render(doc,output/'index.html')
    return doc

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--old-run',required=True);p.add_argument('--new-run',required=True);args=p.parse_args();assess(args.old_run,args.new_run)
