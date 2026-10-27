"""Frozen four-arm reference-domain depth diagnostic; no mesh or adoption."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from .runner import ROOT,binding,make_plan,plan_files,stage_inputs,commands,save_json,_effective_dense_config
from .supervision import run_supervised
from .mask_consumption_probe import need,load,unchanged,resize_nearest,erode_two,authenticate_depth
from integrations.openmvs_depth.dmap import read_dmap

ARMS=('WHOLE','G0','G1','G2')


def binary(path,shape):
    with Image.open(path) as im:
        need(im.mode=='L' and im.size==(shape[1],shape[0]),'same-domain L mask required')
        a=np.array(im)
    need(np.isin(a,[0,255]).all(),'binary mask required')
    return a!=0


def compare_depth(whole,regional,domain,seeded=True):
    shape=regional.shape if regional is not None else whole.shape if whole is not None else domain.shape
    take=resize_nearest(domain,shape);n=int(take.sum())
    row=dict(status='MEASURED_NOT_ACCEPTED' if seeded else 'UNKNOWN_NO_SEED',domain_pixels=n,
        regional_available=regional is not None,whole_available=whole is not None)
    if whole is None or regional is None:
        if seeded:row['status']='UNKNOWN_MISSING_DEPTH'
        row.update(common_valid=None,agree_2pct=None,only_whole=None,only_region=None,both_unknown=None,
                   regional_valid=None if regional is None else int((take&np.isfinite(regional)&(regional>0)).sum()))
        return row
    need(whole.shape==regional.shape,'incomparable depth resolutions')
    w=np.isfinite(whole)&(whole>0);r=np.isfinite(regional)&(regional>0);common=take&w&r
    agree=common&(abs(whole-regional)<=.02*whole)
    row.update(common_valid=int(common.sum()),agree_2pct=int(agree.sum()),disagree_2pct=int((common&~agree).sum()),
        only_whole=int((take&w&~r).sum()),only_region=int((take&r&~w).sum()),both_unknown=int((take&~w&~r).sum()),
        regional_valid=int((take&r).sum()),whole_valid=int((take&w).sum()),
        agree_fraction=None if not common.any() else float(agree.sum()/common.sum()))
    return row


def leakage(depth,allowed):
    if depth is None:return dict(status='UNKNOWN_NO_DEPTH',forbidden_interior_pixels=None,positive_leak_pixels=None)
    forbidden=erode_two(~resize_nearest(allowed,depth.shape));positive=np.isfinite(depth)&(depth>0)
    count=int((forbidden&positive).sum())
    return dict(status='REFERENCE_MASK_LEAK_DETECTED' if count else 'ZERO_MEASURED_INTERIOR_LEAK_NOT_QUALITY',
                forbidden_interior_pixels=int(forbidden.sum()),positive_leak_pixels=count)


def union_counts(depths,domains,foreground):
    present=[d for d in depths if d is not None]
    if not present:return dict(status='UNKNOWN_NO_DEPTH',union_valid_pixels=None,overlap_conflict_pixels=None)
    shape=present[0].shape;need(all(d.shape==shape for d in present),'group depth domains differ')
    fg=resize_nearest(foreground,shape)
    good=[np.zeros(shape,bool) if d is None else np.isfinite(d)&(d>0)&resize_nearest(m,shape)&fg for d,m in zip(depths,domains)]
    multiplicity=np.sum(good,axis=0);conflict=np.zeros(shape,bool)
    for i in range(3):
        for j in range(i+1,3):
            if depths[i] is not None and depths[j] is not None:
                conflict|=good[i]&good[j]&(abs(depths[i]-depths[j])>.02*np.minimum(depths[i],depths[j]))
    return dict(status='DESCRIPTIVE_ONLY_NOT_MERGED',union_valid_pixels=int((multiplicity>0).sum()),
        multiple_group_valid_pixels=int((multiplicity>1).sum()),overlap_conflict_pixels=int(conflict.sum()),
        missing_groups=[i for i,d in enumerate(depths) if d is None],
        conflict_definition='abs(z_i-z_j)>0.02*min(z_i,z_j); any pair; no averaging or winner selection')


def codes():
    return [binding(ROOT/p) for p in ('integrations/mvs_surface/region_dense_experiment.py',
        'scripts/compare_regional_mvs_depth.py','integrations/mvs_surface/mask_consumption_probe.py',
        'integrations/openmvs_depth/dmap.py','integrations/video_masks/face_mask_adapter.py')]


def prepare(source_plan,regions_result,probe_result,output):
    source_plan=Path(source_plan).resolve(strict=True);regions_result=Path(regions_result).resolve(strict=True)
    out=Path(output).resolve();rr=regions_result.parent
    need(not out.exists() and not any(out.is_relative_to(p) or p.is_relative_to(out) for p in (source_plan.parent,rr)),'fresh disjoint output required')
    source=load(source_plan);need(source==make_plan(**source['arguments']),'source plan drift')
    need(len(source['frames'])==12 and source['arguments']['ignore_label']==0,'12-view label-zero source required')
    regions=load(regions_result)
    need(regions['status']=='OVERLAPPING_DOMAINS_PREPARED_NOT_NATIVE_EXECUTED' and regions['native_executions']==0
         and regions['original_rgb_modified'] is False and regions['original_masks_changed'] is False,'regional preparation contract differs')
    rows={r['export_image_id']:r for r in regions['rows']}
    frames=source['frames'];need(len(rows)==len(regions['rows'])==12 and set(rows)=={f['image']['global_id'] for f in frames},'one-to-one region camera coverage required')
    refs=[binding(source_plan),*plan_files(source),*codes()]
    for name in ('RESULT.json','GROUPS.json','SEEDS.json','INPUT_BINDINGS.json','INPUT_BINDINGS_POST.json'):refs.append(binding(rr/name))
    before=load(rr/'INPUT_BINDINGS.json');after=load(rr/'INPUT_BINDINGS_POST.json')
    need(before==after,'regional upstream binding snapshots differ');unchanged(before);refs+=before
    joins=[r for r in before if Path(r['path']).name=='SELECTED_VIEWS.json']
    need(len(joins)==1,'one frozen source selection join required')
    joinrows=load(joins[0]['path'])['rows'];join={r['export_image_id']:r for r in joinrows}
    need(len(join)==len(joinrows)==12 and set(join)==set(rows),'selection camera IDs differ')
    selected=[]
    for f in frames:
        cam=f['image'];name=Path(cam['name']).name;shape=(cam['height'],cam['width']);row=rows[cam['global_id']]
        need(join[cam['global_id']]['decode_index']==row['decode_index'] and join[cam['global_id']]['exported_name']==name,
             'regional decode ordinal/camera/name join differs')
        mask=binary(f['mask']['path'],shape);valid=binary(f['validity']['path'],shape)
        control=rr/'control/masks'/f['mask_name'];crec=binding(control)
        need(crec['sha256']==f['mask']['sha256'] and crec['bytes']==f['mask']['bytes'],'control differs from original mask bytes');refs.append(crec)
        need(row['foreground_pixels']==int((mask&valid).sum()),'foreground count differs')
        groups={g['group']:g for g in row['groups']};need(set(groups)=={0,1,2} and len(row['groups'])==3,'three unique groups required')
        records=[]
        for g in range(3):
            r=groups[g];path=Path(r['mask']).resolve(strict=True)
            need(path==rr/f'group_{g}/masks'/f['mask_name'],'regional mask name/path differs')
            arr=binary(path,shape);need(not np.any(arr&~(mask&valid)) and int(arr.sum())==r['covered_pixels'] and r['forbidden_added_pixels']==0,'regional mask domain/count differs')
            seeded=r['usable_seed_cells']>0
            need(not seeded or (r['actual_seed_observations']>0 and r['status']=='CANDIDATE_NOT_PART_IDENTITY'),'seeded group provenance/status differs')
            need(seeded or (not arr.any() and r['status']=='UNKNOWN_NO_USABLE_SEED'),'no-seed mask must be empty/unknown')
            rec=binding(path);refs.append(rec);records.append(dict(group=g,mask=rec,seeded=seeded,seed_record=r))
        selected.append(dict(global_id=cam['global_id'],decode_index=row['decode_index'],name=name,groups=records))
    out.mkdir(parents=True);arms=[]
    for name in ARMS:
        work=out/name/'work';work.mkdir(parents=True);stage_inputs(source,work,time.monotonic()+120)
        if name!='WHOLE':
            g=int(name[1])
            for f,s in zip(frames,selected):
                need(f['image']['global_id']==s['global_id'],'staging camera order differs')
                (work/'__mvs_masks'/f['mask_name']).write_bytes(Path(s['groups'][g]['mask']['path']).read_bytes())
        arms.append(dict(name=name,work=str(work),argv=commands(source,work)[0][1],
            staged=[binding(p,512<<20) for p in sorted(work.rglob('*')) if p.is_file()]))
    unchanged(refs)
    prepared=dict(schema='regional-dense-only-diagnostic-v1',source_plan=str(source_plan),refs=refs,
        probe_result=str(Path(probe_result).resolve()),views=selected,arms=arms,status='PREPARED_NOT_EXECUTED',
        budget=dict(total_seconds=900,arm_seconds=180,memory_bytes=8<<30,output_bytes=1<<30,threads=2),
        scope='Independent regional reference-mask diagnostic; empty unknown masks are not production admission.')
    save_json(out/'PREPARED.json',prepared,exclusive=True);return binding(out/'PREPARED.json')


def read_arm_depths(work,cameras,check):
    depths={};refs=[];paths=sorted(work.glob('depth*.dmap'));need(len(paths)<=256,'depth-file bound')
    for path in paths:
        check();d=read_dmap(path);name=Path(d.image_name_utf8.replace('\\','/')).name
        need(name in cameras and name not in depths,'unknown/duplicate embedded dmap identity')
        authenticate_depth(d,cameras[name]);depths[name]=d.depth.copy();refs.append(binding(path,512<<20))
    return depths,refs


def render_target(path,frame,depths):
    with Image.open(frame['image_file']['path']) as im:rgb=np.array(im.convert('RGB'))
    h,w=rgb.shape[:2];canvas=Image.new('RGB',(w*4,h+35),(20,25,35));draw=ImageDraw.Draw(canvas)
    for j,name in enumerate(ARMS):
        d=depths.get(name);arr=rgb.copy()
        if d is not None:
            hit=resize_nearest(np.isfinite(d)&(d>0),(h,w));arr[hit]=(arr[hit]*.5+[10,100,70]).astype(np.uint8)
        canvas.paste(Image.fromarray(arr),(j*w,35));draw.text((j*w+5,6),name+(' / NO DMAP' if d is None else ' / POSITIVE DEPTH'),fill='white')
    canvas.save(path)


def execute(prepared,prepared_sha256):
    path=Path(prepared).resolve(strict=True);out=path.parent;need(binding(path)['sha256']==prepared_sha256,'prepared hash differs');doc=load(path)
    need(doc['schema']=='regional-dense-only-diagnostic-v1','wrong diagnostic schema')
    # Gate before creating an attempt marker: preparation may precede probe completion.
    probe=Path(doc['probe_result']).resolve(strict=True);p=load(probe)
    need(p['status']=='REFERENCE_MASK_CONSUMPTION_VERIFIED_NOT_GEOMETRY_ACCEPTANCE' and p.get('source_unchanged') is True,'positive unchanged reference-mask assay required')
    proof=load(probe.parent/'PREPARED.json');need(Path(proof['source_plan']).resolve()==Path(doc['source_plan']),'probe/source plan differs')
    need(binding(doc['source_plan']) in proof['refs'],'probe source-plan bytes differ')
    need(all(p['arms'][a].get('reclaimed') is True and p['arms'][a].get('execution_ok') is True for a in ('DISABLED','RECT','BLACK')),'probe supervision incomplete')
    save_json(out/'EXECUTION_ATTEMPT.json',dict(prepared=binding(path),probe=binding(probe)),exclusive=True)
    start=time.monotonic();deadline=start+900;result=dict(status='EXECUTION_OR_IDENTITY_UNKNOWN',geometry_accepted=False,default_replaced=False,arms={},views=[],new_native_launch_attempts=0)
    refs=[*doc['refs'],binding(probe),binding(probe.parent/'PREPARED.json')];all_depths={}
    def check():
        if time.monotonic()>deadline:raise TimeoutError('900s diagnostic deadline')
    try:
        unchanged(refs);need(all(r in refs for r in codes()),'diagnostic code drift')
        source=load(doc['source_plan']);frames=source['frames'];cams={Path(f['image']['name']).name:f['image'] for f in frames}
        need([a['name'] for a in doc['arms']]==list(ARMS),'four fixed arms required');save_json(out/'BINDINGS_PRE.json',refs)
        for arm in doc['arms']:
            check();work=Path(arm['work']);need(work==out/arm['name']/'work','work directory differs');unchanged(arm['staged'])
            need({str(p.resolve()) for p in work.rglob('*') if p.is_file()}=={r['path'] for r in arm['staged']},'unexpected prior cache/file')
            need(arm['argv']==commands(source,work)[0][1],'fixed dense argv differs')
            evidence=work/'__regional_runtime';evidence.mkdir();result['new_native_launch_attempts']+=1;save_json(out/'RESULT.json',result)
            native=run_supervised(arm['argv'],cwd=work,output=evidence,wall_seconds=min(180,deadline-time.monotonic()),memory_bytes=8<<30,output_bytes=1<<30,label=arm['name'])
            save_json(evidence/'supervisor_result.json',native,exclusive=True)
            ok=(native.get('status')=='completed' and native.get('root_exit_code')==0 and native.get('reclaim_confirmed') is True
                and type(native.get('pid')) is int and native['pid']>0 and not native.get('errors') and not native.get('log_errors')
                and native.get('argv')==arm['argv'] and native.get('cwd')==str(work))
            result['arms'][arm['name']]=dict(supervisor=native,execution_ok=ok)
            need(ok,'native execution/reclamation incomplete; no retry or next arm')
            result['arms'][arm['name']]['effective_config']=_effective_dense_config(source,work,evidence)
            unchanged([r for r in arm['staged'] if Path(r['path']).name!='__mvs_dense.cfg'])
            depths,records=read_arm_depths(work,cams,check);all_depths[arm['name']]=depths;refs+=records
            result['arms'][arm['name']]['depth_bindings']=records;save_json(out/'RESULT.json',result)
        leak=False
        for f,s in zip(frames,doc['views']):
            check();name=s['name'];shape=(f['image']['height'],f['image']['width'])
            fg=binary(f['mask']['path'],shape)&binary(f['validity']['path'],shape);whole=all_depths['WHOLE'].get(name)
            row=dict(decode_index=s['decode_index'],global_id=s['global_id'],whole=compare_depth(whole,whole,fg),
                whole_leakage=leakage(whole,fg),groups=[])
            domains=[];group_depth=[]
            for g in range(3):
                record=s['groups'][g];domain=binary(record['mask']['path'],shape);d=all_depths['G'+str(g)].get(name)
                metrics=compare_depth(whole,d,domain,record['seeded']);lg=leakage(d,domain)
                leak|=lg.get('positive_leak_pixels',0) not in (0,None)
                row['groups'].append(dict(group=g,seed_status='SEEDED_CANDIDATE_NOT_PART_IDENTITY' if record['seeded'] else 'UNKNOWN_NO_SEED',
                    metrics=metrics,leakage=lg,original_foreground_depth=compare_depth(whole,d,fg,record['seeded'])))
                domains.append(domain);group_depth.append(d)
            row['group_union']=union_counts(group_depth,domains,fg);result['views'].append(row)
            leak|=row['whole_leakage'].get('positive_leak_pixels',0) not in (0,None)
            if s['decode_index']==840:render_target(out/'frame_000840_depth.png',f,{a:all_depths[a].get(name) for a in ARMS})
        result['status']='REFERENCE_MASK_LEAK_DETECTED' if leak else 'REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED'
    except Exception as exc:result.update(status='EXECUTION_OR_IDENTITY_UNKNOWN',error=type(exc).__name__+': '+str(exc))
    snapshots=[]
    for rec in refs:
        try:
            current=binding(rec['path'],512<<20);snapshots.append(dict(expected=rec,actual=current,unchanged=current==rec))
        except Exception as exc:snapshots.append(dict(expected=rec,unchanged=False,error=str(exc)))
    result['source_unchanged']=all(s['unchanged'] for s in snapshots)
    if not result['source_unchanged']:result['status']='EXECUTION_OR_IDENTITY_UNKNOWN'
    result.update(elapsed_seconds=time.monotonic()-start,limitations=[
        '同view共享相机/RGB和MVS估计相关；2%相符不是独立精度。','无seed全黑视图始终UNKNOWN，不算区域恢复成功。',
        '新增深度在whole无深度处没有一致性参照；缺失深度不等于几何不存在。','组并集仅统计，不平均、不选赢家、不融合点云、不生成mesh、不替换默认。'])
    save_json(out/'BINDINGS_POST.json',snapshots);save_json(out/'RESULT.json',result)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>分区深度诊断：未接受几何</h1><p>'+result['status']+'</p><p>绿色仅表示正深度覆盖；同源估计相符不等于准确。无seed视图保留未知。</p><a href="RESULT.json">逐视图计数、监督日志与绑定</a><br><img style="max-width:100%" src="frame_000840_depth.png">',encoding='utf-8')
    return result
