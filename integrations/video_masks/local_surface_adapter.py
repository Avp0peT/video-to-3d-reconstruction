"""Conforming local boundary subdivision; reversible candidates, never new observations."""
from __future__ import annotations
import csv
from dataclasses import asdict
import html
from pathlib import Path
import time
import numpy as np
from .face_mask_adapter import (ROOT, LIMITS, _need, _read, _save, _inside, _binary,
    bind, verify_depth, write_partition, _topology)
from .face_mask_gate import Policy, aggregate_views
from .local_subdivision import subdivide_conforming
from .local_face_evidence import LocalPolicy, evaluate_local_view, prepare_probe_membership
from integrations.viewer.ply_data import read_ply
from integrations.mvs_surface.scene import read_scene
from integrations.openmvs_depth.dmap import read_dmap


def seed_candidates(ledger, retained_ids, face_count):
    """Only v2 gap-qualified, v3-restored parents; no original-face vote inheritance."""
    _need(len(ledger) == face_count, 'incomplete parent ledger')
    _need([int(r['source_face_id']) for r in ledger] == list(range(face_count)), 'parent ledger order/identity')
    retained = np.zeros(face_count, bool)
    ids = np.asarray(retained_ids, np.int64)
    _need(len(np.unique(ids)) == len(ids) and (not len(ids) or (ids.min() >= 0 and ids.max() < face_count)), 'retained parent IDs')
    retained[ids] = True
    for i, row in enumerate(ledger):
        _need((row['decision'] == 'RETAINED_NOT_ACCEPTED') == bool(retained[i]), 'ledger/partition mismatch')
    gap = np.array([int(r['gap_conflict_views']) >= 2 and
        r['gap_angle_pair_evaluated'] == 'True' and
        float(r['gap_maximum_conflict_angle_degrees'] or 0) >= 8.0 for r in ledger], bool)
    return gap & retained, ~retained


def area(vertices, faces):
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)*.5


def combine_decisions(parent_ids, eligible, old_quarantine, proposed, protected):
    parent_ids = np.asarray(parent_ids)
    eligible, proposed, protected = [np.asarray(x, bool) for x in (eligible, proposed, protected)]
    _need(all(x.shape == parent_ids.shape for x in (eligible, proposed, protected)), 'decision arrays differ')
    additional = eligible & proposed & ~protected
    return np.asarray(old_quarantine, bool)[parent_ids] | additional, additional


def run(surface_run, gate_run, output, *, footprint_mode='disk'):
    _need(footprint_mode in ('disk', 'closed_cells'), 'unknown footprint mode')
    source, gate = Path(surface_run).resolve(strict=True), Path(gate_run).resolve(strict=True)
    out = Path(output).resolve()
    _need(out.is_relative_to(ROOT/'outputs') and not out.exists(), 'fresh repository output required')
    _need(not source.is_relative_to(out) and not gate.is_relative_to(out), 'output is input ancestor')
    for protected in ('surface','bridge','masks','export'):
        _need(not out.is_relative_to(source/protected), 'output overlaps protected native source')
    _need(not out.is_relative_to(gate), 'output overlaps old gate')
    out.mkdir(parents=True); (out/'per_view').mkdir()
    started = time.monotonic(); bindings = {}; rows = []; views = []
    policy = LocalPolicy()
    result = dict(schema='local-boundary-surface-v1', status='RUNNING', geometry_accepted=False,
        quality_status='NOT_EVALUATED', scale_status='unknown', source_run=str(source), gate_run=str(gate),
        source_unchanged=False, new_sam_executions=0,new_sfm_executions=0,new_mvs_executions=0,
        footprint_mode=footprint_mode,
        new_vertex_kind='INTERPOLATED_ON_PARENT_TRIANGLE_NOT_MEASURED', policy=asdict(policy),
        limitations=['Same masks and camera poses; no segmentation truth or physical visibility acceptance.',
            'Two conforming midpoint rounds only on restored gap candidates. No smoothing, filling or new measured points.',
            'Whole-footprint background certificates are conservative; ambiguous and small children remain.',
            'Original supported probes protect only child triangles that actually contain them.',
            'Retained geometry can still contain wrong joins; this is a reversible local candidate.'])
    def check():
        if time.monotonic()-started > 420:
            raise TimeoutError('local boundary cooperative 420 second limit')
    def add(path, expected=None):
        record = bind(path, expected); bindings[record['path']] = record; return record
    try:
        for name in ('local_surface_adapter.py','local_subdivision.py','local_face_evidence.py','face_mask_adapter.py','face_mask_gate.py','face_edge_gate.py'):
            add(Path(__file__).with_name(name))
        for name in ('RESULT.json','FACE_LEDGER.csv','VIEWS.json','retained.mapping.npz','retained.ply','original.ply'):
            add(gate/name)
        old = _read(gate/'RESULT.json')
        _need(old.get('schema') == 'face-mask-quarantine-v3' and old.get('status') == 'CANDIDATE_SURFACE_NOT_ACCEPTED'
            and old.get('source_unchanged') is True and Path(old['source_run']).resolve() == source, 'accepted v3 source chain required')
        for record in old['source_bindings']:
            add(record['path'], record)
        native = _read(source/'surface/result.json'); masks = _read(source/'masks/RESULT.json')
        _need(native.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED' and not native.get('input_drift'), 'native source refused')
        _need(masks.get('coverage_status') == 'COMPLETE' and masks.get('adapter_source_chain') == 'VERIFIED'
            and not masks.get('input_drift') and not masks.get('output_drift'), 'mask source refused')
        records = [g for s in native['stages'] for g in s.get('outputs',{}).get('geometry',[]) if g.get('role') == 'pre_cleanup_surface']
        _need(len(records) == 1, 'unique raw native mesh required')
        raw = _inside(records[0]['path'], source); raw_binding = add(raw, records[0]); add(gate/'original.ply', raw_binding)
        mesh = read_ply(raw, **LIMITS)
        original_v = np.asarray(mesh['vertices'], float); original_f = np.asarray(mesh['faces'], np.int64)
        colors = np.asarray(mesh['colors'], np.uint8) if mesh['colors'] is not None else None
        with np.load(gate/'retained.mapping.npz', allow_pickle=False) as m:
            retained_ids, vertex_ids = m['source_face_ids'], m['source_vertex_ids']
        retained = read_ply(gate/'retained.ply', **LIMITS)
        _need(np.array_equal(np.asarray(retained['vertices']), original_v[vertex_ids]) and
            np.array_equal(vertex_ids[np.asarray(retained['faces'],np.int64)], original_f[retained_ids]), 'v3 geometry mapping mismatch')
        with (gate/'FACE_LEDGER.csv').open(encoding='utf8', newline='') as f:
            ledger = list(csv.DictReader(f))
        eligible, old_quarantine = seed_candidates(ledger, retained_ids, len(original_f))
        original_eligible = eligible.copy(); v=original_v.copy(); faces=original_f.copy(); parents=np.arange(len(faces), dtype=np.int64)
        lineage = {}; round_counts=[]
        for round_id in range(2):
            check()
            sub = subdivide_conforming(v, faces, parents, eligible, eligible.copy(), colors=colors,
                max_faces=400000, max_vertices=200000)
            lineage['round_'+str(round_id)+'_midpoints'] = sub['midpoint_lineage']
            lineage['round_'+str(round_id)+'_source_faces'] = sub['source_face_indices']
            v,faces,parents,eligible,colors = [sub[k] for k in ('vertices','faces','parent_ids','eligible','colors')]
            round_counts.append(dict(vertices=len(v), faces=len(faces),eligible_children=int(eligible.sum())))
        parent_area = np.bincount(parents, weights=area(v,faces), minlength=len(original_f))
        raw_area = area(original_v,original_f)
        _need(np.allclose(parent_area,raw_area,rtol=1e-10,atol=1e-12), 'subdivision parent area mismatch')
        _need(np.array_equal(v[:len(original_v)], original_v), 'original vertex moved')
        membership = prepare_probe_membership(v,faces,original_v,original_f,parents,check=check)
        np.savez_compressed(out/'LINEAGE.npz',parent_face_ids=parents,eligible=eligible,
            original_parent_candidate=original_eligible,original_parent_quarantined=old_quarantine,
            original_probe_membership=membership,**lineage)
        outputs={'refined_before':write_partition(out/'refined_before.ply',v,faces,colors,np.arange(len(faces)))}
        add(source/'bridge/scene.mvs', masks['scene_binding']); scene=read_scene(source/'bridge/scene.mvs')
        _need(np.array_equal(scene['transform'],np.eye(4)), 'unsupported scene transform')
        images={Path(i['name']).name:i for i in scene['images']}; frames={f['exported_name']:f for f in masks['frames']}
        _need(0 < len(images) <= 256 and len(images)==len(scene['images'])==len(frames)==len(masks['frames'])
            and images.keys()==frames.keys(), 'mask scene set mismatch')
        old_views={v['image_name']:v for v in _read(gate/'VIEWS.json')}
        _need(old_views.keys()==images.keys(), 'old/new views differ')
        for name,image in images.items():
            _need(old_views[name]['camera']==image and frames[name]['target_image_id']==image['global_id']
                and frames[name]['status']=='OK' and frames[name]['readback_verified'] is True, 'camera/frame identity mismatch')
        _save(out/'PLAN.json',dict(schema='local-boundary-plan-v1',rounds=2,max_faces=400000,max_vertices=200000,
            source_mesh=raw_binding,policy=asdict(policy),candidate_parent_count=int(original_eligible.sum()),
            native_pixel_coordinates=True,max_seconds=420,threads=2,geometry_accepted=False,
            footprint_mode=footprint_mode))
        protected_any=np.zeros(len(faces),bool); original_protected_any=np.zeros(len(faces),bool); cameras={}
        def process(image,d=None):
            check(); name=Path(image['name']).name; frame=frames[name]; vid=str(image['global_id'])
            mask=_binary(frame['mask'],source,(image['width'],image['height']),list_bindings:=[])
            valid=_binary(frame['validity'],source,(image['width'],image['height']),list_bindings)
            for ref in list_bindings: bindings[ref['path']]=ref
            _need(not np.any(mask & ~valid), 'mask foreground in unknown domain')
            gp=gate/'gap_per_view'/('view_'+vid+'.npz'); add(gp)
            with np.load(gp,allow_pickle=False) as ar: gap=ar['gap_mask'].copy()
            _need(gap.dtype==bool and gap.shape==mask.shape and not np.any(gap & (mask|~valid)), 'invalid gap domain')
            row=evaluate_local_view(v,faces,K=np.asarray(image['K']),R=np.asarray(image['R']),C=np.asarray(image['C']),
                mask=mask,validity=valid,depth=d.depth if d is not None else None,gap_mask=gap,view_id=vid,
                original_vertices=original_v,original_faces=original_f,parent_ids=parents,eligible=eligible,
                policy=policy,check=check,probe_membership=membership,footprint_mode=footprint_mode)
            protected_any[:] |= row['core_protected']; original_protected_any[:] |= row['original_probe_protected']
            np.savez_compressed(out/'per_view'/('view_'+vid+'.npz'),**{k:x for k,x in row.items() if isinstance(x,np.ndarray)})
            rows.append(dict(view_id=vid,status=row['status'].copy())); cameras[vid]=image['C']
            views.append(dict(view_id=vid,input_index=frame['input_index'],camera=image,depth_available=d is not None,
                negative_children=int(row['negative'].sum()),protected_children=int(row['core_protected'].sum())))
            _save(out/'PROGRESS.json',dict(processed_views=len(views),expected_views=len(images),last_view=vid,elapsed_seconds=time.monotonic()-started))
            print('LOCAL_BOUNDARY_VIEW',vid,'negative',views[-1]['negative_children'],flush=True)
        done=set()
        for p in sorted((source/'surface/work').glob('depth*.dmap')):
            check();d=read_dmap(p); name=Path(d.image_name_utf8.replace('\\','/')).name
            _need(name in images and name not in done,'unknown/duplicate depth')
            verify_depth(d,images[name]);add(p,dict(sha256=d.sha256,bytes=d.file_bytes));process(images[name],d);done.add(name);del d
        for name in sorted(set(images)-done): process(images[name])
        ag=aggregate_views(v,faces,rows,cameras,Policy(min_angle_degrees=policy.min_angle_degrees),check=check)
        proposed=ag['quarantine'] & (ag['conflict_views'] >= policy.min_views)
        quarantine,additional=combine_decisions(parents,eligible,old_quarantine,proposed,protected_any)
        _need(not quarantine.all() and not np.any(additional & (~eligible|protected_any)), 'invalid child decisions')
        np.savez_compressed(out/'DECISIONS.npz',quarantined=quarantine,additional=additional,protected_any=protected_any,
            original_probe_protected_any=original_protected_any,conflict_views=ag['conflict_views'],
            max_angle=ag['max_conflict_angle_degrees'])
        for label,ids in [('retained',np.flatnonzero(~quarantine)),('quarantined',np.flatnonzero(quarantine))]:
            outputs[label]=write_partition(out/(label+'.ply'),v,faces,colors,ids)
        with (out/'FACE_LEDGER.csv').open('x',encoding='utf8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['refined_face_id','original_parent_id','eligible','decision','additional_quarantine',
                'core_protected','original_probe_protected','negative_views','max_angle','witness_1','witness_2'])
            for i in range(len(faces)):
                writer.writerow([i,int(parents[i]),bool(eligible[i]),'QUARANTINED_CANDIDATE' if quarantine[i] else 'RETAINED_NOT_ACCEPTED',
                    bool(additional[i]),bool(protected_any[i]),bool(original_protected_any[i]),int(ag['conflict_views'][i]),
                    float(ag['max_conflict_angle_degrees'][i]),*(ag['witness_view_ids'][i] or [None,None])])
        _save(out/'VIEWS.json',views)
        _need(all(bind(r['path'])==r for r in bindings.values()),'source changed during local evaluation')
        areas=area(v,faces)
        result.update(status='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED',source_unchanged=True,outputs=outputs,
            counts=dict(original_vertices=len(original_v),original_faces=len(original_f),candidate_parents=int(original_eligible.sum()),
                refined_vertices=len(v),refined_faces=len(faces),interpolated_vertices=len(v)-len(original_v),
                eligible_children=int(eligible.sum()),additional_quarantined_children=int(additional.sum()),
                affected_parent_faces=int(len(np.unique(parents[additional]))),retained_faces=int((~quarantine).sum()),
                quarantined_faces=int(quarantine.sum()),protected_children=int(protected_any.sum()),views=len(views),depth_views=len(done)),
            area=dict(original=float(raw_area.sum()),additional_quarantined=float(areas[additional].sum()),
                quarantined_fraction=float(areas[quarantine].sum()/areas.sum())),
            topology=dict(original=_topology(original_f),refined_before=_topology(faces),retained=_topology(faces[~quarantine])),
            rounds=round_counts,source_bindings=list(bindings.values()),elapsed_seconds=time.monotonic()-started)
        _save(out/'EVIDENCE.json',result)
        from integrations.viewer.export import export_html
        def entry(p,title,sha):
            return dict(ply=str(p),sha256=sha,title=title,data_kind='real_photos',declared_unit=None,
                note='Interpolated local boundary candidate. No new measured geometry; holes preserved.',evidence=[str(out/'EVIDENCE.json')])
        new=entry(out/'retained.ply','局部边缘候选 / 未验收',outputs['retained']['sha256'])
        _save(out/'retained_scenes.json',dict(schema='ply-viewer-scenes-v1',scenes=[new]))
        _save(out/'comparison_scenes.json',dict(schema='ply-viewer-scenes-v1',scenes=[entry(gate/'retained.ply','前版边缘候选',bind(gate/'retained.ply')['sha256']),new]))
        export_html(out/'retained_scenes.json',out/'retained_viewer.html',profile='surface400k')
        export_html(out/'comparison_scenes.json',out/'comparison_viewer.html',profile='surface400k')
        check()
        _need(all(bind(r['path'])==r for r in bindings.values()),'source changed during viewer export')
    except Exception as exc:
        result.update(status='FAILED',source_unchanged=False,
            error=type(exc).__name__+': '+str(exc),processed_views=len(views))
    result['elapsed_seconds']=time.monotonic()-started
    _save(out/'RESULT.json',result)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>局部物体边缘</title>'
        '<h1>局部细分与多视图边缘检查</h1><p>'+html.escape(result['status'])+'</p>'
        '<p>新点来自原三角形插值；没有补拍信息、填洞或精度验收。原始掩膜未修改。</p>'
        '<p><a href="retained_viewer.html">单独候选</a> · <a href="comparison_viewer.html">三维对比</a> · '
        '<a href="RESULT.json">完整证据</a> · <a href="FACE_LEDGER.csv">逐面判定</a></p>'
        '<pre>'+html.escape(str(result.get('counts',result.get('error',''))))+'</pre>',encoding='utf8')
    return result
