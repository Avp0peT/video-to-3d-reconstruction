"""Continue a verified unified photo run to bounded faceted CAD delivery."""
from pathlib import Path
from datetime import datetime, timezone
import sys

from integrations.meshprep.provenance import binding, document, validate_source
from integrations.cad_delivery.delivery import prepared_input, verify_delivery
from integrations.mvs_surface.supervision import run_supervised
from .surface import save

ROOT = Path(__file__).resolve().parents[2]
LIGHT = 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe'
FREECAD = 'outputs/toolchains/freecad-1.1.4/FreeCAD_1.1.4-Windows-x86_64-py311/bin/python.exe'


def need(ok, message):
    if not ok:
        raise ValueError('photo CAD: ' + message)


def within(path, parent):
    path = Path(path).resolve(strict=True)
    need(path.is_relative_to(parent), 'source evidence escapes unified run')
    return path


def check(row):
    actual = binding(row['path'], 512 << 20)
    need(all(actual[k] == row[k] for k in ('path', 'bytes', 'sha256')), 'bound file changed: ' + row['path'])
    return actual


def commands(frozen):
    output=Path(frozen['output'])
    return [('prepare',[str(x) for x in [frozen['freecad_python'],'-I','-B',ROOT/'scripts/prepare_mesh.py','--mesh',frozen['mesh'],
        '--source-run',frozen['source_result'],'--source-role','mesh','--output',output/'prepared',
        '--method','quadric','--max-input-faces','400000','--target-faces','2000',
        '--max-relative-deviation','0.01','--max-relative-volume-change','0.02']],60),
        ('cad',[str(x) for x in [frozen['python'],'-B','-X','utf8',ROOT/'scripts/build_cad_delivery.py',
        '--prepared-run',output/'prepared','--freecad-python',frozen['freecad_python'],
        '--output',output/'cad','--wall-seconds','120']],135)]


def prepared_bound(frozen):
    directory=Path(frozen['output'])/'prepared'
    checked=prepared_input(directory)
    record,_=document(directory/'mesh_prep.json')
    expected={key:frozen['source_link'][key] for key in ('mesh','source_record')}
    declared=record.get('inputs')
    need(isinstance(declared,dict) and set(declared)==set(expected), 'prepared source input schema differs')
    for key,row in declared.items():
        need(isinstance(row,dict) and set(row)=={'path','sha256'}, 'prepared source row schema differs')
        actual=binding(row['path'],512<<20)
        need(row['path']==actual['path'] and row['sha256']==actual['sha256']
             and actual==expected[key], 'prepared input belongs to another producer mesh or source record')
    need(record.get('source_validation') == frozen['source_link'], 'prepared source validation differs from frozen producer')
    return checked


def native_ok(native, argv, cwd):
    return (native.get('status')=='completed' and type(native.get('root_exit_code')) is int
            and native['root_exit_code']==0 and native.get('reclaim_confirmed') is True
            and not native.get('errors') and not native.get('log_errors')
            and type(native.get('pid')) is int and native['pid']>0
            and native.get('argv')==argv and native.get('cwd')==cwd)


def delivery_bound(frozen, prepared):
    directory=Path(frozen['output'])/'cad'
    verified=verify_delivery(directory)
    request,_=document(directory/'request.json')
    need(request.get('prepared')==prepared and request.get('prepared_run')==str(Path(frozen['output'])/'prepared'),
         'delivery request belongs to another prepared input')
    return verified


def geometry_checkpoint(run, path):
    """Bind an immutable geometry checkpoint, never the changing pipeline state."""
    run = Path(run).resolve(strict=True)
    expected = run / 'CAD_SOURCE.json'
    path = Path(path).resolve(strict=True)
    need(path == expected and path.is_relative_to(run), 'geometry checkpoint must be this run/CAD_SOURCE.json')
    record, row = document(path)
    need(record.get('schema') == 'photo-cad-source-v1'
         and record.get('source_run') == str(run)
         and record.get('status') == 'GEOMETRY_READY_FOR_CAD'
         and record.get('target') == 'cad'
         and record.get('geometry_deliverable_ready') is True
         and record.get('deliverable_ready') is False, 'invalid geometry checkpoint state')
    current, _ = document(run / 'pipeline.json')
    need(current.get('target') == 'cad' and isinstance(current.get('stages'), dict),
         'current pipeline is not the checkpoint CAD source')
    for name in ('surface', 'blender'):
        value = record.get('stages', {}).get(name)
        need(isinstance(value, dict) and value.get('status') == 'SUCCESS'
             and value == current['stages'].get(name), 'geometry checkpoint differs from current ' + name)
    return record, row


def surface_continuation_source(run, *, root=ROOT):
    """Read-only formal CONTINUATION source, without fabricating pipeline state.

    Reuses current byte bindings and native identity checks. Geometry/Blender
    proof below in plan remains common to old and continuation sources.
    """
    run, root = Path(run).resolve(strict=True), Path(root).resolve()
    files = []
    def doc(path, inside=run):
        value, row = document(within(path, inside)); files.append(row); return value
    meta = doc(run / 'CONTINUATION.json'); frozen = doc(run / 'PLAN.json')
    need(meta.get('schema') == frozen.get('schema') == 'photo-surface-continuation-v1'
         and meta.get('status') in ('SUCCESS', 'PARTIAL')
         and type(meta.get('exit_code')) is int and meta['exit_code'] == (1 if meta['status'] == 'PARTIAL' else 0)
         and meta.get('deliverable_ready') is True and frozen.get('status') == 'READY'
         and frozen.get('output') == str(run) and frozen.get('root') == str(root), 'formal surface continuation incomplete')
    source = Path(frozen.get('source_run', '')).resolve(strict=True)
    need(source.is_relative_to(root / 'outputs') and source != root / 'outputs'
         and not source.is_relative_to(run) and not run.is_relative_to(source)
         and meta.get('source_run') == str(source), 'continuation original source root differs')
    rows = frozen.get('bindings')
    need(isinstance(rows, list) and 0 < len(rows) <= 2048 and rows == meta.get('source_bindings')
         and all(isinstance(r, dict) and set(r) == {'path', 'bytes', 'sha256'} for r in rows)
         and len({r['path'] for r in rows}) == len(rows), 'continuation complete source bindings missing')
    actual_rows = [check(row) for row in rows]; files.extend(actual_rows)
    need(sum(r['bytes'] for r in actual_rows) <= 3 << 30, 'continuation input byte budget exceeded')
    indexed = {r['path']: r for r in actual_rows}
    original = doc(source / 'pipeline.json', source)
    checkpoint = doc(source / 'SURFACE_RESUME_SOURCE.json', source)
    for path in (source / 'pipeline.json', source / 'SURFACE_RESUME_SOURCE.json', source / 'PLAN.json'):
        need(str(path) in indexed and indexed[str(path)] == binding(path, 8 << 20), 'original source/checkpoint not bound by continuation plan')
    need(original.get('status') in ('SUCCESS', 'PARTIAL', 'FAILED', 'CANCELLED')
         and original.get('status') == frozen.get('source_status') == meta.get('source_pipeline_status')
         and type(original.get('exit_code')) is int, 'original producer not terminal or differs')
    upstream = frozen.get('upstream_stages')
    need(isinstance(upstream, dict) and set(upstream) == {'input', 'sfm', 'segmentation'}
         and upstream == checkpoint.get('upstream_stages')
         and upstream == {k: original.get('stages', {}).get(k) for k in upstream}, 'producer stage projection differs')
    need(checkpoint.get('schema') == 'photo-surface-resume-source-v1'
         and checkpoint.get('source_run') == str(source) and checkpoint.get('producer_status') == 'RUNNING', 'producer checkpoint schema/source differs')
    committed = checkpoint.get('bindings')
    need(isinstance(committed, list) and committed and all(indexed.get(r.get('path')) == r for r in committed), 'producer checkpoint dependency not frozen')
    need(meta.get('reused_stage_statuses') == {k: v.get('status') for k, v in upstream.items()}, 'reused upstream status differs')
    need(all(v.get('status') in (('SUCCESS',) if k == 'input' else ('SUCCESS', 'PARTIAL'))
             and v.get('reclaim_confirmed') is True for k, v in upstream.items()), 'upstream reuse incomplete or not reclaimed')
    for name, stage in upstream.items():
        entry_path = source / 'logs' / name / 'ENTRY.json'
        need(stage.get('entry_path') == str(entry_path) and str(entry_path) in indexed, 'upstream supervisor entry not bound')
        entry = doc(entry_path, source)
        code = 1 if stage['status'] == 'PARTIAL' else 0
        need(type(stage.get('exit_code')) is int and stage['exit_code'] == code
             and entry.get('status') == 'completed' and type(entry.get('root_exit_code')) is int
             and entry['root_exit_code'] == code and entry.get('reclaim_confirmed') is True
             and not entry.get('errors') and not entry.get('log_errors')
             and type(entry.get('pid')) is int and entry['pid'] > 0
             and isinstance(stage.get('argv'), list) and stage['argv']
             and entry.get('argv') == stage['argv'] and entry.get('cwd') == str(root), 'upstream actual supervisor identity/reclaim differs')
    partial = any(v['status'] == 'PARTIAL' for v in upstream.values())
    need(meta['status'] == ('PARTIAL' if partial else 'SUCCESS'), 'continuation masks partial upstream status')
    names = {'native_subset': ('derive_general_subset.py', None), 'mvs_bridge': ('prepare_mvs.py', None),
             'mvs_masks': ('prepare_mvs_masks.py', None), 'surface_plan': ('build_mvs_surface.py', 'plan'),
             'surface_reconstruction': ('build_mvs_surface.py', 'run'), 'blender_export': ('export_blender.py', None)}
    stages = meta.get('stages')
    need(isinstance(stages, dict) and set(stages) == set(names), 'continuation native stages incomplete')
    work = run / 'geometry/surface_pipeline'
    arguments = {
        'native_subset': {'--sfm-run': frozen['sfm_run'], '--io-run': frozen['io_run'], '--model-id': frozen['model_id'], '--output': str(work / 'subset')},
        'mvs_bridge': {'--image-root': frozen['sfm_run'], '--output': str(work / 'bridge')},
        'mvs_masks': {'--sfm-run': frozen['sfm_run'], '--bridge': str(work / 'bridge'), '--output': str(work / 'masks')},
        'surface_plan': {'--scene-path': str(work / 'bridge/scene.mvs'), '--plan-output': str(work / 'SURFACE_PLAN.json')},
        'surface_reconstruction': {'--plan': str(work / 'SURFACE_PLAN.json'), '--output': str(work / 'surface')},
        'blender_export': {'--spec': str(work / 'SCENES.json'), '--output': str(work / 'blender')},
    }
    for name, (script, action) in names.items():
        stage = stages[name]; expected = run / (name + '_ENTRY.json')
        need(stage.get('status') == 'SUCCESS' and type(stage.get('exit_code')) is int and stage['exit_code'] == 0
             and stage.get('reclaim_confirmed') is True and stage.get('entry_path') == str(expected), 'continuation stage identity/reclaim differs')
        native = doc(expected)
        log = doc(run / 'logs' / name / 'ENTRY.json')
        need(native == log, 'continuation stage/log entry differs')
        argv = native.get('argv')
        prefix = [frozen['python'], '-B', '-X', 'utf8', str(root / 'scripts' / script)]
        if action is not None: prefix.append(action)
        need(isinstance(argv, list) and argv[:len(prefix)] == prefix
             and native_ok(native, argv, str(root)), 'continuation native command/cwd/pid/reclaim differs')
        for flag, value in arguments[name].items():
            need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                 and argv[argv.index(flag) + 1] == value, 'continuation command input/output association differs')
    outcome = doc(run / 'geometry/surface_pipeline/RESULT.json')
    need(outcome.get('status') == 'SUCCESS' and type(outcome.get('exit_code')) is int and outcome['exit_code'] == 0
         and outcome.get('surface') == meta.get('surface') and outcome.get('blender') == meta.get('blender')
         and outcome.get('stages') == meta.get('geometry_steps'), 'continuation differs from complete_surface result')
    return {**meta, 'stages': {'surface': meta['surface'], 'blender': meta['blender']}}, files


def plan(run, output, *, freecad_python=None, python=None, geometry_record=None, root=ROOT):
    root = Path(root).resolve(); run = Path(run).resolve(strict=True); output = Path(output).resolve()
    need(run.is_relative_to(root / 'outputs'), 'unified run outside project outputs')
    need(output.is_relative_to(root / 'outputs') and output != root / 'outputs', 'output outside project outputs')
    need(not output.exists(), 'output already exists')
    if geometry_record is None:
        need(not output.is_relative_to(run) and not run.is_relative_to(output), 'output overlaps source run')
    else:
        need(output == run / 'cad', 'unified CAD output must be this run/cad')
    files = []
    def doc(path):
        value, row = document(within(path, run)); files.append(row); return value
    if geometry_record is None:
        if (run / 'CONTINUATION.json').is_file():
            need(not (run / 'pipeline.json').exists(), 'ambiguous pipeline and continuation source')
            meta, continuation_files = surface_continuation_source(run, root=root)
            files.extend(continuation_files)
            source_kind = 'surface_continuation'
        else:
            meta = doc(run / 'pipeline.json')
            source_kind = 'pipeline'
        need(meta.get('status') in ('SUCCESS', 'PARTIAL') and meta.get('deliverable_ready') is True, 'unified run has no verified deliverable')
    else:
        meta, checkpoint_binding = geometry_checkpoint(run, geometry_record)
        files.append(checkpoint_binding)
        source_kind = 'geometry_checkpoint'
    stages = meta.get('stages', {}); surface = stages.get('surface', {}); blend = stages.get('blender', {})
    need(surface.get('status') == blend.get('status') == 'SUCCESS' and blend.get('reopen_verified') is True, 'surface or Blender stage incomplete')
    files.append(check(dict(blend, path=str(within(blend['path'], run)))))
    exported = doc(blend['result_path'])
    need(exported.get('status') == 'SUCCESS' and exported.get('blend') == {k:blend[k] for k in ('path','bytes','sha256')}, 'Blender binding differs')
    bs = exported.get('stages', [])
    need([s.get('mode') for s in bs] == ['create','verify'], 'Blender independent readback missing')
    need(all(s.get('data', {}).get('status') == 'PASS' and s.get('supervision', {}).get('status') == 'completed'
             and s.get('supervision', {}).get('root_exit_code') == 0 and s.get('supervision', {}).get('reclaim_confirmed') is True
             and not s.get('supervision', {}).get('errors') and not s.get('supervision', {}).get('log_errors')
             and s.get('data', {}).get('payload_sha256') == exported.get('payload_sha256') for s in bs), 'Blender readback did not pass')
    need(bs[0]['data'].get('scenes') == bs[1]['data'].get('scenes') and exported.get('blend_sha256_before_readback') == blend['sha256'], 'Blender identity changed across readback')
    evidence = surface.get('evidence_paths')
    need(isinstance(evidence, list) and len(evidence) == 1, 'unique successful surface evidence required')
    source_path = within(evidence[0], run); source = doc(source_path)
    need(source.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED' and source.get('exit_code') == 0, 'surface producer failed')
    rows = source.get('stages', [])
    need(len(rows) == 2 and rows[1].get('name') == 'mesh', 'mesh stage evidence missing')
    geometry = rows[1].get('outputs', {}).get('geometry', [])
    selected = [r for r in geometry if r.get('role') == 'mesh']
    need(len(selected) == 1, 'unique final mesh role required')
    mesh = within(selected[0]['path'], run); files.append(check(selected[0]))
    declared = surface.get('artifacts', [])
    need(len(declared) == 1, 'surface declared artifact missing')
    raw = [r for r in geometry if r.get('role') == 'pre_cleanup_surface']
    need(len(raw) == 1 and all(declared[0].get(k) == raw[0].get(k) for k in ('path','bytes','sha256')), 'declared surface is not producer raw output')
    files.append(check(dict(declared[0],path=str(within(declared[0]['path'],run)))))
    scenes = bs[1]['data'].get('scenes', [])
    need(len(scenes) == 1 and scenes[0].get('source_sha256') == declared[0]['sha256'], 'Blender does not contain the unique declared surface')
    link = validate_source(mesh, source_path, synthetic=False, role='mesh')
    files.extend(link[k] for k in ('mesh','source_record','plan') if link.get(k))
    native = Path(freecad_python or root / FREECAD).resolve(strict=True)
    light = Path(python or root / LIGHT).resolve(strict=True)
    files.extend(binding(p, 512 << 20) for p in (native,light))
    code = [root/'scripts/prepare_mesh.py',root/'scripts/build_cad_delivery.py',root/'scripts/mesh_to_cad.py',
            root/'scripts/continue_photo_cad.py',root/'integrations/demo_pipeline/cad.py',
            root/'integrations/demo_pipeline/surface.py',root/'integrations/mvs_surface/supervision.py']
    for folder in ('meshprep','cad_delivery','mesh_scale','viewer'):
        code.extend(sorted((root/'integrations'/folder).glob('*.py')))
    files.extend(binding(p, 8 << 20) for p in code)
    frozen = {r['path']:r for r in files}
    return {'schema':'photo-cad-continuation-v1','run':str(run),'output':str(output),'mesh':str(mesh),
            'source_result':str(source_path),'source_link':link,'python':str(light),'freecad_python':str(native),
            'geometry_record':str(Path(geometry_record).resolve()) if geometry_record is not None else None,
            'source_kind':source_kind,
            'bindings':list(frozen.values()),'recipe':{'method':'quadric','target_faces':2000,'max_input_faces':400000,
            'max_relative_deviation':0.01,'max_relative_volume_change':0.02,'prepare_seconds':60,'cad_seconds':120,
            'memory_bytes':2<<30,'output_bytes':256<<20},'automatic_retry':False,'scale_status':'unknown',
            'physical_accuracy':'NOT_EVALUATED','representation':'faceted solid; no parametric design recovery'}


def verify_cad_output(output, expected_source_run):
    """Read-only current validation for UI downloads; no native launch."""
    output = Path(output).resolve(strict=True)
    need(output.is_relative_to(ROOT / 'outputs'), 'CAD output outside project outputs')
    result, _ = document(output / 'RESULT.json')
    frozen, _ = document(output / 'PLAN.json')
    expected = Path(expected_source_run).resolve(strict=True)
    need(frozen.get('schema') == 'photo-cad-continuation-v1' and Path(frozen['run']).resolve() == expected
         and Path(frozen['output']).resolve() == output, 'CAD result belongs to another source run or output')
    need(result.get('schema') == 'photo-cad-result-v1' and result.get('status') == 'SUCCESS'
         and type(result.get('exit_code')) is int and result['exit_code'] == 0
         and result.get('source_and_code_stable') is True, 'CAD continuation has no accepted result')
    for row in frozen['bindings']:check(row)
    if frozen.get('geometry_record') is not None:
        geometry_checkpoint(expected, frozen['geometry_record'])
    if (expected / 'CONTINUATION.json').exists() or frozen.get('source_kind') == 'surface_continuation':
        need(frozen.get('source_kind') == 'surface_continuation', 'CAD continuation source mode differs')
        surface_continuation_source(expected, root=ROOT)
    need(validate_source(frozen['mesh'],frozen['source_result'],synthetic=False,role='mesh') == frozen['source_link'], 'CAD producer source changed')
    prepared=prepared_bound(frozen)
    need(result.get('preparation')==prepared, 'continuation prepared binding differs')
    stages=result.get('stages',[])
    recipe=commands(frozen)
    need(len(stages)==2, 'continuation supervision stages missing')
    for recorded,(name,argv,seconds) in zip(stages,recipe):
        expected_path=output/(name+'_supervision')/'result.json'
        need(recorded.get('name')==name and recorded.get('argv')==argv and recorded.get('cwd')==str(output)
             and recorded.get('supervisor')==str(expected_path) and type(recorded.get('exit_code')) is int
             and recorded['exit_code']==0 and recorded.get('reclaim_confirmed') is True,
             'continuation supervision association differs')
        native,_=document(expected_path)
        need(native_ok(native,argv,str(output)), 'continuation native supervision did not pass')
    delivery = delivery_bound(frozen,prepared)
    report, _ = document(output / 'cad/cad/conversion_report.json')
    paths = {str((output / 'cad/cad' / name).resolve()) for name in ('model.FCStd','model.step')}
    artifacts = report.get('artifacts', [])
    need(len(artifacts) == 2 and {r.get('path') for r in artifacts} == paths
         and result.get('artifacts') == artifacts, 'CAD artifact association differs')
    checked = [check(row) for row in artifacts]
    return {'status':'CURRENT_VERIFIED_CAD_OUTPUT','source_run':str(expected),'output':str(output),
            'artifacts':checked,'independent_readback':report['checks'],'delivery_verification':delivery,
            'scale_status':'unknown','physical_accuracy':'NOT_EVALUATED','representation':'faceted solid'}


def execute(frozen, *, supervisor=None):
    """At most one preparation and one conditional delivery entry; no retry."""
    supervisor = supervisor or run_supervised
    output = Path(frozen['output']); need(not output.exists(), 'output already exists')
    output.mkdir(parents=True)
    result = {'schema':'photo-cad-result-v1','status':'RUNNING','exit_code':4,'stages':[],
              'started_utc':datetime.now(timezone.utc).isoformat(),'scale_status':'unknown',
              'physical_accuracy':'NOT_EVALUATED','automatic_retry':False,'artifacts':[]}
    save(output/'PLAN.json',frozen)
    def stable():
        for row in frozen['bindings']: check(row)
        if frozen.get('geometry_record') is not None:
            geometry_checkpoint(frozen['run'], frozen['geometry_record'])
        if (frozen.get('source_kind') == 'surface_continuation' or
                frozen.get('run') and (Path(frozen['run']) / 'CONTINUATION.json').exists()):
            need(frozen.get('source_kind') == 'surface_continuation', 'CAD continuation source mode differs')
            surface_continuation_source(frozen['run'], root=ROOT)
        need(validate_source(frozen['mesh'],frozen['source_result'],synthetic=False,role='mesh') == frozen['source_link'], 'producer association changed')
    def launch(name,argv,seconds):
        evidence=output/(name+'_supervision'); evidence.mkdir()
        row={'name':name,'argv':[str(x) for x in argv],'cwd':str(output),'status':'STARTING'}
        result['stages'].append(row);save(output/'RESULT.json',result)
        native=supervisor(row['argv'],cwd=str(output),output=str(evidence),wall_seconds=seconds,
                          memory_bytes=2<<30,output_bytes=256<<20,label='photo_cad_'+name)
        save(evidence/'result.json',native);row.update(status=native.get('status'),exit_code=native.get('root_exit_code'),
                                                     reclaim_confirmed=native.get('reclaim_confirmed'),supervisor=str(evidence/'result.json'))
        save(output/'RESULT.json',result)
        if not native_ok(native,row['argv'],str(output)):
            code=native.get('root_exit_code')
            result['exit_code']=(5 if native.get('status') in ('wall_clock_exceeded','run_dir_threshold','total_threshold')
                                 else code if native.get('status')=='completed' and type(code) is int and code!=0 else 4)
            raise RuntimeError(name+' failed; no retry')
    try:
        stable()
        recipe=commands(frozen)
        launch(*recipe[0])
        preparation=prepared_bound(frozen)
        result['preparation']=preparation
        stable()
        launch(*recipe[1])
        verified=delivery_bound(frozen,preparation)
        report,_=document(output/'cad/cad/conversion_report.json')
        result.update(status='SUCCESS',exit_code=0,delivery_verification=verified,
                      independent_readback=report['checks'],artifacts=report['artifacts'])
        for row in result['artifacts']:check(row)
    except Exception as exc:
        result.update(status='FAILED',exit_code=result['exit_code'] if result['exit_code']!=0 else 4,
                      error=f'{type(exc).__name__}: {exc}')
    finally:
        try:stable();result['source_and_code_stable']=True
        except Exception as exc:result.update(status='SOURCE_DRIFT',exit_code=4,source_and_code_stable=False,drift_error=str(exc))
        result['ended_utc']=datetime.now(timezone.utc).isoformat();save(output/'RESULT.json',result)
    return result
