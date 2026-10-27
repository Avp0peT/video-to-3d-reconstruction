import json
from pathlib import Path
import pytest

from integrations.demo_pipeline import resume as r
from integrations.demo_pipeline.surface import save
from integrations.mvs_surface.photo_inputs import _binding


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'project'; root.mkdir()
    run = root / 'outputs/source'; run.mkdir(parents=True)
    originals = root / 'originals'; originals.mkdir()
    for i in range(2):
        (originals / f'{i}.png').write_bytes(b'original' + bytes([i]))
    io = run / 'input/io'; io.mkdir(parents=True)
    frames = []
    for i in range(2):
        (io / f'{i}.png').write_bytes(b'canonical' + bytes([i]))
        frames.append(dict(input_index=i, frame_id=f'frame{i}', image_output=f'{i}.png',
                           image_sha256=r._binding(io / f'{i}.png')['sha256'],
                           canonical_width=2, canonical_height=2, status='OK'))
    save(io / 'run.json', dict(contract_version='photo-io-v1', run_id='io1', status='SUCCESS'))
    (io / 'frames.jsonl').write_text('\n'.join(json.dumps(x) for x in frames))
    dataset = run / 'input/photo_manifest.json'; save(dataset, {})
    save(run / 'input/RESULT.json', dict(contract_version='object-input-v1', status='INPUTS_PREPARED',
        data_kind='real', exit_code=0, photo_io={'run':_binding(io/'run.json'), 'manifest':_binding(dataset)}))
    # Full input transformation verification is separately covered by existing
    # object_input tests; these fixtures exercise continuation identity linkage.
    import integrations.object_input.session as session
    monkeypatch.setattr(session, 'verify_input', lambda *a, **k: {'status':'INPUT_BINDINGS_VALID'})
    sfm = root / 'outputs/sfm'; sfm.mkdir()
    model = sfm / 'models/0'; (model/'binary').mkdir(parents=True)
    for filename in r.MODEL_BIN:
        (model/'binary'/filename).write_bytes(filename.encode())
    save(model/'descriptions.json', dict(source={}, binary={}, text={}))
    maps=[]
    for i, frame in enumerate(frames):
        staged=sfm/f'{i}.png';staged.write_bytes((io/f'{i}.png').read_bytes())
        maps.append(dict(input_index=i,frame_id=frame['frame_id'],upstream_io_run_id='io1',
            upstream_image_output=frame['image_output'],staged_image_name=f'{i}.png',
            canonical_image_sha256=_binding(staged)['sha256']))
    (sfm/'frames_map.jsonl').write_text('\n'.join(json.dumps(x) for x in maps))
    counts={'input_rows':2}
    save(sfm/'run.json',dict(contract_version='sfm-native-object-v1',status='SUCCESS',data_kind='real',
        upstream_io_run_id='io1',counts=counts,models=[dict(model_id='0',usable=True,roundtrip_verified=True,
        num_registered_images=2,num_points3D=5,binary_path='models/0/binary',
        descriptions_path='models/0/descriptions.json',source_vs_binary_differences=[],source_vs_text_differences=[])]))
    segmentation=run/'segmentation'; masks=segmentation/'product';masks.mkdir(parents=True)
    mask_frames=[]
    for i,frame in enumerate(frames):
        mask=masks/f'{i}.mask.png';mask.write_bytes(b'mask'+bytes([i]));digest=_binding(mask)['sha256']
        mask_frames.append(dict(input_index=i,frame_id=frame['frame_id'],status='OK',
            image_domain='photo_io_canonical',source_image=str(io/f'{i}.png'),
            source_image_sha256=frame['image_sha256'],photo_io_image_sha256=frame['image_sha256'],
            width=2,height=2,candidates=[dict(candidate_id='c0',mask_path=mask.name,mask_sha256=digest,foreground_pixels=1)],
            selected_candidate_id='c0',selected_mask={'path':mask.name,'sha256':digest},
            selection={'method':'unique_candidate','source':'automatic','mask_source':'automatic_sam3'}))
    sam_counts=dict(requested=2,success=2,no_match=0,needs_selection=0,error=0,not_run=0)
    save(masks/'manifest.json',dict(contract_version='object-mask-v1',run_id='sam1',status='SUCCESS',
        started_utc='start',finished_utc='end',source_io=dict(path=str(io),run_id='io1',
        run_json_sha256=_binding(io/'run.json')['sha256'],frames_jsonl_sha256=_binding(io/'frames.jsonl')['sha256']),
        backend={},object={'object_id':'obj'},counts=sam_counts,frames=mask_frames))
    save(segmentation/'PRODUCT_OUTCOME.json',dict(status='SUCCESS',exit_code=0,run_dir=str(masks),
        counts=sam_counts,stop_state={'latched':False}))
    save(segmentation/'CONFIG.json',dict(io_run=str(io),text_prompt='object',object_id='obj',indices=[0,1]))
    tools=root/'tools';tools.mkdir()
    for name in ('python.exe','blender.exe','InterfaceCOLMAP.exe','DensifyPointCloud.exe','ReconstructMesh.exe'):
        (tools/name).write_bytes(name.encode())
    scripts=root/'scripts';scripts.mkdir()
    for name in ('derive_general_subset.py','prepare_mvs.py','prepare_mvs_masks.py','build_mvs_surface.py','export_blender.py'):
        (scripts/name).write_text('# fixture')
    old_plan=dict(schema='photo-demo-plan-v1',source=r.source_info(originals),prompt='object',object_id='obj',
        python=str(tools/'python.exe'),mvs_tool_root=str(tools),blender_exe=str(tools/'blender.exe'),model_id='0')
    save(run/'PLAN.json',old_plan)
    save(run/'INPUT_INVENTORY.json',{'source':str(originals),'files':old_plan['source']['files']})
    stages={}
    for name in ('input','sfm','segmentation'):
        entry_path=run/'logs'/name/'ENTRY.json';entry_path.parent.mkdir(parents=True)
        argv=['fixture',name]
        save(entry_path,dict(status='completed',reclaim_confirmed=True,root_exit_code=0,errors=[],log_errors=[],argv=argv,cwd=str(root),pid=1))
        stages[name]=dict(status='SUCCESS',argv=argv,entry_path=str(entry_path),reclaim_confirmed=True,exit_code=0)
    stages['sfm'].update(run_json=str(sfm/'run.json'),counts=counts)
    stages['segmentation'].update(run_dir=str(masks),counts=sam_counts)
    meta=dict(schema='photo-demo-pipeline-v1',status='RUNNING',exit_code=None,updated_utc='now',
        source=str(originals),prompt='object',object_id='obj',stages=stages)
    save(run/'pipeline.json',meta)
    return root,run,root/'outputs/continued',tools,meta


def commit(source):
    root,run,out,tools,meta=source
    r.write_resume_checkpoint(run,root=root)
    meta.update(status='PARTIAL',exit_code=1);save(run/'pipeline.json',meta)
    return r.plan(run,out,python=tools/'python.exe',tool_root=tools,blender_exe=tools/'blender.exe',root=root)


def test_producer_commit_and_readonly_plan(source):
    frozen=commit(source)
    assert frozen['status']=='READY' and frozen['upstream_reuse']==['input','sfm','segmentation']
    root,run,out,tools,meta=source
    before=(run/'SURFACE_RESUME_SOURCE.json').read_bytes()
    with pytest.raises(ValueError,match='already exists'):
        r.write_resume_checkpoint(run,root=root)
    assert before==(run/'SURFACE_RESUME_SOURCE.json').read_bytes() and not out.exists()


def test_historical_run_cannot_retrofit_checkpoint(source):
    root,run,out,tools,meta=source
    meta.update(status='PARTIAL',exit_code=1);save(run/'pipeline.json',meta)
    with pytest.raises(ValueError,match='running producer'):
        r.write_resume_checkpoint(run,root=root)
    with pytest.raises(FileNotFoundError):
        r.plan(run,out,python=tools/'python.exe',tool_root=tools,blender_exe=tools/'blender.exe',root=root)


@pytest.mark.parametrize('target', ['models/0/binary/images.bin','frames_map.jsonl'])
def test_committed_sfm_drift_refused(source,target):
    frozen=commit(source)
    Path(frozen['sfm_run'],target).write_bytes(b'changed')
    with pytest.raises(ValueError):r.execute(frozen)
    assert not Path(frozen['output']).exists()


def test_committed_mask_drift_refused(source):
    frozen=commit(source)
    Path(frozen['mask_manifests'][0]).parent.joinpath('0.mask.png').write_bytes(b'changed')
    with pytest.raises(ValueError):r.execute(frozen)


@pytest.mark.parametrize('change', ['running','reclaim','counts','source','checkpoint'])
def test_refusal_paths(source,change):
    frozen=commit(source);root,run,out,tools,meta=source
    if change=='running':meta['stages']['surface']={'status':'RUNNING'}
    if change=='reclaim':meta['stages']['sfm']['reclaim_confirmed']=False
    if change=='counts':meta['stages']['sfm']['counts']={'input_rows':3}
    if change=='source':Path(meta['source'],'0.png').write_bytes(b'changed')
    if change=='checkpoint':
        cp=r.read(run/'SURFACE_RESUME_SOURCE.json');cp['upstream_stages']['sfm']['counts']={};save(run/'SURFACE_RESUME_SOURCE.json',cp)
    save(run/'pipeline.json',meta)
    with pytest.raises(ValueError):r.execute(frozen)
    assert not out.exists()


def test_stub_execution_calls_only_surface_and_keeps_old_run(source):
    frozen=commit(source);root,run,out,tools,meta=source
    before={p:_binding(p) for p in run.rglob('*') if p.is_file()}
    calls=[]
    def executor(root,out,name,argv,**limits):
        calls.append((name,limits))
        return dict(status='completed',root_exit_code=0,reclaim_confirmed=True,errors=[],log_errors=[])
    def builder(**kwargs):
        assert kwargs['sfm_run']==Path(frozen['sfm_run'])
        assert kwargs['stage']('native_subset',['stub'],180,2<<30)==0
        geometry=kwargs['run'];geometry.mkdir()
        ply=geometry/'mesh.ply';ply.write_bytes(b'mesh');blend=geometry/'model.blend';blend.write_bytes(b'blend')
        return dict(exit_code=0,surface={'status':'SUCCESS','artifacts':[_binding(ply)]},
                    blender={'status':'SUCCESS',**_binding(blend),'reopen_verified':True},stages={})
    result=r.execute(frozen,executor=executor,surface_builder=builder)
    assert result['status']=='SUCCESS' and result['deliverable_ready'] is True
    assert [c[0] for c in calls]==['native_subset']
    assert all(_binding(p)==binding for p,binding in before.items())
    with pytest.raises(ValueError,match='already exists'):r.execute(frozen,executor=executor,surface_builder=builder)


def test_failed_supervision_stops_no_retry_and_reports_failure(source):
    frozen=commit(source);calls=[]
    def executor(*args,**kw):
        calls.append(args[2]);return {'status':'completed','root_exit_code':0,'reclaim_confirmed':False}
    def builder(**kwargs):
        kwargs['stage']('native_subset',['stub'],180,2<<30)
        kwargs['stage']('mvs_bridge',['stub'],210,2<<30)
    result=r.execute(frozen,executor=executor,surface_builder=builder)
    assert result['status']=='FAILED' and result['deliverable_ready'] is False
    assert calls==['native_subset'] and 'after stop' in result['error']


def test_builder_exception_preserved(source):
    frozen=commit(source)
    def fail(**kw):raise RuntimeError('fixture surface failed')
    result=r.execute(frozen,surface_builder=fail)
    assert result['status']=='FAILED' and 'fixture surface failed' in result['error']


def test_failed_downstream_after_commit_can_resume_but_must_be_reclaimed(source):
    root,run,out,tools,meta=source
    r.write_resume_checkpoint(run,root=root)
    entry_path=run/'logs/native_subset/ENTRY.json';entry_path.parent.mkdir(parents=True)
    argv=['fixture','native_subset']
    save(entry_path,dict(status='completed',reclaim_confirmed=True,root_exit_code=4,
        errors=[],log_errors=[],argv=argv,cwd=str(root),pid=9))
    meta['stages']['native_subset']=dict(status='EXECUTED',argv=argv,entry_path=str(entry_path),
        reclaim_confirmed=True,exit_code=4)
    meta['stages']['surface']={'status':'FAILED'}
    meta.update(status='FAILED',exit_code=4);save(run/'pipeline.json',meta)
    frozen=r.plan(run,out,python=tools/'python.exe',tool_root=tools,blender_exe=tools/'blender.exe',root=root)
    assert frozen['status']=='READY'
    assert str(entry_path) in {b['path'] for b in frozen['bindings']}
    entry=r.read(entry_path);entry['reclaim_confirmed']=False;save(entry_path,entry)
    with pytest.raises(ValueError,match='reclamation'):
        r.execute(frozen)
    assert not out.exists()


def test_partial_upstream_not_promoted_to_full_success(source):
    root,run,out,tools,meta=source
    meta['stages']['sfm'].update(status='PARTIAL',exit_code=1)
    entry_path=Path(meta['stages']['sfm']['entry_path'])
    entry=r.read(entry_path);entry['root_exit_code']=1;save(entry_path,entry)
    sfm_path=Path(meta['stages']['sfm']['run_json'])
    sfm=r.read(sfm_path);sfm['status']='PARTIAL';save(sfm_path,sfm)
    save(run/'pipeline.json',meta)
    frozen=commit(source)
    def builder(**kwargs):
        geometry=kwargs['run'];geometry.mkdir()
        ply=geometry/'mesh.ply';ply.write_bytes(b'mesh');blend=geometry/'model.blend';blend.write_bytes(b'blend')
        return dict(exit_code=0,surface={'status':'SUCCESS','artifacts':[_binding(ply)]},
            blender={'status':'SUCCESS',**_binding(blend),'reopen_verified':True},stages={})
    result=r.execute(frozen,surface_builder=builder)
    assert result['status']=='PARTIAL' and result['exit_code']==1 and result['deliverable_ready']
    assert result['reused_stage_statuses']['sfm']=='PARTIAL'


@pytest.mark.parametrize('code',[5,130])
def test_supervisor_stop_code_not_replaced_by_generic_builder_error(source,code):
    frozen=commit(source);calls=[]
    def executor(*args,**kw):
        calls.append(args[2])
        return dict(status='terminated',root_exit_code=0,reclaim_confirmed=True,errors=[],log_errors=[],
            termination_reason='wall_clock_exceeded' if code==5 else 'user_cancelled',
            cancellation_requested=code==130)
    def builder(**kwargs):
        assert kwargs['stage']('native_subset',['stub'],180,2<<30)==code
        raise RuntimeError('generic builder wrapper')
    result=r.execute(frozen,executor=executor,surface_builder=builder)
    assert result['exit_code']==code and calls==['native_subset']
    assert result['status']==('CANCELLED' if code==130 else 'FAILED')


def test_geometry_description_can_exceed_small_control_record_limit(source):
    root,run,out,tools,meta=source
    sfm=Path(meta['stages']['sfm']['run_json']).parent
    description=sfm/'models/0/descriptions.json'
    description.write_text(json.dumps({'source':{},'binary':{},'text':{}})+' '*(9<<20))
    frozen=commit(source)
    bound=next(row for row in frozen['bindings'] if row['path']==str(description))
    assert bound['bytes']>8<<20 and frozen['status']=='READY'
