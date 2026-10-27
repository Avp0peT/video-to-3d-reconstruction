"""TEST_DATA orchestration only. No model, reconstruction or Blender execution."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline import pipeline as p


@pytest.fixture
def rig(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    source=tmp_path/'photos';source.mkdir()
    for i in range(3):(source/f'{i}.png').write_bytes(b'TEST_DATA'+bytes([i]))
    exe=root/'python.exe';exe.write_bytes(b'TEST_EXE')
    weight=root/'weight.pt';weight.write_bytes(b'TEST_WEIGHT')
    package=root/'package';package.mkdir();p.save(package/'BUNDLE.json',{})
    tool=root/'tools';tool.mkdir()
    for name in ('InterfaceCOLMAP.exe','DensifyPointCloud.exe','ReconstructMesh.exe','blender.exe'):
        (tool/name).write_bytes(b'TEST_EXE')
    options=p.Options(source,'test object',output=root/'outputs/run',python=exe,sam3_python=exe,
        checkpoint=weight,sam3_package=package,mvs_tool_root=tool,blender_exe=tool/'blender.exe',
        automatic_recovery=False,view_geometry_audit=False,post_review=False,research_code6=False)
    # This fixture isolates export orchestration and has no native camera/model.
    # Automatic recovery wiring has its own full branch fixture.
    monkeypatch.setattr(p,'make_viewer',lambda *args:{'status':'SUCCESS','summary':'TEST_DATA'})
    return root,options


def native_fixture(*,seg='SUCCESS',sfm='SUCCESS',reclaim=True,mutate=None):
    calls=[]
    def execute(root,run,name,argv,**kw):
        calls.append(name)
        if name=='input':
            io=run/'input/photo_io/run';io.mkdir(parents=True)
            p.save(io/'run.json',{'status':'SUCCESS'})
            manifest=run/'input/photo_manifest.json';p.save(manifest,{'contract_version':'photo-io-v1'})
            p.save(run/'input/RESULT.json',{'status':'INPUTS_PREPARED','photo_io':{
                'run':{'path':str(io/'run.json'),'sha256':p.sha(io/'run.json')},
                'manifest':{'path':str(manifest),'sha256':p.sha(manifest)},
                'identities':[{'input_index':i} for i in (0,1,2)]}})
        elif name=='sfm':
            out=Path(argv[argv.index('--output')+1])/'native';out.mkdir(parents=True)
            p.save(out/'run.json',{'status':sfm,'counts':{},'models':[{'usable':True}]})
        elif name=='segmentation':
            out=run/'segmentation/product_run/native';out.mkdir(parents=True)
            p.save(out/'manifest.json',{'status':seg,'frames':[{'input_index':i,'status':'OK','selected_mask':{'path':f'{i}.mask.png'}} for i in (0,1,2)]})
            p.save(run/'segmentation/PRODUCT_OUTCOME.json',{'status':seg,'run_dir':str(out),'counts':{'requested':3}})
            if mutate: mutate(run)
        code=1 if (name=='sfm' and sfm=='NO_MODEL') or (name=='segmentation' and seg=='PARTIAL') else 0
        if name=='segmentation' and seg=='RESOURCE_LIMIT':code=5
        return {'status':'completed','root_exit_code':code,'reclaim_confirmed':reclaim,
                'errors':[],'log_errors':[]}
    return execute,calls


def renderer(run):
    out=run/'report';out.mkdir();(out/'index.html').write_text('TEST_DATA')
    return out/'index.html'


def good_surface(**kwargs):
    assert kwargs['mask_manifests']==[kwargs['run'].parent/'segmentation/product_run/native/manifest.json']
    assert kwargs['dataset_manifest']==kwargs['run'].parent/'input/photo_manifest.json'
    kwargs['run'].mkdir(parents=True)
    mesh=kwargs['run']/'mesh.ply';mesh.write_bytes(b'TEST_DATA_MESH')
    blend=kwargs['run']/'model.blend';blend.write_bytes(b'TEST_DATA_BLEND')
    return {'status':'SUCCESS','exit_code':0,'stages':{'mvs_masks':{'status':'SUCCESS'}},
        'surface':{'status':'SUCCESS','selected_indices':[0,1],'missing_indices':[2],
                   'artifacts':[{'path':str(mesh),'sha256':p.sha(mesh)}]},
        'blender':{'status':'SUCCESS','reopen_verified':True,'path':str(blend),'sha256':p.sha(blend)}}


def test_full_target_declares_all_deliverables_without_launching(rig):
    root,opts=rig;plan=p.plan(opts,root)
    assert plan['target']=='blender' and 'surface' in plan['stages'] and 'blender' in plan['stages']
    assert all(plan['dependencies'].values()) and not opts.output.exists()
    assert 'never fill' in plan['missing_mask_policy']


def test_pipeline_calls_downstream_after_bound_inputs(rig):
    root,opts=rig;execute,calls=native_fixture();seen=[]
    def build(**kw):seen.append(kw);return good_surface(**kw)
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,surface_builder=build)
    state=p.read(out/'pipeline.json')
    assert code==0 and state['status']=='SUCCESS' and state['deliverable_ready'] is True
    assert calls==['input','segmentation','sfm'] and len(seen)==1
    assert seen[0]['stage'] is not None


def test_partial_segmentation_can_produce_viewable_subset_but_remains_partial(rig):
    root,opts=rig;execute,_=native_fixture(seg='PARTIAL')
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,surface_builder=good_surface)
    state=p.read(out/'pipeline.json')
    assert code==1 and state['status']=='PARTIAL' and state['deliverable_ready'] is True
    assert state['stages']['surface']['missing_indices']==[2]


@pytest.mark.parametrize('sfm,seg,expected',[('NO_MODEL','SUCCESS',1),('SUCCESS','RESOURCE_LIMIT',5)])
def test_unavailable_or_resource_stopped_prerequisite_never_starts_surface(rig,sfm,seg,expected):
    root,opts=rig;execute,_=native_fixture(sfm=sfm,seg=seg)
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,
        surface_builder=lambda **kw:pytest.fail('downstream launched'))
    assert code==expected and p.read(out/'pipeline.json')['deliverable_ready'] is False


def test_missing_blender_refuses_before_any_native_launch(rig):
    root,opts=rig;opts.blender_exe.unlink();execute,calls=native_fixture()
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer)
    assert code==3 and not calls and p.read(out/'pipeline.json')['status']=='FAILED'


def test_changed_source_stops_before_geometry(rig):
    root,opts=rig
    execute,_=native_fixture(mutate=lambda run:(opts.source/'0.png').write_bytes(b'DRIFT'))
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,
        surface_builder=lambda **kw:pytest.fail('downstream launched after drift'))
    assert code==2 and p.read(out/'pipeline.json')['deliverable_ready'] is False


def test_segmentation_run_cannot_escape_its_stage(rig,tmp_path):
    root,opts=rig
    def drift(run):
        path=run/'segmentation/PRODUCT_OUTCOME.json';doc=p.read(path);doc['run_dir']=str(tmp_path);p.save(path,doc)
    execute,_=native_fixture(mutate=drift)
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,
        surface_builder=lambda **kw:pytest.fail('unbound mask source'))
    assert code==2 and any('escapes' in e for e in p.read(out/'pipeline.json')['errors'])


def test_blender_failure_retains_surface_and_cannot_be_ready(rig):
    root,opts=rig;execute,_=native_fixture()
    def fail(**kw):
        result=good_surface(**kw);result.update(status='FAILED',exit_code=4)
        result['blender']={'status':'FAILED'};return result
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,surface_builder=fail)
    state=p.read(out/'pipeline.json')
    assert code==4 and state['stages']['surface']['status']=='SUCCESS' and not state['deliverable_ready']


def test_native_no_reclaim_never_enters_downstream(rig):
    root,opts=rig;execute,calls=native_fixture(reclaim=False)
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,
        surface_builder=lambda **kw:pytest.fail('unreclaimed work followed by geometry'))
    assert calls==['input'] and code==4 and not p.read(out/'pipeline.json')['deliverable_ready']


@pytest.mark.parametrize('fault',['missing_reopen','bad_hash','outside','empty_artifacts'])
def test_metadata_success_never_substitutes_deliverable_evidence(rig,fault):
    root,opts=rig;execute,_=native_fixture()
    def corrupt(**kw):
        result=good_surface(**kw)
        if fault=='missing_reopen':result['blender']['reopen_verified']=False
        elif fault=='bad_hash':result['blender']['sha256']='0'*64
        elif fault=='outside':
            path=opts.source/'0.png';result['blender'].update(path=str(path),sha256=p.sha(path))
        else:result['surface']['artifacts']=[]
        return result
    out,code=p.run_pipeline(opts,root=root,executor=execute,renderer=renderer,surface_builder=corrupt)
    assert code==4 and not p.read(out/'pipeline.json')['deliverable_ready']
