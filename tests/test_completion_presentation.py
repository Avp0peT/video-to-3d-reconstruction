"""Presentation orchestration and Blender-independent visibility/material guards."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from integrations.blender_export import completion_presentation as p


def fixture(tmp_path,monkeypatch,mutation=None):
    monkeypatch.setattr(p,'OUTPUTS',tmp_path)
    base=tmp_path/'base.blend';base.write_bytes(b'original blend')
    sources=[];scenes=[]
    for i,title in enumerate(('Inferred complete envelope','Observed-depth reconstruction')):
        path=tmp_path/('layer'+str(i)+'.ply');path.write_bytes(('mesh'+str(i)).encode());sources.append(path)
        scenes.append(dict(title=title,source_path=str(path),sha256=p._binding(path)['sha256'],vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],evidence=[]))
    payload=tmp_path/'payload.json';payload.write_text(json.dumps(dict(scenes=scenes)))
    executable=tmp_path/'blender.exe';executable.write_bytes(b'fake executable never launched')
    monkeypatch.setattr(p,'find_blender',lambda:executable)
    calls=[]
    def fake(argv,**kw):
        calls.append((argv,kw))
        assert kw['wall_seconds']==90 and kw['memory_bytes']==4<<30 and kw['output_bytes']==128<<20
        assert Path(argv[1]).name=='isolated_launcher.py'
        request=json.loads(Path(argv[2]).read_text());args=request['blender_argv'];items=args[args.index('--')+1:]
        mode,basepath,pay,sha,blend,target,legacy,legacysha=items
        assert '--background' in args and '--disable-autoexec' in args
        assert request['scratch'].startswith(str(tmp_path/'runtime_scratch'))
        assert request['child_timeout_seconds']==80
        if mode=='create':Path(blend).write_bytes(b'styled blend')
        report=dict(status='PASS',mode=mode,payload_sha256=sha,scenes=[dict(title=s['title'],vertices=3,faces=1) for s in scenes],presentation=[dict(title=s['title'],hidden=i==1) for i,s in enumerate(scenes)])
        if mutation=='readback' and mode=='verify':report['presentation'][1]['hidden']=False
        if mutation=='blend' and mode=='verify':Path(blend).write_bytes(b'changed')
        if mutation=='base' and mode=='create':base.write_bytes(b'changed original')
        Path(target).write_text(json.dumps(report))
        return dict(status='completed',root_exit_code=1 if mutation=='native' else 0,reclaim_confirmed=True,errors=[],log_errors=[])
    monkeypatch.setattr(p,'run_supervised',fake)
    return base,payload,tmp_path/'styled',tmp_path/'runtime_scratch/task',calls


def test_two_isolated_stages_bind_and_independently_verify_without_base_change(tmp_path,monkeypatch):
    base,payload,out,scratch,calls=fixture(tmp_path,monkeypatch)
    before=base.read_bytes()
    result=p.style_completion_blend(base,payload,out,scratch_root=scratch)
    assert result['status']=='SUCCESS' and result['readback_verified']
    assert len(calls)==2
    assert base.read_bytes()==before
    assert (out/'model.blend').read_bytes()==b'styled blend'
    assert (out/'CREATE.json').is_file() and (out/'VERIFY.json').is_file() and (out/'BINDINGS.json').is_file()
    assert result['geometry_changed'] is False and result['metadata_changed'] is False
    assert all(item['supervision']['reclaim_confirmed'] for item in result['stages'])


@pytest.mark.parametrize('mutation',['readback','blend','base','native'])
def test_changed_visibility_geometry_input_or_native_failure_cannot_claim_success(tmp_path,monkeypatch,mutation):
    base,payload,out,scratch,calls=fixture(tmp_path,monkeypatch,mutation)
    result=p.style_completion_blend(base,payload,out,scratch_root=scratch)
    assert result['status']=='FAILED_PRESENTATION_NOT_ADOPTED'
    assert result['geometry_accepted'] is False
    if mutation in ('base','native'):assert len(calls)==1


def test_invalid_titles_or_scratch_fail_before_any_process(tmp_path,monkeypatch):
    base,payload,out,scratch,calls=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='runtime_scratch'):p.style_completion_blend(base,payload,out,scratch_root=out/'temp')
    value=json.loads(payload.read_text());value['scenes'][0]['title']='unlabelled geometry';payload.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='titles'):p.style_completion_blend(base,payload,out,scratch_root=scratch)
    assert not calls and not out.exists()


def fake_worker(monkeypatch):
    objects={};collections={};scenes=[]
    for i,(title,(name,color,hidden)) in enumerate([('Inferred complete envelope',('INFERRED_COMPLETION',(1.,.38,.06,1.),False)),('Observed-depth reconstruction',('OBSERVED_DEPTH',(.1,.4,.9,1.),True))]):
        node=SimpleNamespace(type='BSDF_PRINCIPLED',inputs={'Base Color':SimpleNamespace(default_value=color)})
        material=SimpleNamespace(name='mat'+str(i),use_nodes=True,diffuse_color=color,node_tree=SimpleNamespace(nodes=[node]))
        obj=SimpleNamespace(name='PLY_'+str(i+1).zfill(2),hide_render=hidden,hide_viewport=hidden,data=SimpleNamespace(materials=[material]))
        collection=SimpleNamespace(objects=[obj],hide_render=hidden,hide_viewport=hidden)
        obj.users_collection=[collection];objects[obj.name]=obj;collections[name]=collection;scenes.append(dict(title=title))
    monkeypatch.setitem(sys.modules,'bpy',SimpleNamespace(data=SimpleNamespace(objects=objects,collections=collections)))
    file=Path(p.__file__).with_name('completion_presentation_worker.py')
    spec=importlib.util.spec_from_file_location('_fake_presentation_worker',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,scenes,objects,collections


def test_worker_style_checks_hidden_observation_collection_and_material(monkeypatch):
    worker,scenes,objects,collections=fake_worker(monkeypatch)
    report=worker.check_style(scenes)
    assert report[0]['hide_viewport'] is False and report[1]['hide_viewport'] is True
    objects['PLY_02'].hide_viewport=False
    with pytest.raises(ValueError,match='visibility'):worker.check_style(scenes)
    objects['PLY_02'].hide_viewport=True
    objects['PLY_01'].data.materials[0].diffuse_color=(0.,0.,0.,1.)
    with pytest.raises(ValueError,match='color'):worker.check_style(scenes)
