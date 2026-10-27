import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline import partial_reconstruction as p


def pairs(n=15):
    return [(1,2,[[i,i] for i in range(n)]),(2,3,[[i,i] for i in range(n)])]


def test_graph_three_view_and_isolate():
    result=p.graph_plan([1,2,3,4],pairs())
    assert result['groups'][0]['scheduled'] and result['groups'][0]['three_view_tracks']==15
    assert result['groups'][1]['image_ids']==[4] and not result['groups'][1]['scheduled']


def test_many_to_one_component_refused_not_silently_deduplicated():
    source=pairs();source.append((1,3,[[i,(i+1)%15] for i in range(15)]))
    result=p.graph_plan([1,2,3],source)
    assert not result['groups'][0]['scheduled']
    assert result['groups'][0]['conflicting_tracks']==1
    assert len(result['tracks'][0]['elements'])==45


def test_two_view_only_never_scheduled():
    r=p.graph_plan([1,2],pairs()[:1])
    assert not r['groups'][0]['scheduled']


@pytest.mark.parametrize('ids',[[True,2,3],[1,1,2],[],[0,2,3]])
def test_invalid_image_ids(ids):
    with pytest.raises(ValueError):p.graph_plan(ids,[])


def test_mask_hole_rejects_face_without_editing_inputs():
    uv=[[2,2],[8,2],[2,8]]
    points=[dict(observations=[dict(image_id=i,raw_pixels=q) for i in (1,2,3)]) for q in uv]
    mesh=dict(faces=[[0,1,2]],face_support=[[1,2,3]])
    masks={i:np.ones((12,12),dtype=bool) for i in (1,2,3)}
    assert p.supported_mask_faces(points,mesh,masks)[0]==[[0,1,2]]
    masks[3][4,4]=False
    before=copy.deepcopy(points)
    assert p.supported_mask_faces(points,mesh,masks)[0]==[]
    assert points==before and not masks[3][4,4]


def test_ply_preserves_coordinates_faces_and_color(tmp_path):
    from integrations.viewer.ply_data import read_ply
    points=[dict(xyz=x,color=[20,30,40]) for x in ([.123456789012345,0,1],[1,0,1],[0,1,1])]
    path=tmp_path/'piece.ply';p.write_ply(path,points,[[0,1,2]])
    read=read_ply(path)
    assert read['vertices']==[x['xyz'] for x in points] and read['faces']==[[0,1,2]]
    assert read['colors']==[x['color'] for x in points]


def test_entry_plan_only_does_not_invoke_native(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('partial_entry',ROOT/'scripts/reconstruct_partial.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(p,'prepare',lambda x:(dict(image_root=str(source),graph=dict(groups=[]),source_bindings=[]),{}))
    from integrations.demo_pipeline import pipeline
    monkeypatch.setattr(pipeline,'_invoke',lambda *a,**kw:pytest.fail('native invoked in plan-only'))
    assert module.main(['--sfm-run',str(source),'--output',str(tmp_path/'out')])==0
    assert (tmp_path/'out/PLAN.json').is_file() and not (tmp_path/'out/local').exists()


def test_child_bootstrap_can_import_bound_source():
    import subprocess
    spec=importlib.util.spec_from_file_location('partial_entry_bootstrap',ROOT/'scripts/reconstruct_partial.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bootstrap=module.CHILD.split('from integrations.demo_pipeline.partial_reconstruction import run')[0]
    code=bootstrap+"\nimport kozlov_lab\nassert Path(kozlov_lab.__file__).resolve()==Path.cwd()/'src/kozlov_lab/__init__.py'\n"
    child=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10)
    assert child.returncode==0,child.stderr
