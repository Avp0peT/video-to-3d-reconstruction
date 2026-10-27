"""Explicit CAD UI action and rerun semantics, no native execution."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT/'app/pages/11_Photo_to_3D.py'


@pytest.mark.parametrize('eligible',[True,False])
def test_continuation_cad_button_is_explicit_and_rerun_never_executes(tmp_path,monkeypatch,eligible):
    from integrations.demo_pipeline import cad
    source=tmp_path/'continued';source.mkdir()
    refs=[]
    for name in ('model.FCStd','model.step'):
        p=tmp_path/name;raw=('synthetic-'+name).encode();p.write_bytes(raw)
        refs.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    calls=[]
    def plan(*a,**kw):
        if not eligible:raise ValueError('source continuation lacks verified geometry')
        return {'schema':'synthetic-plan-only'}
    def execute(run):
        calls.append(str(run))
        return {'source_run':str(run),'task':str(tmp_path/'cad_task'),'output':str(tmp_path/'delivery'),'supervision':{}}
    fixture=SimpleNamespace(execute=execute,verified=lambda saved,run:{'artifacts':refs})
    monkeypatch.setitem(sys.modules,'_cad_ui_fixture',fixture)
    monkeypatch.setattr(cad,'plan',plan)
    script='\n'.join([
        'import runpy', 'from pathlib import Path', 'import _cad_ui_fixture as fixture',
        'module=runpy.run_path('+repr(str(PAGE))+',run_name="test_page")',
        'render=module["render_continuation_cad"]',
        'render.__globals__["execute_cad"]=fixture.execute',
        'render.__globals__["verified_cad"]=fixture.verified',
        'render(Path('+repr(str(source))+'))',
    ])
    app=AppTest.from_string(script,default_timeout=15).run()
    assert not app.exception and not calls
    button=app.button(key='continued_surface_cad')
    assert button.disabled is (not eligible)
    if eligible:
        button.click().run()
        assert not app.exception and calls==[str(source)]
        assert len(app.get('download_button'))==2
        app.run()
        assert calls==[str(source)] and len(app.get('download_button'))==2
        # Actual download bytes are rechecked after a successful earlier result.
        Path(refs[0]['path']).write_bytes(b'changed')
        app.run()
        assert calls==[str(source)] and not app.get('download_button')
        assert any('下载核验未通过' in warning.value for warning in app.warning)
    else:
        app.run();assert not calls and not app.get('download_button')
        assert any('条件未满足' in caption.value for caption in app.caption)
