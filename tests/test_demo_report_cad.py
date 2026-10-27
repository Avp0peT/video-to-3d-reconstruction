"""Report authorization joins; fake geometry is never passed off as native CAD."""
from integrations.demo_pipeline import cad,report
import pytest


@pytest.mark.parametrize('fault',[None,'path','native_reject','failed'])
def test_report_cad_requires_bound_same_run_native_verification(tmp_path,monkeypatch,fault):
    calls=[]
    refs=[{'path':str(tmp_path/'cad/cad/cad'/name),'bytes':1,'sha256':'a'*64}
          for name in ('model.FCStd','model.step')]
    def check(output,run):
        calls.append((output,run))
        if fault=='native_reject':raise ValueError('independent readback missing')
        return {'artifacts':refs,'scale_status':'unknown'}
    monkeypatch.setattr(cad,'verify_cad_output',check)
    pipeline={'stages':{'cad':{'status':'FAILED' if fault=='failed' else 'SUCCESS',
                             'output':str(tmp_path/('other' if fault=='path' else 'cad'))}}}
    warnings=[]
    html,result=report._cad_outputs(tmp_path,tmp_path/'report',pipeline,warnings)
    assert result['available'] is (fault is None)
    assert ('下载 .step' in html) is (fault is None)
    assert len(calls)==(0 if fault in ('path','failed') else 1)
    assert '真实尺度与实物精度未验证' in html


def test_report_display_metadata_cannot_invent_cad_success(tmp_path,monkeypatch):
    import json
    (tmp_path/'REPORT_DATA.json').write_text(json.dumps({'cad':{'status':'SUCCESS','output':str(tmp_path/'cad')}}))
    def forbidden(*a,**kw):raise AssertionError('no authoritative CAD stage')
    monkeypatch.setattr(cad,'verify_cad_output',forbidden)
    output=report.render_report(tmp_path)
    validation=json.loads((output.parent/'REPORT_VALIDATION.json').read_text())
    assert validation['cad']['available'] is False
    assert '下载 .step' not in output.read_text(encoding='utf-8')
