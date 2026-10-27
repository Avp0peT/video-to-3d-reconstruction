from scripts import run_region_mask_review as pipeline


def test_failed_consumption_never_starts_regional_native(monkeypatch):
    monkeypatch.setattr(pipeline.probe,'execute',lambda *a:dict(status='UNKNOWN'))
    def forbidden(*args):raise AssertionError('regional native must not start')
    monkeypatch.setattr(pipeline.regional,'execute',forbidden)
    r=pipeline.execute_gated(dict(path='p',sha256='p'),dict(path='r',sha256='r'))
    assert r['regional_status']=='NOT_RUN' and r['regional_launches']==0


def test_verified_consumption_keeps_regional_nonacceptance(monkeypatch):
    monkeypatch.setattr(pipeline.probe,'execute',lambda *a:dict(status='REFERENCE_MASK_CONSUMPTION_VERIFIED_NOT_GEOMETRY_ACCEPTANCE'))
    monkeypatch.setattr(pipeline.regional,'execute',lambda *a:dict(status='REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED',new_native_launch_attempts=4))
    r=pipeline.execute_gated(dict(path='p',sha256='p'),dict(path='r',sha256='r'))
    assert r['status']=='REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED' and r['regional_launches']==4
