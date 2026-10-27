"""Final stage wiring: controlled cancellation and one small stdlib child."""
from pathlib import Path
import signal
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline import pipeline, supervision


def test_interrupt_is_latched_and_handler_restored(tmp_path,monkeypatch):
    previous=signal.getsignal(signal.SIGINT)
    def controlled(*args,**kwargs):
        signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
        assert kwargs['cancel_requested']() is True
        return {'status':'user_cancelled','termination_reason':'user_cancelled',
                'root_exit_code':1,'reclaim_confirmed':True,'errors':[],'log_errors':[]}
    monkeypatch.setattr(supervision,'supervise',controlled)
    entry=pipeline._invoke(tmp_path,tmp_path,'fixture',[],seconds=4,memory=256<<20)
    assert signal.getsignal(signal.SIGINT)==previous
    assert pipeline.entry_code(entry)==130
    assert pipeline.read(tmp_path/'logs/fixture/ENTRY.json')['cancellation_requested'] is True


def test_handler_restored_if_supervisor_cannot_start(tmp_path,monkeypatch):
    previous=signal.getsignal(signal.SIGINT)
    def fail(*args,**kwargs):raise OSError('fixture before native creation')
    monkeypatch.setattr(supervision,'supervise',fail)
    with pytest.raises(OSError):pipeline._invoke(tmp_path,tmp_path,'fixture',[],seconds=4,memory=256<<20)
    assert signal.getsignal(signal.SIGINT)==previous


def test_real_normal_child_through_final_entry(tmp_path):
    entry=pipeline._invoke(ROOT,tmp_path,'fixture',[sys.executable,'-B','-c','print("DEMO_ENTRY_OK")'],
                           seconds=5,memory=256<<20)
    assert pipeline.entry_code(entry)==0 and entry['reclaim_confirmed'] is True
    assert entry['cancellation_requested'] is False
    assert 'DEMO_ENTRY_OK' in (tmp_path/'logs/fixture/stdout.txt').read_text()
