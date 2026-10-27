"""Runtime metadata counterexamples; never native reconstruction."""
from pathlib import Path
from types import SimpleNamespace
import stat
import pytest
from integrations.mvs_surface import supervision as s


def install(monkeypatch, path, sequence):
    original = Path.lstat
    calls = []
    def query(p, *args, **kwargs):
        info = original(p, *args, **kwargs)
        if p != path:
            return info
        action = sequence[min(len(calls), len(sequence)-1)]
        calls.append(action)
        if action == 'missing':
            raise FileNotFoundError(str(p))
        if action == 'denied':
            raise PermissionError(str(p))
        return SimpleNamespace(st_mode=stat.S_IFDIR if action == 'dir' else info.st_mode,
            st_nlink=2 if action == 'hardlink' else 0 if action == 'zero' else 1,
            st_ino=0 if action == 'zero' else info.st_ino, st_dev=info.st_dev,
            st_size=info.st_size, st_file_attributes=s.REPARSE_POINT if action == 'reparse' else 0)
    monkeypatch.setattr(Path, 'lstat', query)
    return calls


def test_zero_then_valid_is_reobserved_and_counted(tmp_path, monkeypatch):
    p=tmp_path/'depth';p.write_bytes(b'123')
    calls=install(monkeypatch,p,['zero','valid'])
    row=s.runtime_directory_observation(tmp_path)
    assert calls == ['zero','valid'] and row['bytes']==3
    assert row['metadata_refresh_count']==1 and row['skipped_entry_count']==0
    assert row['metadata_refreshes'][0]['outcome']=='VALID_REGULAR_METADATA'


@pytest.mark.parametrize('bad',['zero','hardlink','dir','reparse','denied'])
def test_refresh_does_not_allow_invalid_or_unknown(tmp_path,monkeypatch,bad):
    p=tmp_path/'depth';p.write_bytes(b'x')
    calls=install(monkeypatch,p,['zero',bad])
    with pytest.raises(s.ObservationError):s.runtime_directory_observation(tmp_path)
    assert calls==['zero',bad]


def test_zero_then_missing_retains_unknown_bytes(tmp_path,monkeypatch):
    p=tmp_path/'depth';p.write_bytes(b'123');(tmp_path/'steady').write_bytes(b'abcde')
    install(monkeypatch,p,['zero','missing'])
    row=s.runtime_directory_observation(tmp_path)
    assert row['bytes']==5 and row['skipped_entry_count']==1
    assert row['metadata_refreshes'][0]['outcome']=='CONFIRMED_DISAPPEARED_UNKNOWN_BYTES'


def test_strict_scan_and_real_hardlinks_not_relaxed(tmp_path,monkeypatch):
    p=tmp_path/'depth';p.write_bytes(b'x');calls=install(monkeypatch,p,['zero','valid'])
    with pytest.raises(s.ObservationError):s.strict_directory_size(tmp_path)
    assert calls==['zero']


def test_zero_refresh_bound(tmp_path,monkeypatch):
    p=tmp_path/'depth';p.write_bytes(b'x');install(monkeypatch,p,['zero','valid'])
    monkeypatch.setattr(s,'RUNTIME_MAX_SKIPPED_ENTRIES',0)
    with pytest.raises(s.ObservationError,match='bound'):s.runtime_directory_observation(tmp_path)
