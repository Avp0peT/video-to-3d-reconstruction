"""TEST_DATA: concurrent temporary-file rotation; no native processes."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import stat
import pytest
from integrations.mvs_surface import supervision as s


def rename_once(monkeypatch, old, new):
    original=Path.lstat; events=[]
    def observed(path, *args, **kwargs):
        if path==old and not events:
            old.rename(new);events.append(str(old))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',observed)
    return events


def test_regular_file_rotation_returns_explicit_sample_and_skipped_path(tmp_path,monkeypatch):
    old=tmp_path/'depth0006.geo.dmap';old.write_bytes(b'123');(tmp_path/'steady').write_bytes(b'12345')
    events=rename_once(monkeypatch,old,tmp_path/'replacement.dmap')
    result=s.runtime_directory_observation(tmp_path)
    assert result['bytes']>=5 and result['bytes']<=8 and events
    assert result['skipped_entries']==[dict(path=str(old),reason='ENUMERATED_REGULAR_FILE_ENOENT')]
    assert result['semantics']=='NONATOMIC_OBSERVED_BYTES_WITH_DISAPPEARING_REGULAR_FILES' and not result['complete_snapshot']
    assert result['scan_attempts']==1


def test_strict_scan_still_refuses_disappearing_file(tmp_path,monkeypatch):
    old=tmp_path/'depth';old.write_bytes(b'123');rename_once(monkeypatch,old,tmp_path/'new')
    with pytest.raises(s.ObservationError):s.strict_directory_size(tmp_path)


def test_only_vanished_file_does_not_become_zero_on_error(tmp_path,monkeypatch):
    old=tmp_path/'depth';old.write_bytes(b'123');rename_once(monkeypatch,old,tmp_path/'new')
    with pytest.raises(s.ObservationError,match='unknown is not zero'):s.runtime_directory_size(tmp_path)


def test_empty_stable_directory_is_a_real_zero_sample(tmp_path):
    assert s.runtime_directory_size(tmp_path)==0


def test_permission_failure_is_not_skipped(tmp_path,monkeypatch):
    path=tmp_path/'depth';path.write_bytes(b'x');original=Path.lstat
    def denied(p,*a,**k):
        if p==path:raise PermissionError('TEST_DATA denied')
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'lstat',denied)
    with pytest.raises(s.ObservationError) as failure:s.runtime_directory_size(tmp_path)
    assert isinstance(failure.value.__cause__,PermissionError)


def test_other_io_failure_is_not_skipped(tmp_path,monkeypatch):
    path=tmp_path/'depth';path.write_bytes(b'x');original=Path.lstat
    def denied(p,*a,**k):
        if p==path:raise OSError(5,'TEST_DATA IO failure')
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'lstat',denied)
    with pytest.raises(s.ObservationError):s.runtime_directory_size(tmp_path)


def test_root_replacement_is_detected(tmp_path,monkeypatch):
    (tmp_path/'steady').write_bytes(b'123');original=s._runtime_root_identity;calls=[]
    def identity(p):
        result=original(p);calls.append(p)
        return result if len(calls)==1 else (result[0],result[1]+1)
    monkeypatch.setattr(s,'_runtime_root_identity',identity)
    with pytest.raises(s.ObservationError,match='identity changed'):s.runtime_directory_size(tmp_path)


def test_subdirectory_replacement_is_detected(tmp_path,monkeypatch):
    child=tmp_path/'child';child.mkdir();path=child/'entry';path.write_bytes(b'payload');original=Path.lstat;events=[]
    def changed(p,*a,**k):
        if p==path and not events:
            child.rename(tmp_path/'retired');child.mkdir();(child/'entry').write_bytes(b'replacement');events.append(True)
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'lstat',changed)
    with pytest.raises(s.ObservationError,match='identity changed'):s.runtime_directory_size(tmp_path)


def test_disappearing_subdirectory_is_not_a_file_rotation(tmp_path,monkeypatch):
    child=tmp_path/'child';child.mkdir();original=Path.lstat;events=[]
    def vanished(p,*a,**k):
        if p==child and not events:child.rmdir();events.append(True)
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'lstat',vanished)
    with pytest.raises(s.ObservationError):s.runtime_directory_size(tmp_path)


@pytest.mark.parametrize('kind',['symlink','reparse','hardlink'])
def test_forbidden_link_metadata_is_never_ignored(tmp_path,monkeypatch,kind):
    path=tmp_path/'entry';path.write_bytes(b'payload');original=Path.lstat
    def linked(p,*a,**k):
        info=original(p,*a,**k)
        if p!=path:return info
        return SimpleNamespace(st_mode=stat.S_IFLNK if kind=='symlink' else info.st_mode,
            st_file_attributes=s.REPARSE_POINT if kind=='reparse' else 0,
            st_nlink=2 if kind=='hardlink' else 1,st_size=info.st_size)
    monkeypatch.setattr(Path,'lstat',linked)
    with pytest.raises(s.ObservationError,match='forbidden'):s.runtime_directory_size(tmp_path)


def test_skipped_entry_bound_stops_instead_of_unlimited_rotation(tmp_path,monkeypatch):
    (tmp_path/'steady').write_bytes(b'123');old=tmp_path/'depth';old.write_bytes(b'abc')
    rename_once(monkeypatch,old,tmp_path/'new');monkeypatch.setattr(s,'RUNTIME_MAX_SKIPPED_ENTRIES',0)
    with pytest.raises(s.ObservationError,match='bound exceeded'):s.runtime_directory_size(tmp_path)


def test_private_supervisor_records_skips_and_pins_root_across_samples(tmp_path,monkeypatch):
    source=b'class JobObject: pass\nclass EventLog: pass\n'
    sp=tmp_path/'fixture.py';sp.write_bytes(source)
    monkeypatch.setattr(s,'SUPERVISOR_PATH',sp);monkeypatch.setattr(s,'SUPERVISOR_SHA256',hashlib.sha256(source).hexdigest())
    module,_=s._load_supervisor();root=tmp_path/'run';root.mkdir();old=root/'depth';old.write_bytes(b'abc');(root/'steady').write_bytes(b'12345')
    rename_once(monkeypatch,old,root/'new')
    assert module.directory_size(root)>=5
    report=module._mvs_directory_samples
    assert report['samples']==1 and len(report['skipped_samples'])==1
    assert report['skipped_samples'][0]['skipped_entry_count']==1
    root.rename(tmp_path/'retired');root.mkdir()
    with pytest.raises(s.ObservationError,match='between samples'):module.directory_size(root)


def test_skipped_sample_audit_bound_is_fail_closed(tmp_path,monkeypatch):
    source=b'class JobObject: pass\nclass EventLog: pass\n';sp=tmp_path/'fixture.py';sp.write_bytes(source)
    monkeypatch.setattr(s,'SUPERVISOR_PATH',sp);monkeypatch.setattr(s,'SUPERVISOR_SHA256',hashlib.sha256(source).hexdigest())
    module,_=s._load_supervisor();root=tmp_path/'run';root.mkdir();old=root/'depth';old.write_bytes(b'abc');(root/'steady').write_bytes(b'12345')
    rename_once(monkeypatch,old,root/'new');monkeypatch.setattr(s,'RUNTIME_MAX_SKIP_SAMPLES',0)
    with pytest.raises(s.ObservationError,match='audit bound'):module.directory_size(root)
