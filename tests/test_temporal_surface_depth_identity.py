from types import SimpleNamespace
import pytest
from scripts import compare_temporal_surface as c


def fixture(tmp_path,monkeypatch,names):
    work=tmp_path/'work';work.mkdir()
    for name in names:(work/name).write_bytes(b'fixture')
    monkeypatch.setattr(c,'read_dmap',lambda p:SimpleNamespace(image_name_utf8=names[p.name],file_bytes=7,sha256='x'))
    monkeypatch.setattr(c,'_bind',lambda *a,**kw:None)
    calls=[];monkeypatch.setattr(c,'verify_depth',lambda d,cam:calls.append(d.image_name_utf8))
    return calls


def test_filename_different_from_scene_index_still_bound(tmp_path,monkeypatch):
    calls=fixture(tmp_path,monkeypatch,{'depth0080.dmap':'images/image_00000080.png','depth0007.dmap':'images/other.png'})
    path,d=c.find_camera_depth(tmp_path,dict(name='undistorted/images/image_00000080.png',index=7),[])
    assert path.name=='depth0080.dmap' and len(calls)==1


def test_ambiguous_embedded_image_rejected(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,{'depth0080.dmap':'image.png','depth0007.dmap':'image.png'})
    with pytest.raises(ValueError,match='duplicate'):c.find_camera_depth(tmp_path,dict(name='image.png'),[])


def test_missing_only_after_metadata_search(tmp_path,monkeypatch):
    calls=fixture(tmp_path,monkeypatch,{'depth0007.dmap':'wrong.png'})
    assert c.find_camera_depth(tmp_path,dict(name='image.png'),[])==(None,None)
    assert calls==[]
