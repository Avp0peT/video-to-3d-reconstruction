from pathlib import Path
import pytest
from integrations.video_masks import region_surface, local_surface_adapter


@pytest.mark.parametrize('mode', [None, 'closed_cells'])
def test_explicit_footprint_routes_without_changing_default(monkeypatch, mode):
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED'}
    monkeypatch.setattr(local_surface_adapter, 'run', capture)
    argv = ['face-local-worker', '--source', 's', '--gate', 'g', '--output', 'o']
    if mode:
        argv += ['--footprint-mode', mode]
    assert region_surface.main(argv) == 0
    assert calls == [((Path('s'), Path('g'), Path('o')),
                     {} if mode is None else {'footprint_mode': mode})]


def test_no_silent_footprint_option_on_wrong_action():
    with pytest.raises(SystemExit) as error:
        region_surface.main(['run', '--source', 's', '--output', 'o',
                             '--footprint-mode', 'closed_cells'])
    assert error.value.code == 2


def test_invalid_mode_refused_before_output(tmp_path):
    with pytest.raises(ValueError, match='unknown footprint'):
        local_surface_adapter.run(tmp_path, tmp_path, tmp_path/'fresh', footprint_mode='anything')
    assert not (tmp_path/'fresh').exists()
