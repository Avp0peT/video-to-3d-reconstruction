"""Real orchestration on identity-bound tiny images, without video/model runs."""
import json
from pathlib import Path
from integrations.demo_pipeline import temporal_review as review
from test_temporal_review import chain


def test_review_writes_gap_diagnostic_even_when_no_point_can_be_seeded(chain,tmp_path):
    output=tmp_path/'review'
    result=review.review_video(chain['pipeline'],output,chain['discovery'])
    assert result['status']=='DIAGNOSTIC_COMPLETE'
    assert result['source_unchanged'] is True
    assert result['gap_recovery']['counts']['accepted_links']==0
    assert result['gap_recovery']['observations_filled_in_gaps']==0
    for name in ('RECOVERY.json','RECOVERY_ASSESSMENT.json','reassociation/index.html'):
        assert (output/name).is_file()
    original=json.loads((output/'TRACKS.json').read_text())
    recovery=json.loads((output/'RECOVERY.json').read_text())
    assessment=json.loads((output/'RECOVERY_ASSESSMENT.json').read_text())
    assert original['tracks']==recovery['joined_tracks']==[]
    assert assessment['model_refit'] is False
    assert result['geometry_modified'] is result['correspondence_verified'] is False
    assert 'href="reassociation/index.html"' in (output/'index.html').read_text(encoding='utf-8')


def test_recovery_failure_cannot_be_published_as_a_valid_reassociation(chain,tmp_path,monkeypatch):
    def fail(*args):raise ValueError('injected candidate budget error')
    monkeypatch.setattr(review,'reassociate_tracklets',fail)
    output=tmp_path/'failed'
    result=review.review_video(chain['pipeline'],output,chain['discovery'])
    assert result['status']=='FAILED' and 'candidate budget error' in result['reason']
    assert result['geometry_modified'] is False
    assert (output/'TRACKS.json').is_file() and (output/'GEOMETRY.json').is_file()
    assert not (output/'RECOVERY.json').exists() and not (output/'reassociation').exists()
    assert 'href="reassociation/index.html"' not in (output/'index.html').read_text(encoding='utf-8')
