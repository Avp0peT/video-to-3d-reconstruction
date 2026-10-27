"""Real producer-schema fixtures; never load a model or native backend."""
import json
from pathlib import Path
import pytest
from PIL import Image
from test_mask_review import _make_input
from kozlov_lab.reconstruction.colmap_adapter import load_upstream_run
from kozlov_lab.reconstruction.foreground import select_foreground


def test_selected_and_not_requested_complete_ledger(tmp_path):
    fixture = _make_input(tmp_path/'inputs')
    upstream = load_upstream_run(fixture.io)
    result = select_foreground(upstream, fixture.manifest, tmp_path/'ledger')
    assert result['selected_indices'] == [17]
    assert result['masks_by_index'][17].read_bytes() == (fixture.product/'masks/000017_00.png').read_bytes()
    rows = json.loads(result['ledger_path'].read_text())['rows']
    assert [(r['input_index'], r['producer_status'], r['selected']) for r in rows] == [(17,'OK',True),(34,'NOT_REQUESTED',False)]
    assert result['coverage']['upstream_count'] == 2
    with pytest.raises(ValueError, match='new'):
        select_foreground(upstream, fixture.manifest, tmp_path/'ledger')


def test_no_match_is_excluded_without_relabeling(tmp_path):
    fixture = _make_input(tmp_path/'inputs')
    row = fixture.document['frames'][0]
    row.update(status='NO_MATCH', candidates=[], selected_candidate_id=None, selected_mask=None, selection=None)
    fixture.recount(); fixture.save()
    result = select_foreground(load_upstream_run(fixture.io),fixture.manifest,tmp_path/'ledger')
    assert result['selected_indices'] == []
    ledger = json.loads(result['ledger_path'].read_text())['rows']
    assert ledger[0]['producer_status'] == ledger[0]['exclusion_reason'] == 'NO_MATCH'


def test_ambiguous_candidates_are_not_automatically_selected(tmp_path):
    fixture = _make_input(tmp_path/'inputs')
    row = fixture.document['frames'][0]
    row['candidates'].append(fixture.candidate(17, 1))
    row.update(status='NEEDS_SELECTION', selected_candidate_id=None, selected_mask=None, selection=None)
    fixture.recount(); fixture.save()
    result = select_foreground(load_upstream_run(fixture.io), fixture.manifest, tmp_path/'ledger')
    assert result['selected_indices'] == []
    assert json.loads(result['ledger_path'].read_text())['rows'][0]['producer_status'] == 'NEEDS_SELECTION'


def test_in_memory_upstream_identity_drift_refused(tmp_path):
    fixture = _make_input(tmp_path/'inputs')
    upstream = load_upstream_run(fixture.io)
    upstream.rows[0].frame_id = 'modified-upstream'
    with pytest.raises(ValueError):
        select_foreground(upstream, fixture.manifest, tmp_path/'ledger')
    assert not (tmp_path/'ledger').exists()


@pytest.mark.parametrize('damage', ['frame','source_sha','mask_sha','count','object','duplicate_json','empty_mask','labels'])
def test_corrupt_identity_or_mask_rejected(tmp_path, damage):
    fixture = _make_input(tmp_path/'inputs'); row = fixture.document['frames'][0]
    if damage == 'frame': row['frame_id']='wrong'
    elif damage == 'source_sha': row['source_image_sha256']='0'*64
    elif damage == 'mask_sha': row['selected_mask']['sha256']='0'*64
    elif damage == 'count': fixture.document['counts']['success']=2
    elif damage == 'object': row['candidates'][0]['object_id']='wrong-object'
    elif damage in ('empty_mask','labels'):
        mask=fixture.product/row['candidates'][0]['mask_path']
        Image.new('L',(5,5),0 if damage=='empty_mask' else 127).save(mask)
        import hashlib
        digest=hashlib.sha256(mask.read_bytes()).hexdigest()
        row['candidates'][0]['mask_sha256']=row['selected_mask']['sha256']=digest
        row['candidates'][0]['foreground_pixels']=0
    fixture.save()
    if damage == 'duplicate_json':
        fixture.manifest.write_text('{"contract_version":"object-mask-v1","contract_version":"object-mask-v1"}')
    with pytest.raises(ValueError):
        select_foreground(load_upstream_run(fixture.io),fixture.manifest,tmp_path/'ledger')
    assert not (tmp_path/'ledger').exists()


@pytest.mark.parametrize('field', ['root', 'source_io', 'backend'])
def test_non_object_manifest_fields_are_input_errors(tmp_path, field):
    fixture = _make_input(tmp_path/'inputs')
    if field == 'root':
        fixture.manifest.write_text('[]', encoding='utf-8')
    else:
        fixture.document[field] = []
        fixture.save()
    upstream = load_upstream_run(fixture.io)
    with pytest.raises(ValueError, match='must be an object'):
        select_foreground(upstream, fixture.manifest, tmp_path/'ledger')
    assert not (tmp_path/'ledger').exists()
