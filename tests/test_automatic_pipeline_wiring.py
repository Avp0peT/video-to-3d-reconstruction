"""Public pipeline wiring, with artifact-writing workers and no native execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from integrations.demo_pipeline import pipeline as p
from test_object_first_pipeline import rig, injected, renderer


def worker_stub(fault=None):
    basic, _, _ = injected()
    calls = []

    def execute(root, run, stage, argv, **limits):
        calls.append(stage)
        entry = dict(status='completed', root_exit_code=0, reclaim_confirmed=True, errors=[], log_errors=[])
        if stage in ('input', 'segmentation'):
            basic(root, run, stage, argv, **limits)
            if stage == 'segmentation':
                path = run/'segmentation/product/manifest.json'
                doc = p.read(path)
                config = p.read(run/'segmentation/CONFIG.json')
                doc.update(backend=dict(name='sam3'), backend_bindings=dict(revision='pinned'),
                           evidence_level='sam3_real_inference', status='PARTIAL')
                doc['object'].update(text_prompt=config['text_prompt'],
                                     submitted_options=dict(confidence_threshold=0.5))
                doc['frames'][1].update(status='NO_MATCH', candidates=[], selected_candidate_id=None,
                                         selected_mask=None, selection=None)
                doc['counts'].update(success=1, no_match=1)
                p.save(path, doc)
                outcome = p.read(run/'segmentation/PRODUCT_OUTCOME.json')
                outcome.update(status='PARTIAL', exit_code=1, counts=doc['counts'])
                p.save(run/'segmentation/PRODUCT_OUTCOME.json', outcome)
                entry['root_exit_code'] = 1
                if fault == 'memory_reason':
                    entry.update(termination_reason='job_memory_limit')
                elif fault == 'memory_exit':
                    entry['root_exit_code'] = 0xC0000017
        elif stage == 'segmentation_recovery':
            config = p.read(Path(argv[-1]))
            assert config['indices'] == [1]
            folder = Path(config['output_dir'])
            doc = deepcopy(p.read(run/'segmentation/product/manifest.json'))
            recovered = deepcopy(doc['frames'][0])
            source = deepcopy(doc['frames'][1])
            source.update(status='OK', candidates=recovered['candidates'], selected_mask=recovered['selected_mask'],
                          selected_candidate_id=recovered['selected_candidate_id'], selection=recovered['selection'])
            product = folder/'product'; product.mkdir()
            source_mask = run/'segmentation/product'/source['candidates'][0]['mask_path']
            (product/source_mask.name).write_bytes(source_mask.read_bytes())
            doc.update(run_id='sam-recovery', status='SUCCESS', frames=[source],
                       counts=dict(requested=1, success=1, no_match=0, needs_selection=0, error=0, not_run=0))
            doc['object']['text_prompt'] = config['text_prompt']
            p.save(product/'manifest.json', doc)
            p.save(folder/'PRODUCT_OUTCOME.json', dict(status='SUCCESS', exit_code=0, counts=doc['counts'], run_dir=str(product)))
        elif stage in ('sfm', 'sfm_constrained'):
            manifest = Path(argv[argv.index('--object-mask-manifest')+1])
            assert manifest == (run/'segmentation/recovered/manifest.json').resolve()
            assert all(f['selected_mask'] for f in p.read(manifest)['frames'])
            output = Path(argv[argv.index('--output')+1])/'fixture'
            (output/'sparse/0').mkdir(parents=True)
            p.save(output/'run.json', dict(status='PARTIAL',
                models=[dict(model_id='0', usable=True, roundtrip_verified=True, num_registered_images=2,
                             num_points3D=8, binary_path='sparse/0')],
                counts=dict(registered_unique_images=2, unique_images=2),
                foreground=dict(mask_consumption_verified=True)))
            entry['root_exit_code'] = 1
        elif stage in ('camera_domain_original', 'camera_domain_candidate'):
            valid = stage.endswith('candidate')
            p.save(Path(argv[argv.index('--output')+1]), dict(
                status='CAMERA_DOMAIN_VALID' if valid else 'CAMERA_DOMAIN_REJECTED',
                source_unchanged=True, source_model=str(Path(argv[argv.index('--model')+1]).resolve())))
            entry['root_exit_code'] = 0 if valid else 1
        else:
            pytest.fail('Unexpected stage: '+stage)
        log = run/'logs'/stage; log.mkdir(parents=True, exist_ok=True)
        p.save(log/'ENTRY.json', entry)
        return entry

    return execute, calls


def test_default_public_pipeline_recovers_and_selects_constrained_camera(rig, monkeypatch):
    root, options = rig
    assert options.automatic_recovery is True
    monkeypatch.setattr(p, 'make_viewer', lambda sfm_run, run: dict(status='PARTIAL', evidence_paths=[str(sfm_run/'run.json')]))
    executor, calls = worker_stub()
    run, _ = p.run_pipeline(options, root=root, executor=executor, renderer=renderer)
    assert calls == ['input','segmentation','segmentation_recovery','sfm',
                     'camera_domain_original','sfm_constrained','camera_domain_candidate']
    state = p.read(run/'pipeline.json')
    assert p.read(run/'PLAN.json')['automatic_recovery'] is True
    assert state['automatic_segmentation_recovery']['attempts'] == 1
    assert state['automatic_camera_recovery']['status'] == 'CONSTRAINED_DOMAIN_VALID'
    assert state['stages']['sfm']['selected_from'] == 'sfm_constrained'
    assert Path(state['stages']['sfm']['run_json']).is_relative_to(run/'camera_recovery')
    outcome = p.read(run/'segmentation/PRODUCT_OUTCOME.json')
    primary = p.read(run/'segmentation/PRIMARY_PRODUCT_OUTCOME.json')
    assert primary['status'] == 'PARTIAL' and outcome['status'] == 'SUCCESS'
    assert outcome['counts'] == state['stages']['segmentation']['counts']
    assert outcome['execution_stages'] == ['segmentation_primary','segmentation_recovery']
    assert p.read(Path(outcome['run_dir'])/'manifest.json')['counts'] == outcome['counts']
    for stage in calls:
        record = p.read(run/'logs'/stage/'ENTRY.json')
        recorded_stage = 'segmentation_primary' if stage == 'segmentation' else stage
        assert state['stages'][recorded_stage]['reclaim_confirmed'] == record['reclaim_confirmed']
    report = p.read(run/'REPORT_DATA.json')
    assert report['automatic_camera_recovery']['selected_sfm'] == state['stages']['sfm']['run_json']
    assert report['segmentation']['run_dir'] == outcome['run_dir']


@pytest.mark.parametrize('fault', ['memory_reason','memory_exit'])
def test_public_pipeline_memory_stop_never_launches_recovery_or_sfm(rig, fault):
    root, options = rig
    executor, calls = worker_stub(fault)
    run, code = p.run_pipeline(options, root=root, executor=executor, renderer=renderer)
    assert calls == ['input','segmentation']
    assert code == 5
    state = p.read(run/'pipeline.json')
    assert state['stages']['segmentation']['exit_code'] == 5
    assert state['stages']['sfm']['status'] == 'NOT_RUN'
    assert not (run/'segmentation/recovery_attempt').exists()
    assert not (run/'camera_recovery').exists()
