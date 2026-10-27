"""Exercise bounded orchestration with local artifacts and native-stage stubs."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from integrations.demo_pipeline.automatic_recovery import recover_segmentation, select_camera_solution


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def mask_product(folder, image, rows, prompt='glove'):
    frames = []
    for idx, status, intensity in rows:
        candidates = []
        if status == 'OK':
            path = folder / f'masks/{idx}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('L', (2, 2), 255).save(path)
            # A distinguishing immutable metadata value also survives composition.
            candidates = [dict(candidate_id=f'candidate-{intensity}', mask_path=f'masks/{idx}.png',
                               mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), foreground_pixels=4)]
        frames.append(dict(input_index=idx, frame_id=f'frame-{idx}', source_image=str(image),
                           source_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                           photo_io_image_sha256='a'*64, width=2, height=2, status=status, candidates=candidates,
                           selected_candidate_id=candidates[0]['candidate_id'] if candidates else None,
                           selected_mask=dict(path=candidates[0]['mask_path'], sha256=candidates[0]['mask_sha256']) if candidates else None,
                           selection=dict(method='unique_candidate', source='automatic', mask_source='automatic_sam3') if candidates else None))
    counts = dict(requested=len(rows), success=sum(f['status']=='OK' for f in frames),
                  no_match=sum(f['status']=='NO_MATCH' for f in frames), needs_selection=0, error=0, not_run=0)
    status = 'SUCCESS' if counts['success']==len(rows) else 'PARTIAL'
    save(folder/'manifest.json', dict(contract_version='object-mask-v1', run_id=folder.name,
         source_io=dict(path='canonical'), backend=dict(name='sam3'), backend_bindings=dict(revision='pinned'),
         object=dict(object_id='glove', text_prompt=prompt, submitted_options=dict(confidence_threshold=0.5)),
         evidence_level='sam3_real_inference', status=status, counts=counts, frames=frames))
    return dict(status=status, run_dir=str(folder))


@pytest.mark.parametrize('stop', [False, True])
def test_one_segmentation_attempt_and_resource_stop_retains_original(tmp_path, stop):
    run = tmp_path/'run'
    image = tmp_path/'image.png'
    Image.new('RGB', (2, 2)).save(image)
    original = mask_product(run/'segmentation/original', image, [(0,'OK',1),(1,'NO_MATCH',0)])
    before = (Path(original['run_dir'])/'manifest.json').read_bytes()
    calls = []

    def stage(name, command, timeout, memory):
        calls.append(name)
        assert name == 'segmentation_recovery'
        config = json.loads(Path(command[-1]).read_text())
        assert config['indices'] == [1] and config['text_prompt'] == 'black and red glove'
        assert timeout == 950 and memory == 14 << 30
        folder = Path(config['output_dir'])
        outcome = mask_product(folder/'product', image, [(1,'OK',2)], config['text_prompt'])
        save(folder/'PRODUCT_OUTCOME.json', outcome)
        return 1 if stop else 0

    product, detail = recover_segmentation(run=run, root=tmp_path, config=dict(sam3_python='pinned-sam-python'),
        product=original, indices=[0,1], prompt='black and red glove', stage=stage, stopped=lambda:stop)
    assert calls == ['segmentation_recovery'] and detail['attempts'] == 1
    assert (Path(original['run_dir'])/'manifest.json').read_bytes() == before
    if stop:
        assert product == original and detail['status'] == 'FAILED'
        assert not (run/'segmentation/recovered').exists()
    else:
        doc = json.loads(Path(product['manifest']).read_text())
        assert doc['frames'][0]['selected_candidate_id'] == 'candidate-1'
        assert doc['frames'][1]['selected_candidate_id'] == 'candidate-2'
        assert detail['status'] == 'COMPOSED' and detail['recovered'] == 1


def sfm(path, views):
    binary = path.parent/'sparse/0'
    binary.mkdir(parents=True)
    save(path, dict(status='PARTIAL', models=[dict(model_id='0', usable=True, roundtrip_verified=True,
        num_registered_images=views, num_points3D=100, binary_path='sparse/0')]))


def test_recovery_satisfies_real_runtime_fresh_directory_gate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from integrations.demo_pipeline import runtime
    run=tmp_path/'run'; image=tmp_path/'image.png'
    Image.new('RGB',(2,2)).save(image)
    original=mask_product(run/'segmentation/original',image,[(0,'NO_MATCH',0)])
    reached=[]
    def preflight(config):
        reached.append(config.output_dir)
        return dict(status='PRECONDITION_BLOCKED',errors=['EXPLICIT_TEST_STOP_BEFORE_MODEL'])
    monkeypatch.setattr(runtime,'preflight',preflight)
    monkeypatch.setattr(runtime,'load_config',lambda path:SimpleNamespace(
        output_dir=Path(json.loads(Path(path).read_text())['output_dir'])))
    def stage(name, command, timeout, memory):
        assert {p.name for p in Path(command[-1]).parent.iterdir()} == {'CONFIG.json'}
        # Real execute_config gate, then an explicit stop before imports/models.
        return runtime.execute_config(command[-1])
    product,detail=recover_segmentation(run=run,root=tmp_path,config=dict(sam3_python='test'),
        product=original,indices=[0],prompt='black glove',stage=stage,stopped=lambda:False)
    assert len(reached)==1 and detail['exit_code']==3 and product==original
    assert (run/'segmentation/RECOVERY_REQUEST.json').is_file()
    assert not (run/'segmentation/recovered').exists()
    # The prevention of a second execution remains active.
    with pytest.raises(ValueError,match='already has execution artifacts'):
        runtime.execute_config(run/'segmentation/recovery_attempt/CONFIG.json')


@pytest.mark.parametrize('case', ['original_valid', 'fewer_views', 'same_views', 'more_views',
                                  'explicit_model', 'source_drift', 'resource_stop'])
def test_camera_fallback_contract(tmp_path, case):
    run = tmp_path/'run'; run.mkdir()
    original = run/'sfm/original/run.json'
    sfm(original, 5)
    calls = []
    candidate_views = 4 if case == 'fewer_views' else 6 if case == 'more_views' else 5

    def stage(name, command, timeout, memory):
        calls.append(name)
        if name.startswith('camera_domain_'):
            binary = str(Path(command[command.index('--model')+1]).resolve())
            output = Path(command[command.index('--output')+1])
            valid = name.endswith('candidate') or case == 'original_valid'
            save(output, dict(status='CAMERA_DOMAIN_VALID' if valid else 'CAMERA_DOMAIN_REJECTED',
                              source_unchanged=case != 'source_drift', source_model=binary))
            return 0 if valid else 1
        assert name == 'sfm_constrained'
        assert command[command.index('--profile')+1] == 'cad-object-pinhole-v1'
        output = Path(command[command.index('--output')+1])
        sfm(output/'candidate/run.json', candidate_views)
        return 0

    chosen, detail = select_camera_solution(root=tmp_path, run=run,
        frozen=dict(python='pinned-light-python', point_selection='foreground', model_id='0' if case=='explicit_model' else None),
        io_run=tmp_path/'io', manifest=tmp_path/'mask.json', sfm_path=original, stage=stage,
        stopped=lambda:case=='resource_stop')
    fallback = case in ('fewer_views','same_views','more_views')
    assert calls.count('sfm_constrained') == int(fallback)
    assert detail['attempts'] == int(fallback)
    assert detail['components_merged'] is False
    if case in ('same_views','more_views'):
        assert chosen != original and detail['surface_ready'] is True
        assert detail['status'] == 'CONSTRAINED_DOMAIN_VALID'
        assert detail['candidate_selected_views'] >= detail['original_selected_views']
    else:
        assert chosen == original
        assert detail['surface_ready'] == (case == 'original_valid')
    if fallback:
        assert calls == ['camera_domain_original','sfm_constrained','camera_domain_candidate']
    else:
        assert calls == ['camera_domain_original']
