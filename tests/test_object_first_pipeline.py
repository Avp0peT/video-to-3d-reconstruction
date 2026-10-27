"""Object-first orchestration: injected workers, actual bound PNGs, no native calls."""
import json
from pathlib import Path

import pytest
from PIL import Image
from integrations.demo_pipeline import pipeline as p


@pytest.fixture
def rig(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    source = tmp_path / 'photos'
    source.mkdir()
    for index in range(2):
        Image.new('RGB', (2, 2), (index * 20, 40, 80)).save(source / f'{index}.png')
    tool = root / 'python.exe'
    tool.write_bytes(b'EXPLICIT_TEST_TOOL_NOT_EXECUTED')
    git = root / 'git.exe'
    git.write_bytes(b'EXPLICIT_TEST_GIT_NOT_EXECUTED')
    package = root / 'package'
    package.mkdir()
    p.save(package / 'BUNDLE.json', {})
    weight = root / 'weight.pt'
    weight.write_bytes(b'EXPLICIT_TEST_WEIGHT_NOT_LOADED')
    options = p.Options(source, 'can', output=root / 'outputs/demo', python=tool,
                        sam3_python=tool, sam3_package=package, checkpoint=weight,
                        git_exe=git, target='report', view_geometry_audit=False,
                        post_review=False)
    return root, options


def bound(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': p.sha(path)}


def injected(*, fault=None):
    calls, sfm_arguments = [], []

    def execute(root, run, stage, argv, **limits):
        calls.append(stage)
        if stage == 'input':
            io = run / 'input/photo_io/fixture'
            io.mkdir(parents=True)
            frames = []
            for index in range(2):
                image = io / f'{index}.png'
                Image.new('RGB', (2, 2), (index * 20, 40, 80)).save(image)
                frames.append(dict(input_index=index, frame_id=f'frame{index}',
                    image_output=image.name, image_sha256=p.sha(image), canonical_width=2,
                    canonical_height=2, image_mode='RGB', image_dtype='uint8', status='OK'))
            p.save(io / 'run.json', dict(contract_version='photo-io-v1', status='SUCCESS', run_id='io1'))
            (io / 'frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames), encoding='utf-8')
            dataset = run / 'input/photo_manifest.json'
            p.save(dataset, dict(contract_version='photo-io-v1', data_kind='real', frames=[{'image': f'{i}.png'} for i in range(2)]))
            p.save(run / 'input/RESULT.json', dict(status='INPUTS_PREPARED', data_kind='real',
                photo_io={'run': bound(io / 'run.json'), 'manifest': bound(dataset), 'identities': frames}))
        elif stage == 'segmentation':
            config = p.read(run / 'segmentation/CONFIG.json')
            io = Path(config['io_run'])
            product = run / 'segmentation/product'
            product.mkdir()
            frames = []
            for index in range(2):
                mask = product / f'{index}.mask.png'
                Image.new('L', (2, 2), 255).save(mask)
                frames.append(dict(input_index=index, frame_id=f'frame{index}',
                    status='NO_MATCH' if fault == 'no_match' else 'OK', image_domain='photo_io_canonical',
                    source_image=str(io / f'{index}.png'), source_image_sha256=p.sha(io / f'{index}.png'),
                    photo_io_image_sha256=p.sha(io / f'{index}.png'), width=2, height=2,
                    candidates=[dict(candidate_id='c0', mask_path=mask.name, mask_sha256=p.sha(mask), foreground_pixels=4)],
                    selected_candidate_id=None if fault in ('no_match', 'unselected') else 'c0',
                    selected_mask=None if fault in ('no_match', 'unselected') else {'path': mask.name, 'sha256': p.sha(mask)},
                    selection={'method': 'unique_candidate', 'source': 'automatic', 'mask_source': 'automatic_sam3'}))
            counts = dict(requested=2, success=0 if fault == 'no_match' else 2,
                          no_match=2 if fault == 'no_match' else 0, needs_selection=0, error=0, not_run=0)
            if fault != 'missing_manifest':
                p.save(product / 'manifest.json', dict(contract_version='object-mask-v1', run_id='sam1',
                    status='PARTIAL' if fault == 'no_match' else 'SUCCESS', started_utc='start', finished_utc='end',
                    source_io={'path': str(io), 'run_id': 'io1', 'run_json_sha256': p.sha(io / 'run.json'),
                               'frames_jsonl_sha256': p.sha(io / 'frames.jsonl')},
                    backend={}, object={'object_id': config['object_id']}, counts=counts, frames=frames))
            p.save(run / 'segmentation/PRODUCT_OUTCOME.json', dict(status='PARTIAL' if fault == 'no_match' else 'SUCCESS',
                exit_code=1 if fault == 'no_match' else 0, counts=counts, run_dir=str(product), stop_state={'latched': False}))
        elif stage == 'sfm':
            sfm_arguments.append(list(argv))
            assert calls == ['input', 'segmentation', 'sfm']
            assert '--object-mask-manifest' in argv
            mask = Path(argv[argv.index('--object-mask-manifest') + 1])
            assert mask.is_absolute() and mask == (run / 'segmentation/product/manifest.json').resolve()
            assert all(f['selected_mask'] for f in p.read(mask)['frames'])
            output = Path(argv[argv.index('--output') + 1]) / 'fixture'
            output.mkdir(parents=True)
            p.save(output / 'run.json', dict(status='NO_MODEL', models=[], counts={'registered_unique_images': 0, 'unique_images': 2}))
        else:
            pytest.fail('Unexpected injected native stage: ' + stage)
        return dict(status='completed', root_exit_code=1 if stage == 'sfm' else 0,
                    reclaim_confirmed=True, errors=[], log_errors=[])

    return execute, calls, sfm_arguments


def renderer(run):
    output = run / 'report'
    output.mkdir()
    (output / 'index.html').write_text('EXPLICIT_TEST_REPORT', encoding='utf-8')
    return output / 'index.html'


def test_object_first_order_and_absolute_selected_mask_manifest(rig):
    root, options = rig
    executor, calls, argv = injected()
    run, code = p.run_pipeline(options, root=root, executor=executor, renderer=renderer)
    assert calls == ['input', 'segmentation', 'sfm']
    assert len(argv) == 1 and code != 0  # NO_MODEL remains incomplete.
    assert p.read(run / 'pipeline.json')['stages']['sfm']['status'] == 'NO_MODEL'


@pytest.mark.parametrize('fault', ['no_match', 'missing_manifest', 'unselected'])
def test_object_first_missing_selected_support_never_falls_back_to_scene_sfm(rig, fault):
    root, options = rig
    executor, calls, argv = injected(fault=fault)
    run, code = p.run_pipeline(options, root=root, executor=executor, renderer=renderer)
    assert calls == ['input', 'segmentation']
    assert not argv and code != 0
    assert p.read(run / 'pipeline.json')['stages']['sfm']['status'] not in ('SUCCESS', 'PARTIAL', 'NO_MODEL')
    assert (run / 'report/index.html').is_file()
