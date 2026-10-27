"""Bounded, evidence-preserving alternatives used by the public pipeline."""
from pathlib import Path


def recover_segmentation(*, run, root, config, product, indices, prompt, stage, stopped):
    from .pipeline import read, save, sha
    from .segmentation_recovery import build_recovery_request, compose_recovery
    detail = dict(status='NOT_NEEDED', attempts=0, max_attempts=1,
                  original_outcome=product, quality_status='NOT_EVALUATED')
    if product.get('status') not in ('SUCCESS', 'PARTIAL') or not product.get('run_dir'):
        detail.update(status='NOT_RUN', reason='upstream did not complete normally')
        return product, detail
    source = Path(product['run_dir']).resolve(strict=True)
    if not source.is_relative_to((run/'segmentation').resolve()):
        raise ValueError('recovery source escapes segmentation directory')
    manifest = source/'manifest.json'
    document = read(manifest)
    unresolved = [f for f in document['frames'] if f.get('status') in ('NO_MATCH', 'NEEDS_SELECTION')]
    if not unresolved:
        return product, detail
    if len(unresolved) > 96:
        detail.update(status='NOT_RUN', reason='unresolved inputs exceed one recovery batch of 96')
        return product, detail
    request = build_recovery_request(manifest, indices, prompt)
    folder = run/'segmentation'/'recovery_attempt'
    folder.mkdir(exist_ok=False)
    recovery_config = dict(config, output_dir=str(folder), indices=request['input_indices'],
                           text_prompt=request['text_prompt'])
    # Runtime owns REQUEST.json and requires only CONFIG.json before launch.
    # Keep orchestration evidence outside its exclusive fresh output directory.
    save(folder.parent/'RECOVERY_REQUEST.json', request)
    save(folder/'CONFIG.json', recovery_config)
    detail.update(status='RUNNING', attempts=1, request=request)
    save(folder.parent/'RECOVERY.json', detail)
    code = stage('segmentation_recovery', [config['sam3_python'], '-B',
        str(root/'scripts/demo_segment_worker.py'), '--config', str(folder/'CONFIG.json')], 950, 14 << 30)
    outcome = read(folder/'PRODUCT_OUTCOME.json') if (folder/'PRODUCT_OUTCOME.json').is_file() else {}
    detail.update(exit_code=code, attempt_outcome=outcome)
    if code not in (0, 1) or stopped() or outcome.get('status') not in ('SUCCESS', 'PARTIAL'):
        detail.update(status='FAILED', reason='recovery run incomplete; original result retained')
    else:
        attempt = Path(outcome['run_dir']).resolve(strict=True)
        if not attempt.is_relative_to(folder):
            raise ValueError('recovery output escapes attempt directory')
        merged = compose_recovery(manifest, attempt/'manifest.json', run/'segmentation'/'recovered', indices, request)
        detail.update(status='COMPOSED', recovered=merged['recovered'], counts=merged['counts'],
                      manifest=merged['manifest'], manifest_sha256=sha(Path(merged['manifest'])))
        product = merged
    save(folder.parent/'RECOVERY.json', detail)
    return product, detail


def select_camera_solution(*, root, run, frozen, io_run, manifest, sfm_path, stage, stopped):
    """One domain audit, at most one independently recomputed pinhole candidate.

    Numerical domain validity is necessary, not sufficient for shape accuracy.
    Components are never joined. Explicit model selection disables fallback.
    """
    from .pipeline import read, save, sha
    from .surface import choose_model
    original_path = Path(sfm_path).resolve(strict=True)
    original = read(original_path)
    detail = dict(status='NOT_RUN', attempts=0, max_attempts=1, surface_ready=False,
                  calibration_accuracy='NOT_EVALUATED', original_sfm=str(original_path),
                  original_sfm_sha256=sha(original_path), audits=[])
    directory = run/'camera_recovery'
    directory.mkdir(exist_ok=False)

    def audit(path, model, label):
        binary = (path.parent/model['binary_path']).resolve(strict=True)
        if not binary.is_relative_to(path.parent):
            raise ValueError('camera source escapes model run')
        output = directory/(label+'.json')
        code = stage('camera_domain_'+label, [frozen['python'], '-B', '-X', 'utf8',
            str(root/'scripts/audit_camera_domain.py'), '--model', str(binary), '--output', str(output)], 90, 1 << 30)
        result = read(output) if output.is_file() else {}
        valid = (code == 0 and result.get('status') == 'CAMERA_DOMAIN_VALID'
                 and result.get('source_unchanged') is True and result.get('source_model') == str(binary))
        detail['audits'].append(dict(label=label, exit_code=code, path=str(output),
                                    sha256=sha(output) if output.is_file() else None, result=result))
        return valid, code, result

    selected, _ = choose_model(original.get('models', []), frozen.get('model_id'))
    valid, code, result = audit(original_path, selected, 'original')
    chosen = original_path
    if valid and not stopped():
        detail.update(status='ORIGINAL_DOMAIN_VALID', surface_ready=True)
    elif stopped() or code != 1 or result.get('status') != 'CAMERA_DOMAIN_REJECTED' or result.get('source_unchanged') is not True:
        detail.update(status='BLOCKED', reason='audit failed, source drift, or resource stop')
    elif frozen.get('model_id') is not None or frozen.get('all_frames'):
        detail.update(status='BLOCKED', reason='explicit model or >96-frame profile does not allow automatic camera substitution')
    else:
        detail.update(status='RUNNING', attempts=1, profile='cad-object-pinhole-v1',
                      assumption='zero radial distortion is an alternative model, not a calibration fact')
        save(directory/'RESULT.json', detail)
        output = directory/'sfm_candidate'
        code = stage('sfm_constrained', [frozen['python'], '-B', str(root/'scripts/reconstruct_object.py'),
            '--io-run', str(io_run), '--output', str(output), '--profile', 'cad-object-pinhole-v1',
            '--object-mask-manifest', str(manifest), '--point-selection', frozen['point_selection']], 720, 2 << 30)
        candidates = list(output.glob('*/run.json'))
        detail['candidate_exit_code'] = code
        if code in (0, 1) and not stopped() and len(candidates) == 1:
            path = candidates[0].resolve(); candidate = read(path)
            detail.update(candidate_sfm=str(path), candidate_sfm_sha256=sha(path))
            if candidate.get('status') in ('SUCCESS', 'PARTIAL'):
                model, _ = choose_model(candidate.get('models', []), None)
                valid, _, _ = audit(path, model, 'candidate')
                baseline_views, candidate_views = selected['num_registered_images'], model['num_registered_images']
                detail.update(original_selected_views=baseline_views, candidate_selected_views=candidate_views)
                if valid and not stopped() and candidate_views >= baseline_views:
                    chosen = path
                    detail.update(status='CONSTRAINED_DOMAIN_VALID', surface_ready=True)
                else:
                    detail.update(status='BLOCKED', reason='candidate domain invalid or fewer selected registered views')
            else:
                detail.update(status='BLOCKED', reason='candidate produced no usable model')
        else:
            detail.update(status='BLOCKED', reason='candidate did not complete normally')
    detail.update(selected_sfm=str(chosen), selected_sfm_sha256=sha(chosen), components_merged=False)
    save(directory/'RESULT.json', detail)
    return chosen, detail
