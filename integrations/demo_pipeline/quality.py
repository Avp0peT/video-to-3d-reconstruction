"""Conservative object-quality facts, independent of technical file delivery.

Protocol: persist build_quality() as <run>/QUALITY.json, then optionally bind it
in pipeline quality_reference={path, sha256}. Display independently recomputes
facts and never accepts a caller's full-object PASS. Partial review references
use object-shape-review-v1, status PARTIAL_SHAPE, reviewer, notes and exact
current surface artifact {path, sha256} records. No accuracy threshold exists.
"""
from pathlib import Path
import hashlib
import json
import math

SCHEMA = 'object-quality-v1'


def _read(path):
    path = Path(path)
    if path.stat().st_size > 4 << 20:
        raise ValueError('quality record exceeds bound')
    def pairs(items):
        value = {}
        for k, v in items:
            if k in value: raise ValueError('duplicate quality key')
            value[k] = v
        return value
    def bad(v): raise ValueError('nonfinite quality value')
    def number(v):
        value = float(v)
        if not math.isfinite(value): bad(v)
        return value
    value = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
                       parse_constant=bad, parse_float=number)
    if not isinstance(value, dict): raise ValueError('quality object required')
    return value


def _count(v):
    return v if type(v) is int and v >= 0 else None


def _indices(v):
    return sorted(v) if isinstance(v, list) and all(type(i) is int and i >= 0 for i in v) and len(set(v)) == len(v) else None


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20): digest.update(block)
    return digest.hexdigest()


def _reference(ref, *, within=None):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('quality reference requires path/hash')
    p = Path(ref['path']).resolve(strict=True)
    if not Path(ref['path']).is_absolute() or not p.is_file(): raise ValueError('absolute quality file required')
    if within is not None and not p.is_relative_to(within): raise ValueError('quality reference escapes run')
    before = _hash(p)
    if before != ref['sha256']: raise ValueError('quality reference hash differs')
    value = _read(p)
    if _hash(p) != before: raise ValueError('quality record drift')
    return value


def build_quality(run, state=None):
    """Produce declared counts, unknowns and at most a bound partial review."""
    run = Path(run).resolve()
    if state is None:
        try: state = _read(run / 'pipeline.json')
        except (OSError, ValueError, TypeError): state = {}
    state = state if isinstance(state, dict) else {}
    stages = state.get('stages', {})
    stages = stages if isinstance(stages, dict) else {}
    def stage(name):
        v = stages.get(name, {})
        return v if isinstance(v, dict) else {}
    result_bindings = []
    fact_sources = {}
    bound_records = {}
    def counts(name):
        record = stage(name)
        reference = record.get('result_reference')
        if reference is not None:
            try:
                bound = _reference(reference)
                value = bound.get('counts')
                if not isinstance(value, dict): raise ValueError('bound counts missing')
                result_bindings.append(dict(reference))
                bound_records[name] = bound
                fact_sources[name] = 'HASH_BOUND_RESULT'
                return value
            except (OSError, ValueError, TypeError, KeyError):
                fact_sources[name] = 'INVALID_RESULT_REFERENCE_COUNTS_UNKNOWN'
                return {}
        fact_sources[name] = 'PIPELINE_DECLARED'
        return record.get('counts', {})
    sfm = counts('sfm')
    seg = counts('segmentation')
    sfm = sfm if isinstance(sfm, dict) else {}
    seg = seg if isinstance(seg, dict) else {}
    surface = stage('surface')
    selected = _indices(surface.get('selected_indices'))
    missing = _indices(surface.get('missing_indices'))
    requested = _count(sfm.get('input_rows'))
    if requested is None:
        try:
            prepared = _read(run / 'input' / 'RESULT.json')
            identities = prepared.get('photo_io', {}).get('identities')
            idx = _indices([r.get('input_index') for r in identities]) if isinstance(identities, list) and all(isinstance(r, dict) for r in identities) else None
            requested = len(idx) if idx is not None else None
        except (OSError, ValueError, TypeError, AttributeError): pass
    scale_status = bound_records.get('sfm', {}).get('scale_status', state.get('scale_status', surface.get('scale_status')))
    scale_known = True if scale_status == 'known' else False if scale_status in ('unknown', 'uncalibrated') else None
    artifacts = surface.get('artifacts', [])
    if not isinstance(artifacts, list): artifacts = []
    refs = [{'path': r['path'], 'sha256': r['sha256']} for r in artifacts
            if isinstance(r, dict) and isinstance(r.get('path'), str) and isinstance(r.get('sha256'), str)]
    facts = {'requested_views': requested, 'registered_views': _count(sfm.get('registered_unique_images')),
             'component_count': _count(sfm.get('model_count')), 'selected_model_id': surface.get('selected_model_id') if isinstance(surface.get('selected_model_id'), str) else None,
             'selected_view_count': len(selected) if selected is not None else None,
             'selected_input_indices': selected, 'excluded_input_indices': missing,
             'segmentation_no_match': _count(seg.get('no_match')),
             'segmentation_needs_selection': _count(seg.get('needs_selection')),
             'segmentation_error': _count(seg.get('error')), 'scale_known': scale_known,
             'object_supported_observations': None, 'duplicate_image_tracks': None,
             'fact_sources': fact_sources,
             'facts_scope': 'Declared or explicitly hash-bound counts; no new native geometry or mask-truth acceptance'}
    result = {'schema_version': SCHEMA, 'context': {'run': str(run), 'run_id': state.get('run_id'), 'surface_artifacts': refs, 'result_bindings': result_bindings},
              'facts': facts,
              'object_shape_status': 'NOT_EVALUATED', 'full_object_accepted': False,
              'object_goal_achieved': False, 'review': None,
              'limitations': ['Holes are allowed but do not prove whole-object coverage.',
                              'Reprojection fit, closed topology and file reopening do not prove physical accuracy.']}
    review_ref = state.get('shape_review')
    if review_ref is not None:
        try:
            review = _reference(review_ref, within=run)
            if (review.get('schema_version') != 'object-shape-review-v1' or review.get('status') != 'PARTIAL_SHAPE'
                    or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                    or not isinstance(review.get('notes'), str) or not review['notes'].strip()
                    or not refs or review.get('surface_artifacts') != refs):
                raise ValueError('partial shape review incomplete')
            for ref in refs:
                p = Path(ref['path']).resolve(strict=True)
                if not p.is_relative_to(run) or _hash(p) != ref['sha256']: raise ValueError('review artifact drift')
            result.update(object_shape_status='PARTIAL_SHAPE', review=dict(review_ref))
        except (OSError, ValueError, TypeError, KeyError): pass
    return result


def read_for_display(run, state=None):
    """QUALITY cannot elevate file status or authorize a full-object claim."""
    run = Path(run).resolve()
    if state is None:
        try: state = _read(run / 'pipeline.json')
        except (OSError, ValueError, TypeError): state = {}
    actual = build_quality(run, state)
    path = run / 'QUALITY.json'
    if path.exists():
        try:
            ref = state.get('quality_reference') if isinstance(state, dict) else None
            if ref is not None:
                if Path(ref['path']).resolve() != path: raise ValueError('another quality record')
                record = _reference(ref, within=run)
            else: record = _read(path)
            if record != actual: raise ValueError('quality does not match current facts')
        except (OSError, ValueError, TypeError, KeyError):
            actual.update(object_shape_status='NOT_EVALUATED', full_object_accepted=False,
                          object_goal_achieved=False, review=None,
                          quality_warning='QUALITY record invalid or stale; object shape remains unevaluated.')
    return actual


def display_message(quality):
    return ('局部外形；全物体目标未完成。文件可查看，允许的孔洞不代表整体覆盖。'
            if quality.get('object_shape_status') == 'PARTIAL_SHAPE' else
            '物体外形未评估；全物体目标尚未验收。文件验证通过不代表外形或尺寸准确。')
