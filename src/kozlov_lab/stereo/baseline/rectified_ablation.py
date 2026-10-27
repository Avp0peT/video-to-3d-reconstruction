"""Frozen six-point ablation: numeric stereo versus exact AlgT witness.

A and B are independent engineering stereo baselines. C adds an exact witness
and independent geometry verification, not a new numerical triangulation rule.
No physical 3D truth, fitting, point replacement or camera refinement is used.
"""
from fractions import Fraction
import json
import numpy as np
from . import rectified_parallel as rect
from .radial_diagnostics import _pixels
from ..official import given_parallel_correspondence as given
from ...kozlov2025.exact_center import encode_exact, decode_exact


def _world(first, second, calibration):
    try:
        return dict(status='WORLD_CONVERTED_NOT_ACCEPTED', reason='',
                    **rect.rectified_to_world(first, second, calibration))
    except (ValueError, ArithmeticError, TypeError, KeyError) as exc:
        return dict(status='WORLD_CONVERSION_REFUSED', reason=str(exc), world_xyz=None)


def _stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), rms=float(np.sqrt(np.mean(a*a))),
                median_abs=float(np.median(np.abs(a))), max_abs=float(np.max(np.abs(a))))


def _projection_readback(certificate, first, second, labels):
    proof = certificate['candidate']['certificate']
    body = dict(proof['body'])
    matrices = proof['projections']
    if len(matrices) != 2:
        raise ValueError('two certificate projections required')
    projected = []
    for matrix, wanted in zip(matrices, (first, second)):
        view = []
        for label in labels:
            homogeneous = body[label] + (Fraction(1),)
            xy = tuple(sum(a*b for a, b in zip(row, homogeneous)) for row in matrix)
            if xy != wanted[label]:
                raise ValueError('certificate body projection does not reproduce complete input')
            view.append(xy)
        projected.append(tuple(view))
    return body, projected


def evaluate_six(first_rectified_pixels, second_rectified_pixels, rectification, labels):
    """Evaluate all six in declared order, with fixed scale=1 and grid_bits=18.

    Invalid incomplete observations/labels raise ValueError. Stage failures are
    retained independently. ``exact_result`` is identity-preserving encoded JSON;
    C's independent verifier consumes a JSON encode/decode roundtrip certificate.
    Construction and verification each receive 2,000,000 work and 10 seconds.
    Numerical errors compare methods only; they are not ground-truth errors.
    """
    labels = tuple(labels)
    if len(labels) != 6 or any(type(i) is not int for i in labels) or len(set(labels)) != 6:
        raise ValueError('exactly six unique pure integer labels required')
    a, b = _pixels(first_rectified_pixels), _pixels(second_rectified_pixels)
    if a.shape != (6, 2) or b.shape != (6, 2):
        raise ValueError('exactly six complete paired observations required')
    common1, common2 = a.copy(), b.copy()
    common1[:, 1] = common2[:, 1] = a[:, 1]/2 + b[:, 1]/2
    out = dict(schema='rectified-six-ablation-v1', labels=labels,
               raw_rectified1=a.copy(), raw_rectified2=b.copy(),
               A=_world(common1, common2, rectification),
               B=dict(status='NOT_EVALUATED', world_xyz=None),
               C=dict(status='NOT_EVALUATED', world_xyz=None,
                      algorithm_called='NOT_CALLED', independent=dict(status='NOT_EVALUATED')),
               quantization=None, exact_result=None,
               errors=dict(A_vs_B=None, B_vs_C=None),
               scope='engineering world conversion; unknown physical scale; no geometry acceptance',
               added_capability='exact witness and independent verification; B/C numerical equivalence expected')
    out['A'].update(first_pixels=common1, second_pixels=common2)
    try:
        q = rect.quantize_rectified_six(a, b, rectification['intrinsics_rectified'], scale=1, grid_bits=18)
        out['quantization'] = q
        first, second = dict(zip(labels, q['first'])), dict(zip(labels, q['second']))
        pixel1 = rect.surrogate_to_rectified_pixels(q['first'], rectification['intrinsics_rectified'])
        pixel2 = rect.surrogate_to_rectified_pixels(q['second'], rectification['intrinsics_rectified'])
        out['B'] = _world(pixel1, pixel2, rectification)
        out['B'].update(first_pixels=pixel1, second_pixels=pixel2)
    except (ValueError, ArithmeticError, TypeError, KeyError) as exc:
        out['B'].update(status='QUANTIZATION_REFUSED', reason=str(exc))
        out['C'].update(status='SURROGATE_UNAVAILABLE', reason=str(exc))
        return out
    if out['A']['world_xyz'] is not None and out['B']['world_xyz'] is not None:
        out['errors']['A_vs_B'] = _stats(np.linalg.norm(out['A']['world_xyz']-out['B']['world_xyz'], axis=1))
    c = out['C']
    try:
        result = given.assess_given_parallel_input(first, second, dict(zip(labels, labels)), max_work=2000000, seconds=10)
        out['exact_result'] = encode_exact(result)
        restored = decode_exact(json.loads(json.dumps(out['exact_result'])))
        c.update(status=restored['status'], exact_status=restored['status'], reason=restored.get('reason', ''),
                 algorithm_called=restored.get('algorithm_called', 'NOT_REPORTED'))
        if restored.get('certificate') is None:
            c['independent'] = dict(status='NO_CERTIFICATE')
            return out
        check = given.assess_given_parallel_certificate(first, second, dict(zip(labels, labels)),
                    restored['certificate'], max_work=2000000, seconds=10)
        c['independent'] = check
        if check['status'] != 'VERIFIED':
            c.update(status='INDEPENDENT_'+check['status'], reason=check.get('reason', ''))
            return out
        if restored['status'] != 'CONSTRUCTED':
            return out
        body, views = _projection_readback(restored['certificate'], first, second, labels)
        recovered1 = rect.surrogate_to_rectified_pixels(views[0], rectification['intrinsics_rectified'])
        recovered2 = rect.surrogate_to_rectified_pixels(views[1], rectification['intrinsics_rectified'])
        c.update(_world(recovered1, recovered2, rectification))
        c.update(exact_status=restored['status'], affine_gauge_body=tuple(body.items()),
                 projection_readback_exact=True, first_pixels=recovered1, second_pixels=recovered2)
        if c['world_xyz'] is not None and out['B']['world_xyz'] is not None:
            out['errors']['B_vs_C'] = _stats(np.linalg.norm(out['B']['world_xyz']-c['world_xyz'], axis=1))
    except (ValueError, ArithmeticError, TypeError, KeyError, IndexError) as exc:
        c.update(status='C_STAGE_INVALID', reason=str(exc), world_xyz=None)
    return out
