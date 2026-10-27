"""Given zero-centre-error sufficient domain of S-2025 PDF9--11 fits.

No minimax search or numerical perturbation; all supported geometry is rational.
"""
from fractions import Fraction as F
import math
import time

from .independent_cover import Budget, BudgetExceeded
from .affine_subimage import _rows
from .exact_center import solve_exact_center, assess_exact_center_result
from .real_algebraic import Root, AN
from .finite_environment import _canonical_rows

SCHEMA = 'zero-error-fits-v1'
PROOF = 'zero_matching_error_and_nonnegative_competitors_preserved_by_common_affine_map'


class NonRational(Exception):
    pass


def _budget(max_work, seconds):
    if (type(max_work) is not int or not 0 <= max_work <= 2000000
            or type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not 0 <= seconds <= 30):
        raise ValueError('bounded work/time budget required')
    return Budget(max_work, seconds)


def _call(budget, function, *args):
    budget.spend(0)
    result = function(*args, max_work=max(0, budget.maximum-budget.used),
                      seconds=max(0, budget.deadline-time.monotonic()))
    budget.spend(result['work_used'])
    return result


def _parts(raw, parts):
    if type(parts) not in (list, tuple) or not 2 <= len(parts) <= 7:
        raise ValueError('given 2..7 nonempty blocks required')
    seen, normalized = set(), []
    for part in parts:
        if type(part) not in (list, tuple) or not part or any(type(i) is not int for i in part):
            raise ValueError('nonempty integer-labelled parts required')
        ids = set(part)
        if len(ids) != len(part) or ids & seen or not ids <= raw.keys():
            raise ValueError('parts must be label-disjoint without duplicate or foreign labels')
        seen |= ids
        normalized.append(tuple(sorted(ids)))
    if seen != raw.keys():
        raise ValueError('given parts must preserve every supplied point')
    return tuple(normalized)


def _rational(x):
    if type(x) not in (int, F):
        raise ValueError('int/Fraction required; bool/float rejected')
    q = F(x)
    if max(q.numerator.bit_length(), q.denominator.bit_length()) > 8192:
        raise ValueError('given witness bit bound')
    return q


def _input(reference, reference_parts, source, source_parts, matrix, translation, mapping):
    ar, br = dict(_rows(reference, 7)), dict(_rows(source, 7))
    ap, bp = _parts(ar, reference_parts), _parts(br, source_parts)
    if len(ap) != len(bp):
        raise ValueError('same number of blocks required')
    if type(matrix) not in (list, tuple) or len(matrix) != 2 or any(type(row) not in (list, tuple) or len(row) != 2 for row in matrix):
        raise ValueError('2x2 rational matrix required')
    m = tuple(tuple(_rational(x) for x in row) for row in matrix)
    if type(translation) not in (list, tuple) or len(translation) != 2:
        raise ValueError('two rational translation coordinates required')
    t = tuple(_rational(x) for x in translation)
    if type(mapping) is not dict or set(mapping) != set(range(len(bp))) or any(type(i) is not int or type(j) is not int for i,j in mapping.items()):
        raise ValueError('explicit source-index to reference-index bijection required')
    if set(mapping.values()) != set(range(len(ap))):
        raise ValueError('part correspondence must be bijective')
    if m[0][0]*m[1][1]-m[0][1]*m[1][0] == 0:
        raise ValueError('given matrix must be invertible')
    return ar, ap, br, bp, m, t, tuple(sorted(mapping.items()))


def _center(result, budget):
    context = result['number_context']
    if context['domain'] == 'rational':
        return tuple(_rational(x) for x in result['center'])
    if context['domain'] != 'real-algebraic':
        raise NonRational('unsupported exact center number context')
    root = Root.from_data(context['algebraic_root'], budget)
    values = [AN.from_data(x, root) for x in result['center']]
    if any(len(x.numerator) != 1 or len(x.denominator) != 1 for x in values):
        raise NonRational('nonconstant algebraic center outside rational sufficient profile')
    return tuple(x.numerator[0]/x.denominator[0] for x in values)


def _check_geometry(ar, ap, br, bp, m, t, mapping, centers_a, centers_b, radius2, budget):
    rows, transformed = [], []
    for j, k in mapping:
        budget.spend(16)
        c = centers_b[j]
        image = tuple(m[h][0]*c[0]+m[h][1]*c[1]+t[h] for h in (0, 1))
        if image != centers_a[k]:
            return False, 'given part centers do not match exactly', (), ()
        rows.append((j, k, c, image, centers_a[k]))
        for label in bp[j]:
            p = br[label]
            q = tuple(m[h][0]*p[0]+m[h][1]*p[1]+t[h] for h in (0, 1))
            distances = tuple(sum((q[h]-v[h])**2 for h in (0, 1)) for v in centers_a)
            budget.spend(8+8*len(centers_a))
            if distances[k] >= radius2 or any(distances[k] >= d for other,d in enumerate(distances) if other != k):
                return False, 'given transformed whole block not in strict truncated circle interior', (), ()
            transformed.append((label, j, k, q, distances))
    return True, 'all given centers match and complete transformed blocks lie strictly inside', tuple(rows), tuple(transformed)


def given_zero_error_fits(reference, reference_parts, source, source_parts,
                          matrix, translation, part_bijection, *, max_work=2000000, seconds=30):
    """Verify given witness; failure never proves general fits impossible."""
    result = dict(schema=SCHEMA, status='UNSUPPORTED', complete=False, work_used=0)
    budget = None
    try:
        budget = _budget(max_work, seconds)
        args = _input(reference, reference_parts, source, source_parts, matrix, translation, part_bijection)
        ar, ap, br, bp, m, t, mapping = args
        from .primary_futlyar import assess_primary_futlyar
        primary = _call(budget, assess_primary_futlyar, ar, ap)
        if primary['status'] != 'TRUE':
            state = primary['status']
            return dict(result, status='FALSE_GIVEN_WITNESS' if state == 'FALSE' else 'UNKNOWN' if state == 'UNKNOWN' else 'UNSUPPORTED',
                        reason='reference primary not certified: '+state, work_used=budget.used)
        blocks = []
        for part in bp:
            answer = _call(budget, solve_exact_center, {i: br[i] for i in part})
            if answer['complete'] is not True or answer['exact_optimum'] is not True:
                raise NonRational('source full-block exact center construction unresolved')
            blocks.append(answer)
        c = dict(schema=SCHEMA, reference_points=tuple(sorted(ar.items())), source_points=tuple(sorted(br.items())),
                 reference_parts=ap, source_parts=bp, matrix=m, translation=t, part_bijection=mapping,
                 primary_certificate=primary['certificate'], source_blocks=tuple(blocks),
                 proof=PROOF, epsilon1=F(1), epsilon2=F(1))
        assessment = _call(budget, assess_zero_error_fits_certificate, ar, br, c)
        if assessment['status'] != 'VERIFIED':
            return dict(result, status='FALSE_GIVEN_WITNESS' if assessment['status']=='INVALID' else 'UNKNOWN',
                        reason=assessment.get('reason', 'given witness not verified'), work_used=budget.used)
        c.update(center_checks=assessment['center_checks'], point_checks=assessment['point_checks'])
        return dict(result, status='TRUE', complete=True, certificate=c,
                    reason='zero-error stable-center proof and complete strict block containment', work_used=budget.used)
    except (BudgetExceeded, NonRational) as exc:
        return dict(result, status='UNKNOWN', reason=str(exc), work_used=budget.used if budget else 0)
    except (ValueError, TypeError, KeyError, ArithmeticError, AttributeError, IndexError) as exc:
        return dict(result, reason=str(exc), work_used=budget.used if budget else 0)


def assess_zero_error_fits_certificate(reference, source, certificate, *, max_work=2000000, seconds=30):
    """Independent full-block/primary/zero-error proof; invokes no producer."""
    budget = None
    def finish(status, reason, **fields):
        return dict(status=status, reason=reason, work_used=budget.used if budget else 0, **fields)
    try:
        budget = _budget(max_work, seconds)
        c = certificate
        for field in ('center_checks', 'point_checks'):
            if field in c:
                if type(c[field]) is not tuple:
                    return finish('INVALID', 'cached geometry must be canonical tuples')
                for row in c[field]:
                    if type(row) is not tuple or len(row) != 5:
                        return finish('INVALID', 'cached geometry row shape invalid')
                    ints = row[:2] if field=='center_checks' else row[:3]
                    vectors = row[2:] if field=='center_checks' else row[3:]
                    if any(type(x) is not int for x in ints) or any(type(v) is not tuple or any(type(x) is not F for x in v) for v in vectors):
                        return finish('INVALID', 'cached identity/rational types invalid')
        if c['schema'] != SCHEMA or c['proof'] != PROOF or type(c['epsilon1']) is not F or type(c['epsilon2']) is not F or c['epsilon1'] != 1 or c['epsilon2'] != 1:
            return finish('INVALID', 'zero-error symbolic proof malformed')
        args = _input(reference, c['reference_parts'], source, c['source_parts'], c['matrix'], c['translation'], dict(c['part_bijection']))
        ar, ap, br, bp, m, t, mapping = args
        if (not _canonical_rows(c['reference_points']) or not _canonical_rows(c['source_points'])
                or c['reference_points'] != tuple(sorted(ar.items())) or c['source_points'] != tuple(sorted(br.items())) or c['part_bijection'] != mapping):
            return finish('INVALID', 'complete external input or mapping changed')
        from .primary_futlyar import assess_primary_futlyar_certificate
        primary = _call(budget, assess_primary_futlyar_certificate, ar, c['primary_certificate'])
        if primary['status'] != 'VERIFIED':
            return finish('UNKNOWN' if primary['status']=='UNKNOWN' else 'INVALID', 'reference primary independent verification failed')
        pc = c['primary_certificate']
        if tuple(tuple(part) for part in pc['parts']) != ap or len(c['source_blocks']) != len(bp):
            return finish('INVALID', 'full block certificate count or reference parts changed')
        ca = tuple(tuple(_rational(x) for x in center) for center in pc['centers'])
        r2 = _rational(pc['radius_squared'])
        cb = []
        for part, answer in zip(bp, c['source_blocks']):
            check = _call(budget, assess_exact_center_result, {i: br[i] for i in part}, answer)
            if check['status'] != 'VERIFIED':
                return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID', 'source exact full-block certificate failed')
            cb.append(_center(answer, budget))
        valid, reason, centers, points = _check_geometry(ar, ap, br, bp, m, t, mapping, ca, cb, r2, budget)
        if not valid:
            return finish('INVALID', reason)
        def exact_geometry(value):
            if type(value) in (tuple,list):
                return all(exact_geometry(x) for x in value)
            return type(value) in (int,F)
        if any(key in c and not exact_geometry(c[key]) for key in ('center_checks','point_checks')):
            return finish('INVALID', 'stored geometry requires exact numbers, not booleans')
        if ('center_checks' in c and c['center_checks'] != centers) or ('point_checks' in c and c['point_checks'] != points):
            return finish('INVALID', 'stored full geometry checks tampered')
        budget.spend(0)
        return finish('VERIFIED', reason, center_checks=centers, point_checks=points)
    except (BudgetExceeded, NonRational) as exc:
        return finish('UNKNOWN', str(exc))
    except (ValueError, TypeError, KeyError, ArithmeticError, AttributeError, IndexError) as exc:
        return finish('INVALID', str(exc))


def verify_zero_error_fits_certificate(reference, source, certificate, *, max_work=2000000, seconds=30):
    return assess_zero_error_fits_certificate(reference, source, certificate, max_work=max_work, seconds=seconds)['status']=='VERIFIED'
