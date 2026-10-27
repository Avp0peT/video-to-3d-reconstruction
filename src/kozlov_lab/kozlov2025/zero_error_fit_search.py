"""Finite zero-center-error witness search, not general S-2025 fits."""
from fractions import Fraction as F
from itertools import permutations

from .zero_error_fits import (_budget, _call, _parts, _center,
    given_zero_error_fits, assess_zero_error_fits_certificate, NonRational)
from .affine_subimage import _rows, _sub, _det, _solve, _linear, _map
from .exact_center import solve_exact_center, assess_exact_center_result
from .independent_cover import BudgetExceeded

SCHEMA = 'zero-error-fit-search-v1'


def search_zero_error_fits(reference, reference_parts, source, source_parts,
                           *, max_work=2000000, seconds=30):
    budget = None
    tried = 0
    rank = None
    def finish(status, reason, **kw):
        return dict(schema=SCHEMA, status=status, reason=reason, center_rank=rank,
                    candidates_tried=tried, work_used=budget.used if budget else 0,
                    general_fits_decided=False, **kw)
    try:
        budget = _budget(max_work, seconds)
        a, b = dict(_rows(reference, 7)), dict(_rows(source, 7))
        ap, bp = _parts(a, reference_parts), _parts(b, source_parts)
        if len(ap) != len(bp):
            return finish('UNSUPPORTED', 'same block count required')
        from .primary_futlyar import assess_primary_futlyar
        primary = _call(budget, assess_primary_futlyar, a, ap)
        if primary['status'] != 'TRUE':
            return finish('UNKNOWN' if primary['status']=='UNKNOWN' else 'UNSUPPORTED', 'reference primary not certified')
        ca = tuple(tuple(F(x) for x in c) for c in primary['certificate']['centers'])
        cb = []
        for part in bp:
            block = {i:b[i] for i in part}
            answer = _call(budget, solve_exact_center, block)
            if answer['complete'] is not True or answer['exact_optimum'] is not True:
                raise NonRational('complete block center unresolved')
            check = _call(budget, assess_exact_center_result, block, answer)
            if check['status'] != 'VERIFIED':
                raise NonRational('complete block center certificate unresolved')
            cb.append(_center(answer, budget))
        # Coincident source centers cannot map bijectively to distinct primary centers.
        if len(set(cb)) != len(cb):
            unique = list(dict.fromkeys(cb))
            rank = 0 if len(unique) == 1 else 1
            if len(unique)>2 and any(_det(_sub(unique[1],unique[0]),_sub(p,unique[0])) for p in unique[2:]):
                rank = 2
            return finish('NO_SUPPORTED_WITNESS', 'coincident source centers cannot match primary centers', search_complete=True)
        basis = [0, 1]
        u = _sub(cb[1], cb[0])
        rank = 1
        for k in range(2, len(cb)):
            budget.spend()
            if _det(u, _sub(cb[k], cb[0])):
                basis.append(k)
                rank = 2
                break
        lookup = {c:i for i,c in enumerate(ca)}
        for anchors in permutations(range(len(ca)), rank+1):
            budget.spend(50)
            tried += 1
            du = _sub(ca[anchors[1]], ca[anchors[0]])
            if rank == 2:
                dv = _sub(ca[anchors[2]], ca[anchors[0]])
                if not _det(du, dv):
                    continue
                matrix = _solve(u, _sub(cb[basis[2]], cb[0]), du, dv)
            else:
                matrix = _solve(u, (-u[1], u[0]), du, (-du[1], du[0]))
            translation = _sub(ca[anchors[0]], _linear(matrix, cb[0]))
            mapping = {}
            for i,c in enumerate(cb):
                budget.spend(10)
                q = _map(matrix, translation, c)
                if q not in lookup:
                    break
                mapping[i] = lookup[q]
            if len(mapping) != len(cb) or len(set(mapping.values())) != len(cb):
                continue
            answer = _call(budget, given_zero_error_fits, a, ap, b, bp, matrix, translation, mapping)
            if answer['status'] == 'TRUE':
                return finish('TRUE', 'complete certified zero-error witness', search_complete=False,
                              certificate=answer['certificate'], matrix=matrix,
                              translation=translation, part_bijection=tuple(sorted(mapping.items())))
            if answer['status'] == 'UNKNOWN':
                return finish('UNKNOWN', 'candidate verification incomplete', search_complete=False)
        return finish('NO_SUPPORTED_WITNESS', 'finite anchor candidates exhausted; rank1 extensions restricted' if rank==1
                      else 'all rank2 zero-error center maps exhausted; not general fits', search_complete=True)
    except (BudgetExceeded, NonRational) as exc:
        return finish('UNKNOWN', str(exc), search_complete=False)
    except (ValueError, TypeError, KeyError, ArithmeticError, AttributeError, IndexError) as exc:
        return finish('UNSUPPORTED', str(exc), search_complete=False)


def assess_zero_error_fit_search_result(reference, source, result, *, max_work=2000000, seconds=30):
    """Independently verifies TRUE geometry, never calls search or constructors."""
    try:
        if result['schema'] != SCHEMA or result['status'] != 'TRUE' or result['general_fits_decided'] is not False:
            return dict(status='INVALID', work_used=0)
        c = result['certificate']
        matrix, translation, mapping = result['matrix'], result['translation'], result['part_bijection']
        if (type(matrix) is not tuple or len(matrix)!=2
                or any(type(row) is not tuple or len(row)!=2 or any(type(x) is not F for x in row) for row in matrix)
                or type(translation) is not tuple or len(translation)!=2 or any(type(x) is not F for x in translation)
                or type(mapping) is not tuple or any(type(row) is not tuple or len(row)!=2 or any(type(x) is not int for x in row) for row in mapping)):
            return dict(status='INVALID',work_used=0)
        if (result['matrix'] != c['matrix'] or result['translation'] != c['translation']
                or result['part_bijection'] != c['part_bijection']):
            return dict(status='INVALID', work_used=0)
        return assess_zero_error_fits_certificate(reference, source, c, max_work=max_work, seconds=seconds)
    except (ValueError, TypeError, KeyError, AttributeError):
        return dict(status='INVALID', work_used=0)
