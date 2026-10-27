"""Exact rational five-contact profile of a complete supplied 1..7 point set.

Rank-five conic construction and positive covariance equality certify the
complete input. Unsupported candidates say nothing about general four-contact
algebraic optima. The independent verifier never solves a linear system.
"""
from fractions import Fraction as F
from itertools import combinations, permutations
import math

from .independent_cover import normalize, Budget, BudgetExceeded
from .exact_triangle_ellipse import _digest

MAX_INTERMEDIATE_BITS = 16384
PROOF = 'positive_contact_covariance_equality_and_complete_input_containment'


def _input(points):
    if not isinstance(points, dict) or not 1 <= len(points) <= 7:
        raise ValueError('complete input must contain 1..7 distinct labelled points; no sampling')
    labels = sorted(points) if all(type(i) is int for i in points) else []
    return normalize(points, [labels])[0]


def _bounded(x):
    if type(x) not in (int, F):
        raise ValueError('exact int/Fraction required; bool/float certificates unsupported')
    x = F(x)
    if max(x.numerator.bit_length(), x.denominator.bit_length()) > MAX_INTERMEDIATE_BITS:
        raise BudgetExceeded('intermediate rational bit bound exhausted')
    return x


def _rref(rows, columns, budget):
    """Constructor only; exact elimination with shared work/bit/time limits."""
    matrix = [[_bounded(v) for v in row] for row in rows]
    pivots = []
    for column in range(columns):
        budget.spend()
        pivot = next((i for i in range(len(pivots),len(matrix)) if matrix[i][column]), None)
        if pivot is None:
            continue
        j = len(pivots)
        matrix[j], matrix[pivot] = matrix[pivot], matrix[j]
        factor = matrix[j][column]
        for k in range(len(matrix[j])):
            budget.spend(); matrix[j][k] = _bounded(matrix[j][k]/factor)
        for i in range(len(matrix)):
            if i == j or not matrix[i][column]:
                continue
            factor = matrix[i][column]
            for k in range(len(matrix[i])):
                budget.spend(); matrix[i][k] = _bounded(matrix[i][k]-factor*matrix[j][k])
        pivots.append(column)
        if len(pivots) == len(matrix):
            break
    return matrix, pivots


def _evaluation(p):
    x,y = p
    return tuple(_bounded(v) for v in (x*x,2*x*y,y*y,2*x,2*y,1))


def _determinant(matrix, budget=None):
    """Direct permutation identity; no elimination or linear-system solver."""
    total = F(0)
    for order in permutations(range(len(matrix))):
        value = F(1)
        for i,j in enumerate(order):
            if budget is not None: budget.spend()
            value = _bounded(value*matrix[i][j])
        inversions = sum(order[i]>order[j] for i in range(len(order)) for j in range(i+1,len(order)))
        total = _bounded(total + (-value if inversions % 2 else value))
    return total


def _q(p,c,h):
    x,y = p[0]-c[0],p[1]-c[1]
    return _bounded(h[0][0]*x*x+2*h[0][1]*x*y+h[1][1]*y*y)


def complete_input_contact5_ellipse(points, *, max_work=2000000, seconds=30):
    result = dict(schema='complete-input-contact5-ellipse-v1', status='UNSUPPORTED_INPUT',
        complete=False,exact_optimum=False,work_used=0,visited_quintuples=0,
        candidate_failures={},scope='complete supplied 1..7 input; five labels are support only')
    if (type(max_work) is not int or not 0 <= max_work <= 2000000 or
            type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0 <= seconds <= 30):
        return dict(result,reason='invalid shared bounded budget')
    budget = Budget(max_work,seconds)
    validated = False
    def skip(reason):
        result['candidate_failures'][reason] = result['candidate_failures'].get(reason,0)+1
    try:
        budget.spend()
        raw = _input(points)
        validated = True
        budget.spend(len(raw))
        result.update(declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw))
        for ids in combinations(sorted(raw),5):
            budget.spend(); result['visited_quintuples'] += 1
            evaluation = [_evaluation(raw[i]) for i in ids]
            rr,pivots = _rref(evaluation,6,budget)
            if len(pivots) != 5:
                skip('rank_not_five'); continue
            free = next(i for i in range(6) if i not in pivots)
            conic = [F(0)]*6; conic[free] = F(1)
            for row,pivot in zip(rr,pivots): conic[pivot] = _bounded(-row[free])
            a,b,e,d,f,g = conic
            determinant = _bounded(a*e-b*b)
            if a == 0 or determinant <= 0:
                skip('not_definite_conic'); continue
            # Definite sign and projective scale normalize to A=1.
            conic = tuple(_bounded(v/a) for v in conic)
            a,b,e,d,f,g = conic
            determinant = _bounded(a*e-b*b)
            inverse_m = tuple(tuple(_bounded(v) for v in row) for row in
                              ((e/determinant,-b/determinant),(-b/determinant,a/determinant)))
            c = tuple(_bounded(-sum(inverse_m[j][k]*(d,f)[k] for k in (0,1))) for j in (0,1))
            rho = _bounded(-d*c[0]-f*c[1]-g)
            if rho <= 0:
                skip('nonpositive_radius'); continue
            s = tuple(tuple(_bounded(rho*inverse_m[j][k]) for k in (0,1)) for j in (0,1))
            h = tuple(tuple(_bounded(v) for v in row) for row in ((a/rho,b/rho),(b/rho,e/rho)))
            z = [tuple(_bounded(raw[i][k]-c[k]) for k in (0,1)) for i in ids]
            w = [[F(1) for _ in ids], [p[0] for p in z], [p[1] for p in z],
                 [_bounded(p[0]*p[0]) for p in z], [_bounded(p[0]*p[1]) for p in z],
                 [_bounded(p[1]*p[1]) for p in z]]
            target = (F(1),F(0),F(0),s[0][0]/2,s[0][1]/2,s[1][1]/2)
            wr,wp = _rref([row+[value] for row,value in zip(w,target)],5,budget)
            if len(wp) != 5 or any(not any(row[:5]) and row[5] != 0 for row in wr):
                skip('moment_system_inconsistent'); continue
            weights = tuple(_bounded(wr[i][5]) for i in range(5))
            if any(v <= 0 for v in weights):
                skip('nonpositive_weight'); continue
            budget.spend(100)
            evaluations = {}
            for i,p in raw.items():
                budget.spend(10); evaluations[i] = _q(p,c,h)
            if max(evaluations.values()) > 1:
                skip('complete_input_outside'); continue
            minor = [[row[k] for k in pivots] for row in evaluation]
            minor_det = _determinant(minor,budget)
            det_s = _bounded(s[0][0]*s[1][1]-s[0][1]*s[1][0])
            trace = _bounded(s[0][0]+s[1][1]); disc = _bounded((s[0][0]-s[1][1])**2+4*s[0][1]**2)
            certificate = dict(schema='complete-input-contact5-certificate-v1',
                declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw),support_labels=list(ids),
                conic_coefficients=conic,rank5_minor_columns=pivots,rank5_minor_determinant=minor_det,
                radius_squared_level=rho,contact_weights=weights,center=c,shape_matrix=s,inverse_shape_matrix=h,
                all_input_quadratic_values=evaluations,determinant_shape=det_s,covariance_determinant=det_s/4,
                area_squared_over_pi_squared=det_s,
                semiaxis_squared=dict(expression='(trace +/- sqrt(discriminant))/2',trace=trace,discriminant=disc),
                major_axis_length=dict(expression='sqrt(2*(trace+sqrt(discriminant)))',trace=trace,discriminant=disc),
                proof=PROOF,exact_optimum=True)
            budget.spend(1000+len(raw)*200)
            if not verify_complete_input_contact5_ellipse(raw,certificate):
                raise ArithmeticError('independent five-contact certificate verification failed')
            budget.spend(0)
            return dict(result,status='EXACT_OPTIMUM_COMPLETE_INPUT',complete=True,exact_optimum=True,
                        center=c,certificate=certificate,work_used=budget.used)
        budget.spend(0)
        return dict(result,status='UNSUPPORTED_CONTACT_DOMAIN',work_used=budget.used,
                    reason='no strictly positive rational rank5 certificate found; does not classify contact count or disprove any ellipse')
    except BudgetExceeded as exc:
        return dict(result,status='PARTIAL_UNKNOWN',reason=str(exc),work_used=budget.used)
    except (ValueError,TypeError,ArithmeticError,AttributeError,KeyError,IndexError) as exc:
        return dict(result,status='PARTIAL_UNKNOWN' if validated else 'UNSUPPORTED_INPUT',
                    reason=str(exc),work_used=budget.used)


def verify_complete_input_contact5_ellipse(points,certificate):
    """Independent identities only: no RREF, candidate search, hull or solver."""
    try:
        raw = _input(points); z = certificate
        if z['schema'] != 'complete-input-contact5-certificate-v1': return False
        ids,declared = z['support_labels'],z['declared_labels']
        if (not isinstance(ids,(list,tuple)) or len(ids) != 5 or any(type(i) is not int for i in ids) or
                len(set(ids)) != 5 or not set(ids) <= raw.keys() or
                any(type(i) is not int for i in declared) or declared != sorted(raw) or
                z['source_coordinate_sha256'] != _digest(raw)): return False
        def vector(value,n):
            if not isinstance(value,(list,tuple)) or len(value) != n: raise ValueError('vector dimension')
            return tuple(_bounded(v) for v in value)
        def matrix(value):
            if not isinstance(value,(list,tuple)) or len(value) != 2: raise ValueError('matrix dimension')
            return tuple(vector(row,2) for row in value)
        conic = vector(z['conic_coefficients'],6)
        a,b,e,d,f,g = conic
        if a != 1 or a*e-b*b <= 0: return False
        evaluation = [_evaluation(raw[i]) for i in ids]
        if any(sum(v*co for v,co in zip(row,conic)) != 0 for row in evaluation): return False
        cols = z['rank5_minor_columns']
        if (not isinstance(cols,(list,tuple)) or len(cols)!=5 or any(type(i) is not int for i in cols) or
                list(cols) != sorted(set(cols)) or min(cols)<0 or max(cols)>5): return False
        minor_det = _determinant([[row[k] for k in cols] for row in evaluation])
        if minor_det == 0 or _bounded(z['rank5_minor_determinant']) != minor_det: return False
        c = vector(z['center'],2); s = matrix(z['shape_matrix']); h = matrix(z['inverse_shape_matrix'])
        rho = _bounded(z['radius_squared_level'])
        if rho <= 0 or rho != -d*c[0]-f*c[1]-g: return False
        m = ((a,b),(b,e))
        if any(sum(m[j][k]*c[k] for k in (0,1)) != -(d,f)[j] for j in (0,1)): return False
        for matrix_value in (s,h):
            if (matrix_value[0][1] != matrix_value[1][0] or matrix_value[0][0] <= 0 or
                    matrix_value[0][0]*matrix_value[1][1]-matrix_value[0][1]**2 <= 0): return False
        for j in (0,1):
            for k in (0,1):
                if sum(m[j][l]*s[l][k] for l in (0,1)) != rho*int(j==k): return False
                if sum(s[j][l]*h[l][k] for l in (0,1)) != int(j==k): return False
        weights = vector(z['contact_weights'],5)
        if min(weights) <= 0 or sum(weights) != 1: return False
        for j in (0,1):
            if sum(weights[t]*(raw[i][j]-c[j]) for t,i in enumerate(ids)) != 0: return False
            for k in (0,1):
                if sum(weights[t]*(raw[i][j]-c[j])*(raw[i][k]-c[k]) for t,i in enumerate(ids)) != s[j][k]/2: return False
        qs = z['all_input_quadratic_values']
        if not isinstance(qs,dict) or any(type(i) is not int for i in qs) or set(qs)!=set(raw): return False
        for i,p in raw.items():
            q = _q(p,c,h)
            if _bounded(qs[i]) != q or q > 1 or (i in ids and q != 1): return False
        det_s = _bounded(s[0][0]*s[1][1]-s[0][1]**2)
        for key,value in (('determinant_shape',det_s),('area_squared_over_pi_squared',det_s),('covariance_determinant',det_s/4)):
            if _bounded(z[key]) != value: return False
        trace,disc = s[0][0]+s[1][1],(s[0][0]-s[1][1])**2+4*s[0][1]**2
        for key,expression in (('semiaxis_squared','(trace +/- sqrt(discriminant))/2'),
                               ('major_axis_length','sqrt(2*(trace+sqrt(discriminant)))')):
            value = z[key]
            if (not isinstance(value,dict) or set(value)!= {'expression','trace','discriminant'} or
                    value['expression'] != expression or _bounded(value['trace']) != trace or _bounded(value['discriminant']) != disc): return False
        return z['proof'] == PROOF and z['exact_optimum'] is True
    except (ValueError,TypeError,ArithmeticError,AttributeError,KeyError,IndexError,BudgetExceeded):
        return False
