"""Bounded exact four-contact constructor for complete rational 1..7 inputs."""
from fractions import Fraction as F
from itertools import combinations
import math
import time

from .independent_cover import Budget, BudgetExceeded
from .exact_contact5_ellipse import _input, _bounded, _rref, _determinant, PROOF
from .exact_triangle_ellipse import _digest
from .real_algebraic import isolate_real_roots


def _poly(values):
    values = list(values)
    while len(values)>1 and not values[-1]: values.pop()
    return tuple(_bounded(v) for v in values)


def _add(a,b):
    return _poly([(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0)
                  for i in range(max(len(a),len(b)))])


def _mul(a,b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b): out[i+j] = _bounded(out[i+j]+x*y)
    return _poly(out)


def _scale(a,k): return _poly([k*x for x in a])
def _derivative(a): return _poly([i*a[i] for i in range(1,len(a))] or [0])


def _weights(points,center,s,root,budget):
    # The coordinate moment coefficient matrix is rational; only its RHS is
    # algebraic. Row transformations preserve all six equations exactly.
    rows = [[F(1) for _ in points], [p[0] for p in points], [p[1] for p in points],
            [p[0]**2 for p in points], [p[0]*p[1] for p in points], [p[1]**2 for p in points]]
    cx,cy = center
    target = (root.num((F(1),)),cx,cy,s[0][0]/2+cx**2,s[0][1]/2+cx*cy,s[1][1]/2+cy**2)
    rr,pivots = _rref([row+[F(i==j) for j in range(6)] for i,row in enumerate(rows)],4,budget)
    if len(pivots)!=4: return None
    weights = tuple(sum((row[4+i]*target[i] for i in range(6)),0) for row in rr[:4])
    if any(not sum((row[i]*weights[i] for i in range(4)),0).equals(value)
           for row,value in zip(rows,target)): return None
    return weights


def complete_input_contact4_ellipse(points, *, max_work=2000000, seconds=30):
    result = dict(schema='complete-input-contact4-ellipse-v1',status='UNSUPPORTED_INPUT',
        complete=False,exact_optimum=False,work_used=0,visited_quadruples=0,
        visited_roots=0,candidate_failures={},scope='complete supplied 1..7 rational input; four labels are support only')
    if (type(max_work) is not int or not 0<=max_work<=2000000 or
        type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=30):
        return dict(result,reason='invalid shared bounded budget')
    budget = Budget(max_work,seconds); validated = False
    def skip(reason): result['candidate_failures'][reason] = result['candidate_failures'].get(reason,0)+1
    try:
        budget.spend()
        if not isinstance(points,dict) or any(type(v) not in (int,F) for p in points.values() for v in p):
            raise ValueError('exact int/Fraction rational coordinates required')
        raw = _input(points); validated = True
        result.update(declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw))
        for ids in combinations(sorted(raw),4):
            budget.spend(); result['visited_quadruples'] += 1
            origin = tuple(_bounded(sum(raw[i][k] for i in ids)/4) for k in (0,1))
            local = [(raw[i][0]-origin[0],raw[i][1]-origin[1]) for i in ids]
            evaluation = [[x*x,2*x*y,y*y,2*x,2*y] for x,y in local]
            rr,pivots = _rref([row+[F(1)] for row in evaluation],5,budget)
            if len(pivots)!=4: skip('rank_not_four'); continue
            free = next(i for i in range(5) if i not in pivots)
            pencil = [(F(0),F(0)) for _ in range(5)]; pencil[free]=(F(0),F(1))
            for row,col in zip(rr,pivots): pencil[col] = (_bounded(row[5]),_bounded(-row[free]))
            a,b,c,d,e = pencil
            k = _add(_mul(a,c),_scale(_mul(b,b),-1))
            j = _add(_add(_mul(c,_mul(d,d)),_scale(_mul(b,_mul(d,e)),-2)),_mul(a,_mul(e,e)))
            u = _add(j,k)
            n = _add(_scale(_mul(_derivative(u),k),2),_scale(_mul(u,_derivative(k)),-3))
            budget.spend(100)
            if len(n)>4: raise ArithmeticError('stationary polynomial degree exceeds three')
            if n==(F(0),): skip('unsupported_identically_stationary_pencil'); continue
            for root in isolate_real_roots(n,budget):
                result['visited_roots'] += 1
                aa,bb,cc,dd,ee = [root.num(v) for v in pencil]
                kk,uu = root.num(k),root.num(u)
                if aa.sign()<=0 or kk.sign()<=0 or uu.sign()<=0:
                    skip('not_positive_definite_ellipse'); continue
                cl = ((bb*ee-cc*dd)/kk,(bb*dd-aa*ee)/kk)
                center = tuple(cl[i]+origin[i] for i in (0,1))
                s = ((uu*cc/kk**2,-uu*bb/kk**2),(-uu*bb/kk**2,uu*aa/kk**2))
                h = ((kk*aa/uu,kk*bb/uu),(kk*bb/uu,kk*cc/uu))
                weights = _weights([raw[i] for i in ids],center,s,root,budget)
                if weights is None: skip('moment_system_inconsistent'); continue
                if any(v.sign()<=0 for v in weights): skip('nonpositive_weight'); continue
                qs = {}
                for i,p in raw.items():
                    budget.spend(10); x,y = p[0]-center[0],p[1]-center[1]
                    qs[i] = h[0][0]*x*x+2*h[0][1]*x*y+h[1][1]*y*y
                if any((v-1).sign()>0 for v in qs.values()): skip('complete_input_outside'); continue
                det = s[0][0]*s[1][1]-s[0][1]**2
                trace = s[0][0]+s[1][1]; disc=(s[0][0]-s[1][1])**2+4*s[0][1]**2
                data = lambda v:v.to_data()
                certificate = dict(schema='complete-input-contact4-certificate-v1',
                    algebraic_root=root.to_data(),declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw),
                    support_labels=list(ids),center=[data(v) for v in center],
                    shape_matrix=[[data(v) for v in row] for row in s],
                    inverse_shape_matrix=[[data(v) for v in row] for row in h],
                    contact_weights=[data(v) for v in weights],all_input_quadratic_values={i:data(v) for i,v in qs.items()},
                    determinant_shape=data(det),covariance_determinant=data(det/4),area_squared_over_pi_squared=data(det),
                    semiaxis_squared=dict(expression='(trace +/- sqrt(discriminant))/2',trace=data(trace),discriminant=data(disc)),
                    major_axis_length=dict(expression='sqrt(2*(trace+sqrt(discriminant)))',trace=data(trace),discriminant=data(disc)),
                    proof=PROOF,exact_optimum=True,
                    construction=dict(normalization='support_mean_origin_F_minus_one',free_column=free,
                        rank4_minor_columns=pivots,rank4_minor_determinant=str(_determinant([[row[k] for k in pivots] for row in evaluation],budget)),
                        support_mean=[str(v) for v in origin],pencil=[[str(v) for v in row] for row in pencil],
                        stationary_polynomial=[str(v) for v in n]))
                from .exact_contact4_verify import assess_contact4_certificate
                assessment = assess_contact4_certificate(raw,certificate,max_work=max(0,budget.maximum-budget.used),
                    seconds=max(0,budget.deadline-time.monotonic()))
                budget.spend(assessment.get('work_used',0))
                if assessment['status']=='UNKNOWN': raise BudgetExceeded('independent verification incomplete')
                if assessment['status']!='VERIFIED': raise ArithmeticError('independent four-contact certificate verification failed')
                return dict(result,status='EXACT_OPTIMUM_COMPLETE_INPUT',complete=True,exact_optimum=True,
                            center=certificate['center'],certificate=certificate,work_used=budget.used)
        budget.spend(0)
        return dict(result,status='UNSUPPORTED_CONTACT_DOMAIN',work_used=budget.used,
                    reason='no strictly positive rank4 certificate found; does not disprove any ellipse')
    except BudgetExceeded as exc:
        return dict(result,status='PARTIAL_UNKNOWN',reason=str(exc),work_used=budget.used)
    except (ValueError,TypeError,ArithmeticError,AttributeError,KeyError,IndexError) as exc:
        return dict(result,status='PARTIAL_UNKNOWN' if validated else 'UNSUPPORTED_INPUT',reason=str(exc),work_used=budget.used)
