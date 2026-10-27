"""Complete1..7-point exact ellipse certificate with a three-point support.

The support does not replace the complete input. Missing a certificate is UNKNOWN
outside this supported profile, never evidence that an enclosing ellipse fails.
"""
from fractions import Fraction as F
from itertools import combinations
import math,time
from .independent_cover import normalize,orient,Budget,BudgetExceeded
from .exact_triangle_ellipse import exact_triangle_enclosing_ellipse,verify_exact_triangle_ellipse,_digest


def _input(points):
    if not isinstance(points,dict) or not 1<=len(points)<=7:
        raise ValueError('complete input must have1..7 distinct labelled points; no sampling')
    ids=sorted(points) if all(type(i) is int for i in points) else []
    return normalize(points,[ids])[0]


def _quadratic(p,c,h):
    x,y=p[0]-c[0],p[1]-c[1]
    return h[0][0]*x*x+2*h[0][1]*x*y+h[1][1]*y*y


def complete_input_contact3_ellipse(points,*,max_work=2000000,seconds=30):
    result=dict(schema='complete-input-contact3-ellipse-v1',status='UNSUPPORTED_INPUT',
        complete=False,exact_optimum=False,work_used=0,visited_triples=0,
        scope='complete supplied1..7 input; three labels are support only')
    if (type(max_work) is not int or not 0<=max_work<=2000000 or type(seconds) not in (int,float)
            or not math.isfinite(seconds) or not 0<=seconds<=30):
        return dict(result,reason='invalid shared bounded budget')
    budget=Budget(max_work,seconds)
    try:
        budget.spend()
        raw=_input(points)
        budget.spend(len(raw))
        result.update(declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw))
        for ids in combinations(sorted(raw),3):
            budget.spend()
            result['visited_triples']+=1
            if orient(*(raw[i] for i in ids))==0:continue
            subset={i:raw[i] for i in ids}
            answer=exact_triangle_enclosing_ellipse(subset,max_work=max_work-budget.used,
                seconds=max(0,budget.deadline-time.monotonic()))
            budget.spend(answer['work_used'])
            if answer['status']!='EXACT_OPTIMUM':
                return dict(result,status='PARTIAL_UNKNOWN',reason='support certificate did not complete',work_used=budget.used)
            z=answer['certificate'];c=z['center'];h=z['inverse_shape_matrix']
            evaluations={}
            for i,p in raw.items():
                budget.spend(10)
                evaluations[i]=_quadratic(p,c,h)
            if max(evaluations.values())>1:continue
            certificate=dict(schema='complete-input-contact3-certificate-v1',
                declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw),
                support_labels=list(ids),support_triangle_certificate=z,
                all_input_quadratic_values=evaluations,center=c,
                shape_matrix=z['shape_matrix'],inverse_shape_matrix=h,
                area_squared_over_pi_squared=z['area_squared_over_pi_squared'],
                semiaxis_squared=z['semiaxis_squared'],major_axis_length=z['major_axis_length'],
                exact_optimum=True,proof='support_optimality_plus_complete_input_exact_containment')
            budget.spend(len(raw)*10+40)
            if not verify_complete_input_contact3_ellipse(raw,certificate):
                raise ArithmeticError('independent complete-input certificate verification failed')
            budget.spend(0)
            return dict(result,status='EXACT_OPTIMUM_COMPLETE_INPUT',complete=True,exact_optimum=True,
                center=c,certificate=certificate,work_used=budget.used)
        budget.spend(0)
        return dict(result,status='UNSUPPORTED_CONTACT_DOMAIN',reason='all at most35 support triples checked; no complete-input contact3 certificate found; does not disprove any ellipse or ostov',work_used=budget.used)
    except BudgetExceeded as exc:
        return dict(result,status='PARTIAL_UNKNOWN',reason=str(exc),work_used=budget.used)
    except (ValueError,TypeError,ArithmeticError,AttributeError,KeyError,IndexError) as exc:
        return dict(result,reason=str(exc),work_used=budget.used)


def verify_complete_input_contact3_ellipse(points,certificate):
    """No solver/enumeration/hull. Check complete binding and support optimality.

    Independent support optimum is a lower bound for ANY ellipse containing all
    inputs. Full-input containment attains that bound and proves the same unique
    optimum; no barycentric-in-support-hull premise for other input points.
    """
    try:
        raw=_input(points);z=certificate
        if z['schema']!='complete-input-contact3-certificate-v1':return False
        ids=z['support_labels'];declared=z['declared_labels'];qs=z['all_input_quadratic_values']
        if (any(type(i) is not int for i in declared) or declared!=sorted(raw) or
            any(type(i) is not int for i in ids) or len(ids)!=3 or len(set(ids))!=3 or
            not set(ids)<=raw.keys() or any(type(i) is not int for i in qs) or set(qs)!=set(raw) or
            z['source_coordinate_sha256']!=_digest(raw)):
            return False
        support={i:raw[i] for i in ids};tri=z['support_triangle_certificate']
        if not verify_exact_triangle_ellipse(support,tri):return False
        for key in ('center','shape_matrix','inverse_shape_matrix','area_squared_over_pi_squared','semiaxis_squared','major_axis_length'):
            if z[key]!=tri[key]:return False
        def exact_tree(x):
            if isinstance(x,dict):return all(exact_tree(v) for v in x.values())
            if isinstance(x,(tuple,list)):return all(exact_tree(v) for v in x)
            return type(x) in (F,int,str)
        if not all(exact_tree(z[k]) for k in ('center','shape_matrix','inverse_shape_matrix','area_squared_over_pi_squared','semiaxis_squared','major_axis_length')):
            return False
        c,h=tri['center'],tri['inverse_shape_matrix']
        for i,p in raw.items():
            if type(qs[i]) not in (int,F):return False
            q=_quadratic(p,c,h)
            if qs[i]!=q or q>1:return False
        return z['exact_optimum'] is True and z['proof']=='support_optimality_plus_complete_input_exact_containment'
    except (ValueError,TypeError,ArithmeticError,AttributeError,KeyError,IndexError):
        return False
