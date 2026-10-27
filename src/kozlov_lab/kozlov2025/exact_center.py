"""Finite bounded exact center API; no floating solver or production integration."""
from fractions import Fraction as F
import json
import math
import time

from .independent_cover import Budget, BudgetExceeded, normalize, orient
from .exact_triangle_ellipse import _digest
from . import exact_triangle_ellipse as triangle
from . import exact_contact3_ellipse as contact3
from . import exact_contact5_ellipse as contact5
from . import exact_contact4_ellipse as contact4
from .exact_contact4_verify import assess_contact4_certificate

MAX_BITS = 16384


def _input(points):
    if type(points) is not dict or not 1<=len(points)<=7:
        raise ValueError('complete 1..7 labelled rational points required')
    for label,p in points.items():
        if type(label) is not int or type(p) not in (list,tuple) or len(p)!=2:
            raise ValueError('integer labels and two coordinates required')
        if any(type(v) not in (int,F) for v in p):
            raise ValueError('int/Fraction only; bool and float rejected')
    return normalize(points,[sorted(points)])[0]


def _budget(work,seconds):
    if (type(work) is not int or not 0<=work<=2000000 or type(seconds) not in (int,float)
        or not math.isfinite(seconds) or not 0<=seconds<=30):
        raise ValueError('invalid bounded work/time budget')
    return Budget(work,seconds)


def _tree(value,budget,depth=0):
    budget.spend()
    if depth>32: raise ValueError('certificate nesting bound')
    if type(value) in (int,F):
        v=F(value)
        if max(v.numerator.bit_length(),v.denominator.bit_length())>MAX_BITS:
            raise BudgetExceeded('stored rational bit bound')
    elif type(value) in (tuple,list):
        if len(value)>128: raise ValueError('certificate array bound')
        for v in value: _tree(v,budget,depth+1)
    elif type(value) is dict:
        if len(value)>128 or any(type(k) not in (str,int) for k in value): raise ValueError('certificate mapping bound')
        for v in value.values(): _tree(v,budget,depth+1)
    elif type(value) is str:
        if len(value)>6000: raise ValueError('certificate string bound')
    elif type(value) is not bool: raise ValueError('unsupported exact certificate value')


def _rational_tree(value):
    if type(value) in (tuple,list): return all(_rational_tree(v) for v in value)
    if type(value) is dict: return all(_rational_tree(v) for v in value.values())
    return type(value) in (int,F,str)


def _degenerate(raw,budget):
    ids=sorted(raw)
    if len(ids)==1:
        p=raw[ids[0]]
        return dict(schema='exact-center-degenerate-certificate-v1',kind='point',
                    declared_labels=ids,source_coordinate_sha256=_digest(raw),endpoint_labels=ids,
                    center=p,major_axis_length=dict(expression='sqrt(length_squared)',length_squared=F(0)))
    a,b=sorted(ids,key=lambda i:raw[i])[0],sorted(ids,key=lambda i:raw[i])[-1]
    if any(orient(raw[a],raw[b],raw[i])!=0 for i in ids): return None
    budget.spend(len(raw)*10)
    center=tuple((raw[a][k]+raw[b][k])/2 for k in (0,1))
    length=sum((raw[b][k]-raw[a][k])**2 for k in (0,1))
    return dict(schema='exact-center-degenerate-certificate-v1',kind='segment',
                declared_labels=ids,source_coordinate_sha256=_digest(raw),endpoint_labels=[a,b],
                center=center,major_axis_length=dict(expression='sqrt(length_squared)',length_squared=length))


def _verify(raw,result,budget):
    _tree(result,budget)
    if (result['schema']!='exact-center-v1' or result['complete'] is not True or result['exact_optimum'] is not True
        or result['declared_labels']!=sorted(raw) or any(type(i) is not int for i in result['declared_labels'])
        or result['source_coordinate_sha256']!=_digest(raw)): return False
    cert=result['certificate']; profile=result['profile']; context=result['number_context']
    if result['center']!=cert['center'] or result['major_axis_length']!=cert['major_axis_length']: return False
    if profile=='contact4':
        def typed(value):
            return json.dumps(encode_exact(value),sort_keys=True,separators=(',',':'))
        if typed(context)!=typed(dict(domain='real-algebraic',algebraic_root=cert['algebraic_root'])): return False
        if any(typed(result[key])!=typed(cert[key]) for key in ('center','major_axis_length')): return False
        check=assess_contact4_certificate(raw,cert,max_work=max(0,budget.maximum-budget.used),seconds=max(0,budget.deadline-time.monotonic()))
        budget.spend(check['work_used'])
        if check['status']=='UNKNOWN': raise BudgetExceeded('independent algebraic verification incomplete')
        return result['status']=='EXACT_OPTIMUM_COMPLETE_INPUT' and check['status']=='VERIFIED'
    if context!=dict(domain='rational'): return False
    # Python equality alone would accept False == Fraction(0) inside the
    # nested certificate. Validate both sides before comparing identities.
    if not all(_rational_tree(owner[key]) for owner in (result,cert)
               for key in ('center','major_axis_length')): return False
    # Fixed bounded legacy identity checks: <=7 points, bounded tree/bit gate.
    # This is a cooperative work charge, not a count of CPU instructions.
    budget.spend(2000+200*len(raw))
    if profile in ('point','segment'):
        if result['status']!='EXACT_DEGENERATE_COMPLETE_INPUT' or cert['schema']!='exact-center-degenerate-certificate-v1': return False
        if cert['declared_labels']!=sorted(raw) or cert['source_coordinate_sha256']!=_digest(raw) or cert['kind']!=profile: return False
        ids=cert['endpoint_labels']
        if any(type(i) is not int for i in ids) or not set(ids)<=raw.keys(): return False
        if profile=='point':
            return len(raw)==1 and ids==sorted(raw) and cert['center']==raw[ids[0]] and cert['major_axis_length']==dict(expression='sqrt(length_squared)',length_squared=0)
        if len(ids)!=2 or len(set(ids))!=2: return False
        a,b=(raw[i] for i in ids)
        v=(b[0]-a[0],b[1]-a[1]); length=v[0]**2+v[1]**2
        if length<=0 or cert['center']!=tuple((a[k]+b[k])/2 for k in (0,1)): return False
        if cert['major_axis_length']!=dict(expression='sqrt(length_squared)',length_squared=length): return False
        for p in raw.values():
            projection=sum((p[k]-a[k])*v[k] for k in (0,1))
            if orient(a,b,p)!=0 or not 0<=projection<=length: return False
        return True
    if result['status']!='EXACT_OPTIMUM_COMPLETE_INPUT': return False
    if profile=='triangle': return triangle.verify_exact_triangle_ellipse(raw,cert)
    if profile=='contact3': return contact3.verify_complete_input_contact3_ellipse(raw,cert)
    if profile=='contact5': return contact5.verify_complete_input_contact5_ellipse(raw,cert)
    return False


def assess_exact_center_result(points,result,*,max_work=2000000,seconds=30):
    """Independent certificate identities; never invokes any constructor."""
    budget=None
    try:
        budget=_budget(max_work,seconds); budget.spend()
        raw=_input(points)
        valid=_verify(raw,result,budget); budget.spend(0)
        return dict(status='VERIFIED' if valid else 'INVALID',work_used=budget.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN',reason=str(exc),work_used=budget.used if budget else 0)
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(status='INVALID',reason=str(exc),work_used=budget.used if budget else 0)


def verify_exact_center_result(points,result,*,max_work=2000000,seconds=30):
    return assess_exact_center_result(points,result,max_work=max_work,seconds=seconds)['status']=='VERIFIED'


def solve_exact_center(points,*,max_work=2000000,seconds=30):
    """One shared budget across degenerate/triangle/contact3/contact5/contact4."""
    result=dict(schema='exact-center-v1',status='UNSUPPORTED_INPUT',complete=False,exact_optimum=False,
                work_used=0,attempts=[],scope='complete rational labelled 1..7 points; bounded exact research profile')
    budget=None; validated=False
    try:
        budget=_budget(max_work,seconds); budget.spend()
        raw=_input(points); validated=True
        result.update(declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw))
        cert=_degenerate(raw,budget)
        if cert:
            profile=cert['kind']; status='EXACT_DEGENERATE_COMPLETE_INPUT'
        else:
            cert=None
            stages=(('triangle',triangle.exact_triangle_enclosing_ellipse),('contact3',contact3.complete_input_contact3_ellipse),
                    ('contact5',contact5.complete_input_contact5_ellipse),('contact4',contact4.complete_input_contact4_ellipse))
            for profile,constructor in stages:
                budget.spend()
                candidate=constructor(raw,max_work=max(0,budget.maximum-budget.used),seconds=max(0,budget.deadline-time.monotonic()))
                budget.spend(candidate['work_used'])
                result['attempts'].append(dict(profile=profile,status=candidate['status'],work_used=candidate['work_used']))
                if candidate['exact_optimum'] is True:
                    cert=candidate['certificate']; status='EXACT_OPTIMUM_COMPLETE_INPUT'; break
                if candidate['status'] not in ('UNSUPPORTED_DOMAIN','UNSUPPORTED_CONTACT_DOMAIN'):
                    raise BudgetExceeded('bounded '+profile+' construction incomplete: '+candidate.get('reason','unknown'))
            if cert is None:
                return dict(result,status='PARTIAL_UNKNOWN',reason='no supported finite exact certificate found; does not disprove an ellipse',work_used=budget.used)
        context=dict(domain='real-algebraic',algebraic_root=cert['algebraic_root']) if profile=='contact4' else dict(domain='rational')
        result.update(status=status,complete=True,exact_optimum=True,profile=profile,number_context=context,
                      center=cert['center'],major_axis_length=cert['major_axis_length'],certificate=cert)
        if not _verify(raw,result,budget): raise ArithmeticError('independent exact center verification failed')
        budget.spend(0)
        result['work_used']=budget.used
        return result
    except BudgetExceeded as exc:
        return dict(result,status='PARTIAL_UNKNOWN',complete=False,exact_optimum=False,reason=str(exc),work_used=budget.used if budget else 0)
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(result,status='PARTIAL_UNKNOWN' if validated else 'UNSUPPORTED_INPUT',complete=False,exact_optimum=False,reason=str(exc),work_used=budget.used if budget else 0)


def encode_exact(value):
    """Identity-preserving JSON tags: no implicit integer-key stringification."""
    if type(value) is F: return {'$exact':'rational','numerator':str(value.numerator),'denominator':str(value.denominator)}
    if type(value) is tuple: return {'$exact':'tuple','items':[encode_exact(v) for v in value]}
    if type(value) is list: return [encode_exact(v) for v in value]
    if type(value) is dict:
        if value and all(type(k) is int for k in value):
            return {'$exact':'integer-map','items':[[k,encode_exact(v)] for k,v in value.items()]}
        if any(type(k) is not str for k in value) or '$exact' in value: raise ValueError('unsupported mapping keys')
        return {k:encode_exact(v) for k,v in value.items()}
    if type(value) in (int,str,bool) or value is None: return value
    raise ValueError('unsupported JSON exact value')


def decode_exact(value):
    remaining=[20000]
    def restore(v,depth=0):
        remaining[0]-=1
        if remaining[0]<0 or depth>32: raise ValueError('bounded exact JSON tree required')
        if type(v) is list: return [restore(x,depth+1) for x in v]
        if type(v) is dict:
            if '$exact' not in v: return {k:restore(x,depth+1) for k,x in v.items()}
            kind=v['$exact']
            if kind=='rational' and set(v)=={'$exact','numerator','denominator'}:
                def integer(x):
                    if type(x) is not str or not 1<=len(x)<=4300 or x!=str(int(x)): raise ValueError('canonical rational decimal required')
                    return int(x)
                n,d=integer(v['numerator']),integer(v['denominator'])
                if d<=0 or max(abs(n).bit_length(),d.bit_length())>MAX_BITS: raise ValueError('bounded positive rational denominator required')
                q=F(n,d)
                if (q.numerator,q.denominator)!=(n,d): raise ValueError('reduced rational required')
                return q
            if kind=='tuple' and set(v)=={'$exact','items'} and type(v['items']) is list:
                return tuple(restore(x,depth+1) for x in v['items'])
            if kind=='integer-map' and set(v)=={'$exact','items'} and type(v['items']) is list:
                out={}
                for item in v['items']:
                    if type(item) is not list or len(item)!=2 or type(item[0]) is not int or item[0] in out:
                        raise ValueError('unique integer-map labels required')
                    out[item[0]]=restore(item[1],depth+1)
                return out
            raise ValueError('invalid exact JSON tag')
        if type(v) in (int,str,bool) or v is None: return v
        raise ValueError('unsupported JSON exact value')
    return restore(value)
