"""S-2019-CODES p59: exact ellipse ratios on noncollinear-triple images.

The source's constant-ratio lemma cancels pi and sqrt(3). Project-derived
analytic triangle ellipses provide separately checkable geometric witnesses.
No degenerate ellipse convention, numeric optimizer, or renumbering is used.
"""
from fractions import Fraction as F
from itertools import combinations
from ..registration.fixed_orientation import _Budget, _Exhausted, _points, _fraction
from .environment_code import build_environment_code, assess_environment_code

SCHEMA='ellipse-image-code-2019-v1'


def _input(points):
    raw=_points(points)
    if not 3<=len(raw)<=8:raise ValueError('complete 3..8 distinct rational 2D points required')
    return raw


def _area(a,b,c):
    return abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))


def _key(ids):return ','.join(str(i) for i in ids)


def _tree(v,budget,depth=0):
    budget.spend()
    if depth>24:raise ValueError('certificate depth bound')
    if type(v) in (int,F):_fraction(v,16384)
    elif type(v) is bool or v is None:pass
    elif type(v) is str:
        if len(v)>256:raise ValueError('certificate string bound')
    elif type(v) in (tuple,list,dict):
        if len(v)>4096:raise ValueError('certificate container bound')
        if type(v) is dict:
            for k,x in v.items():_tree(k,budget,depth+1);_tree(x,budget,depth+1)
        else:
            for x in v:_tree(x,budget,depth+1)
    else:raise ValueError('bounded pure exact certificate required')


def _native_triangle(raw,ids):
    c=tuple(sum(raw[i][k] for i in ids)/3 for k in (0,1))
    delta=[tuple(raw[i][k]-c[k] for k in (0,1)) for i in ids]
    s=tuple(tuple(F(2,3)*sum(d[j]*d[k] for d in delta) for k in (0,1)) for j in (0,1))
    det=s[0][0]*s[1][1]-s[0][1]**2
    inv=((s[1][1]/det,-s[0][1]/det),(-s[0][1]/det,s[0][0]/det))
    return dict(labels=ids,center=c,shape_matrix=s,inverse_shape_matrix=inv,
                area_squared_over_pi_squared=det,contact_weights=(F(1,3),)*3)


def _verify_triangle(raw,ids,c):
    if type(c) is not dict or set(c)!={'labels','center','shape_matrix','inverse_shape_matrix','area_squared_over_pi_squared','contact_weights'}:raise ValueError('triangle fields')
    if type(c['labels']) is not tuple or any(type(i) is not int for i in c['labels']) or c['labels']!=ids:raise ValueError('triangle labels')
    center,s,inv,w=c['center'],c['shape_matrix'],c['inverse_shape_matrix'],c['contact_weights']
    if type(center) is not tuple or len(center)!=2:raise ValueError('center shape')
    for matrix in (s,inv):
        if type(matrix) is not tuple or len(matrix)!=2 or any(type(row) is not tuple or len(row)!=2 for row in matrix):raise ValueError('matrix shape')
    if type(w) is not tuple or len(w)!=3:raise ValueError('weight shape')
    scalars=list(center)+[v for m in (s,inv) for row in m for v in row]+list(w)+[c['area_squared_over_pi_squared']]
    if any(type(v) not in (int,F) for v in scalars):raise ValueError('exact geometry scalars')
    if w!=(F(1,3),)*3 or s[0][1]!=s[1][0] or s[0][0]<=0:raise ValueError('positive symmetric triangle witness')
    det=s[0][0]*s[1][1]-s[0][1]**2
    d=_area(*(raw[i] for i in ids))
    if d==0 or det<=0 or det!=c['area_squared_over_pi_squared'] or det!=F(4,27)*d*d:raise ValueError('ellipse area identity')
    for j in (0,1):
        if sum(w[t]*raw[i][j] for t,i in enumerate(ids))!=center[j]:raise ValueError('weighted center')
        for k in (0,1):
            if sum(s[j][l]*inv[l][k] for l in (0,1))!=int(j==k):raise ValueError('inverse identity')
            cov=sum(w[t]*(raw[i][j]-center[j])*(raw[i][k]-center[k]) for t,i in enumerate(ids))
            if 2*cov!=s[j][k]:raise ValueError('covariance equality')
    for i in ids:
        d=tuple(raw[i][k]-center[k] for k in (0,1))
        if sum(d[j]*inv[j][k]*d[k] for j in (0,1) for k in (0,1))!=1:raise ValueError('all vertices must be contacts')


def _remaining(budget):
    import time
    return dict(max_work=max(0,budget.maximum-budget.used),seconds=max(0,budget.deadline-time.monotonic()))


def _verify(raw,c,budget):
    _tree(c,budget)
    if type(c) is not dict or set(c)!={'schema','source','domain_policy','area_code','ellipses','ellipse_ratios','geometric_meaning'}:raise ValueError('ellipse code fields')
    if c['schema']!=SCHEMA or c['source']!='S-2019-CODES:59' or c['domain_policy']!='every_triple_noncollinear':raise ValueError('ellipse source/domain')
    if c['geometric_meaning']!='(x-center)^T inverse_shape (x-center) <= 1; area^2/pi^2 = det(shape)':raise ValueError('ellipse semantics')
    check=assess_environment_code(raw,c['area_code'],**_remaining(budget));budget.spend(check['work_used'])
    if check['status']=='UNKNOWN':raise _Exhausted('area code verification budget')
    if check['status']!='VERIFIED':raise ValueError('complete area code invalid')
    if c['area_code']['zero_denominators']:raise ValueError('collinear triple: no positive-area minimum ellipse')
    keys={_key(ids):ids for ids in combinations(sorted(raw),3)}
    if type(c['ellipses']) is not dict or set(c['ellipses'])!=set(keys):raise ValueError('complete triangle ellipse set')
    for key,ids in keys.items():
        budget.spend(100);_verify_triangle(raw,ids,c['ellipses'][key])
    ratios=c['ellipse_ratios']; expected=c['area_code']['ratios']
    if type(ratios) is not dict or set(ratios)!=set(expected):raise ValueError('complete indexed ellipse ratios')
    for key,v in ratios.items():
        budget.spend()
        if type(v) not in (int,F) or v<=0 or v!=expected[key]:raise ValueError('exact lemma-cancelled ratio')
        numerator,denominator=key.split('/')
        if v*v*c['ellipses'][denominator]['area_squared_over_pi_squared']!=c['ellipses'][numerator]['area_squared_over_pi_squared']:raise ValueError('ratio geometry mismatch')


def build_ellipse_code2019(points,*,max_work=2000000,seconds=30):
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend()
        raw=_input(points)
        for ids in combinations(sorted(raw),3):
            budget.spend()
            if _area(*(raw[i] for i in ids))==0:raise ValueError('every triple must be noncollinear; no degenerate ellipse convention')
        area=build_environment_code(raw,**_remaining(budget));budget.spend(area['work_used'])
        if area['status']=='UNKNOWN':raise _Exhausted('environment construction budget')
        if area.get('certificate') is None:raise ValueError('environment construction failed: '+area['status'])
        ellipses={}
        for ids in combinations(sorted(raw),3):
            budget.spend(100);ellipses[_key(ids)]=_native_triangle(raw,ids)
        c=dict(schema=SCHEMA,source='S-2019-CODES:59',domain_policy='every_triple_noncollinear',
               area_code=area['certificate'],ellipses=ellipses,ellipse_ratios=dict(area['certificate']['ratios']),
               geometric_meaning='(x-center)^T inverse_shape (x-center) <= 1; area^2/pi^2 = det(shape)')
        _verify(raw,c,budget)
        return dict(status='EXACT_ELLIPSE_CODE',certificate=c,work_used=budget.used,
                    theorem_domain=c['area_code']['plane_domain'],affine_equivalence_decided=False)
    except _Exhausted as e:return dict(status='UNKNOWN',reason=str(e),certificate=None,work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:return dict(status='UNSUPPORTED',reason=str(e),certificate=None,work_used=budget.used if budget else 0)


def assess_ellipse_code2019(points,certificate,*,max_work=2000000,seconds=30):
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend()
        _verify(_input(points),certificate,budget)
        return dict(status='VERIFIED',work_used=budget.used)
    except _Exhausted as e:return dict(status='UNKNOWN',reason=str(e),work_used=budget.used)
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError) as e:return dict(status='INVALID',reason=str(e),work_used=budget.used if budget else 0)
