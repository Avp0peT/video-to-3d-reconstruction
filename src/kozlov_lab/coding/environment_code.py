"""S-2019-CODES pp57-58 exact indexed area ratios and literal environment restriction.

The finite enumeration, storage and certificates are project implementations of
source definitions. This module does not decide affine equivalence or produce ellipses.
"""
from fractions import Fraction as F
from itertools import combinations
from ..registration.fixed_orientation import _Budget, _Exhausted, _fraction, _label

ENV_SCHEMA = 'indexed-environment-code-v1'
SUB_SCHEMA = 'restricted-environment-code-v1'
SOURCE = 'S-2019-CODES:57-58'


def _points(raw):
    if type(raw) is not dict or not 1 <= len(raw) <= 8:
        raise ValueError('complete labelled 1..8 point dictionary required')
    out = {}
    for label, point in raw.items():
        _label(label)
        if type(point) not in (list, tuple) or len(point) != 2:
            raise ValueError('two exact rational coordinates required')
        out[label] = tuple(_fraction(v, 256) for v in point)
    if len(set(out.values())) != len(out):
        raise ValueError('duplicate environment coordinates')
    return tuple(sorted(out.items()))


def _labels(raw, points):
    if type(raw) not in (tuple, list) or not 1 <= len(raw) <= len(points):
        raise ValueError('nonempty bounded subset labels required')
    for label in raw:
        _label(label)
    if len(set(raw)) != len(raw) or not set(raw) <= {i for i, _ in points}:
        raise ValueError('unique environment subset labels required')
    return tuple(sorted(raw))


def _tree(v, budget, depth=0):
    budget.spend()
    if depth > 16:
        raise ValueError('certificate depth bound')
    if type(v) in (int, F):
        _fraction(v, 16384)
    elif type(v) is bool:
        pass
    elif type(v) is str:
        if len(v) > 128:
            raise ValueError('certificate string bound')
    elif type(v) in (tuple, dict):
        if len(v) > 4096:
            raise ValueError('certificate container bound')
        if type(v) is dict:
            if any(type(k) is not str for k in v):
                raise ValueError('string certificate keys required')
            for k, x in v.items():
                _tree(k, budget, depth+1)
                _tree(x, budget, depth+1)
        else:
            for x in v:
                _tree(x, budget, depth+1)
    else:
        raise ValueError('pure exact dict/tuple certificate required')


def _same(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is tuple:
        return len(a) == len(b) and all(_same(x,y) for x,y in zip(a,b))
    if type(a) is dict:
        return set(a) == set(b) and all(_same(a[k],b[k]) for k in a)
    return a == b


def _key(triple):
    return ','.join(str(i) for i in triple)


def _domain(points, budget):
    if len(points) < 3:
        return 'collinear'
    for (_, p), (_, q) in combinations(points, 2):
        dx, dy = q[0]-p[0], q[1]-p[1]
        offsets = set()
        for _, r in points:
            budget.spend()
            offsets.add(dx*r[1]-dy*r[0])
        if len(offsets) == 1:
            return 'collinear'
        if len(offsets) == 2:
            return 'two_parallel_lines'
    return 'plane'


def _build(points, budget):
    coords = dict(points)
    triples = tuple(combinations(coords, 3))
    areas = {}
    for triple in triples:
        budget.spend()
        a,b,c = (coords[i] for i in triple)
        areas[_key(triple)] = abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    ratios = {}
    for numerator, num in areas.items():
        for denominator, den in areas.items():
            budget.spend()
            if numerator != denominator and den != 0:
                ratios[numerator+'/'+denominator] = num/den
    domain = _domain(points,budget)
    return dict(schema=ENV_SCHEMA,source=SOURCE,environment_points=points,
                labels=tuple(coords),double_areas=areas,
                zero_denominators=tuple(k for k,v in areas.items() if v == 0),
                ratios=ratios,code_defined=any(v != 0 for v in areas.values()),
                domain=domain,plane_domain=domain == 'plane')


def _verify(points, c, budget, tree=True):
    if tree:
        _tree(c,budget)
    fields={'schema','source','environment_points','labels','double_areas',
            'zero_denominators','ratios','code_defined','domain','plane_domain'}
    if type(c) is not dict or set(c) != fields or c['schema'] != ENV_SCHEMA or c['source'] != SOURCE:
        raise ValueError('environment schema/fields/source')
    if not _same(c['environment_points'],points) or not _same(c['labels'],tuple(i for i,_ in points)):
        raise ValueError('complete global input binding')
    # Independent recomputation uses the expanded determinant, not the constructor.
    areas = {}
    for first,second,third in combinations(points,3):
        budget.spend()
        i,(ax,ay)=first;j,(bx,by)=second;k,(cx,cy)=third
        areas[_key((i,j,k))] = abs(ax*by+bx*cy+cx*ay-ay*bx-by*cx-cy*ax)
    if not _same(c['double_areas'],areas):
        raise ValueError('complete double-area entries')
    zeros=tuple(k for k,v in areas.items() if v == 0)
    if not _same(c['zero_denominators'],zeros):
        raise ValueError('complete degenerate denominators')
    ratios={}
    for denominator, den in areas.items():
        for numerator, num in areas.items():
            budget.spend()
            if den and numerator != denominator:
                ratios[numerator+'/'+denominator]=num/den
    if not _same(c['ratios'],ratios):
        raise ValueError('complete indexed ratios; nonzero denominators, no self pairs')
    defined=any(v != 0 for v in areas.values())
    domain=_domain(points,budget)
    if (type(c['code_defined']) is not bool or c['code_defined'] != defined
            or type(c['plane_domain']) is not bool or c['plane_domain'] != (domain == 'plane')
            or c['domain'] != domain):
        raise ValueError('exact code/domain predicates')
    return c


def _projection(env, labels, budget):
    allowed=set(labels)
    areas={}
    for key,value in env['double_areas'].items():
        budget.spend()
        if set(map(int,key.split(','))) <= allowed:
            areas[key]=value
    ratios={}
    for key,value in env['ratios'].items():
        budget.spend()
        if all(set(map(int,part.split(','))) <= allowed for part in key.split('/')):
            ratios[key]=value
    points=tuple(row for row in env['environment_points'] if row[0] in allowed)
    domain=_domain(points,budget)
    return dict(labels=labels,double_areas=areas,
                zero_denominators=tuple(k for k,v in areas.items() if v == 0),
                ratios=ratios,code_defined=any(v != 0 for v in areas.values()),
                domain=domain,plane_domain=domain == 'plane')


def _verify_restriction(points, labels, c, budget):
    _tree(c,budget)
    fields={'schema','source','environment','subset_labels','restricted'}
    if type(c) is not dict or set(c) != fields or c['schema'] != SUB_SCHEMA or c['source'] != SOURCE:
        raise ValueError('restriction schema/fields/source')
    if not _same(c['subset_labels'],labels):
        raise ValueError('global subset binding')
    env=_verify(points,c['environment'],budget,False)
    # Re-select by exhaustive environment indices, independent of projection builder.
    selected_areas={}
    for triple in combinations(labels,3):
        budget.spend()
        key=_key(triple)
        selected_areas[key]=env['double_areas'][key]
    selected_ratios={}
    for left in selected_areas:
        for right in selected_areas:
            budget.spend()
            key=left+'/'+right
            if key in env['ratios']:
                selected_ratios[key]=env['ratios'][key]
    subpoints=tuple((i,p) for i,p in points if i in set(labels))
    domain=_domain(subpoints,budget)
    expected=dict(labels=labels,double_areas=selected_areas,
                  zero_denominators=tuple(k for k,v in selected_areas.items() if v == 0),
                  ratios=selected_ratios,code_defined=any(v != 0 for v in selected_areas.values()),
                  domain=domain,plane_domain=domain == 'plane')
    if not _same(c['restricted'],expected):
        raise ValueError('complete literal global restriction')
    return c


def _run(points, labels, certificate, mode, max_work, seconds):
    budget=None
    verify=mode.startswith('assess')
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        rows=_points(points)
        subset=_labels(labels,rows) if mode.endswith('restricted') else None
        if mode == 'build':
            c=_build(rows,budget);_verify(rows,c,budget)
        elif mode == 'restrict_restricted':
            env=_build(rows,budget)
            c=dict(schema=SUB_SCHEMA,source=SOURCE,environment=env,
                   subset_labels=subset,restricted=_projection(env,subset,budget))
            _verify_restriction(rows,subset,c,budget)
        elif mode == 'assess':
            c=_verify(rows,certificate,budget)
        else:
            c=_verify_restriction(rows,subset,certificate,budget)
        core=c['restricted'] if subset is not None else c
        return dict(status='VERIFIED' if verify else ('EXACT_RESTRICTED_CODE' if subset is not None else 'EXACT_ENVIRONMENT_CODE'),
                    certificate=c,work_used=budget.used,code_defined=core['code_defined'],
                    plane_domain=core['plane_domain'],affine_equivalence_decided=False)
    except _Exhausted as e:
        return dict(status='UNKNOWN',reason=str(e),certificate=None,work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:
        return dict(status='INVALID' if verify else 'UNSUPPORTED',reason=str(e),
                    certificate=None,work_used=budget.used if budget else 0)


def build_environment_code(points, *, max_work=2000000, seconds=30):
    return _run(points,None,None,'build',max_work,seconds)


def restrict_environment_code(points, labels, *, max_work=2000000, seconds=30):
    return _run(points,labels,None,'restrict_restricted',max_work,seconds)


def assess_environment_code(points, certificate, *, max_work=2000000, seconds=30):
    return _run(points,None,certificate,'assess',max_work,seconds)


def assess_restricted_code(points, labels, certificate, *, max_work=2000000, seconds=30):
    return _run(points,labels,certificate,'assess_restricted',max_work,seconds)
