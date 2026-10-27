"""S-2009 pp106-107: acceptance of a supplied placement, not its generation.

The source defines ordinary and narrow acceptance. Augmenting paths and Hall
certificates are this project's exact implementation of those finite predicates.
"""
from fractions import Fraction as F
from .fixed_orientation import _Budget, _Exhausted, _points, _fraction

CANDIDATE_SCHEMA='placed-candidate-matching-v1'
BOTTLENECK_SCHEMA='placed-bottleneck-matching-v1'


def _rows(points):
    return tuple(sorted(_points(points).items()))


def _tree(x,budget,depth=0):
    budget.spend()
    if depth>20:raise ValueError('certificate nesting bound')
    if type(x) in (int,F):_fraction(x,16384)
    elif x is None or type(x) is bool:pass
    elif type(x) is str:
        if len(x)>128:raise ValueError('certificate string bound')
    elif type(x) in (tuple,list,dict):
        if len(x)>2048:raise ValueError('certificate container bound')
        if type(x) is dict:
            for k,v in x.items():_tree(k,budget,depth+1);_tree(v,budget,depth+1)
        else:
            for v in x:_tree(v,budget,depth+1)
    else:raise ValueError('exact bounded certificate required')


def _same(a,b):
    if type(a) is not type(b):return False
    if type(a) in (tuple,list):return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    if type(a) is dict:return set(a)==set(b) and all(_same(a[k],b[k]) for k in a)
    return a==b


def _distances(a,b,budget):
    d={}
    for i,p in a:
        for j,q in b:
            budget.spend()
            d[i,j]=(p[0]-q[0])**2+(p[1]-q[1])**2
    return d


def _graph(a,b,distances,radius,budget):
    graph={i:[] for i,_ in a}
    for i,_ in a:
        for j,_ in b:
            budget.spend()
            if distances[i,j]<=radius:graph[i].append(j)
    return {i:tuple(js) for i,js in graph.items()}


def _matching(graph,budget):
    right={}
    def augment(i,seen):
        for j in graph[i]:
            budget.spend()
            if j in seen:continue
            seen.add(j)
            if j not in right or augment(right[j],seen):
                right[j]=i;return True
        return False
    for i in graph:augment(i,set())
    return tuple(sorted((i,j) for j,i in right.items()))


def _hall(graph,matching,budget):
    matched=dict(matching);reverse={j:i for i,j in matching}
    left=set(graph)-set(matched);right=set();queue=sorted(left)
    for i in queue:
        for j in graph[i]:
            budget.spend()
            if j in right:continue
            right.add(j)
            if j in reverse and reverse[j] not in left:
                left.add(reverse[j]);queue.append(reverse[j])
    if not left or len(right)>=len(left):raise ArithmeticError('no Hall deficiency from maximum matching')
    return dict(kind='hall_deficiency',source_subset=tuple(sorted(left)),target_neighbors=tuple(sorted(right)))


def _certificate(a,b,distances,radius,budget):
    graph=_graph(a,b,distances,radius,budget)
    edges=tuple((i,j) for i,js in graph.items() for j in js)
    acceptable=all(graph.values()) and {j for _,j in edges}=={j for j,_ in b}
    if len(a)!=len(b):
        mapping=None;obstruction=dict(kind='unequal_cardinality',source_count=len(a),target_count=len(b))
    else:
        result=_matching(graph,budget)
        if len(result)==len(a):mapping=result;obstruction=None
        else:mapping=None;obstruction=_hall(graph,result,budget)
    return dict(schema=CANDIDATE_SCHEMA,source_points=a,target_points=b,radius_squared=radius,
                edges=edges,acceptable=acceptable,narrow_acceptable=mapping is not None,
                matching=mapping,obstruction=obstruction)


def _verify_candidate(a,b,certificate,budget,tree=True):
    c=certificate
    if tree:_tree(c,budget)
    keys={'schema','source_points','target_points','radius_squared','edges','acceptable','narrow_acceptable','matching','obstruction'}
    if type(c) is not dict or set(c)!=keys or c['schema']!=CANDIDATE_SCHEMA:raise ValueError('candidate schema/fields')
    if not _same(c['source_points'],a) or not _same(c['target_points'],b):raise ValueError('complete input binding')
    r=_fraction(c['radius_squared'],16384)
    if r<0:raise ValueError('negative radius squared')
    graph=_graph(a,b,_distances(a,b,budget),r,budget)
    edges=tuple((i,j) for i,js in graph.items() for j in js)
    expected=all(graph.values()) and {j for _,j in edges}=={j for j,_ in b}
    if not _same(c['edges'],edges) or type(c['acceptable']) is not bool or c['acceptable']!=expected:raise ValueError('complete threshold graph')
    if type(c['narrow_acceptable']) is not bool:raise ValueError('typed narrow predicate')
    if c['narrow_acceptable']:
        m=c['matching']
        if type(m) is not tuple or len(m)!=len(a) or len(a)!=len(b) or c['obstruction'] is not None:raise ValueError('complete perfect matching required')
        if any(type(pair) is not tuple or len(pair)!=2 or any(type(i) is not int for i in pair) for pair in m):raise ValueError('integer matching labels')
        if {i for i,j in m}!={i for i,_ in a} or {j for i,j in m}!={j for j,_ in b}:raise ValueError('bijection')
        if any(j not in graph[i] for i,j in m):raise ValueError('matching edge exceeds threshold')
    else:
        if c['matching'] is not None or type(c['obstruction']) is not dict:raise ValueError('negative certificate required')
        o=c['obstruction']
        if len(a)!=len(b):
            if not _same(o,dict(kind='unequal_cardinality',source_count=len(a),target_count=len(b))):raise ValueError('cardinality witness')
        else:
            if set(o)!={'kind','source_subset','target_neighbors'} or o['kind']!='hall_deficiency':raise ValueError('Hall schema')
            subset=o['source_subset']
            if type(subset) is not tuple or not subset or any(type(i) is not int for i in subset):raise ValueError('typed Hall subset')
            if len(set(subset))!=len(subset) or not set(subset)<=set(graph):raise ValueError('Hall subset membership')
            neighbors=tuple(sorted({j for i in subset for j in graph[i]}))
            if not _same(o['target_neighbors'],neighbors) or len(neighbors)>=len(subset):raise ValueError('strict Hall deficiency')
    return c


def candidate_acceptance_exact(source,target,radius_squared,*,max_work=2000000,seconds=30):
    """Complete exact ordinary/narrow acceptance of an already placed candidate."""
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        a,b=_rows(source),_rows(target);r=_fraction(radius_squared)
        if r<0:raise ValueError('nonnegative radius squared required')
        c=_certificate(a,b,_distances(a,b,budget),r,budget);_verify_candidate(a,b,c,budget)
        return dict(status='EXACT_PLACED_CANDIDATE',acceptable=c['acceptable'],narrow_acceptable=c['narrow_acceptable'],
                    certificate=c,work_used=budget.used,global_registration_decided=False)
    except _Exhausted as e:return dict(status='UNKNOWN',reason=str(e),certificate=None,work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:return dict(status='UNSUPPORTED',reason=str(e),certificate=None,work_used=budget.used if budget else 0)


def assess_candidate_acceptance_certificate(source,target,certificate,*,max_work=2000000,seconds=30):
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        c=_verify_candidate(_rows(source),_rows(target),certificate,budget)
        return dict(status='VERIFIED',acceptable=c['acceptable'],narrow_acceptable=c['narrow_acceptable'],work_used=budget.used)
    except _Exhausted as e:return dict(status='UNKNOWN',reason=str(e),work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:return dict(status='INVALID',reason=str(e),work_used=budget.used if budget else 0)


def _verify_bottleneck(a,b,c,budget):
    _tree(c,budget)
    if type(c) is not dict or set(c)!={'schema','source_points','target_points','distance_squared','upper','lower'} or c['schema']!=BOTTLENECK_SCHEMA:raise ValueError('bottleneck schema')
    if len(a)!=len(b) or not _same(c['source_points'],a) or not _same(c['target_points'],b):raise ValueError('complete equal-sized binding')
    values=sorted(set(_distances(a,b,budget).values()))
    r=_fraction(c['distance_squared'],16384)
    if r not in values:raise ValueError('optimum must be an attained edge length')
    upper=_verify_candidate(a,b,c['upper'],budget,False)
    if not upper['narrow_acceptable'] or upper['radius_squared']!=r:raise ValueError('upper bound matching')
    k=values.index(r)
    if k==0:
        if not _same(c['lower'],dict(kind='minimum_edge_squared',value=r)):raise ValueError('no shorter edge exists')
    else:
        lower=_verify_candidate(a,b,c['lower'],budget,False)
        if lower['narrow_acceptable'] or lower['radius_squared']!=values[k-1]:raise ValueError('preceding threshold Hall bound')
    return c


def placed_bottleneck_exact(source,target,*,max_work=2000000,seconds=30):
    """Min over all bijections at the supplied placement, with no transform search."""
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        a,b=_rows(source),_rows(target)
        if len(a)!=len(b):raise ValueError('bottleneck bijection needs equal cardinality')
        distances=_distances(a,b,budget);values=sorted(set(distances.values()))
        lo,hi=0,len(values)-1
        while lo<hi:
            middle=(lo+hi)//2
            if len(_matching(_graph(a,b,distances,values[middle],budget),budget))==len(a):hi=middle
            else:lo=middle+1
        r=values[lo]
        upper=_certificate(a,b,distances,r,budget)
        lower=_certificate(a,b,distances,values[lo-1],budget) if lo else dict(kind='minimum_edge_squared',value=r)
        c=dict(schema=BOTTLENECK_SCHEMA,source_points=a,target_points=b,distance_squared=r,upper=upper,lower=lower)
        _verify_bottleneck(a,b,c,budget)
        return dict(status='EXACT_PLACED_BOTTLENECK',distance_squared=r,certificate=c,work_used=budget.used,global_registration_decided=False)
    except _Exhausted as e:return dict(status='UNKNOWN',reason=str(e),certificate=None,work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:return dict(status='UNSUPPORTED',reason=str(e),certificate=None,work_used=budget.used if budget else 0)


def assess_placed_bottleneck_certificate(source,target,certificate,*,max_work=2000000,seconds=30):
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        c=_verify_bottleneck(_rows(source),_rows(target),certificate,budget)
        return dict(status='VERIFIED',distance_squared=c['distance_squared'],work_used=budget.used)
    except _Exhausted as e:return dict(status='UNKNOWN',reason=str(e),work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:return dict(status='INVALID',reason=str(e),work_used=budget.used if budget else 0)
