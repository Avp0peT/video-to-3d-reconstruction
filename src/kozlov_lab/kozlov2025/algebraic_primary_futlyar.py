"""Given primary futlyar in one exact real algebraic domain (S-2025 PDF10).

A complete <=7 point partition has at most one block with >=4 points.
Consequently the supported center constructors introduce at most one cubic
root. Other centers embed as rationals; no compositum or floating comparison.
"""
import json
from .primary_futlyar import _budget, _remaining, _input, UnsupportedDomain
from .independent_cover import (BudgetExceeded, verify_independent_cover_2d,
                               verify_cover_certificate)
from .exact_center import solve_exact_center, assess_exact_center_result, _tree, encode_exact
from .exact_triangle_ellipse import _digest
from .real_algebraic import Root, AN


def _call(b,function,*args):
    answer=function(*args,**_remaining(b)); b.spend(answer['work_used'])
    return answer


def _domain(blocks,b):
    contexts=[row['result']['number_context'] for row in blocks]
    algebraic=[x['algebraic_root'] for x in contexts if x['domain']=='real-algebraic']
    if len(algebraic)>1:
        raise UnsupportedDomain('more than one algebraic center domain; no compositum implemented')
    # Rational-only inputs use the single real root of X for uniform encoding.
    data=algebraic[0] if algebraic else {'poly':[[0,1],[1,1]],'lo':[-1,1],'hi':[1,1]}
    return data,Root.from_data(data,b)


def _pieces(blocks,root):
    centers=[]; sizes=[]
    for row in blocks:
        answer=row['result']; context=answer['number_context']
        algebraic=context['domain']=='real-algebraic'
        def number(x):
            return AN.from_data(x,root) if algebraic else root.num((x,))
        center=tuple(number(x) for x in answer['center'])
        if len(center)!=2: raise ValueError('two coordinates required')
        size=answer['major_axis_length']
        if answer['profile'] in ('point','segment'):
            sizes.append(dict(kind='squared',squared=number(size['length_squared'])))
        else:
            sizes.append(dict(kind='ellipse',trace=number(size['trace']),discriminant=number(size['discriminant'])))
        centers.append(center)
    return centers,sizes


def _dist(a,b):
    return (a[0]-b[0])**2+(a[1]-b[1])**2


def _geometry(raw,groups,centers,sizes,root):
    if len(centers)<2: raise UnsupportedDomain('m=1 has no minimum center separation')
    radius=min(_dist(a,c) for i,a in enumerate(centers) for c in centers[i+1:])
    if radius<=0: raise UnsupportedDomain('coincident centers have unresolved primary radius')
    screens=[]
    for size in sizes:
        if size['kind']=='squared':
            screens.append(4*radius>=size['squared'])
        else:
            # R >= full major-axis/2 iff 2 R²-tr(S) >= sqrt(discriminant).
            u=2*radius-size['trace']
            screens.append(u>=0 and u*u>=size['discriminant'])
    checks=[]; failure=None; boundary=False
    for owner,group in enumerate(groups):
        for label in group:
            point=tuple(root.num((x,)) for x in raw[label])
            distances=[_dist(point,c) for c in centers]
            own=distances[owner]
            nearest=all(own<d for j,d in enumerate(distances) if j!=owner)
            inside=own<radius
            checks.append(dict(label=label,owner=owner,
                distances_squared=[d.to_data() for d in distances],
                uniquely_nearest=nearest,strictly_inside=inside))
            if not nearest: failure='point is not uniquely nearest to its assigned center'
            if own>radius: failure='point is outside assigned primary circle'
            if own==radius: boundary=True
    if not all(screens): failure='R is less than half the cover size'
    status='FALSE' if failure else 'UNSUPPORTED' if boundary else 'TRUE'
    reason=failure or ('primary circle equality boundary unresolved by source' if boundary else 'complete strict primary conditions certified')
    fields=dict(centers=[[x.to_data() for x in c] for c in centers],radius_squared=radius.to_data(),
                radius_screens=screens,point_checks=checks)
    return status,reason,fields


def assess_algebraic_primary_futlyar(points,parts,*,max_work=2000000,seconds=30):
    result=dict(schema='algebraic-primary-futlyar-v1',status='INVALID_INPUT',complete=False,work_used=0)
    b=None
    try:
        b=_budget(max_work,seconds); b.spend()
        raw,groups=_input(points,parts)
        if len(groups)<2: raise UnsupportedDomain('m=1 has no minimum center separation')
        cover=_call(b,verify_independent_cover_2d,raw,groups)
        if cover['status']=='NOT_INDEPENDENT':
            return dict(result,status='FALSE',complete=True,reason='given cover is not independent',work_used=b.used)
        if cover['status']!='VALID_INDEPENDENT_COVER':
            raise BudgetExceeded('independent cover unresolved')
        blocks=[]
        for group in groups:
            answer=_call(b,solve_exact_center,{i:raw[i] for i in group})
            if not answer['complete']: raise BudgetExceeded('full-block exact center unresolved')
            blocks.append(dict(labels=list(group),result=answer))
        data,root=_domain(blocks,b)
        centers,sizes=_pieces(blocks,root)
        status,reason,fields=_geometry(raw,groups,centers,sizes,root)
        certificate=dict(schema='algebraic-primary-certificate-v1',status=status,
            declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw),
            parts=[list(g) for g in groups],cover_certificate=cover,blocks=blocks,
            number_context=dict(domain='real-algebraic',algebraic_root=data),**fields)
        if status=='TRUE':
            check=_call(b,assess_algebraic_primary_certificate,raw,certificate)
            if check['status']=='UNKNOWN': raise BudgetExceeded('independent primary verification incomplete')
            if check['status']!='VERIFIED': raise ValueError('independent primary certificate rejected')
        b.spend(0)
        return dict(result,status=status,complete=status in ('TRUE','FALSE'),reason=reason,
                    certificate=certificate,work_used=b.used)
    except UnsupportedDomain as exc:
        return dict(result,status='UNSUPPORTED',reason=str(exc),work_used=b.used if b else 0)
    except BudgetExceeded as exc:
        return dict(result,status='UNKNOWN',reason=str(exc),work_used=b.used if b else 0)
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(result,reason=str(exc),work_used=b.used if b else 0)


def assess_algebraic_primary_certificate(points,certificate,*,max_work=2000000,seconds=30):
    """Independent exact identities and comparisons; never calls a constructor."""
    b=None
    try:
        b=_budget(max_work,seconds); b.spend(); z=certificate; _tree(z,b)
        required={'schema','status','declared_labels','source_coordinate_sha256','parts','cover_certificate',
                  'blocks','number_context','centers','radius_squared','radius_screens','point_checks'}
        if type(z) is not dict or set(z)!=required or z['schema']!='algebraic-primary-certificate-v1' or z['status']!='TRUE':
            raise ValueError('strict successful primary certificate required')
        raw,groups=_input(points,z['parts'])
        if z['declared_labels']!=sorted(raw) or any(type(i) is not int for i in z['declared_labels']) or z['source_coordinate_sha256']!=_digest(raw):
            raise ValueError('full input identity mismatch')
        b.spend(500+len(raw)*len(groups)**2)
        if z['cover_certificate']['status']!='VALID_INDEPENDENT_COVER' or not verify_cover_certificate(raw,groups,z['cover_certificate']):
            raise ValueError('independent cover certificate invalid')
        if type(z['blocks']) is not list or len(z['blocks'])!=len(groups): raise ValueError('complete block list required')
        for group,row in zip(groups,z['blocks']):
            if type(row) is not dict or set(row)!={'labels','result'} or row['labels']!=list(group) or any(type(i) is not int for i in row['labels']):
                raise ValueError('block identity mismatch')
            check=_call(b,assess_exact_center_result,{i:raw[i] for i in group},row['result'])
            if check['status']=='UNKNOWN': raise BudgetExceeded('center verification unfinished')
            if check['status']!='VERIFIED': raise ValueError('center certificate invalid')
        data,root=_domain(z['blocks'],b)
        context=dict(domain='real-algebraic',algebraic_root=data)
        # Encoded equality distinguishes JSON booleans from integer 0/1. Root
        # identity is bound to the verified block certificate, not a name.
        def canonical(value): return json.dumps(encode_exact(value),sort_keys=True,separators=(',',':'))
        if canonical(z['number_context'])!=canonical(context): raise ValueError('algebraic root context mismatch')
        centers,sizes=_pieces(z['blocks'],root)
        status,_,fields=_geometry(raw,groups,centers,sizes,root)
        if status!='TRUE' or any(canonical(z[key])!=canonical(value) for key,value in fields.items()):
            raise ValueError('exact primary geometry or typed cached fields mismatch')
        b.spend(0)
        return dict(status='VERIFIED',work_used=b.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN',reason=str(exc),work_used=b.used if b else 0)
    except (UnsupportedDomain,ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(status='INVALID',reason=str(exc),work_used=b.used if b else 0)


def verify_algebraic_primary_certificate(points,certificate,**kwargs):
    return assess_algebraic_primary_certificate(points,certificate,**kwargs)['status']=='VERIFIED'
