"""Given independent cover -> exact centres for point/line/triangle-hull blocks.

No enumeration, sampling, general ellipse solver, or existing API changes.
"""
from fractions import Fraction as F
import math
import time
from .independent_cover import (normalize, hull, orient, Budget, BudgetExceeded,
    verify_independent_cover_2d, verify_cover_certificate)
from .exact_triangle_ellipse import (exact_triangle_enclosing_ellipse,
    verify_exact_triangle_ellipse, _digest)


def given_cover_supported_block_ostov(points, parts, *, max_work=2000000, seconds=30):
    result = dict(schema='supported-block-given-ostov-v1', status='UNSUPPORTED_INPUT',
                  complete=False, exact_centers=False, work_used=0,
                  scope='one given cover of complete supplied image; not all covers')
    if (type(max_work) is not int or not 0 <= max_work <= 2000000 or
            type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0 <= seconds <= 30):
        return dict(result, reason='invalid shared bounded budget')
    budget = Budget(max_work, seconds)
    def remaining():
        budget.spend(0)
        return max_work-budget.used, max(0,budget.deadline-time.monotonic())
    try:
        budget.spend()
        raw, groups, missing = normalize(points,parts)
        if missing:
            return dict(result,status='INCOMPLETE_COVER',reason='missing supplied labels',missing_labels=missing)
        budget.spend(len(raw))
        work, sec = remaining()
        cover = verify_independent_cover_2d(raw,groups,max_work=work,seconds=sec)
        budget.spend(cover['work_used'])
        if cover['status'] != 'VALID_INDEPENDENT_COVER':
            status = 'PARTIAL_UNKNOWN' if cover['status'].startswith('UNKNOWN') else cover['status']
            return dict(result,status=status,reason='given cover not certified independent',cover_certificate=cover,work_used=budget.used)
        if not verify_cover_certificate(raw,groups,cover):
            raise ArithmeticError('independent cover certificate failed')
        budget.spend(len(raw)*len(groups)**2)
        rows = []
        for group in groups:
            ids = hull(group,raw,budget)
            block = {i:raw[i] for i in group}
            if len(ids) > 3:
                return dict(result,status='UNSUPPORTED_DOMAIN',reason='a supplied block hull has more than three extreme points',unsupported_block=list(group),work_used=budget.used)
            if len(ids) == 3:
                work, sec = remaining()
                answer = exact_triangle_enclosing_ellipse(block,max_work=work,seconds=sec)
                budget.spend(answer['work_used'])
                if answer['status'] != 'EXACT_OPTIMUM':
                    return dict(result,status='PARTIAL_UNKNOWN',reason='triangle exact certificate incomplete',work_used=budget.used)
                rows.append(dict(labels=list(group),kind='triangle',center=answer['center'],
                    size=answer['certificate']['major_axis_length'],certificate=answer['certificate']))
            else:
                a,b = min(group,key=lambda i:raw[i]),max(group,key=lambda i:raw[i])
                c = tuple((raw[a][k]+raw[b][k])/2 for k in (0,1))
                squared = sum((raw[a][k]-raw[b][k])**2 for k in (0,1))
                rows.append(dict(labels=list(group),kind='point' if a==b else 'segment',
                    endpoint_labels=[a,b],center=c,size=dict(expression='sqrt(squared)',squared=squared)))
            budget.spend(len(group)*10+10)
        result.update(declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw),
            parts=[list(g) for g in groups],pieces=rows,m=len(rows),
            centers=[r['center'] for r in rows],
            cover_size=dict(expression='max(nonnegative_lengths)',lengths=[r['size'] for r in rows]),
            cover_certificate=cover,status='EXACT_GIVEN_COVER_OSTOV',complete=True,exact_centers=True)
        budget.spend(len(raw)*20+len(groups)**2*len(raw))
        if not verify_supported_block_ostov(raw,result):
            raise ArithmeticError('independent ostov result failed')
        budget.spend(0)
        result['work_used'] = budget.used
        return result
    except BudgetExceeded as exc:
        return dict(result,status='PARTIAL_UNKNOWN',complete=False,exact_centers=False,reason=str(exc),work_used=budget.used)
    except (ValueError,TypeError,ArithmeticError,AttributeError,KeyError,IndexError) as exc:
        return dict(result,status='UNSUPPORTED_INPUT',complete=False,exact_centers=False,reason=str(exc),work_used=budget.used)


def verify_supported_block_ostov(points,result):
    """Recheck complete input, cover separation, exact block centres and sizes.

    No cover solver, hull or centre constructor is invoked here.
    """
    try:
        r = result
        if (r['schema'] != 'supported-block-given-ostov-v1' or
                r['status'] != 'EXACT_GIVEN_COVER_OSTOV' or r['complete'] is not True or r['exact_centers'] is not True):
            return False
        raw, groups, missing = normalize(points,r['parts'])
        if any(type(i) is not int for i in r['declared_labels']):
            return False
        if missing or r['declared_labels'] != sorted(raw) or r['source_coordinate_sha256'] != _digest(raw):
            return False
        if not verify_cover_certificate(raw,groups,r['cover_certificate']):
            return False
        rows = r['pieces']
        if len(rows) != len(groups) or type(r['m']) is not int or r['m'] != len(groups):
            return False
        def exact_tree(value):
            if isinstance(value,dict):
                return all(exact_tree(v) for v in value.values())
            if isinstance(value,(list,tuple)):
                return all(exact_tree(v) for v in value)
            return type(value) in (int,F,str)
        if not exact_tree(r['centers']) or not exact_tree(r['cover_size']):
            return False
        for group,row in zip(groups,rows):
            if not exact_tree(row['center']) or not exact_tree(row['size']):
                return False
            if any(type(i) is not int for i in row['labels']):
                return False
            if row['labels'] != list(group):
                return False
            block = {i:raw[i] for i in group}
            if row['kind'] == 'triangle':
                z = row['certificate']
                if not verify_exact_triangle_ellipse(block,z) or row['center'] != z['center'] or row['size'] != z['major_axis_length']:
                    return False
            elif row['kind'] in ('point','segment'):
                ids = row['endpoint_labels']
                if len(ids) != 2 or any(type(i) is not int or i not in group for i in ids):
                    return False
                a,b = (raw[i] for i in ids)
                if ids != [min(group,key=lambda i:raw[i]),max(group,key=lambda i:raw[i])]:
                    return False
                if row['kind'] != ('point' if len(group)==1 else 'segment'):
                    return False
                if any(orient(a,b,p) != 0 for p in block.values()):
                    return False
                c = tuple((a[k]+b[k])/2 for k in (0,1))
                sq = sum((a[k]-b[k])**2 for k in (0,1))
                if row['center'] != c or any(type(x) not in (int,F) for x in row['center']):
                    return False
                if row['size'] != dict(expression='sqrt(squared)',squared=sq) or type(row['size']['squared']) not in (int,F):
                    return False
            else:
                return False
        return (r['centers'] == [row['center'] for row in rows] and
            r['cover_size'] == dict(expression='max(nonnegative_lengths)',lengths=[row['size'] for row in rows]))
    except (ValueError,TypeError,ArithmeticError,AttributeError,KeyError,IndexError):
        return False
