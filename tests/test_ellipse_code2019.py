from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import json
import pytest
from kozlov_lab.coding import ellipse_code2019 as e
from kozlov_lab.coding.exact import build_kozlov_image_code_exact
from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact
from kozlov_lab.kozlov2025.exact_triangle_ellipse import exact_triangle_enclosing_ellipse

P={11:(0,0),22:(2,0),33:(0,2),44:(3,3)}


def test_triangle_hand_geometry_and_area_identity():
    p={-7:(0,0),14:(1,0),2**63-1:(0,1)}
    result=e.build_ellipse_code2019(p)
    assert result['status']=='EXACT_ELLIPSE_CODE' and not result['theorem_domain']
    c=result['certificate'];g=next(iter(c['ellipses'].values()))
    assert g['center']==(F(1,3),F(1,3))
    assert g['shape_matrix']==((F(4,9),-F(2,9)),(-F(2,9),F(4,9)))
    assert g['area_squared_over_pi_squared']==F(4,27)
    assert c['ellipse_ratios']=={}
    assert e.assess_ellipse_code2019(p,c)['status']=='VERIFIED'


def test_complete_code_matches_existing_area_oracle_and_ellipse_geometry():
    r=e.build_ellipse_code2019(P);c=r['certificate']
    assert r['status']=='EXACT_ELLIPSE_CODE' and r['theorem_domain']
    assert len(c['ellipse_ratios'])==12 and len(c['ellipses'])==4
    ids=sorted(P)
    old=build_kozlov_image_code_exact([P[i] for i in ids])
    expected={','.join(str(ids[j]) for j in sorted(row.numerator))+'/'+','.join(str(ids[j]) for j in sorted(row.denominator)):row.value for row in old.entries}
    assert c['ellipse_ratios']==expected
    for triple in combinations(ids,3):
        old=exact_triangle_enclosing_ellipse({i:P[i] for i in triple})
        g=c['ellipses'][','.join(map(str,triple))]
        assert old['status']=='EXACT_OPTIMUM'
        for name in ('center','shape_matrix','inverse_shape_matrix','area_squared_over_pi_squared'):
            assert g[name]==old['certificate'][name]
    restored=decode_exact(json.loads(json.dumps(encode_exact(c))))
    assert e.assess_ellipse_code2019(P,restored)['status']=='VERIFIED'


def test_affine_covariance_and_invariant_indexed_ratios():
    c=e.build_ellipse_code2019(P)['certificate']
    q=((F(2),F(1)),(F(-1),F(3)));t=(F(7),F(-4))
    transformed={i:tuple(sum(q[k][j]*p[j] for j in (0,1))+t[k] for k in (0,1)) for i,p in P.items()}
    d=e.build_ellipse_code2019(transformed)['certificate']
    assert d['ellipse_ratios']==c['ellipse_ratios']
    for key,g in c['ellipses'].items():
        h=d['ellipses'][key]
        assert h['center']==tuple(sum(q[k][j]*g['center'][j] for j in (0,1))+t[k] for k in (0,1))
        qsqt=tuple(tuple(sum(q[j][a]*g['shape_matrix'][a][b]*q[k][b] for a in (0,1) for b in (0,1)) for k in (0,1)) for j in (0,1))
        assert h['shape_matrix']==qsqt
        assert h['area_squared_over_pi_squared']==49*g['area_squared_over_pi_squared']


def test_independent_verifier_cannot_construct(monkeypatch):
    c=e.build_ellipse_code2019(P)['certificate']
    def forbidden(*a,**k):raise AssertionError('constructor called')
    monkeypatch.setattr(e,'_native_triangle',forbidden)
    monkeypatch.setattr(e,'build_environment_code',forbidden)
    monkeypatch.setattr(e,'build_ellipse_code2019',forbidden)
    assert e.assess_ellipse_code2019(P,c)['status']=='VERIFIED'


@pytest.mark.parametrize('mutation',['missing_triangle','extra_triangle','label_bool','scale_shape','center','det','ratio','ratio_bool','missing_ratio','self_ratio','area_point','area_missing','source','policy','extra','zero_weight','float_inverse','extra_geometry'])
def test_forgery_is_rejected(mutation):
    c=deepcopy(e.build_ellipse_code2019(P)['certificate'])
    key=next(iter(c['ellipses']));g=c['ellipses'][key];ratio=next(iter(c['ellipse_ratios']))
    if mutation=='missing_triangle':del c['ellipses'][key]
    elif mutation=='extra_triangle':c['ellipses']['11,22,99']=deepcopy(g)
    elif mutation=='label_bool':g['labels']=(True,22,33)
    elif mutation=='scale_shape':g['shape_matrix']=tuple(tuple(2*v for v in row) for row in g['shape_matrix'])
    elif mutation=='center':g['center']=(0,0)
    elif mutation=='det':g['area_squared_over_pi_squared']*=2
    elif mutation=='ratio':c['ellipse_ratios'][ratio]*=2
    elif mutation=='ratio_bool':c['ellipse_ratios'][ratio]=True
    elif mutation=='missing_ratio':del c['ellipse_ratios'][ratio]
    elif mutation=='self_ratio':c['ellipse_ratios'][key+'/'+key]=F(1)
    elif mutation=='area_point':c['area_code']['environment_points']=()
    elif mutation=='area_missing':del c['area_code']['ratios'][ratio]
    elif mutation=='source':c['source']='unbound'
    elif mutation=='policy':c['domain_policy']='allow_degeneracy'
    elif mutation=='extra':c['invented_theorem']=True
    elif mutation=='zero_weight':g['contact_weights']=(F(1),F(0),F(0))
    elif mutation=='float_inverse':g['inverse_shape_matrix']=((1.0,0),(0,1))
    else:g['invented']=0
    assert e.assess_ellipse_code2019(P,c)['status']=='INVALID'


@pytest.mark.parametrize('points',[{}, {1:(0,0),2:(1,0)}, {0:(0,0),1:(1,0),2:(2,0)},
    {0:(0,0),1:(1,0),2:(2,0),3:(0,1),4:(2,3)},
    {True:(0,0),2:(1,0),3:(0,1)}, {1:(0.0,0),2:(1,0),3:(0,1)},
    {i:(i,i*i) for i in range(9)}, {1:(2**256,0),2:(1,0),3:(0,1)}])
def test_strict_domain_and_bounds(points):
    r=e.build_ellipse_code2019(points)
    assert r['status']=='UNSUPPORTED' and r['certificate'] is None


def test_no_three_collinear_is_not_source_plane_domain():
    p={0:(0,0),1:(1,0),2:(1,1),3:(0,1)}
    r=e.build_ellipse_code2019(p)
    assert r['status']=='EXACT_ELLIPSE_CODE' and not r['theorem_domain']
    assert r['certificate']['area_code']['domain']=='two_parallel_lines'
    assert r['affine_equivalence_decided'] is False


def test_eight_complete_points_and_budget():
    p={-8+i:(i,i*i) for i in range(8)}
    r=e.build_ellipse_code2019(p);c=r['certificate']
    assert r['status']=='EXACT_ELLIPSE_CODE'
    assert len(c['ellipses'])==56 and len(c['ellipse_ratios'])==3080
    restored=decode_exact(json.loads(json.dumps(encode_exact(c))))
    assert e.assess_ellipse_code2019(p,restored)['status']=='VERIFIED'
    assert e.build_ellipse_code2019(p,max_work=0)['status']=='UNKNOWN'
    assert e.build_ellipse_code2019(p,seconds=0)['status']=='UNKNOWN'
    assert e.assess_ellipse_code2019(p,c,max_work=5)['status']=='UNKNOWN'
    assert e.build_ellipse_code2019(p,max_work=True)['status']=='UNSUPPORTED'
