"""Independent adversarial certificates and exact mathematics, not solver snapshots."""
from fractions import Fraction as F
from copy import deepcopy
from kozlov_lab.kozlov2025.exact_triangle_ellipse import _digest
from kozlov_lab.kozlov2025.exact_center import assess_exact_center_result


def point_center_witness(label,point):
    raw={label:tuple(F(x) for x in point)}
    size=dict(expression='sqrt(length_squared)',length_squared=F(0))
    cert=dict(schema='exact-center-degenerate-certificate-v1',kind='point',
              declared_labels=[label],source_coordinate_sha256=_digest(raw),
              endpoint_labels=[label],center=raw[label],major_axis_length=deepcopy(size))
    answer=dict(schema='exact-center-v1',status='EXACT_DEGENERATE_COMPLETE_INPUT',
                complete=True,exact_optimum=True,declared_labels=[label],
                source_coordinate_sha256=_digest(raw),profile='point',number_context={'domain':'rational'},
                center=raw[label],major_axis_length=size,certificate=cert)
    return raw,answer


def test_independent_point_certificate_bool_values_are_rejected():
    raw,answer=point_center_witness(0,(0,0))
    assert assess_exact_center_result(raw,answer)['status']=='VERIFIED'
    answer['certificate']['center']=(False,False)
    answer['certificate']['major_axis_length']['length_squared']=False
    assert assess_exact_center_result(raw,answer)['status']=='INVALID'


def independent_primary_witness():
    from kozlov_lab.kozlov2025.independent_cover import verify_cover_certificate
    raw={0:(F(0),F(0)),1:(F(10),F(0))}
    cover=dict(schema='independent-cover-certificate-v1',contract='independent-cover-v1',
               status='VALID_INDEPENDENT_COVER',coverage_complete=True,decision_complete=True,
               pairwise_independence=True,missing_labels=[],pair_certificates=[
                   dict(left_part=0,right_part=1,kind='strict_separation',axis=(F(1),F(0)))])
    assert verify_cover_certificate(raw,[[0],[1]],cover)
    centers=[raw[0],raw[1]]
    blocks=[dict(labels=[i],result=point_center_witness(i,raw[i])[1]) for i in (0,1)]
    checks=[dict(label=i,owner=i,distances_squared=[F(0),F(100)] if i==0 else [F(100),F(0)],
                 uniquely_nearest=True,strictly_inside=True) for i in (0,1)]
    return raw,dict(schema='primary-futlyar-certificate-v1',status='TRUE',parts=[[0],[1]],
                    declared_labels=[0,1],source_coordinate_sha256=_digest(raw),cover_certificate=cover,
                    blocks=blocks,centers=centers,radius_squared=F(100),
                    size_descriptors=[dict(kind='squared',squared=F(0)) for _ in centers],
                    radius_screens=[True,True],point_checks=checks)


def independent_zero_witness():
    from kozlov_lab.kozlov2025.zero_error_fits import SCHEMA,PROOF
    raw,primary=independent_primary_witness()
    source={2:raw[0],3:raw[1]}
    cert=dict(schema=SCHEMA,proof=PROOF,epsilon1=F(1),epsilon2=F(1),
              reference_points=tuple(sorted(raw.items())),source_points=tuple(sorted(source.items())),
              reference_parts=((0,),(1,)),source_parts=((2,),(3,)),matrix=((F(1),F(0)),(F(0),F(1))),
              translation=(F(0),F(0)),part_bijection=((0,0),(1,1)),primary_certificate=primary,
              source_blocks=tuple(point_center_witness(i,source[i])[1] for i in (2,3)),
              center_checks=((0,0,source[2],raw[0],raw[0]),(1,1,source[3],raw[1],raw[1])),
              point_checks=((2,0,0,source[2],(F(0),F(100))),
                            (3,1,1,source[3],(F(100),F(0)))))
    return raw,source,cert


def test_independent_zero_geometry_check_bool_corruption_rejected():
    from kozlov_lab.kozlov2025.zero_error_fits import assess_zero_error_fits_certificate
    raw,source,cert=independent_zero_witness()
    assert assess_zero_error_fits_certificate(raw,source,cert)['status']=='VERIFIED'
    cert['center_checks']=((False,False,(False,False),(False,False),(False,False)),
                           cert['center_checks'][1])
    assert assess_zero_error_fits_certificate(raw,source,cert)['status']=='INVALID'


def test_independent_zero_full_input_matrix_and_containment_checks():
    from kozlov_lab.kozlov2025.zero_error_fits import assess_zero_error_fits_certificate
    raw,source,cert=independent_zero_witness()
    assert assess_zero_error_fits_certificate(raw,source,cert)['status']=='VERIFIED'
    assert assess_zero_error_fits_certificate(raw,{**source,4:(F(20),F(0))},cert)['status']=='INVALID'
    assert assess_zero_error_fits_certificate({**raw,5:(F(30),F(0))},source,cert)['status']=='INVALID'
    bad=deepcopy(cert); bad['source_blocks']=bad['source_blocks'][:1]
    assert assess_zero_error_fits_certificate(raw,source,bad)['status']=='INVALID'
    bad=deepcopy(cert); bad['matrix']=((F(1),F(1)),(F(2),F(2)))
    assert assess_zero_error_fits_certificate(raw,source,bad)['status']=='INVALID'
    bad=deepcopy(cert); bad['part_bijection']=((0,1),(1,0))
    assert assess_zero_error_fits_certificate(raw,source,bad)['status']=='INVALID'
    assert assess_zero_error_fits_certificate(raw,source,cert,max_work=1)['status']=='UNKNOWN'
    assert assess_zero_error_fits_certificate(raw,source,cert,seconds=0)['status']=='UNKNOWN'


def test_zero_error_nontrivial_affine_witness_and_common_perturbation_identity():
    from kozlov_lab.kozlov2025.zero_error_fits import assess_zero_error_fits_certificate
    raw,source,cert=independent_zero_witness()
    # Invert T(x,y)=(2x+y+3,x+y-2) by hand; det(T)=1.
    source={2:(F(-5),F(7)),3:(F(5),F(-3))}
    cert['source_points']=tuple(sorted(source.items()))
    cert['source_blocks']=tuple(point_center_witness(i,source[i])[1] for i in (2,3))
    cert['matrix']=((F(2),F(1)),(F(1),F(1))); cert['translation']=(F(3),F(-2))
    del cert['center_checks']; del cert['point_checks']
    answer=assess_zero_error_fits_certificate(raw,source,cert)
    assert answer['status']=='VERIFIED'
    # Objective zero is a global minimum since every Euclidean length >=0.
    # For any common affine H, H(a)-H(T(b)) = linear_H*(a-T(b)) = 0.
    # This exact instance additionally checks a nonorthogonal common H.
    h=((F(1),F(7,3)),(F(-2),F(1))); shift=(F(5,7),F(-9,11))
    for _,_,_,transformed,reference_center in answer['center_checks']:
        lhs=tuple(sum(h[j][k]*reference_center[k] for k in (0,1))+shift[j] for j in (0,1))
        rhs=tuple(sum(h[j][k]*transformed[k] for k in (0,1))+shift[j] for j in (0,1))
        assert lhs==rhs


def test_independent_segment_complete_and_source_outside_radius():
    from kozlov_lab.kozlov2025.zero_error_fits import assess_zero_error_fits_certificate
    raw,_,cert=independent_zero_witness()
    source={2:(F(-20),F(0)),3:(F(20),F(0)),4:(F(10),F(0))}
    segment_raw={i:source[i] for i in (2,3)}
    size=dict(expression='sqrt(length_squared)',length_squared=F(1600))
    witness=dict(schema='exact-center-degenerate-certificate-v1',kind='segment',
                 declared_labels=[2,3],source_coordinate_sha256=_digest(segment_raw),
                 endpoint_labels=[2,3],center=(F(0),F(0)),major_axis_length=deepcopy(size))
    result=dict(schema='exact-center-v1',status='EXACT_DEGENERATE_COMPLETE_INPUT',
                complete=True,exact_optimum=True,declared_labels=[2,3],source_coordinate_sha256=_digest(segment_raw),
                profile='segment',number_context={'domain':'rational'},center=(F(0),F(0)),
                major_axis_length=size,certificate=witness)
    assert assess_exact_center_result(segment_raw,result)['status']=='VERIFIED'
    assert assess_exact_center_result({**segment_raw,5:(F(21),F(0))},result)['status']=='INVALID'
    cert.update(source_points=tuple(sorted(source.items())),source_parts=((2,3),(4,)),
                source_blocks=(result,point_center_witness(4,source[4])[1]))
    del cert['center_checks']; del cert['point_checks']
    assert assess_zero_error_fits_certificate(raw,source,cert)['status']=='INVALID'


def test_independent_primary_full_distance_and_numeric_type_checks():
    from kozlov_lab.kozlov2025.primary_futlyar import assess_primary_futlyar_certificate
    raw,cert=independent_primary_witness()
    assert assess_primary_futlyar_certificate(raw,cert)['status']=='VERIFIED'
    bad=deepcopy(cert); bad['radius_squared']=F(25)
    assert assess_primary_futlyar_certificate(raw,bad)['status']=='INVALID'
    bad=deepcopy(cert); bad['size_descriptors'][0]['squared']=False
    assert assess_primary_futlyar_certificate(raw,bad)['status']=='INVALID'
    bad=deepcopy(cert); bad['point_checks'][0]['distances_squared'][0]=False
    assert assess_primary_futlyar_certificate(raw,bad)['status']=='INVALID'
    assert assess_primary_futlyar_certificate({**raw,2:(F(30),F(0))},cert)['status']=='INVALID'


def test_independent_reducible_root_local_zero_and_denominator():
    from kozlov_lab.kozlov2025.real_algebraic import Root,Budget
    root=Root((0,-1,0,1),F(1,2),F(3,2),Budget())
    # alpha=1 selected, denominator alpha*(alpha+1) vanishes at other roots.
    assert root.num((1,),(0,1,1)).equals(F(1,2))
    assert root.num((-1,1)).sign()==0
    assert root.num((0,1)).equals(1)
    import pytest
    with pytest.raises(ZeroDivisionError): root.num((1,),(-1,1))
