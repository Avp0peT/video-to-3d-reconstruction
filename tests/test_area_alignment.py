"""Explicit TEST_DATA rasters; fakes capture masks and one-shot native calls."""
from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.area_alignment import align_area,compose_cv,map_cv


class CV:
    TERM_CRITERIA_COUNT=1;TERM_CRITERIA_EPS=2;TM_CCOEFF_NORMED=5;MOTION_AFFINE=2
    class error(Exception):
        def __init__(self,message,code=-7):super().__init__(message);self.code=code
    def __init__(self,matrix=None,mode=None):self.calls=[];self.matrix=np.array(matrix if matrix is not None else [[1,0,24],[0,1,32]],np.float32);self.mode=mode
    def matchTemplate(self,search,template,method,mask=None):
        self.calls.append(('match',search.copy(),template.copy(),mask.copy()))
        import cv2
        cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
        result=cv2.matchTemplate(search,template,method,mask=mask)
        if self.mode=='nanmatch':result[:]=np.nan
        return result
    def findTransformECCWithMask(self,template,search,tm,sm,initial,motion,criteria,gauss):
        self.calls.append(('ecc',template.copy(),search.copy(),tm.copy(),sm.copy(),initial.copy(),motion,criteria,gauss))
        if self.mode=='nonconverge':raise self.error('Iterations do not converge',-7)
        if self.mode=='programming':raise self.error('bad argument',-215)
        return .99,self.matrix.copy()


def sample_data():
    rng=np.random.default_rng(991)
    template=rng.integers(30,210,(33,33),dtype=np.uint8)
    search=rng.integers(30,210,(97,97),dtype=np.uint8);search[32:65,24:57]=template
    return template,search,np.full((33,33),255,np.uint8),np.full((97,97),255,np.uint8)


def test_whole_area_identity_translation_and_native_contract():
    values=sample_data();before=[v.copy() for v in values];cv=CV();result=align_area(*values,cv)
    assert result['status']=='ACCEPTED',result['reasons']
    assert result['initialization']['translation_cv_xy']==[24,32]
    assert result['holdout']['ncc']> .999999 and result['holdout']['normalized_rmse']<1e-10
    assert result['native_calls']==dict(matchTemplate=1,findTransformECCWithMask=1)
    ecc=cv.calls[1];assert ecc[-3:]==(2,(3,40,1e-4),1)
    train=np.asarray(result['pixel_evidence']['train_mask_uint8']);held=np.asarray(result['pixel_evidence']['holdout_mask_uint8'])
    assert not np.any((train>0)&(held>0)) and np.array_equal(ecc[3],train)
    assert all(np.array_equal(a,b) for a,b in zip(values,before))
    assert not result['identity_verified'] and not result['geometry_accepted']
    json.dumps(result,allow_nan=False)


def test_holdout_pixels_do_not_enter_translation_or_photometric_fit():
    values=sample_data();baseline=align_area(*values,CV());template,search,tm,sm=values
    yy,xx=np.indices((33,33));held=((xx//8+yy//8)%4)==3
    search[32:65,24:57][held]=255-search[32:65,24:57][held]
    changed=align_area(template,search,tm,sm,CV())
    assert changed['initialization']['translation_cv_xy']==baseline['initialization']['translation_cv_xy']
    assert changed['photometric']==baseline['photometric']
    assert changed['status']=='REJECTED' and 'HOLDOUT_NCC_GATE' in changed['reasons']


def test_partial_search_mask_is_counted_not_deleted():
    template,search,tm,sm=sample_data();sm[32:35,24:57]=0
    result=align_area(template,search,tm,sm,CV())
    assert result['holdout']['coverage']<1 and result['holdout']['count']>result['holdout']['valid_count']
    assert result['status']=='ACCEPTED'


def test_large_mask_occlusion_rejected_or_no_supported_initialization():
    template,search,tm,sm=sample_data();sm[32:55,24:57]=0
    result=align_area(template,search,tm,sm,CV())
    assert result['status']!='ACCEPTED'


def test_source_mask_and_flat_template_block_before_native():
    template,search,tm,sm=sample_data();tm[:12]=0;cv=CV()
    result=align_area(template,search,tm,sm,cv)
    assert result['status']=='BLOCKED' and result['reasons']==['SOURCE_MASK_COVERAGE'] and not cv.calls
    template[:]=90;tm[:]=255
    assert align_area(template,search,tm,sm,cv)['reasons']==['FLAT_TEMPLATE_TRAINING_SUPPORT']


def test_missing_exact_api_never_falls_back():
    class Missing:pass
    assert align_area(*sample_data(),Missing())['reasons']==['MISSING_FIND_TRANSFORM_ECC_WITH_MASK']


@pytest.mark.parametrize('matrix,reason',[([[-1,0,50],[0,1,32]],'AFFINE_SHAPE_GATE'),([[1.6,0,24],[0,1,32]],'AFFINE_SHAPE_GATE'),([[1,0,-1],[0,1,32]],'WARP_OUTSIDE_SEARCH_IMAGE')])
def test_invalid_affine_rejected(matrix,reason):
    result=align_area(*sample_data(),CV(matrix))
    assert result['status']=='REJECTED' and reason in result['reasons']


def test_nonconvergence_retained_but_programming_errors_propagate():
    result=align_area(*sample_data(),CV(mode='nonconverge'))
    assert result['status']=='REJECTED' and result['reasons']==['NATIVE_NONCONVERGENCE']
    with pytest.raises(CV.error,match='bad argument'):align_area(*sample_data(),CV(mode='programming'))


def test_nan_match_has_no_candidate_and_no_optimizer():
    cv=CV(mode='nanmatch');result=align_area(*sample_data(),cv)
    assert result['status']=='BLOCKED' and result['reasons']==['NO_VALID_TRAINING_TRANSLATION'] and len(cv.calls)==1


def test_budget_after_native_is_primary_with_partial():
    cv=CV()
    def budget():
        if cv.calls:raise TimeoutError('deadline')
    with pytest.raises(TimeoutError,match='deadline') as exc:align_area(*sample_data(),cv,budget=budget)
    assert exc.value.partial_result['native_calls']['matchTemplate']==1


def test_invalid_dtype_shape_and_mask_are_not_accepted():
    a,b,m,n=sample_data()
    with pytest.raises(ValueError):align_area(a.astype(float),b,m,n,CV())
    with pytest.raises(ValueError):align_area(a,b,m[:32],n,CV())
    with pytest.raises(ValueError):align_area(a,np.zeros((130,97),np.uint8),m,np.zeros((130,97),np.uint8),CV())


def test_compose_and_point_mapping_use_template_to_search_direction():
    a=[[1.,0.,24.],[0.,1.,32.]];b=[[1.,0.,-24.],[0.,1.,-32.]]
    assert np.allclose(compose_cv(b,a),[[1,0,0],[0,1,0]])
    assert map_cv(a,[[0,0],[16,16]])==[[24.,32.],[40.,48.]]


def test_real_opencv_identity_translation_when_api_available():
    import cv2 as cv
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    assert hasattr(cv,'findTransformECCWithMask'),'Toolchain must supply exact masked ECC API'
    result=align_area(*sample_data(),cv)
    assert result['status']=='ACCEPTED',result['reasons']
    assert np.allclose(result['matrix_cv2x3'],[[1,0,24],[0,1,32]],atol=.1)


@pytest.mark.parametrize('angle_degrees,shear,brightness,offset',[(5.,0.,1.,0.),(0.,.08,1.,0.),(3.,.04,1.12,7.)])
def test_real_opencv_affine_shape_and_brightness_corner_accuracy(angle_degrees,shear,brightness,offset):
    import cv2 as cv
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    assert hasattr(cv,'findTransformECCWithMask'),'Exact installed masked ECC API required'
    # TEST_DATA comes from an analytic continuous intensity field, sampled
    # separately on template and inverse-transformed search grids. No product
    # warp/map/fitting helper creates the ground-truth raster or expected map.
    def intensity(x,y):
        return 115+25*np.sin(.23*x+.11*y)+22*np.cos(.17*y-.08*x)+18*np.sin(.31*x-.19*y)+20*np.exp(-((x-9)**2+(y-21)**2)/70)+8*np.sin(.067*x+.077*y)
    angle=np.deg2rad(angle_degrees);c,s=np.cos(angle),np.sin(angle)
    A=np.array([[c,-s+shear],[s,c]],float);translation=np.array([34.,28.]);expected=np.column_stack((A,translation))
    ty,tx=np.indices((33,33));template=np.rint(np.clip(intensity(tx,ty),0,255)).astype(np.uint8)
    sy,sx=np.indices((97,97));target=np.stack((sx-translation[0],sy-translation[1]),axis=-1)
    source=target@np.linalg.inv(A).T
    search=np.rint(np.clip(brightness*intensity(source[:,:,0],source[:,:,1])+offset,0,255)).astype(np.uint8)
    result=align_area(template,search,np.full(template.shape,255,np.uint8),np.full(search.shape,255,np.uint8),cv)
    assert result['status']=='ACCEPTED',result['reasons']
    corners=np.array([[0.,0.,1.],[32.,0.,1.],[0.,32.,1.],[32.,32.,1.],[16.,16.,1.]])
    displacement=np.linalg.norm(corners@np.asarray(result['matrix_cv2x3']).T-corners@expected.T,axis=1)
    assert displacement.max()<.5,displacement
    assert result['native_calls']['findTransformECCWithMask']==1


def test_full_search_binary_mask_uses_native_max_fast_path():
    result=align_area(*sample_data(),CV())
    assert result['initialization']['selection_path']=='FINITE_NATIVE_MAX_FULL_SEARCH_BINARY_TRAIN_MASK'
    template,search,tm,sm=sample_data();sm[0,0]=0
    supported=align_area(template,search,tm,sm,CV())
    assert supported['initialization']['selection_path']=='SUPPORTED_MASK_INTERSECTION_SEARCH'
    assert supported['initialization']['translation_cv_xy']==result['initialization']['translation_cv_xy']
