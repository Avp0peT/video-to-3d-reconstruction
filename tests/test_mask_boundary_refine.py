"""TEST_DATA: fixed CPU priors and conservative publication guards."""
import numpy as np
import pytest
from integrations.video_masks.boundary_refine import RefinePolicy,evaluate_proposal,refine_candidate
from integrations.demo_pipeline.temporal_review import load_cv
from integrations.video_masks import boundary_refine as refinement


def inputs():
    union=np.zeros((100,100),np.uint8);union[10:90,10:90]=255
    trimap=np.zeros_like(union);trimap[7:93,7:93]=128;trimap[15:85,15:85]=255
    return union,trimap


def test_only_unknown_contraction_under_cap_is_candidate_and_hole_unchanged():
    union,trimap=inputs();union[30,30]=0;trimap[30,30]=128
    proposal=union.copy();proposal[10,10:20]=0
    snapshot=union.copy();r=evaluate_proposal(union,trimap,proposal)
    assert r['status']=='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED'
    assert r['added_pixels']==0 and r['removed_pixels']==10 and r['core_unchanged']
    assert np.array_equal(union,snapshot) and proposal[30,30]==0


@pytest.mark.parametrize('case',['core','background','cap'])
def test_refuses_known_label_mutation_addition_or_large_change(case):
    union,trimap=inputs();proposal=union.copy()
    if case=='core':proposal[30,30]=0
    elif case=='background':proposal[0,0]=255
    else:proposal[10:12,10:90]=0
    assert evaluate_proposal(union,trimap,proposal)['status']=='REFUSED'


def test_tiny_original_component_cannot_be_fully_erased_under_area_cap():
    union,trimap=inputs();union[1,1]=255;trimap[1,1]=128;proposal=union.copy();proposal[1,1]=0
    r=evaluate_proposal(union,trimap,proposal)
    assert r['change_fraction_of_original_foreground']<.01
    assert 'ORIGINAL_COMPONENT_ELIMINATED' in r['reasons']


def test_original_hole_cannot_be_filled_even_if_unknown():
    union,trimap=inputs();union[30,30]=0;trimap[30,30]=128;proposal=union.copy();proposal[30,30]=255
    assert 'BACKGROUND_OR_HOLE_ADDITION' in evaluate_proposal(union,trimap,proposal)['reasons']


def test_unknown_initialization_and_frozen_iteration_contract():
    actual=load_cv([]);seen={}
    class Fake:
        GC_PR_FGD=actual.GC_PR_FGD;GC_PR_BGD=actual.GC_PR_BGD;GC_FGD=actual.GC_FGD;GC_BGD=actual.GC_BGD;GC_INIT_WITH_MASK=actual.GC_INIT_WITH_MASK;error=actual.error
        ocl=actual.ocl;connectedComponentsWithStats=staticmethod(actual.connectedComponentsWithStats)
        @staticmethod
        def setNumThreads(x):seen['threads']=x
        @staticmethod
        def setRNGSeed(x):seen['seed']=x
        @staticmethod
        def grabCut(image,mask,rect,bg,fg,iters,mode):seen.update(mask=mask.copy(),iterations=iters,mode=mode)
        def __getattr__(self,name):return getattr(actual,name)
    union,trimap=inputs();rgb=np.zeros((100,100,3),np.uint8)
    r=refine_candidate(rgb,union,trimap,cv=Fake())
    assert seen['threads']==1 and seen['seed']==0 and seen['iterations']==2
    assert seen['mask'][10,10]==actual.GC_PR_FGD
    assert seen['mask'][8,8]==actual.GC_PR_BGD
    assert seen['mask'][30,30]==actual.GC_FGD
    assert seen['mask'][0,0]==actual.GC_BGD
    assert r['unknown_is_background'] is False and r['changed_pixels']==0


@pytest.mark.parametrize('kwargs',[dict(iterations=3),dict(threads=2),dict(rng_seed=1),dict(max_changed_foreground_fraction=.02)])
def test_policy_cannot_silently_expand_experiment(kwargs):
    with pytest.raises(ValueError):RefinePolicy(**kwargs)


def make_review(tmp_path):
    from PIL import Image
    import json,hashlib
    review=tmp_path/'review';review.mkdir();(review/'union_upper_bound').mkdir();(review/'trimaps').mkdir()
    union,trimap=inputs();original=tmp_path/'original.png';Image.fromarray(union).save(original)
    rgb=tmp_path/'source_rgb.png';Image.new('RGB',union.shape[::-1],(20,30,40)).save(rgb)
    (review/'union_upper_bound/000000.png').write_bytes(original.read_bytes())
    Image.fromarray(trimap).save(review/'trimaps/000000.png')
    refs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in (original,rgb)]
    (review/'PLAN.json').write_text(json.dumps(dict(source_bindings=refs,frames=[dict(decode_index=0,image=str(rgb),union=str(original))])),encoding='utf8')
    (review/'RESULT.json').write_text(json.dumps(dict(schema='mask-boundary-review-v1',source_unchanged=True,published_for_reconstruction=False)),encoding='utf8')
    return review,original


def test_success_keeps_raw_defaults_and_separate_unaccepted_proposal(tmp_path,monkeypatch):
    review,original=make_review(tmp_path);out=tmp_path/'refine'
    refinement.prepare(review,out,[0])
    def fake(rgb,union,trimap,*args):
        proposal=union.copy();proposal[10,10]=0
        return dict(proposal=proposal,status='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED',grabcut_calls=0)
    monkeypatch.setattr(refinement,'refine_candidate',fake)
    result=refinement.execute(out)
    assert result['status']=='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED'
    assert result['proposal_promotion'] is False
    assert (out/'default_masks/000000.png').read_bytes()==original.read_bytes()
    from PIL import Image
    assert np.asarray(Image.open(out/'proposals/000000.png'))[10,10]==0


def test_late_source_drift_retains_runtime_frozen_raw_default(tmp_path,monkeypatch):
    from PIL import Image
    review,original=make_review(tmp_path);out=tmp_path/'refine';safe=original.read_bytes()
    refinement.prepare(review,out,[0])
    def fake(rgb,union,trimap,*args):
        proposal=union.copy();proposal[10,10]=0
        Image.new('L',union.shape[::-1],0).save(review/'union_upper_bound/000000.png')
        return dict(proposal=proposal,status='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED',grabcut_calls=0)
    monkeypatch.setattr(refinement,'refine_candidate',fake)
    with pytest.raises(ValueError,match='drift'):refinement.execute(out)
    assert (out/'default_masks/000000.png').read_bytes()==safe
    assert (review/'union_upper_bound/000000.png').read_bytes()!=safe
    import json
    result=json.loads((out/'RESULT.json').read_text(encoding='utf8'))
    assert result['status']=='REFUSED' and result['source_unchanged'] is False
    assert result['proposal_promotion'] is False


def test_cli_refused_result_is_nonzero(tmp_path,monkeypatch):
    from scripts import refine_mask_boundary as cli
    monkeypatch.setattr(cli,'execute',lambda output:dict(status='REFUSED'))
    assert cli.main(['execute','--output',str(tmp_path/'refine')])==1


def gap_inputs(kind):
    union=np.zeros((100,100),np.uint8);union[10:90,10:90]=255
    if kind=='hole':
        union[45:49,45:49]=0
        proposal=union.copy();proposal[45:49,45:49]=255
    elif kind=='bridge':
        union[10:65,48:52]=0
        proposal=union.copy();proposal[30,48:52]=255
    elif kind=='mouth':
        union[10:40,48:52]=0;union[40:70,35:65]=0
        proposal=union.copy();proposal[20,48:52]=255
    else:raise AssertionError(kind)
    # Changed pixels are deliberately UNKNOWN; the preservation guard must reject
    # the actual source-background damage without borrowing definite trimap labels.
    return union,np.full(union.shape,128,np.uint8),proposal


@pytest.mark.parametrize('kind,violation',[
    ('bridge','PROTECTED_BACKGROUND_FILLED'),
    ('hole','SOURCE_BACKGROUND_COMPONENT_ERASED'),
    ('mouth','EXTERIOR_BACKGROUND_PATH_DISCONNECTED'),
])
def test_actual_gap_kernel_refuses_bridge_hole_fill_and_closed_external_mouth(kind,violation):
    union,trimap,proposal=gap_inputs(kind)
    before=[x.copy() for x in (union,trimap,proposal)]
    result=evaluate_proposal(union,trimap,proposal)
    assert result['change_fraction_of_original_foreground']<.01
    assert result['status']=='REFUSED'
    assert 'GAP_PRESERVATION_NOT_VERIFIED' in result['reasons']
    assert result['gap_guard']['status']=='GAP_PRESERVATION_REJECTED'
    assert result['gap_guard']['ok'] is False
    assert violation in result['gap_guard']['violations']
    if kind=='mouth':assert result['gap_guard']['path_disconnected_pixels']>0
    for value,snapshot in zip((union,trimap,proposal),before):
        np.testing.assert_array_equal(value,snapshot)


def test_evaluate_proposal_calls_real_gap_kernels_and_keeps_trimap_unknown_distinct(monkeypatch):
    from integrations.video_masks.narrow_gap import detect_narrow_gaps,evaluate_gap_preservation
    union,trimap=inputs();proposal=union.copy();proposal[10,10]=0
    seen=[]
    def detect(source,validity,**kwargs):
        assert source.dtype==validity.dtype==np.bool_
        np.testing.assert_array_equal(source,union!=0)
        assert validity.all() and np.any(trimap==128)
        seen.append('detect')
        return detect_narrow_gaps(source,validity,**kwargs)
    def preserve(source,candidate,validity,diagnostics,**kwargs):
        np.testing.assert_array_equal(candidate,proposal!=0)
        seen.append('preserve')
        return evaluate_gap_preservation(source,candidate,validity,diagnostics,**kwargs)
    monkeypatch.setattr(refinement,'detect_narrow_gaps',detect)
    monkeypatch.setattr(refinement,'evaluate_gap_preservation',preserve)
    result=evaluate_proposal(union,trimap,proposal)
    assert seen==['detect','preserve']
    assert result['status']=='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED'
    guard=result['gap_guard']
    assert guard['status']=='GAP_PRESERVATION_PASSED_NOT_GEOMETRY_ACCEPTED'
    assert guard['ok'] is True and guard['preserved'] is True
    assert guard['validity_scope']=='FULL_SUPPLIED_MASK_GRID'
    assert guard['trimap_unknown_is_invalid'] is False
    assert result['removed_pixels']==1
    assert guard['stats']['foreground_removed_pixels']==1
    assert guard['stats']['foreground_added_pixels']==0
    import json
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('failure',['unknown_status','missing_ok','contradictory_map','invalid_map','nonfinite_stats','exception'])
def test_gap_guard_unknown_invalid_or_unverifiable_cannot_pass(monkeypatch,failure):
    union,trimap=inputs()
    def preserve(source,candidate,validity,diagnostics,**kwargs):
        if failure=='exception':raise ValueError('synthetic invalid evidence')
        row=dict(status='GAP_PRESERVATION_PASSED_NOT_GEOMETRY_ACCEPTED',ok=True,preserved=True,
                 violations=[],stats={},protected_filled=np.zeros_like(source),path_disconnected=np.zeros_like(source))
        if failure=='unknown_status':row['status']='UNKNOWN'
        elif failure=='missing_ok':row.pop('ok')
        elif failure=='contradictory_map':row['protected_filled'][50,50]=True
        elif failure=='invalid_map':row['protected_filled']=np.zeros((1,1),bool)
        elif failure=='nonfinite_stats':row['stats']={'value':float('nan')}
        return row
    monkeypatch.setattr(refinement,'evaluate_gap_preservation',preserve)
    result=evaluate_proposal(union,trimap,union.copy())
    assert result['status']=='REFUSED'
    assert 'GAP_PRESERVATION_NOT_VERIFIED' in result['reasons']
    assert result['gap_guard']['ok'] is False and result['gap_guard']['preserved'] is False
    assert result['gap_guard']['status']!='GAP_PRESERVATION_PASSED_NOT_GEOMETRY_ACCEPTED'


def test_prepare_binds_actual_gap_kernel_and_preserves_default_policy(tmp_path):
    import hashlib
    from pathlib import Path
    from integrations.video_masks import narrow_gap
    review,_=make_review(tmp_path)
    plan=refinement.prepare(review,tmp_path/'refine',[0])
    path=Path(narrow_gap.__file__).resolve()
    bindings=[row for row in plan['source_bindings'] if row['path']==str(path)]
    assert len(bindings)==1
    assert bindings[0]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert bindings[0]['bytes']==path.stat().st_size
    assert plan['default_mask_policy']=='ALWAYS_UNCHANGED_ORIGINAL_UNION'
    assert plan['published_for_reconstruction'] is False and plan['proposal_promotion'] is False
    assert plan['gap_guard_validity']=='FULL_SUPPLIED_MASK_GRID'


def test_cli_candidate_return_code_stays_zero(tmp_path,monkeypatch):
    from scripts import refine_mask_boundary as cli
    monkeypatch.setattr(cli,'execute',lambda output:dict(status='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED'))
    assert cli.main(['execute','--output',str(tmp_path/'refine')])==0


def test_backend_expansion_is_still_clipped_to_source_before_gap_guard():
    actual=load_cv([])
    class Fake:
        def __getattr__(self,name):return getattr(actual,name)
        def grabCut(self,image,mask,rect,bg,fg,iters,mode):mask[:]=actual.GC_FGD
    union,trimap=inputs();union[30:34,30:34]=0;trimap[30:34,30:34]=128
    result=refine_candidate(np.zeros((*union.shape,3),np.uint8),union,trimap,cv=Fake())
    np.testing.assert_array_equal(result['proposal'],union)
    assert result['added_pixels']==0 and result['changed_pixels']==0
    assert result['gap_guard']['ok'] is True


def test_refused_without_candidate_does_not_claim_gap_preservation():
    union,trimap=inputs();trimap[:]=128
    result=refine_candidate(np.zeros((*union.shape,3),np.uint8),union,trimap)
    assert result['status']=='REFUSED' and result['grabcut_calls']==0
    assert result['gap_guard']['status']=='NOT_EVALUATED'
    assert result['gap_guard']['ok'] is False
