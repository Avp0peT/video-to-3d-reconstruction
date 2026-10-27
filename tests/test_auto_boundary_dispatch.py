"""Synthetic dispatch only: no video decoder, model, GPU or review child runs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from integrations.demo_pipeline.pipeline import read, save, sha
from integrations.video_masks import auto_boundary, full_pipeline, fusion, video_export


READY = 'AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED'


def row_fixture(folder):
    mask=folder/'mask.png';image=folder/'image.jpg'
    pixels=np.zeros((8,8),np.uint8);pixels[2:6,2:6]=255
    Image.fromarray(pixels).save(mask);Image.new('RGB',(8,8),(20,30,40)).save(image)
    row=dict(decode_index=0,status='ORIGINAL_RETAINED',original_status='OK',
             mask_path=str(mask),mask_sha256=sha(mask),selected_root_seed_decode_index=0,
             warning_flags=[],bidirectional_metrics=None)
    return row,image


def hook_spy(monkeypatch, child_status=READY, raises=False):
    calls=[];children=[];original=auto_boundary.attach_auto_review
    def child(root,run,label,argv,**kwargs):
        # These assertions enforce dispatch ordering, independently of the return dict.
        assert (run/'RESULT.json').is_file() and (run/'index.html').is_file()
        assert (run/'ASSEMBLY.json').is_file() and (run/'FUSION.json').is_file()
        frozen=(run/'RESULT.json').read_bytes()
        assert 'automatic_boundary_review' not in json.loads(frozen)
        children.append(dict(run=run,source_bytes=frozen,argv=argv,limits=kwargs))
        if raises:raise RuntimeError('synthetic child refusal')
        target=Path(argv[argv.index('--output')+1]);target.mkdir()
        save(target/'RESULT.json',dict(status=child_status))
        (target/'index.html').write_text('synthetic review only',encoding='utf8')
        return dict(status='completed',root_exit_code=0,reclaim_confirmed=True)
    def attach(run,result):
        before=(run/'RESULT.json').read_bytes()
        snapshot=json.loads(json.dumps(result))
        calls.append((run,snapshot))
        value=original(run,result)
        assert (run/'RESULT.json').read_bytes()==before
        assert result==snapshot
        assert value is not result
        return value
    monkeypatch.setattr(auto_boundary,'_invoke',child)
    monkeypatch.setattr(auto_boundary,'attach_auto_review',attach)
    return calls,children


def fusion_source(tmp_path,monkeypatch,fail=False):
    source=tmp_path/'source';source.mkdir();(source/'worker').mkdir()
    row,image=row_fixture(source)
    pipeline_plan=source/'pipeline-plan.json'
    save(pipeline_plan,dict(video_tools=dict(ffmpeg='synthetic',ffprobe='synthetic')))
    plan=dict(frame_count=1,width=8,height=8,pipeline_plan_path=str(pipeline_plan))
    save(source/'PLAN.json',plan)
    save(source/'RESULT.json',dict(status='FULL_TIMELINE_EXPORTED',input_drift=[],code_drift=[]))
    save(source/'ASSEMBLY.json',dict(rows=[row]))
    save(source/'START_BINDINGS.json',dict(input_files={}))
    save(source/'DECODE.json',dict(frames=[dict(decode_index=0,path=str(image),sha256=sha(image))]))
    (source/'worker/events.jsonl').write_text('',encoding='utf8')
    monkeypatch.setattr(fusion,'gather_predictions',lambda *a:{})
    def exporter(*a,**k):
        if fail:raise RuntimeError('synthetic export failure')
        return dict(placeholder_count=0)
    monkeypatch.setattr(video_export,'export_video',exporter)
    return source


@pytest.mark.parametrize('mode,fail,expected',[('union',False,READY),
    ('supported-holes',False,'NOT_APPLICABLE'),('union',True,'NOT_APPLICABLE')])
def test_fusion_hook_once_after_frozen_source_and_no_child_for_inapplicable(tmp_path,monkeypatch,mode,fail,expected):
    source=fusion_source(tmp_path,monkeypatch,fail);out=tmp_path/'fused'
    calls,children=hook_spy(monkeypatch)
    result=fusion.execute(source,out,mode)
    assert len(calls)==1
    assert len(children)==(expected==READY)
    assert result['automatic_boundary_review']['status']==expected
    assert read(out/'RESULT.json')['status']==('FAILED' if fail else 'FULL_TIMELINE_FUSED_EXPORTED')
    assert 'automatic_boundary_review' not in read(out/'RESULT.json')
    assert read(out/'AUTOMATIC_BOUNDARY.json')==result['automatic_boundary_review']
    page=(out/'index.html').read_text(encoding='utf8')
    assert 'AUTOMATIC_BOUNDARY.json' in page and expected in page
    # Both report renders must preserve the previously generated comparison card.
    assert 'comparisons/000000.jpg' in page
    if children:
        assert (out/'RESULT.json').read_bytes()==children[0]['source_bytes']
        assert '../fused__boundary_review/index.html' in page


def full_fixture(tmp_path,monkeypatch,fail=False):
    # Point only the execute() binding inventory at a tiny synthetic repository.
    root=tmp_path/'synthetic_repo'
    relatives=['integrations/video_masks/full_pipeline.py','scripts/run_photo_demo.py',
        'scripts/propagate_video_masks.py','integrations/demo_pipeline/pipeline.py',
        'integrations/demo_pipeline/supervision.py','integrations/sam3/sam3_adapter/observer.py',
        'integrations/sam3/sam3_adapter/resources.py']
    for relative in relatives:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# TEST_DATA\n')
    monkeypatch.setattr(full_pipeline,'__file__',str(root/relatives[0]))
    pipeline=tmp_path/'pipeline';pipeline.mkdir();row,image=row_fixture(pipeline)
    checkpoint=pipeline/'synthetic-checkpoint';checkpoint.write_bytes(b'not a model')
    save(pipeline/'PLAN.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                                 video_tools=dict(ffmpeg='synthetic',ffprobe='synthetic')))
    common=pipeline/'fixture-input';common.write_bytes(b'TEST_DATA')
    plan=dict(width=8,height=8,frame_count=1,reuse_inputs={},coverage_ledger=[])
    for key in ('source_video','source_manifest','video_result','pipeline_plan','io_run','io_frames','timeline'):
        plan[key+'_path']=str(common);plan[key+'_sha256']=sha(common)
    monkeypatch.setattr(full_pipeline,'make_full_plan',lambda *a:dict(plan))
    def decode(root,out,plan,ffmpeg):
        if fail:raise RuntimeError('synthetic pre-worker failure')
        (out/'frames').mkdir();(out/'frames/000000.jpg').write_bytes(image.read_bytes())
    def assemble(out,plan):
        save(out/'ASSEMBLY.json',dict(rows=[row]));return [dict(row)]
    monkeypatch.setattr(full_pipeline,'decode_all',decode)
    monkeypatch.setattr(full_pipeline,'prepare_tasks',lambda *a:[])
    monkeypatch.setattr(full_pipeline,'assemble',assemble)
    monkeypatch.setattr(full_pipeline,'gather_predictions',lambda *a:{})
    monkeypatch.setattr(video_export,'export_video',lambda *a:dict(placeholder_count=0))
    return pipeline


@pytest.mark.parametrize('execute_model,mode,fail,expected',[
    (False,'union',False,'NOT_APPLICABLE'),(True,'union',False,READY),
    (True,'none',False,'NOT_APPLICABLE'),(True,'supported-holes',False,'NOT_APPLICABLE'),
    (True,'union',True,'NOT_APPLICABLE')])
def test_full_pipeline_dispatch_is_once_and_only_complete_union_starts_child(tmp_path,monkeypatch,execute_model,mode,fail,expected):
    pipeline=full_fixture(tmp_path,monkeypatch,fail);out=tmp_path/'full'
    calls,children=hook_spy(monkeypatch)
    result=full_pipeline.execute(pipeline,out,execute_model=execute_model,fusion_mode=mode)
    assert len(calls)==1 and len(children)==(expected==READY)
    assert result['automatic_boundary_review']['status']==expected
    assert 'automatic_boundary_review' not in read(out/'RESULT.json')
    assert read(out/'AUTOMATIC_BOUNDARY.json')==result['automatic_boundary_review']
    assert 'AUTOMATIC_BOUNDARY.json' in (out/'index.html').read_text(encoding='utf8')
    if children:assert (out/'RESULT.json').read_bytes()==children[0]['source_bytes']


@pytest.mark.parametrize('raises,child_status',[(True,READY),(False,'FAILED'),(False,'NOT_EVALUATED')])
def test_failed_child_sidecar_is_explicit_without_relabeling_source_stage(tmp_path,monkeypatch,raises,child_status):
    source=fusion_source(tmp_path,monkeypatch);out=tmp_path/'fused'
    calls,children=hook_spy(monkeypatch,child_status,raises)
    result=fusion.execute(source,out)
    assert len(calls)==len(children)==1
    assert result['status']==read(out/'RESULT.json')['status']=='FULL_TIMELINE_FUSED_EXPORTED'
    assert result['automatic_boundary_review']['required'] is True
    assert result['automatic_boundary_review']['status']=='FAILED'
    assert read(out/'AUTOMATIC_BOUNDARY.json')['status']=='FAILED'
    assert not full_pipeline.automatic_boundary_complete(result,True)


@pytest.mark.parametrize('entry',['public_fusion','standalone_fusion','public_full','standalone_full'])
@pytest.mark.parametrize('review_status,expected',[(READY,0),('FAILED',1),('NOT_EVALUATED',1),(None,1)])
def test_all_union_cli_paths_require_explicit_review_completion(tmp_path,monkeypatch,capsys,entry,review_status,expected):
    is_fusion='fusion' in entry
    result=dict(status='FULL_TIMELINE_FUSED_EXPORTED' if is_fusion else 'FULL_TIMELINE_EXPORTED')
    if review_status is not None:result['automatic_boundary_review']=dict(required=True,status=review_status)
    module=fusion if is_fusion else full_pipeline
    calls=[]
    monkeypatch.setattr(module,'execute',lambda *a,**k:(calls.append((a,k)) or result))
    if entry.startswith('public'):
        from scripts.run_photo_demo import main
        args=['fuse-masks'] if is_fusion else ['propagate-masks','--full-video','--execute']
    else:
        if is_fusion:from scripts.fuse_video_masks import main
        else:from scripts.propagate_video_masks import main
        args=[] if is_fusion else ['--full-video','--execute']
    args+=['--run' if is_fusion else '--pipeline',str(tmp_path/'source'),'--output',str(tmp_path/'out')]
    assert main(args)==expected
    assert len(calls)==1 and calls[0][0][-1]=='union'
    printed=json.loads(capsys.readouterr().out)
    assert printed['status']==result['status']
    if review_status is not None:assert printed['automatic_boundary_review']['status']==review_status


@pytest.mark.parametrize('entry,mode',[('fusion','supported-holes'),('full','none'),('full','supported-holes')])
def test_public_nonunion_cli_keeps_stage_exit_code(tmp_path,monkeypatch,entry,mode):
    from scripts.run_photo_demo import main
    stage='FULL_TIMELINE_FUSED_EXPORTED' if entry=='fusion' else 'FULL_TIMELINE_EXPORTED'
    monkeypatch.setattr(fusion if entry=='fusion' else full_pipeline,'execute',lambda *a:dict(status=stage,
                        automatic_boundary_review=dict(required=False,status='NOT_APPLICABLE')))
    args=(['fuse-masks','--run'] if entry=='fusion' else ['propagate-masks','--full-video','--execute','--pipeline'])
    args+=[str(tmp_path/'source'),'--output',str(tmp_path/'out'),'--mode' if entry=='fusion' else '--mask-fusion',mode]
    assert main(args)==0


@pytest.mark.parametrize('status,code',[(READY,0),('FAILED',1),('NOT_EVALUATED',1),('NOT_APPLICABLE',1)])
@pytest.mark.parametrize('indices',[None,[3,8]])
def test_manual_review_command_dispatches_bounded_default(tmp_path,monkeypatch,status,code,indices):
    from scripts.run_photo_demo import main
    calls=[]
    def execute(source,output,**kwargs):calls.append((source,output,kwargs));return dict(status=status)
    monkeypatch.setattr(auto_boundary,'execute_auto_boundary',execute)
    args=['review-mask-boundaries','--run',str(tmp_path/'source'),'--output',str(tmp_path/'review')]
    if indices is not None:args+=['--indices',*map(str,indices)]
    assert main(args)==code
    assert calls==[(tmp_path/'source',tmp_path/'review',dict(indices=indices,max_frames=16))]


def test_report_rerender_reads_manifest_preserves_images_and_review_link(tmp_path,monkeypatch):
    run=tmp_path/'report';run.mkdir()
    review=tmp_path/'review';review.mkdir();(review/'index.html').write_text('TEST_DATA',encoding='utf8')
    result=dict(status='FULL_TIMELINE_FUSED_EXPORTED')
    save(run/'RESULT.json',result);source_bytes=(run/'RESULT.json').read_bytes()
    item=dict(decode_index=7,added_pixels=2,path=str(run/'comparisons/000007.jpg'))
    generations=[]
    def generate(target,rows):
        generations.append(target);(target/'comparisons').mkdir()
        Image.new('RGB',(2,2),(1,2,3)).save(target/'comparisons/000007.jpg')
        save(target/'COMPARISONS.json',dict(rows=[item]));return [item]
    monkeypatch.setattr(fusion,'make_comparisons',generate)
    fusion.report(run,result,[dict(decode_index=7)])
    manifest_bytes=(run/'COMPARISONS.json').read_bytes()
    image_bytes=(run/'comparisons/000007.jpg').read_bytes()
    enriched=dict(result,automatic_boundary_review=dict(status=READY,required=True,report=str(review/'index.html')))
    def forbid_regeneration(*args):raise AssertionError('existing comparison evidence must only be read')
    monkeypatch.setattr(fusion,'make_comparisons',forbid_regeneration)
    fusion.report(run,enriched,[dict(decode_index=7)])
    page=(run/'index.html').read_text(encoding='utf8')
    assert generations==[run]
    assert '<img loading="lazy" src="comparisons/000007.jpg">' in page
    assert 'href="../review/index.html"' in page and 'AUTOMATIC_BOUNDARY.json' in page
    assert (run/'COMPARISONS.json').read_bytes()==manifest_bytes
    assert (run/'comparisons/000007.jpg').read_bytes()==image_bytes
    assert (run/'RESULT.json').read_bytes()==source_bytes
    # A manifest remains authoritative even if its image directory is missing.
    missing_image=tmp_path/'missing_image';missing_image.mkdir()
    (missing_image/'COMPARISONS.json').write_bytes(manifest_bytes)
    fusion.report(missing_image,enriched,[dict(decode_index=7)])
    assert 'COMPARISON_IMAGE_MISSING: comparisons/000007.jpg' in (missing_image/'index.html').read_text(encoding='utf8')
    # Conversely, an existing image directory cannot replace its absent manifest.
    missing_manifest=tmp_path/'missing_manifest';missing_manifest.mkdir();(missing_manifest/'comparisons').mkdir()
    fusion.report(missing_manifest,enriched,[dict(decode_index=7)])
    assert 'COMPARISON_EVIDENCE_UNAVAILABLE' in (missing_manifest/'index.html').read_text(encoding='utf8')
