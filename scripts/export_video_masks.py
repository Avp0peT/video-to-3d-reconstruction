"""Re-export verified committed masks in a new directory; no model execution."""
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from integrations.demo_pipeline.pipeline import read,save,sha
from integrations.video_masks.video_export import export_video
from integrations.video_masks.full_pipeline import report


def execute(source:Path,output:Path):
    source=source.resolve(strict=True);output=output.resolve()
    if output.exists():raise ValueError('output must be new')
    source_result=read(source/'RESULT.json')
    if source_result['status']!='FULL_TIMELINE_EXPORTED' or source_result['code_drift'] or source_result['input_drift']:
        raise ValueError('source result incomplete or drifted at its recorded execution')
    source_export=read(source/'EXPORT.json')
    if not source_export['pixel_readback_verified'] or source_export['placeholder_count']:
        raise ValueError('source export has missing/unverified masks')
    plan=read(source/'PLAN.json');assembly=read(source/'ASSEMBLY.json');rows=assembly['rows']
    if [r['decode_index'] for r in rows]!=list(range(plan['frame_count'])):raise ValueError('incomplete assembly')
    frozen={str(source/name):sha(source/name) for name in ('PLAN.json','RESULT.json','ASSEMBLY.json','EXPORT.json','DECODE.json')}
    for row in rows:
        if not row['mask_path'] or sha(Path(row['mask_path']))!=row['mask_sha256']:raise ValueError('source mask drift')
        frozen[row['mask_path']]=row['mask_sha256']
    images=read(source/'DECODE.json')['frames']
    for image in images:
        if sha(Path(image['path']))!=image['sha256']:raise ValueError('source image drift')
        frozen[image['path']]=image['sha256']
    output.mkdir(parents=True);(output/'frames').mkdir()
    save(output/'PLAN.json',plan);save(output/'ASSEMBLY.json',assembly)
    save(output/'DERIVATION.json',dict(source_run=str(source),source_bindings=frozen,
        model_entries=0,reason='CPU-only presentation-label correction; mask pixels and selection unchanged',
        code={str(ROOT/p):sha(ROOT/p) for p in ('scripts/export_video_masks.py','integrations/video_masks/video_export.py',
                                              'integrations/video_masks/full_pipeline.py')}))
    result=dict(status='EXPORTING',model_entries=0,inference_source_run=str(source),quality_status='NOT_EVALUATED',
                published_for_reconstruction=False)
    save(output/'RESULT.json',result)
    try:
        for image in images:
            os.link(image['path'],output/'frames'/f'{image["decode_index"]:06d}.jpg')
        source_plan=read(Path(plan['pipeline_plan_path']))
        result['export']=export_video(ROOT,output,plan,rows,source_plan['video_tools']['ffmpeg'],source_plan['video_tools']['ffprobe'])
        result['status']='FULL_TIMELINE_EXPORTED'
    except Exception as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        result['source_drift']=[p for p,digest in frozen.items() if not Path(p).is_file() or sha(Path(p))!=digest]
        if result['source_drift']:result['status']='FAILED'
        save(output/'RESULT.json',result);report(output,result,rows)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=execute(args.run,args.output)
    print(result['status'])
    raise SystemExit(0 if result['status']=='FULL_TIMELINE_EXPORTED' else 1)
