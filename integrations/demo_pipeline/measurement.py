"""Reference-measurement continuation; existing prepared geometry stays immutable."""
from pathlib import Path
import sys

from .cad import verify_cad_output, native_ok
from integrations.cad_delivery.delivery import prepared_input, verify_delivery, validate_scale_link
from integrations.meshprep.ply_io import read_ply
from integrations.mesh_scale.ledger import (file_binding, verify_binding, read_json, load_request,
                                           evidence, verify_ledger, same_json)
from integrations.mesh_scale.calibration import fit_transform, apply_transform, measure_checks
from integrations.mvs_surface.supervision import run_supervised
from .surface import save

ROOT=Path(__file__).resolve().parents[2]
MAX_POINTS=20000


def need(value,message):
    if not value:raise ValueError('photo measurement: '+message)


def _check_status(checks):
    need(isinstance(checks,list),'independent check ledger must be a list')
    if any(row.get('within_threshold') is False for row in checks):
        error=ValueError('photo measurement: independent check exceeds declared threshold')
        error.independent_checks=checks
        raise error
    return 'PASS' if checks and all(row.get('within_threshold') is True for row in checks) else 'NOT_EVALUATED'


def inspect_source(run,cad_output=None):
    run=Path(run).resolve(strict=True)
    output=Path(cad_output).resolve(strict=True) if cad_output else run/'cad'
    verified=verify_cad_output(output,run)
    prepared=prepared_input(output/'prepared')
    need(prepared.get('synthetic_input') is False,'real measured continuation requires real source')
    mesh=prepared['mesh'];verify_binding(mesh,64<<20)
    points,faces,_=read_ply(mesh['path'],max_vertices=MAX_POINTS,max_faces=10000)
    need(len(points)>1 and faces,'prepared mesh has no measurable surface')
    verify_binding(mesh,64<<20)
    files=[file_binding(run/'pipeline.json'),file_binding(output/'PLAN.json'),file_binding(output/'RESULT.json'),
           prepared['record'],mesh,*prepared['artifacts'],*prepared['upstream']]
    files += verified['artifacts']
    return {'schema':'photo-measurement-source-v1','run':str(run),'cad_output':str(output),
            'prepared_dir':str(output/'prepared'),'prepared_run':str(output/'prepared'),'prepared':prepared,'mesh':mesh,
            'vertex_count':len(points),'face_count':len(faces),'points':[list(p) for p in points],
            'source_bindings':list({r['path']:r for r in files}.values()),'scale_status':'unknown',
            'physical_accuracy':'NOT_EVALUATED'}


def create_request(source,vertex_indices,length,unit,label,correspondence,evidence_path,checks=None):
    need(source.get('schema')=='photo-measurement-source-v1','unsupported inspected source')
    mesh=source['mesh'];verify_binding(mesh,64<<20)
    points,_,_=read_ply(mesh['path'],max_vertices=MAX_POINTS,max_faces=10000)
    need(len(points)==source['vertex_count'] and [list(p) for p in points]==source['points'],'inspected vertices changed')
    proof=file_binding(Path(evidence_path).resolve(strict=True),16<<20)
    ref={'vertex_indices':vertex_indices,'length':length,'unit':unit,'label':label,
         'correspondence':correspondence,'evidence':{'path':proof['path'],'sha256':proof['sha256']}}
    request={'schema_version':'mesh-scale-request-v1','source_mesh_sha256':mesh['sha256'],
             'data_kind':'real','reference':ref,'checks':[] if checks is None else checks}
    _validate_request(request,points,Path(proof['path']))
    return request


def _validate_request(request,points,path):
    need(request.get('data_kind')=='real','reference must declare real source geometry')
    checks=request.get('checks',[])
    need(isinstance(checks,list) and len(checks)<=64,'checks must be a list of at most 64')
    transform=fit_transform(points,request['reference'],request.get('origin_vertex'))
    scaled,_=apply_transform(points,transform)
    _check_status(measure_checks(scaled,checks,request['reference']['vertex_indices']))
    evidence(request['reference'],path)
    for check in checks:evidence(check,path)
    return transform


def plan(run,reference,output,cad_output=None,freecad_python=None):
    source=inspect_source(run,cad_output)
    reference=Path(reference).resolve(strict=True);output=Path(output).resolve()
    need(output.is_relative_to(ROOT/'outputs') and output!=ROOT/'outputs','output must be new project outputs child')
    need(not output.exists(),'output already exists')
    for protected in (Path(source['run']),Path(source['cad_output']),reference):
        need(not output.is_relative_to(protected) and not protected.is_relative_to(output),'output overlaps source/reference')
    request,proofs=load_request(reference,source['mesh']['path'])
    _validate_request(request,source['points'],reference)
    freecad=Path(freecad_python).resolve(strict=True) if freecad_python else Path(read_json(Path(source['cad_output'])/'PLAN.json')['freecad_python']).resolve(strict=True)
    files=[*source['source_bindings'],file_binding(reference),*proofs,file_binding(freecad),file_binding(Path(sys.executable))]
    files += [file_binding(p) for p in (Path(__file__),ROOT/'scripts/measure_photo_cad.py',ROOT/'scripts/build_cad_delivery.py')]
    # Existing source verifiers bind their own tool/code sets; preserve that proof.
    summary={k:v for k,v in source.items() if k!='points'}
    return {'schema':'photo-measurement-plan-v1','source':summary,'source_run':source['run'],'reference':str(reference),'output':str(output),
            'freecad_python':str(freecad),'python':str(Path(sys.executable).resolve()),
            'bindings':list({r['path']:r for r in files}.values()),'automatic_retry':False,
            'budgets':{'internal_seconds':120,'outer_seconds':135,'memory_bytes':2<<30,'output_bytes':256<<20},
            'physical_accuracy':'NOT_EVALUATED'}


def _source_check(frozen):
    for row in frozen['bindings']:verify_binding(row)
    source=inspect_source(frozen['source']['run'],frozen['source']['cad_output'])
    need(same_json({k:v for k,v in source.items() if k!='points'},frozen['source']),'source identity changed')
    return source


def _outputs(frozen):
    output=Path(frozen['output']);source=frozen['source']
    verified=verify_delivery(output/'delivery')
    request=read_json(output/'delivery/request.json')
    need(same_json(request.get('prepared'),source['prepared']) and request.get('prepared_run')==source['prepared_dir'],
         'delivery substituted a different prepared source')
    need(request.get('reference')==frozen['reference'],'delivery substituted a foreign reference')
    ledger=verify_ledger(output/'delivery/scale/scale_ledger.json',synthetic=False)
    validate_scale_link(request,ledger)
    need(ledger['source_mesh']==source['mesh'] and ledger['request']==file_binding(frozen['reference']),
         'scale ledger source/reference identity differs')
    check_status=_check_status(ledger['independent_checks'])
    report=read_json(output/'delivery/cad/conversion_report.json')
    refs=report.get('artifacts',[])
    expected={str(output/'delivery/cad'/n) for n in ('model.FCStd','model.step')}
    need(len(refs)==2 and {r['path'] for r in refs}==expected,'scaled CAD artifact association differs')
    for row in refs:verify_binding(row)
    return {'artifacts':[file_binding(row['path']) for row in refs],'scale_ledger':file_binding(output/'delivery/scale/scale_ledger.json'),
            'independent_readback':report['checks'],'delivery_verification':verified,
            'independent_check_status':check_status,'independent_checks':ledger['independent_checks'],
            'scale_status':'reference_scaled','length_unit':'mm','physical_accuracy':'NOT_EVALUATED',
            'notice':'Scale follows declared measurement and correspondence; evidence hash does not authenticate physical truth.'}


def execute(frozen,supervisor=None):
    output=Path(frozen['output']);need(not output.exists(),'output already exists');output.mkdir(parents=True)
    save(output/'PLAN.json',frozen)
    result={'schema':'photo-measurement-result-v1','status':'RUNNING','exit_code':4,'artifacts':[],'automatic_retry':False,
            'source_run':frozen['source']['run'],'source_cad_output':frozen['source']['cad_output'],
            'output':str(output),'scale_status':'unknown','physical_accuracy':'NOT_EVALUATED'}
    result.update(independent_check_status='NOT_EVALUATED',independent_checks=[])
    argv=[frozen['python'],'-B','-X','utf8',str(ROOT/'scripts/build_cad_delivery.py'),
          '--prepared-run',frozen['source']['prepared_dir'],'--reference',frozen['reference'],
          '--freecad-python',frozen['freecad_python'],'--output',str(output/'delivery'),'--wall-seconds','120']
    result.update(argv=argv,cwd=str(output));save(output/'RESULT.json',result)
    try:
        _source_check(frozen)
        logs=output/'supervision';logs.mkdir()
        native=(supervisor or run_supervised)(argv,cwd=str(output),output=str(logs),wall_seconds=135,
                     memory_bytes=2<<30,output_bytes=256<<20,label='photo_reference_cad')
        save(output/'SUPERVISOR_RESULT.json',native)
        if not native_ok(native,argv,str(output)):
            code=native.get('root_exit_code')
            result['exit_code']=(5 if native.get('status') in ('wall_clock_exceeded','run_dir_threshold','total_threshold')
                                 else code if native.get('status')=='completed' and type(code) is int and code!=0 else 4)
            raise ValueError('scaled delivery failed supervision; no retry')
        result.update(_outputs(frozen));_source_check(frozen)
        result.update(status='SUCCESS',exit_code=0)
    except Exception as exc:
        result.update(status='FAILED',exit_code=result['exit_code'] if result['exit_code'] else 4,error=f'{type(exc).__name__}: {exc}')
        if hasattr(exc,'independent_checks'):
            result.update(independent_check_status='FAILED',independent_checks=exc.independent_checks)
    finally:
        try:_source_check(frozen);result['source_and_code_stable']=True
        except Exception as exc:result.update(status='SOURCE_DRIFT',exit_code=4,source_and_code_stable=False,drift_error=str(exc))
        save(output/'RESULT.json',result)
    return result


def verify_result(output):
    output=Path(output).resolve(strict=True);frozen=read_json(output/'PLAN.json');result=read_json(output/'RESULT.json')
    need(frozen.get('schema')=='photo-measurement-plan-v1' and frozen.get('output')==str(output),'measurement output association differs')
    need(output.is_relative_to(ROOT/'outputs'),'measurement output escapes project outputs')
    need(result.get('status')=='SUCCESS' and type(result.get('exit_code')) is int and result['exit_code']==0
         and result.get('source_and_code_stable') is True,'measurement result did not succeed')
    _source_check(frozen)
    native=read_json(output/'SUPERVISOR_RESULT.json')
    need(native_ok(native,result.get('argv'),str(output)),'measurement supervisor evidence differs')
    current=_outputs(frozen)
    need(current['artifacts']==result['artifacts'] and current['scale_ledger']==result['scale_ledger'],'measurement download identity changed')
    need(current['independent_check_status']==result.get('independent_check_status') and
         same_json(current['independent_checks'],result.get('independent_checks')),'independent check result changed')
    return {'status':'CURRENT_VERIFIED_MEASURED_CAD','output':str(output),'source_run':frozen['source']['run'],
            'source_cad_output':frozen['source']['cad_output'],**current}


def make_preview(source,output):
    from integrations.viewer.export import export_html
    output=Path(output).resolve();need(output.is_relative_to(ROOT/'outputs') and not output.exists(),'preview output must be new outputs child')
    for protected in (Path(source['run']),Path(source['cad_output'])):
        need(not output.is_relative_to(protected),'preview cannot write original run')
    verify_binding(source['mesh']);output.mkdir(parents=True)
    spec=output/'SCENES.json'
    save(spec,{'schema':'ply-viewer-scenes-v1','scenes':[{'ply':source['mesh']['path'],'sha256':source['mesh']['sha256'],
        'title':'Prepared mesh vertex indices','data_kind':'real_photos','declared_unit':None,
        'note':'Choose indices on this prepared mesh; original measurement must identify the same physical endpoints.',
        'evidence':[source['prepared']['record']['path']]}]})
    report=export_html(spec,output/'index.html');verify_binding(source['mesh'])
    return {'path':str(output/'index.html'),'binding':file_binding(output/'index.html'),'viewer':report,
            'mesh':source['mesh'],'vertex_count':source['vertex_count']}
