#!/usr/bin/env python3
"""Export a hash-bound mesh as faceted OPEN_SURFACE/SHELL CAD; never a solid."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

_spec = importlib.util.spec_from_file_location('_closed_cad_runtime', Path(__file__).with_name('mesh_to_cad.py'))
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)
Rejected = _runtime.Rejected
sha256 = _runtime.sha256
configure_runtime = _runtime.configure_runtime
RTOL, ATOL = 1e-7, 1e-9
_provenance_spec = importlib.util.spec_from_file_location('_open_cad_provenance', Path(__file__).resolve().parents[1] / 'integrations/meshprep/provenance.py')
identity = importlib.util.module_from_spec(_provenance_spec)
_provenance_spec.loader.exec_module(identity)

def document(path):
    try:
        return identity.document(path)
    except (ValueError, OSError) as exc:
        raise Rejected('INVALID_SOURCE_DOCUMENT', str(exc)) from exc

def verify_binding(row, limit=64*1024**2):
    if not isinstance(row,dict) or not isinstance(row.get('path'),str):
        raise Rejected('SOURCE_BINDING','Readable artifact path required.')
    try:
        actual=identity.binding(row['path'],limit)
    except (ValueError,OSError) as exc:
        raise Rejected('SOURCE_BINDING',str(exc)) from exc
    if any(actual[k]!=row.get(k) for k in actual):
        raise Rejected('SOURCE_BINDING','Artifact path/sha256/bytes differ.')
    return actual

def verify_face_map(report, loaded_faces):
    mapping,_=document(report['artifacts']['face_map']['path'])
    retained=mapping.get('output_face_to_source_face'); deleted=mapping.get('deleted_source_faces')
    if mapping.get('schema_version')!='open-mesh-face-map-v1' or not isinstance(retained,list) or not isinstance(deleted,list):
        raise Rejected('FACE_MAP','Expected ordered source face map.')
    if any(type(i) is not int or i<0 for i in retained+deleted) or retained!=sorted(set(retained)) or deleted!=sorted(set(deleted)) or set(retained)&set(deleted):
        raise Rejected('FACE_MAP','Face IDs must be unique ordered nonnegative disjoint integers.')
    if type(report.get('retained_count')) is not int or type(report.get('removed_count')) is not int or report['retained_count']!=len(retained) or report['removed_count']!=len(deleted) or len(retained)!=loaded_faces or report.get('deleted_face_ids')!=deleted:
        raise Rejected('FACE_MAP','Counts/deleted IDs disagree with loaded mesh.')
    total=len(retained)+len(deleted)
    if sorted(retained+deleted)!=list(range(total)):
        raise Rejected('FACE_MAP','Mapping must partition the original source faces.')
    return mapping

def topology(mesh):
    points, faces = mesh.Topology
    xyz = [(float(p.x), float(p.y), float(p.z)) for p in points]
    edges, links, seen = defaultdict(list), defaultdict(lambda: defaultdict(set)), set()
    bad = 0
    area = 0.0
    for face in faces:
        if len(face) != 3 or any(type(i) is not int or i < 0 or i >= len(xyz) for i in face):
            bad += 1
            continue
        a,b,c = face
        key = tuple(sorted(face))
        if len(set(face)) != 3 or key in seen:
            bad += 1
        seen.add(key)
        p,q,r = xyz[a],xyz[b],xyz[c]
        u,v = [q[i]-p[i] for i in range(3)],[r[i]-p[i] for i in range(3)]
        cross = [u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
        length = math.hypot(*cross)
        if not math.isfinite(length) or length <= 0:
            bad += 1
        else:
            area += length/2
        for l,r in ((a,b),(b,c),(c,a)):
            edges[tuple(sorted((l,r)))].append(1 if l < r else -1)
        for center,l,r in ((a,b,c),(b,c,a),(c,a,b)):
            links[center][l].add(r); links[center][r].add(l)
    bad_links = 0
    for neighbors in links.values():
        degrees = [len(v) for v in neighbors.values()]
        reached, pending = set(),[next(iter(neighbors))]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node); pending.extend(neighbors[node]-reached)
        if len(reached) != len(neighbors) or any(d not in (1,2) for d in degrees) or degrees.count(1) not in (0,2):
            bad_links += 1
    finite = bool(xyz) and all(math.isfinite(x) for p in xyz for x in p)
    return dict(faces=len(faces), finite=finite, invalid_faces=bad, nonmanifold_vertices=bad_links,
                nonmanifold_edges=sum(len(v)>2 for v in edges.values()),
                orientation_errors=sum(len(v)==2 and sum(v)!=0 for v in edges.values()),
                boundary_edges=sum(len(v)==1 for v in edges.values()), area=area,
                bbox=([min(p[i] for p in xyz) for i in range(3)]+[max(p[i] for p in xyz) for i in range(3)]) if finite else None)

def require_mesh(stats):
    if not stats['finite'] or not 0 < stats['faces'] <= 10000 or not math.isfinite(stats['area']) or stats['area'] <= 0 or any(stats[k] for k in ('invalid_faces','nonmanifold_vertices','nonmanifold_edges','orientation_errors')):
        raise Rejected('INVALID_OPEN_MESH', 'Finite nondegenerate consistently oriented manifold triangles required; boundary edges allowed.')
    for key in ('native_has_self_intersections','native_has_nonmanifolds','native_has_nonuniform_orientation','native_has_invalid_neighbourhood'):
        if key in stats and stats[key]:
            raise Rejected('INVALID_OPEN_MESH', key)

def boundary_count(shape):
    buckets = defaultdict(list)
    for face in shape.Faces:
        for edge in face.Edges:
            bucket = buckets[edge.hashCode()]
            for record in bucket:
                if edge.isSame(record[0]):
                    record[1] += 1
                    break
            else:
                bucket.append([edge, 1])
    counts = [record[1] for bucket in buckets.values() for record in bucket]
    if any(count > 2 for count in counts):
        raise Rejected('NONMANIFOLD_BREP', 'BRep edge belongs to more than two faces.', 4)
    return sum(count == 1 for count in counts)

def metrics(shape):
    if shape.isNull():
        raise Rejected('NULL_BREP', 'Empty BRep.', 4)
    box = shape.BoundBox
    values = dict(bbox=[float(getattr(box,k)) for k in ('XMin','YMin','ZMin','XMax','YMax','ZMax')],area=float(shape.Area),face_count=len(shape.Faces),solid_count=len(shape.Solids),is_valid=bool(shape.isValid()))
    if not values['is_valid'] or values['solid_count'] != 0 or values['face_count'] < 1 or not math.isfinite(values['area']) or values['area'] <= 0 or not all(math.isfinite(v) for v in values['bbox']):
        raise Rejected('INVALID_OPEN_BREP', str(values), 4)
    values['boundary_edges'] = boundary_count(shape)
    values['is_open'] = values['boundary_edges'] > 0
    try:
        check=shape.check(True)
    except Exception as exc:
        raise Rejected('BREP_INTERFERENCE',str(exc),4) from exc
    if check is not None:
        raise Rejected('UNEXPECTED_BREP_CHECK','shape.check(True) must return None.',4)
    values['detailed_bop_check'] = 'PASS'
    return values

def compare(source,target):
    for value in (source,target):
        box=value.get('bbox'); area=value.get('area')
        if not isinstance(box,list) or len(box)!=6 or any(type(v) not in (int,float) or not math.isfinite(v) for v in [*box,area]) or area<=0 or any(box[i]>box[i+3] for i in range(3)):
            raise Rejected('INVALID_METRICS','Finite ordered bbox and positive area required.',4)
    tolerance=max(ATOL,RTOL*max(source['bbox'][i+3]-source['bbox'][i] for i in range(3)))
    return all(abs(a-b)<=tolerance for a,b in zip(source['bbox'],target['bbox'])) and math.isclose(source['area'],target['area'],rel_tol=RTOL,abs_tol=ATOL) and source['face_count']==target['face_count'] and target.get('solid_count',0)==0 and source['boundary_edges']==target['boundary_edges']

def bind_source(args):
    mesh,source=args.mesh.resolve(),args.source_report.resolve()
    for path,expected in ((mesh,args.mesh_sha256),(source,args.source_report_sha256)):
        if not path.is_file() or len(expected)!=64 or sha256(path)!=expected.lower():
            raise Rejected('HASH_BINDING','Input file missing or SHA256 mismatch.')
    if mesh.stat().st_size > 64*1024*1024 or source.stat().st_size > 16*1024*1024:
        raise Rejected('INPUT_SIZE_LIMIT','Mesh/report exceeds bounded input size.')
    report,record=document(source)
    if record['sha256']!=args.source_report_sha256.lower():
        raise Rejected('SOURCE_BINDING','Source report changed.')
    if report.get('schema') != 'open-mesh-repair-v1' or report.get('status') != 'CANDIDATE_WRITTEN_NATIVE_VALIDATION_PENDING':
        raise Rejected('SOURCE_SCHEMA','Expected repair candidate report, not native acceptance.')
    item=report.get('artifacts',{}).get('mesh',{})
    if item.get('path')!=str(mesh) or item.get('sha256')!=args.mesh_sha256.lower() or item.get('bytes') != mesh.stat().st_size:
        raise Rejected('SOURCE_BINDING','artifacts.mesh must bind this exact mesh, hash and size.')
    for key in ('mesh','diagnostic'):
        verify_binding(report.get('input',{}).get(key), identity.MAX_JSON if key=='diagnostic' else 64*1024**2)
    verify_binding(report.get('artifacts',{}).get('face_map'),identity.MAX_JSON)
    if report.get('source_surface_type')!='simplified_reconstruction' or report.get('gap_origin')!='repair_created_excision' or report.get('scale_status')!='unknown':
        raise Rejected('SOURCE_PROVENANCE','Reconstruction surface, excision gaps and unknown scale must be explicit.')
    upstream=verify_binding(report.get('upstream_report'),identity.MAX_JSON)
    upstream_value,_=document(upstream['path'])
    rows=upstream_value.get('artifacts')
    original=report['input']['mesh']
    if not isinstance(rows,list) or sum(isinstance(row,dict) and all(row.get(k)==original[k] for k in ('path','sha256','bytes')) for row in rows)!=1:
        raise Rejected('UPSTREAM_BINDING','Upstream report must uniquely bind original mesh path, SHA256 and bytes.')
    verify_face_map(report,report.get('retained_count'))
    return mesh,source,report

def convert(args,output,report,save):
    mesh_path,source_path,source=bind_source(args)
    report['input']={'mesh':str(mesh_path),'mesh_sha256':args.mesh_sha256.lower(),'source_report':str(source_path),'source_report_sha256':args.source_report_sha256.lower()}
    report['source']=source
    report['mesh_geometry_validation']['status']='RUNNING'; save()
    report['runtime']=configure_runtime(output)
    import FreeCAD as App
    import Mesh
    import Part
    if list(App.Version())[:3]!=['1','1','4']:
        raise Rejected('BACKEND_VERSION','FreeCAD 1.1.4 required.',3)
    mesh=Mesh.Mesh(str(mesh_path))
    verify_face_map(source,int(mesh.CountFacets))
    stats=topology(mesh); report['mesh_geometry_validation']['metrics']=stats; save()
    require_mesh(stats)
    stats.update(native_has_self_intersections=bool(mesh.hasSelfIntersections()),native_has_nonmanifolds=bool(mesh.hasNonManifolds()),native_has_nonuniform_orientation=bool(mesh.hasNonUniformOrientedFacets()),native_has_invalid_neighbourhood=bool(mesh.hasInvalidNeighbourhood()))
    require_mesh(stats)
    report['mesh_geometry_validation']['status']='PASS'; save()
    report['cad_export']['status']='RUNNING'; save()
    shape=Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology,1e-7) # existing conversion tolerance; no filling or repair
    original=metrics(shape)
    mesh_metrics={'bbox':stats['bbox'],'area':stats['area'],'face_count':stats['faces'],'boundary_edges':stats['boundary_edges']}
    if not compare(mesh_metrics,original):
        raise Rejected('MESH_BREP_CHANGED','Area, bounds or face count changed.',4)
    report['cad_export']['metrics']=original
    doc=App.newDocument('OpenMeshCAD')
    try:
        obj=doc.addObject('Part::Feature','OpenFacetedSurface'); obj.Shape=shape
        obj.Label='OPEN SURFACE - arbitrary coordinates - scale unknown'
        for name,value in (('SourceMeshSHA256',args.mesh_sha256.lower()),('SourceReportSHA256',args.source_report_sha256.lower()),('ScaleStatus','unknown'),('GeometryType','OPEN_SURFACE/SHELL')):
            obj.addProperty('App::PropertyString',name,'Provenance'); setattr(obj,name,value)
        doc.recompute(); doc.saveAs(str(output/'model.FCStd')); shape.exportStep(str(output/'model.step'))
    finally:
        App.closeDocument(doc.Name)
    report['cad_export']['status']='PASS'; save()
    report['reload']['status']='RUNNING'; save()
    reopened=App.openDocument(str(output/'model.FCStd'))
    try:
        objects=[obj for obj in reopened.Objects if hasattr(obj,'Shape')]
        if len(objects)!=1: raise Rejected('FCSTD_OBJECT_COUNT','Exactly one shape required.',4)
        fcstd=metrics(objects[0].Shape)
    finally: App.closeDocument(reopened.Name)
    step=metrics(Part.read(str(output/'model.step')))
    report['reload']['metrics']={'fcstd':fcstd,'step':step}
    if not compare(original,fcstd) or not compare(original,step):
        raise Rejected('RELOAD_CHANGED','Reload geometry changed.',4)
    if sha256(mesh_path)!=args.mesh_sha256.lower() or sha256(source_path)!=args.source_report_sha256.lower():
        raise Rejected('INPUT_CHANGED','Bound input changed during export.',4)
    report['reload']['status']='PASS'
    report['artifacts']=[{'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size} for p in (output/'model.FCStd',output/'model.step')]

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mesh','source-report','output'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('mesh-sha256','source-report-sha256'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv); output=args.output.resolve()
    try: output.mkdir(parents=True,exist_ok=False)
    except OSError as exc:
        print('OUTPUT_REFUSED: '+str(exc),file=sys.stderr); return 2
    report={'status':'STARTED','geometry_type':'OPEN_SURFACE/SHELL','scale':{'status':'unknown','source_unit':'arbitrary','cad_file_unit':'mm','coordinate_factor':1.0,'notice':'CAD mm convention does not establish physical scale.'},'evidence_scope':'Geometry and export only; photo fidelity and GUI viewing not evaluated.','converter_sha256':sha256(Path(__file__)),'mesh_geometry_validation':{'status':'NOT_RUN'},'cad_export':{'status':'NOT_RUN'},'reload':{'status':'NOT_RUN'}}
    def save(): (output/'conversion_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    code=3
    try:
        save(); convert(args,output,report,save); report['status']='PASS_OPEN_SURFACE_CAD_CONVERSION'; code=0
    except Exception as exc:
        report.update(status='REJECTED' if isinstance(exc,Rejected) else 'BACKEND_ERROR',reason=getattr(exc,'reason',type(exc).__name__),error=str(exc)); code=getattr(exc,'exit_code',3)
        for channel in ('mesh_geometry_validation','cad_export','reload'):
            if report[channel]['status']=='RUNNING': report[channel]['status']='FAIL'
        (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    report['script_exit_code']=code; save(); print(json.dumps({'status':report['status'],'report':str(output/'conversion_report.json')})); return code
if __name__=='__main__': raise SystemExit(main())
