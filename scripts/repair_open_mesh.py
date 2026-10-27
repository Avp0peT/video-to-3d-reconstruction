"""Copy a diagnosed mesh while removing only all faces named by complete pairs.

Creates repair-created gaps, never fills holes, edits coordinates, or certifies
that intersections have been eliminated. No native geometry code is imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

ply=load('open_repair_ply',ROOT/'integrations/meshprep/ply_io.py')
geo=load('open_repair_geometry',ROOT/'integrations/meshprep/geometry.py')
identity=load('open_repair_identity',ROOT/'integrations/meshprep/provenance.py')

def need(value,message):
    if not value:raise ValueError(message)

def sha(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise argparse.ArgumentTypeError('expected lowercase SHA256')
    return value

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def finite_number(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value)
    except OverflowError:return False

def diagnosed_faces(diagnostic,mesh,face_count):
    need(diagnostic.get('status')=='READONLY_DIAGNOSTIC_COMPLETE','diagnostic incomplete')
    for key in ('has_self_intersections','api_exists','input_unchanged','loaded_face_order_matches_source_rows'):
        need(diagnostic.get(key) is True,'diagnostic '+key+' not proven')
    need(diagnostic.get('truncated') is False,'diagnostic truncation must be explicitly false')
    need(diagnostic.get('face_index_scope')=='loaded FreeCAD mesh; exact source row mapping proven','source face mapping scope missing')
    source=diagnostic.get('input');need(isinstance(source,dict),'diagnostic input missing')
    need(isinstance(source.get('path'),str) and Path(source['path']).is_absolute()
         and str(Path(source['path']).resolve())==mesh['path'] and source.get('sha256')==mesh['sha256'],'diagnostic mesh identity differs')
    for count in (source.get('faces'),diagnostic.get('loaded_faces')):
        need(type(count) is int and count==face_count,'diagnostic face count differs')
    pairs=diagnostic.get('pairs');need(isinstance(pairs,list) and 0<len(pairs)<=100000,'nonempty complete pair list required')
    for key in ('pair_count','pairs_written'):
        need(type(diagnostic.get(key)) is int and diagnostic[key]==len(pairs),'incomplete pair enumeration')
    unique=set()
    for pair in pairs:
        need(isinstance(pair,dict),'pair must be object')
        a,b=pair.get('face_a'),pair.get('face_b')
        need(type(a) is int and type(b) is int and 0<=a<face_count and 0<=b<face_count and a!=b,'invalid face indices')
        for key in ('endpoint_a','endpoint_b'):
            point=pair.get(key)
            need(isinstance(point,list) and len(point)==3 and all(finite_number(v) for v in point),'invalid intersection endpoint')
        unique.add(tuple(sorted((a,b))))
    unique=sorted(unique);deleted=sorted({i for pair in unique for i in pair})
    need(len(deleted)<face_count,'refuse removal of every face')
    return unique,deleted

def area(points,face):
    a,b,c=(points[i] for i in face);u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
    value=math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])/2
    need(math.isfinite(value),'nonfinite face area')
    return value

def edges(faces):
    return Counter(tuple(sorted((a,b))) for x,y,z in faces for a,b in ((x,y),(y,z),(z,x)))

def boundary_stats(counts):
    boundary=sorted(edge for edge,count in counts.items() if count==1)
    degrees=Counter(vertex for edge in boundary for vertex in edge)
    return {'boundary_edge_count':len(boundary),'boundary_vertex_count':len(degrees),
            'boundary_vertex_degree_histogram':{str(k):v for k,v in sorted(Counter(degrees.values()).items())},
            'edges_with_more_than_two_faces':sum(count>2 for count in counts.values()),
            'scope':'edge incidence only; not hole-loop, topology, or intersection certification'},boundary

def repair(mesh_path,diagnostic_path,expected_mesh_sha,expected_diagnostic_sha,output,*,upstream_report=None,expected_upstream_sha=None):
    sha(expected_mesh_sha);sha(expected_diagnostic_sha)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def deadline():need(time.monotonic()-started<=60,'60 second cooperative data budget exceeded')
    mesh=identity.binding(mesh_path,64*1024**2);diagnostic,db=identity.document(diagnostic_path)
    need(mesh['sha256']==expected_mesh_sha and db['sha256']==expected_diagnostic_sha,'expected input SHA256 mismatch')
    need((upstream_report is None)==(expected_upstream_sha is None),'upstream report and SHA256 must be supplied together')
    upstream=None;upstream_value=None
    if upstream_report is not None:
        sha(expected_upstream_sha);upstream_value,upstream=identity.document(upstream_report)
        need(upstream['sha256']==expected_upstream_sha,'upstream report SHA256 mismatch')
        rows=upstream_value.get('artifacts')
        need(isinstance(rows,list) and sum(isinstance(row,dict) and all(row.get(k)==mesh[k] for k in mesh) for row in rows)==1,
             'upstream report does not uniquely bind input mesh')
    points,faces,metadata=ply.read_ply(mesh['path'])
    unique,deleted=diagnosed_faces(diagnostic,mesh,len(faces));deleted_set=set(deleted)
    retained=[i for i in range(len(faces)) if i not in deleted_set];new_faces=[faces[i] for i in retained]
    deadline()
    total_area=math.fsum(area(points,face) for face in faces)
    deleted_area=math.fsum(area(points,faces[i]) for i in deleted)
    need(math.isfinite(total_area) and total_area>0 and math.isfinite(deleted_area),'invalid area denominator')
    before,before_edges=boundary_stats(edges(faces));after,after_edges=boundary_stats(edges(new_faces))
    created=sorted(set(after_edges)-set(before_edges));deadline()
    target=output/'open_surface.ply';geo.write_double_ply(target,points,new_faces)
    reread_points,reread_faces,_=ply.read_ply(target)
    need(reread_points==points and reread_faces==new_faces,'output coordinate/face roundtrip differs')
    need(identity.binding(mesh['path'],64*1024**2)==mesh and identity.binding(db['path'],identity.MAX_JSON)==db,'source input drift')
    if upstream is not None:need(identity.binding(upstream['path'],identity.MAX_JSON)==upstream,'upstream report drift')
    deadline()
    mapping={'schema_version':'open-mesh-face-map-v1','output_face_to_source_face':retained,
             'deleted_source_faces':deleted,'vertex_mapping':'identity; all source vertices retained'}
    write(output/'face_map.json',mapping)
    report={'schema':'open-mesh-repair-v1','status':'CANDIDATE_WRITTEN_NATIVE_VALIDATION_PENDING',
        'input':{'mesh':mesh,'diagnostic':db},'upstream_report':upstream,
        'upstream_original_status':None if upstream_value is None else upstream_value.get('status'),
        'source_surface_type':'simplified_reconstruction','gap_origin':'repair_created_excision','scale_status':'unknown',
        'deleted_face_ids':deleted,'retained_count':len(new_faces),'removed_count':len(deleted),
        'source_ply':metadata,'original_pair_count':len(diagnostic['pairs']),
        'original_pairs':diagnostic['pairs'],'unique_pairs':[list(pair) for pair in unique],
        'unique_pair_count':len(unique),'deleted_source_faces':deleted,'deleted_face_count':len(deleted),
        'source_vertex_count':len(points),'output_vertex_count':len(points),'source_face_count':len(faces),'output_face_count':len(new_faces),
        'deleted_area':deleted_area,'source_total_area':total_area,'deleted_area_fraction':deleted_area/total_area,
        'area_units':'input coordinate units squared; physical scale unknown','volume_status':'NOT_EVALUATED_OPEN_SURFACE',
        'boundary_before':before,'boundary_after':after,'repair_created_boundary_edges':[list(edge) for edge in created],
        'gap_scope':'repair-created gaps; not identified physical-object holes or unrecognized regions',
        'native_self_intersection_status':'NOT_EVALUATED','coordinates_changed':False,'retained_face_order_and_winding_changed':False,
        'vertices_compacted':False,'hole_filling':False,'welding':False,'smoothing':False,'simplification':False,
        'implementation':[identity.binding(path,identity.MAX_JSON) for path in (Path(__file__),ROOT/'integrations/meshprep/ply_io.py',ROOT/'integrations/meshprep/geometry.py',ROOT/'integrations/meshprep/provenance.py')],
        'artifacts':{'mesh':identity.binding(target,64*1024**2),'face_map':identity.binding(output/'face_map.json',64*1024**2)},
        'elapsed_seconds':time.monotonic()-started}
    write(output/'repair_report.json',report)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True);parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--expected-mesh-sha256',type=sha,required=True);parser.add_argument('--expected-diagnostic-sha256',type=sha,required=True)
    parser.add_argument('--upstream-report',type=Path);parser.add_argument('--expected-upstream-sha256',type=sha)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    try:
        result=repair(args.mesh,args.diagnostic,args.expected_mesh_sha256,args.expected_diagnostic_sha256,args.output,
            upstream_report=args.upstream_report,expected_upstream_sha=args.expected_upstream_sha256)
        print(json.dumps({'status':result['status'],'deleted_face_count':result['deleted_face_count']}));return 0
    except (ValueError,OSError) as exc:
        print('REFUSED: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
