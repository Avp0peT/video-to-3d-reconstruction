import json,sqlite3,hashlib,statistics,collections
from pathlib import Path
from PIL import Image,ImageDraw
root=Path.cwd(); out=root/'dev/analysis/20261022_union_registration'
sfm=root/'outputs/delegation/T02A/FUSED_20261022_union_model01_7592a75d/20261022T144734Z_9b020b3b'
report=root/'outputs/delegation/HAND_VIDEO/20261022_union_model01'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[sfm/'run.json',sfm/'frames_map.jsonl',sfm/'database.db',sfm/'models/0/descriptions.json',report/'view_geometry/RESULT.json',report/'view_geometry/TRACKS.json',sfm/'logs/incremental_mapping.stderr.txt',report/'input_masks/manifest.json']
before={str(p):sha(p) for p in paths}
frames=[json.loads(s) for s in (sfm/'frames_map.jsonl').read_text().splitlines()]
manifest=read(report/'input_masks/manifest.json'); timeline={r['input_index']:r for r in manifest['derivation']['rows']}
doc=read(sfm/'models/0/descriptions.json')['source']; print('description keys',doc.keys()); print('image',str(next(iter(doc['images'].values())))[:450]);print('point',str(next(iter(doc['points3D'].values())))[:450])
con=sqlite3.connect('file:'+ (sfm/'database.db').as_posix()+'?mode=ro&immutable=1',uri=True)
print('tables',con.execute("select name from sqlite_master where type='table'").fetchall())
kps=dict(con.execute('select image_id,rows from keypoints'))
byid={r['database_image_id']:r for r in frames}; reg={r['database_image_id'] for r in frames if r['registrations']}
pairs=[]
for pair,count,config in con.execute('select pair_id,rows,config from two_view_geometries'):
 a,b=divmod(pair,2147483647)
 if count: pairs.append({'a':byid[a]['input_index'],'b':byid[b]['input_index'],'matches':count,'config':config,'a_registered':a in reg,'b_registered':b in reg})
raw=list(con.execute('select rows from matches'));con.close()
rows=[]
for f in frames:
 n=f['input_index']; incident=[p for p in pairs if n in (p['a'],p['b'])]; accepted=[p for p in incident if p['matches']>=15]
 rows.append({'input_index':n,'time_seconds':timeline[n]['time_seconds'],'decode_index':timeline[n]['decode_index'],'registered':bool(f['registrations']),'features':kps[f['database_image_id']],'verified_neighbors':len(incident),'verified_correspondences_incident':sum(p['matches'] for p in incident),'neighbors_ge15':len(accepted),'max_pair_matches':max([p['matches'] for p in incident],default=0),'neighbors':sorted([{'input_index':p['b'] if p['a']==n else p['a'],'matches':p['matches']} for p in incident],key=lambda p:-p['matches'])})
summary={'schema':'union-registration-diagnostic-v1','status':'READ_ONLY_DIAGNOSTIC','sfm_run':str(sfm),'source_validation_scope':'This analysis hashes only the explicitly listed source artifacts. Reconstruction derivation validation checks adopted masks/images and selected lineage bindings; not every producer START_BINDINGS input was revalidated. No claim of complete source-chain revalidation.','counts':{'inputs':len(rows),'registered':len(reg),'features':sum(kps.values()),'raw_matches':sum(x[0] for x in raw),'verified_pair_rows':len(pairs),'verified_correspondences':sum(p['matches'] for p in pairs),'pairs_ge15':sum(p['matches']>=15 for p in pairs),'images_ge15_neighbor':sum(r['neighbors_ge15']>0 for r in rows),'points3D':len(doc['points3D'])},'registered_indices':[r['input_index'] for r in rows if r['registered']],'unregistered_indices':[r['input_index'] for r in rows if not r['registered']],'no_ge15_neighbor_indices':[r['input_index'] for r in rows if r['neighbors_ge15']==0],'frame_rows':rows,'strongest_pairs':sorted(pairs,key=lambda p:-p['matches'])[:30],'geometry_audit':read(report/'view_geometry/RESULT.json')['counts'],'source_bindings_before':before}
summary['source_unchanged']=all(sha(p)==before[str(p)] for p in paths)
(out/'registration.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
canvas=Image.new('RGB',(8*144,8*224),'#222222');draw=ImageDraw.Draw(canvas)
for r in rows:
 n=r['input_index']; im=Image.open(sfm/frames[n]['staged_image_name']).convert('RGB'); im.thumbnail((140,192)); x=(n%8)*144;y=(n//8)*224;canvas.paste(im,(x+(144-im.width)//2,y));color='#7fff7f' if r['registered'] else '#ffffff'; draw.text((x+3,y+194),f"{n:02d} {'REG' if r['registered'] else '-'} F{r['features']}",fill=color);draw.text((x+3,y+208),f"N15={r['neighbors_ge15']} max={r['max_pair_matches']}",fill=color)
canvas.save(out/'frames_contact.jpg')
print(json.dumps({k:v for k,v in summary.items() if k not in ('frame_rows','source_bindings_before')},ensure_ascii=False,indent=2))
