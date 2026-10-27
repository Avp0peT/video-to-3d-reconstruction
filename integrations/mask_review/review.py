"""Bound canonical masks to their photos and render human review aids.

Only descriptive statistics are computed. No silhouette/instance-quality verdict,
automatic selection, mask modification, inference or native reconstruction occurs.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONTRACT = 'mask-review-v1'
MAX_FRAMES = 128
MAX_PIXELS = 4_194_304
MAX_TOTAL_MASK_PIXELS = 64 * 1024**2
MAX_JSON = 8 * 1024**2
MAX_FILE = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
MAX_TOTAL_SOURCE_PIXELS = 64 * 1024**2
MAX_PANELS = 256
MAX_OUTPUT_BYTES = 64 * 1024**2
METADATA_RESERVE = 4 * 1024**2
FRAME_STATES = {'OK','NO_MATCH','NEEDS_SELECTION','INPUT_ERROR','BACKEND_ERROR','RESOURCE_LIMIT','NOT_RUN'}
RUN_STATES = {'SUCCESS','PARTIAL','INPUT_ERROR','BACKEND_UNAVAILABLE','BACKEND_ERROR','AUTH_REQUIRED','RESOURCE_LIMIT'}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def integer(value: Any, name: str, minimum: int = 0, maximum: int = 2**31-1) -> int:
    require(type(value) is int and minimum <= value <= maximum, f'invalid integer {name}')
    return value


def text(value: Any, name: str) -> str:
    require(isinstance(value,str) and 0 < len(value) <= 4096, f'invalid {name}')
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_value(value: Any) -> str:
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'invalid sha256')
    return value


def strict_json(data: bytes) -> Any:
    def pairs(rows):
        d={}
        for k,v in rows:
            require(k not in d,'duplicate JSON key')
            d[k]=v
        return d
    def bad(value):
        raise ValueError('nonfinite JSON: '+value)
    try:
        return json.loads(data.decode('utf-8'),object_pairs_hook=pairs,parse_constant=bad)
    except (UnicodeError,json.JSONDecodeError,RecursionError) as exc:
        raise ValueError('invalid UTF-8 JSON') from exc


def contained(root: Path, name: Any) -> Path:
    name=text(name,'relative path').replace('\\','/')
    parts=name.split('/')
    require(not name.startswith('/') and not PureWindowsPath(name).drive,'absolute child path')
    require(all(p not in ('','.','..') and not p.endswith((' ','.')) and ':' not in p
                and not re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',p) for p in parts),'unsafe relative path')
    result=(root/name).resolve()
    require(result.is_relative_to(root.resolve()),'path escapes input root')
    return result


class Bindings:
    def __init__(self):
        self.rows: dict[str,dict]={}
        self.cache: dict[str,bytes]={}
        self.total=0

    def read(self, path: Path, limit: int = MAX_FILE) -> bytes:
        path=Path(path).absolute()
        require(not path.is_symlink(),'symlink input file')
        key=str(path.resolve())
        if key in self.cache:
            require(len(self.cache[key])<=limit,'file exceeds byte limit')
            return self.cache[key]
        with path.open('rb') as stream:
            before=os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size<=limit,'invalid/oversized input file')
            data=stream.read(limit+1)
            after=os.fstat(stream.fileno())
        current=path.stat()
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        require(len(data)<=limit and len(data)==before.st_size and fields(before)==fields(after)==fields(current),'input changed while reading')
        self.total+=len(data)
        require(self.total<=MAX_TOTAL_BYTES,'total input bytes exceeded')
        self.cache[key]=data
        self.rows[key]={'path':key,'bytes':len(data),'sha256':digest(data)}
        return data

    def verify(self) -> None:
        for key,row in self.rows.items():
            path=Path(key)
            require(path.is_file() and not path.is_symlink(),'input disappeared or became link')
            require(path.stat().st_size==row['bytes'],'input bytes changed')
            with path.open('rb') as stream:
                sha=hashlib.sha256()
                for block in iter(lambda:stream.read(1024**2),b''):
                    sha.update(block)
            require(sha.hexdigest()==row['sha256'],'input hash drift: '+key)


def decode_png(data: bytes, mode: str, width: int, height: int) -> bytes:
    require(data[:8]==b'\x89PNG\r\n\x1a\n' and len(data)>=33 and data[12:16]==b'IHDR','not a PNG')
    require(data[24]==8 and data[25]==(0 if mode=='L' else 2),'PNG source encoding is not RGB8/L8')
    require(width*height<=MAX_PIXELS,'image pixel limit exceeded')
    try:
        with Image.open(io.BytesIO(data)) as im:
            require(im.format=='PNG' and im.mode==mode and im.size==(width,height),'PNG mode/size mismatch')
            require(getattr(im,'n_frames',1)==1 and 'transparency' not in im.info,'animated/transparent PNG')
            require(im.getexif().get(274) in (None,1),'PNG orientation must be identity')
            im.load();raw=im.tobytes()
    except (OSError,SyntaxError,Image.DecompressionBombError) as exc:
        raise ValueError('invalid PNG image') from exc
    if mode=='L':
        require(set(raw)<= {0,255},'mask labels must be 0/255')
    return raw


def _components(binary: np.ndarray, connectivity: int, max_runs: int = 200_000) -> list[tuple[int,bool]]:
    """Run-length union-find; bounded adversarial checkerboard work."""
    parent=[];areas=[];edges=[];previous=[]
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    h,w=binary.shape
    for y,row in enumerate(binary):
        change=np.flatnonzero(np.diff(np.r_[False,row,False]))
        current=[];j=0
        for x0,x1 in zip(change[::2],change[1::2],strict=True):
            require(len(parent)<max_runs,'TOPOLOGY_RUN_LIMIT')
            idx=len(parent);parent.append(idx);areas.append(int(x1-x0));edges.append(bool(y in (0,h-1) or x0==0 or x1==w))
            pad=1 if connectivity==8 else 0
            while j<len(previous) and previous[j][1]<=x0-pad:j+=1
            k=j
            while k<len(previous) and previous[k][0]<x1+pad:
                old=find(previous[k][2]);new=find(idx)
                if old!=new:
                    parent[old]=new;areas[new]+=areas[old];edges[new]=edges[new] or edges[old]
                k+=1
            current.append((int(x0),int(x1),idx))
        previous=current
    return [(areas[i],edges[i]) for i in range(len(parent)) if find(i)==i]


def mask_statistics(mask: bytes, width: int, height: int) -> dict:
    integer(width,'width',1,MAX_PIXELS);integer(height,'height',1,MAX_PIXELS)
    require(width*height<=MAX_PIXELS and len(mask)==width*height and set(mask)<= {0,255},'invalid mask array')
    a=np.frombuffer(mask,dtype=np.uint8).reshape(height,width)>0
    count=int(np.count_nonzero(a));ys,xs=np.nonzero(a)
    result={'foreground_pixels':count,'foreground_fraction':count/(width*height),
            'bbox_xyxy_exclusive':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if count else None,
            'touches_border':bool(a[0].any() or a[-1].any() or a[:,0].any() or a[:,-1].any()),
            'foreground_components_8':None,'background_holes_4':None,'largest_component_pixels':None,'topology_status':'NOT_COMPUTED'}
    try:
        fg=_components(a,8);bg=_components(~a,4)
        result.update(foreground_components_8=len(fg),background_holes_4=sum(not boundary for _,boundary in bg),
                      largest_component_pixels=max((area for area,_ in fg),default=0),topology_status='COMPUTED')
    except ValueError as exc:
        if str(exc)!='TOPOLOGY_RUN_LIMIT':raise
        result.update(topology_status='NOT_COMPUTED_RUN_LIMIT',topology_reason=str(exc))
    return result


def _frame_counts(frames: list[dict]) -> dict:
    counts=Counter(f.get('status') for f in frames)
    return {'requested':len(frames),'success':counts['OK'],'no_match':counts['NO_MATCH'],'needs_selection':counts['NEEDS_SELECTION'],
            'error':sum(counts[s] for s in ('INPUT_ERROR','BACKEND_ERROR','RESOURCE_LIMIT')),'not_run':counts['NOT_RUN']}


def load_inputs(paths: list[Path], bindings: Bindings, *, limits_profile: str = "default") -> tuple[list[dict],dict,list[dict]]:
    require(limits_profile in ("default", "whole-video-192-v1"), "unknown input limits profile")
    max_frames = 192 if limits_profile == "whole-video-192-v1" else MAX_FRAMES
    max_source_pixels = 128 * 1024**2 if limits_profile == "whole-video-192-v1" else MAX_TOTAL_SOURCE_PIXELS
    max_mask_pixels = 128 * 1024**2 if limits_profile == "whole-video-192-v1" else MAX_TOTAL_MASK_PIXELS
    require(1<=len(paths)<=max_frames,'manifest count limit')
    rows=[];runs=[];common=None;common_object=None;seen_indices=set();seen_manifest=set();seen_masks=set();pixels=0;source_pixels=0;panel_count=0
    for path in paths:
        path=Path(path).resolve();require(str(path).casefold() not in seen_manifest,'duplicate manifest')
        seen_manifest.add(str(path).casefold());m=strict_json(bindings.read(path,MAX_JSON))
        require(isinstance(m,dict) and m.get('contract_version')=='object-mask-v1','not an object-mask-v1 manifest')
        temporal_union = (m.get('backend', {}).get('name') == 'video_temporal_union'
                          or any(isinstance(f, dict) and isinstance(f.get('selection'), dict)
                                 and f['selection'].get('mask_source') == 'video_temporal_union'
                                 for f in m.get('frames', [])))
        if temporal_union:
            from integrations.video_masks.reconstruction_input import validate_derivation
            validate_derivation(m, path.parent)
        require(m.get('status') in RUN_STATES,'invalid manifest status')
        source=m.get('source_io');obj=m.get('object');backend=m.get('backend')
        require(isinstance(source,dict) and isinstance(obj,dict) and isinstance(backend,dict),'missing source/object/backend')
        root=Path(text(source.get('path'),'source IO path'));require(root.is_absolute(),'source IO must be absolute');root=root.resolve()
        runbytes=bindings.read(root/'run.json',MAX_JSON);framebytes=bindings.read(root/'frames.jsonl',MAX_JSON)
        require(digest(runbytes)==hash_value(source.get('run_json_sha256')) and digest(framebytes)==hash_value(source.get('frames_jsonl_sha256')),'source IO hash mismatch')
        run=strict_json(runbytes)
        require(isinstance(run,dict) and run.get('contract_version')=='photo-io-v1' and run.get('status')=='SUCCESS','source IO not successful photo-io-v1')
        rid=text(source.get('run_id'),'source run ID');require(run.get('run_id')==rid,'source run ID mismatch')
        identity=(str(root).casefold(),rid,source['run_json_sha256'],source['frames_jsonl_sha256'],text(obj.get('object_id'),'object ID'))
        if common is None:common=identity;common_object=obj.copy()
        require(identity==common,'manifests disagree on IO or object identity')
        require(obj.get('selection_policy')=='unique_candidate_or_explicit','unsupported selection policy')
        io_rows=[strict_json(line) for line in framebytes.splitlines() if line.strip()]
        byindex={}
        for r in io_rows:
            require(isinstance(r,dict),'invalid IO row');idx=integer(r.get('input_index'),'IO index')
            require(idx not in byindex,'duplicate IO index');byindex[idx]=r
        require(integer(run.get('input_count'),'IO count')==len(io_rows),'IO input count mismatch')
        require(integer(run.get('ok_count'),'IO OK count')==len(io_rows) and integer(run.get('error_count'),'IO errors')==0
                and all(r.get('status')=='OK' for r in io_rows),'successful IO rows inconsistent')
        fs=m.get('frames');require(isinstance(fs,list) and 0<len(fs)<=max_frames,'invalid manifest frames')
        require(all(isinstance(f,dict) and f.get('status') in FRAME_STATES for f in fs),'invalid frame status')
        expected=_frame_counts(fs);counts=m.get('counts');require(isinstance(counts,dict),'missing counts')
        require(all(integer(counts.get(k),'count '+k)==v for k,v in expected.items()),'manifest counts mismatch')
        if m['status']=='SUCCESS':require(expected['success']==len(fs),'SUCCESS with incomplete frames')
        if m['status']=='PARTIAL':require(expected['error']==expected['not_run']==0 and expected['success']<len(fs),'PARTIAL inconsistent')
        runs.append({'manifest':str(path),'run_id':text(m.get('run_id'),'mask run ID'),'status':m['status'],
                     'evidence_level':m.get('evidence_level','UNDECLARED'), 'backend_name':backend.get('name'),
                     'quality_status':m.get('quality_status'),'cross_view_status':m.get('cross_view_status'),'object':obj})
        for f in fs:
            idx=integer(f.get('input_index'),'input index');require(idx not in seen_indices,'duplicate requested input index')
            seen_indices.add(idx);require(len(seen_indices)<=max_frames,'frame limit exceeded')
            require(isinstance(f.get('errors'),list),'frame errors must be a list')
            if f['status']=='NOT_RUN':text(f.get('not_run_reason'),'NOT_RUN reason')
            metadata=('frame_id','source_image','source_image_sha256','photo_io_image_sha256','image_domain','width','height')
            if f['status'] in ('INPUT_ERROR','NOT_RUN') and any(f.get(k) is None for k in metadata):
                require(f.get('candidates')==[] and f.get('selected_candidate_id') is None
                        and f.get('selected_mask') is None and f.get('selection') is None,'unbound failure has candidates/selection')
                up=byindex.get(idx)
                if up is None:
                    require(all(f.get(k) is None for k in metadata if k!='image_domain')
                            and f.get('image_domain') in (None,'photo_io_canonical'),'unknown input with asserted image metadata')
                else:
                    checks={'frame_id':up.get('frame_id'),'photo_io_image_sha256':up.get('image_sha256'),
                            'image_domain':'photo_io_canonical','width':up.get('canonical_width'),'height':up.get('canonical_height')}
                    for k,v in checks.items():require(f.get(k) is None or (type(f[k]) is type(v) and f[k]==v),'unbound metadata conflict: '+k)
                    if f.get('source_image') is not None or f.get('source_image_sha256') is not None:
                        image=contained(root,up.get('image_output'))
                        require(f.get('source_image') is None or Path(f['source_image']).resolve()==image,'unbound source path conflict')
                        require(f.get('source_image_sha256') is None or f['source_image_sha256']==digest(bindings.read(image,MAX_FILE)),'unbound source hash conflict')
                rows.append({'input_index':idx,**{k:f.get(k) for k in metadata},'image_output':None,
                             'status':f['status'],'binding_status':'UNAVAILABLE','candidates':[],'mask_bytes':None,
                             'selected_candidate_id':None,'selection':None,'errors':f['errors'],
                             'not_run_reason':f.get('not_run_reason'),'manifest':str(path),'source_run_status':m['status'],
                             'evidence_level':m.get('evidence_level','UNDECLARED')})
                panel_count+=1;require(panel_count<=MAX_PANELS,'panel count limit exceeded')
                continue
            require(idx in byindex,'frame absent from source IO');up=byindex[idx]
            w=integer(f.get('width'),'width',1,MAX_PIXELS);h=integer(f.get('height'),'height',1,MAX_PIXELS)
            require(w*h<=MAX_PIXELS,'image dimensions limit')
            source_pixels+=w*h;require(source_pixels<=max_source_pixels,'source pixel total exceeded')
            require((w,h)==(up.get('canonical_width'),up.get('canonical_height')),'canonical dimensions mismatch')
            require(f.get('frame_id')==text(up.get('frame_id'),'frame ID'),'frame identity mismatch')
            require(f.get('image_domain')=='photo_io_canonical','mask image domain is not canonical')
            require(hash_value(f.get('photo_io_image_sha256'))==hash_value(up.get('image_sha256')),'upstream source image hash mismatch')
            image=contained(root,up.get('image_output'))
            require(Path(text(f.get('source_image'),'source image')).is_absolute() and Path(f['source_image']).resolve()==image,'source image path mismatch')
            imagebytes=bindings.read(image,MAX_FILE)
            require(digest(imagebytes)==hash_value(f.get('source_image_sha256')),'canonical file hash mismatch')
            decode_png(imagebytes,'RGB',w,h)
            candidates=f.get('candidates');require(isinstance(candidates,list) and len(candidates)<=16,'invalid candidate count')
            panel_count+=max(1,len(candidates));require(panel_count<=MAX_PANELS,'panel count limit exceeded')
            found={};candidate_rows=[]
            for ordinal,c in enumerate(candidates):
                require(isinstance(c,dict),'invalid candidate');cid=text(c.get('candidate_id'),'candidate ID');require(cid not in found,'duplicate candidate ID')
                mp=contained(path.parent,c.get('mask_path'));mk=str(mp).casefold();require(mk not in seen_masks,'duplicate mask file alias');seen_masks.add(mk)
                raw=bindings.read(mp,MAX_FILE);require(digest(raw)==hash_value(c.get('mask_sha256')),'mask hash mismatch')
                flat=decode_png(raw,'L',w,h);count=flat.count(255)
                require(integer(c.get('foreground_pixels'),'foreground count')==count,'foreground count mismatch')
                score=c.get('score');require(score is None or (type(score) in (int,float) and math.isfinite(score)),'invalid candidate score')
                require(score is None or isinstance(c.get('score_semantics'),str),'missing score semantics')
                pixels+=w*h;require(pixels<=max_mask_pixels,'mask pixel total exceeded')
                entry={'candidate_id':cid,'mask_path':c['mask_path'],'mask_sha256':c['mask_sha256'],'flat':flat,
                       'score':score,'score_semantics':c.get('score_semantics'),'statistics':mask_statistics(flat,w,h),'ordinal':ordinal}
                found[cid]=entry;candidate_rows.append(entry)
            positive=[c for c in candidate_rows if c['statistics']['foreground_pixels']>0]
            selected=None;sid=f.get('selected_candidate_id')
            if f['status']=='OK':
                require(sid in found,'selected candidate absent');selected=found[sid];sm=f.get('selected_mask');selection=f.get('selection')
                require(isinstance(sm,dict) and sm.get('path')==selected['mask_path'] and sm.get('sha256')==selected['mask_sha256'],'selected mask identity mismatch')
                require(selected['statistics']['foreground_pixels']>0,'empty selected mask')
                require(isinstance(selection,dict) and selection.get('method') in ('unique_candidate','explicit_candidate'),'invalid selection')
                require(selection.get('mask_source') in ('automatic_sam3','automatic_sam3_human_edited')
                        or (temporal_union and selection.get('mask_source') == 'video_temporal_union'),'invalid mask provenance')
                text(selection.get('source'),'selection source')
                if selection['method']=='unique_candidate':require(len(positive)==1,'unique selection with multiple candidates')
            else:
                require(sid is None and f.get('selected_mask') is None and f.get('selection') is None,'non-OK frame has selection')
                if f['status']=='NO_MATCH':require(not positive,'NO_MATCH with usable candidates')
                if f['status']=='NEEDS_SELECTION':require(len(positive)>1,'NEEDS_SELECTION without ambiguity')
                if f['status']=='NOT_RUN':text(f.get('not_run_reason'),'NOT_RUN reason')
            require(isinstance(f.get('errors'),list),'frame errors must be a list')
            rows.append({'input_index':idx,'frame_id':f['frame_id'],'image_output':up['image_output'],'source_image':str(image),
                         'source_image_sha256':f['source_image_sha256'],'photo_io_image_sha256':f['photo_io_image_sha256'],
                         'image_domain':f['image_domain'],'width':w,'height':h,'status':f['status'],'candidates':candidate_rows,
                         'mask_bytes':None if selected is None else selected['flat'],'selected_candidate_id':sid,
                         'errors':f['errors'],'not_run_reason':f.get('not_run_reason'),'manifest':str(path),
                         'source_run_status':m['status'],'binding_status':'BOUND','selection':f.get('selection'),
                         'evidence_level':m.get('evidence_level','UNDECLARED')})
    return sorted(rows,key=lambda f:f['input_index']),{'path':common[0],'run_id':common[1],'object_id':common[4],'object':common_object},runs


def _font(size: int):
    for p in ('C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/msyh.ttc'):
        if Path(p).is_file():return ImageFont.truetype(p,size)
    return ImageFont.load_default(size=size)


def _thumb(image: Image.Image, width: int = 320, height: int = 240) -> Image.Image:
    thumb=image.copy();thumb.thumbnail((width,height),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(width,height),(25,30,37));canvas.paste(thumb,((width-thumb.width)//2,(height-thumb.height)//2));return canvas


def render_frame(frame: dict, out: Path, bindings: Bindings) -> list[dict]:
    if frame['binding_status']=='UNAVAILABLE':
        panel=Image.new('RGB',(960,294),(25,30,37));draw=ImageDraw.Draw(panel)
        draw.text((16,20),f"index {frame['input_index']} | {frame['status']} | IMAGE BINDING UNAVAILABLE",fill='white',font=_font(18))
        draw.text((16,65),'No source image or mask is inferred. See the original error in report.json / report.html.',fill=(220,190,110),font=_font(16))
        target=out/'panels'/f"{frame['input_index']}_none.png";panel.save(target)
        return [{'path':'panels/'+target.name,'sha256':digest(target.read_bytes()),'candidate_id':None,'selected':False}]
    w,h=frame['width'],frame['height']
    rgb=decode_png(bindings.read(Path(frame['source_image'])),'RGB',w,h)
    base=Image.frombytes('RGB',(w,h),rgb);images=[]
    for c in frame['candidates'] or [None]:
        panel=Image.new('RGB',(960,294),(25,30,37));panel.paste(_thumb(base),(0,48))
        selected=c is not None and c['candidate_id']==frame['selected_candidate_id']
        if c is not None:
            a=np.frombuffer(c['flat'],dtype=np.uint8).reshape(h,w)>0
            raw=np.frombuffer(rgb,dtype=np.uint8).reshape(h,w,3).copy()
            raw[a]=(raw[a].astype(np.uint16)*2+np.array([0,210,140],dtype=np.uint16))//3
            inner=a.copy();inner[1:] &= a[:-1];inner[:-1] &= a[1:];inner[:,1:] &= a[:,:-1];inner[:,:-1] &= a[:,1:]
            inner[0]=False;inner[-1]=False;inner[:,0]=False;inner[:,-1]=False
            raw[a & ~inner]=[255,208,0]
            overlay=Image.fromarray(raw);mask=Image.frombytes('L',(w,h),c['flat']).convert('RGB')
            panel.paste(_thumb(mask),(320,48));panel.paste(_thumb(overlay),(640,48))
            label=f"index {frame['input_index']} | {c['candidate_id'][:80]} | {'SELECTED' if selected else 'UNSELECTED'}"
            name=f"{frame['input_index']}_{c['ordinal']}.png"
        else:
            label=f"index {frame['input_index']} | {frame['status']} | NO MASK AVAILABLE";name=f"{frame['input_index']}_none.png"
        draw=ImageDraw.Draw(panel);draw.text((10,8),label,fill='white',font=_font(18))
        draw.text((10,31),'Source | Mask | Overlay (display only) — evidence: '+str(frame['evidence_level'])[:95],fill=(185,196,208),font=_font(13))
        target=out/'panels'/name;panel.save(target)
        images.append({'path':'panels/'+name,'sha256':digest(target.read_bytes()),'candidate_id':None if c is None else c['candidate_id'],'selected':selected})
    return images


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _output_limit(out: Path) -> None:
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > MAX_OUTPUT_BYTES-METADATA_RESERVE:
        raise OSError('output byte limit exceeded; partial outputs are not accepted')


def _save_html(out: Path, report: dict, cards: list[str]) -> None:
    diagnostic=report['cross_view_diagnostic']
    summary={k:v for k,v in report.items() if k in ('status','counts','errors','input_stability','source_runs','object')}
    summary['cross_view_diagnostic']={k:v for k,v in diagnostic.items() if k not in ('pairs','bound_frames','excluded_duplicate_points')}
    payload='<!doctype html><meta charset="utf-8"><title>Object mask review</title>' + \
        '<style>body{font:16px system-ui;background:#141920;color:#eee;max-width:1100px;margin:30px auto}section{padding:16px;margin:20px 0;background:#202833}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{color:#72dcba}</style>' + \
        '<h1>Object mask review — '+html.escape(report['status'])+'</h1><p>Human decision pending. Quality and cross-view identity: NOT_EVALUATED. Panels are display aids, not authoritative masks. No automatic selection or approval.</p>' + \
        '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)
    require(len(payload.encode('utf-8'))<=METADATA_RESERVE//2,'HTML metadata limit exceeded')
    (out/'report.html').write_text(payload,encoding='utf-8')


def create_review(manifests: list[Path], output: Path, *, sfm_run: Path | None = None, model_id: str | None = None) -> tuple[dict,int]:
    out=Path(output).absolute().resolve();require(not out.exists(),'output already exists')
    require(out.parent.is_dir(),'output parent must exist')
    paths=[Path(p).absolute().resolve() for p in manifests]
    for p in paths:require(not out.is_relative_to(p.parent),'output must not be inside manifest input root')
    bindings=Bindings();pre_error=None
    try:
        for p in paths:
            m=strict_json(bindings.read(p,MAX_JSON))
            if isinstance(m,dict) and isinstance(m.get('source_io'),dict) and isinstance(m['source_io'].get('path'),str):
                root=Path(m['source_io']['path']).resolve()
                require(not out.is_relative_to(root),'output must not be inside source IO root')
    except (ValueError,OSError) as exc:
        if 'output must not' in str(exc):raise
        pre_error=exc
    if sfm_run is not None:require(not out.is_relative_to(Path(sfm_run).resolve()),'output must not be inside SfM input root')
    out.mkdir();report={'contract_version':CONTRACT,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'INPUT_ERROR',
                       'quality_status':'NOT_EVALUATED','cross_view_status':'NOT_EVALUATED','human_review_status':'PENDING',
                       'counts':{'requested':0,'selected':0,'candidates':0,'unreviewable':0},'frames':[],'errors':[],
                       'cross_view_diagnostic':{'diagnostic_status':'NOT_REQUESTED'},'input_stability':'NOT_VERIFIED'}
    code=2;phase='input';cards=[]
    try:
        if pre_error is not None:raise pre_error
        require((sfm_run is None)==(model_id is None),'sfm-run and model-id must be supplied together')
        frames,source,runs=load_inputs(paths,bindings);report.update(source_io={k:v for k,v in source.items() if k!='object'},source_runs=runs,object=source['object'])
        report['counts']={'requested':len(frames),'selected':sum(f['mask_bytes'] is not None for f in frames),
                          'candidates':sum(len(f['candidates']) for f in frames),'unreviewable':sum(f['mask_bytes'] is None for f in frames)}
        if sfm_run is not None:
            from .tracks import diagnose
            bound=[f for f in frames if f['binding_status']=='BOUND']
            report['cross_view_diagnostic']=diagnose(Path(sfm_run),str(model_id),bound,source['run_id'],bindings.read)
            report['cross_view_diagnostic']['unbound_requested_frames']=[f['input_index'] for f in frames if f['binding_status']!='BOUND']
        phase='output'
        (out/'panels').mkdir();cards=[];sheet=[]
        for f in frames:
            panels=render_frame(f,out,bindings)
            _output_limit(out)
            public={k:v for k,v in f.items() if k not in ('mask_bytes','candidates')}
            public['candidates']=[{k:v for k,v in c.items() if k not in ('flat','ordinal')} for c in f['candidates']]
            public['panels']=panels;report['frames'].append(public)
            chosen=next((p for p in panels if p['selected']),panels[0])
            with Image.open(out/chosen['path']) as im:sheet.append(_thumb(im,480,147))
            esc=lambda v:html.escape(str(v),quote=True)
            cards.append('<section><h2>Input '+str(f['input_index'])+' — '+esc(f['status'])+'</h2>'+
                         ''.join('<img src="'+esc(p['path'])+'" alt="Mask review panel">' for p in panels)+
                         '<pre>'+esc(json.dumps({'frame_id':f['frame_id'],'binding_status':f['binding_status'],'selection':f['selection'],'errors':f['errors'],'not_run_reason':f['not_run_reason'],'candidates':public['candidates']},ensure_ascii=False,indent=2))+'</pre></section>')
        contact=Image.new('RGB',(960,147*math.ceil(len(sheet)/2)),(25,30,37))
        for i,im in enumerate(sheet):contact.paste(im,((i%2)*480,(i//2)*147))
        contact.save(out/'contact_sheet.png')
        _output_limit(out)
        report['review_questions']=['Correct physical object in every view?','Important outline, openings and grooves retained?',
                                    'Background or other instances included?','Ambiguous candidates require explicit selection?',
                                    'Sparse-track disagreement caused by masks, occlusion, boundaries or bad SfM tracks?']
        report['limitations']=['Descriptive morphology is not segmentation accuracy.','Shared-track labels do not certify instance identity.',
                               'Background-background agreement cannot prove object consistency.','No MVS mask consumption or CAD test is performed.',
                               'Source backend/evidence labels are preserved declarations, not authenticated inference proof.']
        phase='input'
        bindings.verify();report['input_stability']='UNCHANGED'
        selected=report['counts']['selected']
        report['status']='NO_REVIEWABLE_MASKS' if selected==0 else ('REPORT_READY' if selected==len(frames) and all(r['status']=='SUCCESS' for r in runs) else 'PARTIAL_REVIEW')
        code=0 if report['status']=='REPORT_READY' else 1
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        report['status']='INPUT_ERROR';report['errors'].append({'type':type(exc).__name__,'message':str(exc)});code=2
    except OSError as exc:
        report['status']='INPUT_ERROR' if phase=='input' else 'INTERNAL_ERROR';report['errors'].append({'type':type(exc).__name__,'message':str(exc)});code=2 if phase=='input' else 4
    except Exception as exc:
        report['status']='INTERNAL_ERROR';report['errors'].append({'type':type(exc).__name__,'message':str(exc)});code=4
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),inputs=list(bindings.rows.values()))
    report['implementation']=[{'path':str(p),'sha256':digest(p.read_bytes())} for p in (Path(__file__),Path(__file__).with_name('tracks.py')) if p.exists()]
    try:
        _save_html(out,report,cards)
        report['artifacts']=[{'path':str(p.relative_to(out)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())}
                             for p in sorted(out.rglob('*')) if p.is_file()]
        raw=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
        if len(raw)>METADATA_RESERVE//2:raise OSError('JSON metadata limit exceeded')
        _save_json(out/'report.json',report)
    except (OSError,ValueError) as exc:
        report['status']='INTERNAL_ERROR';report['errors'].append({'type':type(exc).__name__,'message':'finalisation: '+str(exc)});code=4
        # Best effort honest failure record; a failed disk write cannot promise an artifact.
        for writer in (lambda:_save_html(out,report,[]),lambda:_save_json(out/'report.json',report)):
            try:writer()
            except (OSError,ValueError):pass
    return report,code
