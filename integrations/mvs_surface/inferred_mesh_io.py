"""Inference-specific mesh export; legacy evidence exporters remain immutable."""
from pathlib import Path
import struct
import numpy as np
from .global_local_pipeline import binding
from integrations.viewer.ply_data import read_ply

def write_mesh(path,vertices,faces,colors,*,provenance_note):
    v=np.asarray(vertices,np.float64);f=np.asarray(faces);c=np.asarray(colors);note=provenance_note
    if v.ndim!=2 or v.shape[1]!=3 or not 0<len(v)<=200000 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or not 0<len(f)<=400000 or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v) or c.dtype!=np.uint8 or c.shape!=v.shape:raise ValueError('bounded finite indexed nonempty mesh required')
    if not isinstance(note,str) or not note or len(note)>400 or not note.isascii() or '\n' in note or '\r' in note:raise ValueError('bounded one-line ASCII provenance note required')
    header=('ply\nformat binary_little_endian 1.0\ncomment '+note+'\n'+f'element vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n')
    with Path(path).open('xb') as s:
        s.write(header.encode('ascii'))
        for pt,col in zip(v,c):s.write(struct.pack('<3d3B',*pt,*col))
        for face in f:s.write(struct.pack('<B3i',3,*face))
    actual=read_ply(path,max_bytes=16<<20,max_vertices=200000,max_faces=400000)
    if any(not np.array_equal(np.asarray(actual[k]),value) for k,value in (('vertices',v),('faces',f),('colors',c))):raise ValueError('mesh readback differs')
    obj=Path(path).with_suffix('.obj')
    with obj.open('x',encoding='ascii',newline='\n') as s:
        s.write('# '+note+'\n')
        for pt in v:s.write('v '+' '.join(format(float(x),'.17g') for x in pt)+'\n')
        for face in f:s.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    return dict(**binding(path),obj=binding(obj),vertices=len(v),faces=len(f),exact_readback=True,geometry_kind='INFERRED_NOT_DIRECT_OBSERVATIONS')
