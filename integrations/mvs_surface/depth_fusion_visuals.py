"""Image-domain projected footprints and coverage panels, never visibility proof."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .local_depth_search import _view
from .global_local_pipeline import binding

MAX_PIXELS=16000000


def _checked(view):
    v=_view(view);h,w=v['mask'].shape
    if h*w>MAX_PIXELS or max(h,w)>8192:raise ValueError('bounded image domain required')
    valid=v.get('validity')
    if not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=(h,w):raise ValueError('explicit validity required')
    if type(v.get('depth_available')) is not bool:raise ValueError('explicit depth_available required')
    return v


def _clip(polygon,normal,offset=0.):
    if not len(polygon):return polygon
    values=polygon@normal-offset
    result=[]
    a=polygon[-1];va=values[-1]
    for b,vb in zip(polygon,values):
        ina=va>=0;inb=vb>=0
        if ina!=inb:
            denominator=va-vb
            result.append(a+(b-a)*(va/denominator))
        if inb:result.append(b)
        a,va=b,vb
    return np.asarray(result,np.float64).reshape(-1,3)


def project_footprint(vertices,faces,view,check=lambda:None):
    """Rasterize clipped projected triangle union; no visibility or closure claim.

    Clips the actual triangle polygon in camera space against positive axial z
    and image frustum planes. It neither fills a projected bounding box nor caps
    outside the image. Subpixel rasterization is display sampling, not coverage truth.
    """
    check();v=_checked(view);h,w=v['mask'].shape
    if not isinstance(vertices,np.ndarray) or vertices.ndim!=2 or vertices.shape[1]!=3 or vertices.dtype.kind!='f' or len(vertices)>200000 or not np.isfinite(vertices).all():raise ValueError('finite floating Nx3 vertices at most 200000 required')
    if not isinstance(faces,np.ndarray) or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or len(faces)>400000 or (faces.size and (faces.min()<0 or faces.max()>=len(vertices))):raise ValueError('valid triangle indices at most 400000 required')
    camera=(vertices-v['C'])@v['R'].T
    if not np.isfinite(camera).all() or (np.abs(camera)>1e12).any():raise ValueError('unsafe camera coordinate magnitude')
    k=v['K'];planes=[(np.array([0.,0.,1.]),1e-9),(k[0],0.),((w-1)*k[2]-k[0],0.),(k[1],0.),((h-1)*k[2]-k[1],0.)]
    mask=np.zeros((h,w),np.uint8)
    for start in range(0,len(faces),2048):
        check()
        chunk=camera[faces[start:start+2048].astype(np.int64)]
        halfspaces=np.stack([chunk@normal-offset for normal,offset in planes],axis=-1)
        wholly_inside=(halfspaces>=0).all(axis=(1,2))
        wholly_outside=(halfspaces<0).all(axis=1).any(axis=1)
        for index in np.flatnonzero(~wholly_outside):
            polygon=chunk[index]
            if not wholly_inside[index]:
                for normal,offset in planes:
                    polygon=_clip(polygon,normal,offset)
                    if len(polygon)<3:break
            if len(polygon)<3:continue
            projected=polygon@k.T
            xy=projected[:,:2]/projected[:,2,None]
            if not np.isfinite(xy).all() or (np.abs(xy)>1e6).any():raise ValueError('unsafe projected coordinates after clipping')
            if abs(cv2.contourArea(xy.astype(np.float32)))<1e-8:continue
            # Fixed 1/256-pixel coordinates avoid premature vertex rounding.
            raster=np.rint(xy*256).astype(np.int32)
            cv2.fillConvexPoly(mask,raster,1,lineType=cv2.LINE_8,shift=8)
    return mask.astype(bool)


def _font(size):
    for path in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/arial.ttf'):
        if Path(path).is_file():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def write_coverage_panel(out,view,coverage_map,old_footprint,new_footprint):
    """Save a four-panel image and count valid-FG projection overlap only.

    Mesh projections can occlude each other, overlap background, and include
    back-facing or hidden surfaces. Counts are projected pixels, not new measured
    geometry, geometric completeness, or independent correctness.
    """
    v=_checked(view);shape=v['mask'].shape
    if not isinstance(coverage_map,np.ndarray) or coverage_map.shape!=shape or coverage_map.dtype!=np.uint8 or (coverage_map>8).any():raise ValueError('uint8 coverage map with codes0..8 required')
    for mask in (old_footprint,new_footprint):
        if not isinstance(mask,np.ndarray) or mask.shape!=shape or mask.dtype!=bool:raise ValueError('boolean footprint in native domain required')
    target=Path(out)
    if target.exists():raise ValueError('fresh coverage panel path required')
    if not target.parent.is_dir():raise ValueError('existing output directory required')
    valid=v['validity'];fg=v['mask']&valid;bg=~v['mask']&valid
    native=fg&(v['depth']>0)&v['depth_available']
    counts=dict(view_id=v['view_id'],total_pixels=int(valid.size),valid_foreground_pixels=int(fg.sum()),foreground_native_depth=int(native.sum()),foreground_without_depth=int((fg&~native).sum()),old_foreground_covered=int((old_footprint&fg).sum()),new_foreground_covered=int((new_footprint&fg).sum()),old_background_projection=int((old_footprint&bg).sum()),new_background_projection=int((new_footprint&bg).sum()),old_invalid_projection=int((old_footprint&~valid).sum()),new_invalid_projection=int((new_footprint&~valid).sum()),old_total_footprint=int(old_footprint.sum()),new_total_footprint=int(new_footprint.sum()))
    colors=np.array([[45,45,50],[95,95,100],[240,70,60],[45,140,210],[230,190,55],[50,215,100],[220,75,155],[145,85,220],[210,210,210]],np.uint8)
    availability=v['image'].copy();availability[fg]=(.35*availability[fg]+.65*colors[coverage_map[fg]]).astype(np.uint8);availability[~valid]=colors[1]
    def overlay(footprint,color):
        image=v['image'].copy()
        image[footprint]=(.35*image[footprint]+.65*np.array(color)).astype(np.uint8)
        edge=cv2.morphologyEx(v['mask'].astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))!=0
        image[edge]=(100,245,120)
        return image
    panels=[('原照片 / original',v['image']),('原生深度分类 / native depth',availability),('旧面投影 / old projection',overlay(old_footprint,(255,165,40))),('新面投影 / new projection',overlay(new_footprint,(65,180,245)))]
    h,w=shape;header=48;footer=205;panel_width=max(w,360)
    canvas=Image.new('RGB',(panel_width*4,h+header+footer),(20,31,45));draw=ImageDraw.Draw(canvas);font=_font(17)
    for index,(title,image) in enumerate(panels):
        canvas.paste(Image.fromarray(image),(index*panel_width,header));draw.text((index*panel_width+8,12),title,fill='white',font=font)
    legends=[('2 无原生深度FG / FG no depth',(240,70,60)),('3 未采样原深度 / unsampled native',(45,140,210)),('4 支持不足 / insufficient support',(230,190,55)),('5 采样支持 / sampled supported',(50,215,100)),('6 证据冲突 / evidence conflict',(220,75,155)),('7 被遮挡 / occluded',(145,85,220)),('8 未知 / unknown',(210,210,210))]
    # Keep native photos unscaled; minimum panel widths preserve readable legends.
    y=h+header+8
    for index,(label,color) in enumerate(legends):
        row=index//2;col=index%2;x=8+col*(2*panel_width)
        draw.rectangle((x,y+row*25,x+12,y+row*25+12),fill=color)
        draw.text((x+18,y+row*25-3),label,fill='white',font=font)
    note='投影并集仅为二维占用，不判断遮挡、完整度或几何正确 / Projection only; not visibility or completeness'
    draw.text((8,y+103),note,fill=(245,220,125),font=font)
    draw.text((8,y+132),'FG native='+str(counts['foreground_native_depth'])+'; FG without depth='+str(counts['foreground_without_depth'])+'; old/new FG='+str(counts['old_foreground_covered'])+'/'+str(counts['new_foreground_covered']),fill='white',font=font)
    draw.text((8,y+157),'old/new known BG projection='+str(counts['old_background_projection'])+'/'+str(counts['new_background_projection'])+'; BG is not an object depth gap',fill='white',font=font)
    canvas.save(target,format='PNG')
    return dict(**binding(target),counts=counts,geometry_accepted=False,semantics='Four image panels in native pixel domains; footprints are triangle projection union, not visibility. Known-background projection is diagnostic, not independent truth. No holes or new geometry are inferred.')
