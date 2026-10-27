"""Read-only reconstruction diagnostics; writes only ``run/diagnostics``.

No solver is invoked. Keypoint centres and correspondence indices are read from
the actual COLMAP SQLite database; registration is read from frames_map.jsonl.
All visual sampling is explicit and is not a geometry/segmentation acceptance.
"""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import os
import sqlite3
import shutil
from urllib.parse import quote


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(value, base):
    p = Path(value)
    return p if p.is_absolute() else base / p


def _mask(row, base):
    value = row.get("mask_path")
    if not value:
        selected = row.get("selected_mask") or {}
        value = selected.get("path") or selected.get("mask_path")
    if not value:
        candidates = row.get("candidates") or row.get("objects") or row.get("masks") or []
        if "selected_candidate_id" in row and row["selected_candidate_id"] is None:
            return None
        selected_id = row.get("selected_candidate_id")
        if selected_id is not None:
            candidates = [r for r in candidates if r.get("candidate_id") == selected_id]
        if isinstance(candidates, list) and len(candidates) == 1:
            value = candidates[0].get("mask_path")
    return _resolve(value, base) if value else None


def _surface(path):
    result = {}
    for name, file in (("pipeline", "RESULT.json"), ("native", "surface/result.json")):
        p = path / file
        if p.is_file(): result[name] = _json(p)
    return result


def _native(sfm):
    p = sfm / "run.json"
    data = _json(p) if p.exists() else {}
    models = []
    import pycolmap
    for model in data.get("models", []):
        source = _resolve(model["binary_path"], sfm)
        rec = pycolmap.Reconstruction(str(source))
        models.append(dict(model_id=model["model_id"], binary_path=str(source),
                           registered_images=int(rec.num_reg_images()),
                           points3D=int(rec.num_points3D())))
    return dict(status=data.get("status", "NOT_AVAILABLE"), models=models)


def build_report(run: Path, summary: dict) -> dict:
    """Render diagnostics. Required: input_manifest; optional sfm_run,
    surface_run, baseline_run, viewer_links, status/stages/errors.
    Existing report files may be refreshed as stages finish.
    """
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    import numpy as np
    from PIL import Image, ImageDraw
    run = Path(run).resolve()
    out = run / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(summary["input_manifest"]).resolve()
    manifest = _json(manifest_path)
    frames = manifest.get("frames", manifest.get("rows", []))
    byindex = {r["input_index"]: r for r in frames}
    report = dict(schema="reconstruction-diagnostics-v1", quality_status="NOT_EVALUATED",
                  status=summary.get("status", "IN_PROGRESS"), stages=summary.get("stages", {}),
                  reused_stages=summary.get("reused_stages", {}),
                  errors=list(summary.get("errors", [])), warnings=[],
                  input_manifest=str(manifest_path), manifest_sha256=_sha(manifest_path),
                  counts={"manifest_frames": len(frames)}, frame_statuses=[],
                  features=[], correspondences=[], models=[], baseline={},
                  viewer_links=summary.get("viewer_links", {}), visualizations=[])
    sfm = Path(summary["sfm_run"]) if summary.get("sfm_run") else None
    maps = []
    if sfm and (sfm / "frames_map.jsonl").exists():
        maps = [json.loads(s) for s in (sfm / "frames_map.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        report["sfm"] = _native(sfm)
        report["models"] = report["sfm"]["models"]
    image_by_id = {}
    tiles = []
    detail_dir = out / "frames"
    detail_dir.mkdir(exist_ok=True)
    for row in (maps or frames):
        idx = row["input_index"]
        source = byindex.get(idx, {})
        imgpath = source.get("source_image") or source.get("image_path")
        if not imgpath and sfm and row.get("staged_image_name"):
            imgpath = str(sfm / row["staged_image_name"])
        p = _resolve(imgpath, manifest_path.parent) if imgpath else None
        maskpath = _mask(source, manifest_path.parent)
        record = dict(input_index=idx, status=row.get("status", "NOT_EVALUATED"),
                      registrations=row.get("registrations", []), image_path=str(p) if p else None,
                      mask_path=str(maskpath) if maskpath else None)
        report["frame_statuses"].append(record)
        if not p or not p.is_file():
            report["warnings"].append("image unavailable: " + str(p))
            continue
        with Image.open(p) as opened:
            original = opened.convert("RGB")
        mask = None
        if maskpath and maskpath.is_file():
            with Image.open(maskpath) as opened:
                mask = np.asarray(opened.convert("L")) > 0
            if mask.shape != (original.height, original.width):
                report["warnings"].append("mask/image domain mismatch: " + str(idx))
                mask = None
        if row.get("database_image_id") is not None:
            image_by_id[row["database_image_id"]] = (idx, original, mask)
        overlay = original.copy()
        if mask is not None:
            a = np.asarray(original).copy()
            a[mask] = (.60 * a[mask] + .40 * np.array([30, 255, 90])).astype(np.uint8)
            overlay = Image.fromarray(a)
        tile = Image.new("RGB", (320, 306), "#15191e")
        for x, im in ((0, original), (160, overlay)):
            im = im.copy(); im.thumbnail((158, 274))
            tile.paste(im, (x, 26))
        ImageDraw.Draw(tile).text((4, 5), str(idx) + " " + record["status"], fill="white")
        tiles.append(tile)
        # Native-resolution single-view evidence stays inspectable beyond contacts.
        original.save(detail_dir / f"{idx:06}_original.jpg", quality=94)
        overlay.save(detail_dir / f"{idx:06}_overlay.jpg", quality=94)
        record["detail_original"] = f"frames/{idx:06}_original.jpg"
        record["detail_overlay"] = f"frames/{idx:06}_overlay.jpg"
    report["counts"]["frame_statuses"] = dict(Counter(r["status"] for r in report["frame_statuses"]))
    # All selected input views appear in the contact sheet.
    if tiles:
        sheet = Image.new("RGB", (320 * 4, 306 * ((len(tiles) + 3) // 4)), "#15191e")
        for n, tile in enumerate(tiles): sheet.paste(tile, (n % 4 * 320, n // 4 * 306))
        sheet.save(out / "input_overlay_contact.jpg", quality=90)
        report["visualizations"].append("input_overlay_contact.jpg")
    dbpath = Path(summary.get("database_path", sfm / "database.db" if sfm else out / "absent.db"))
    if dbpath.is_file():
        report["database"] = dict(path=str(dbpath), sha256=_sha(dbpath))
        db = sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            keypoints = {}
            feature_tiles = []
            for iid, rows, cols, blob in db.execute("SELECT image_id,rows,cols,data FROM keypoints"):
                if rows > 100000 or cols not in (2, 4, 6): raise ValueError("invalid keypoint shape")
                xy = np.frombuffer(blob or b"", np.float32).reshape(rows, cols)[:, :2]
                keypoints[iid] = xy
                if iid not in image_by_id: continue
                idx, im, mask = image_by_id[iid]
                inside = np.zeros(rows, dtype=bool)
                if mask is not None:
                    ij = np.floor(xy).astype(int)
                    valid = (ij[:, 0] >= 0) & (ij[:, 0] < im.width) & (ij[:, 1] >= 0) & (ij[:, 1] < im.height)
                    inside[valid] = mask[ij[valid, 1], ij[valid, 0]]
                report["features"].append(dict(input_index=idx, database_image_id=iid, centres=rows,
                    foreground_centres=int(inside.sum()) if mask is not None else None,
                    foreground_evaluated=mask is not None))
                if mask is not None and mask.any():
                    yy,xx=np.where(mask);x0,y0=int(xx.min()),int(yy.min());w,h=int(xx.max()-x0+1),int(yy.max()-y0+1)
                    support=set(zip(np.minimum(3,(xx-x0)*4//w).tolist(),np.minimum(3,(yy-y0)*4//h).tolist()))
                    occupied=set();hist=np.zeros((4,4),dtype=int)
                    for x,y in xy[inside]:
                        gx=min(3,max(0,int((x-x0)*4/w)));gy=min(3,max(0,int((y-y0)*4/h)))
                        occupied.add((gx,gy));hist[gy,gx]+=1
                    report["features"][-1]["foreground_bbox_grid4"] = dict(counts=hist.tolist(), occupied_cells=len(occupied & support), mask_support_cells=len(support), definition="4x4 bins over exact foreground bounding box; centres from database")
                show = im.copy(); show.thumbnail((240, 426)); draw = ImageDraw.Draw(show)
                sx, sy = show.width / im.width, show.height / im.height
                for n, (x, y) in enumerate(xy):
                    x, y = x * sx, y * sy
                    draw.ellipse((x-1, y-1, x+1, y+1), fill="#33ff66" if inside[n] else "#ff8844")
                tile = Image.new("RGB", (244, 452), "#15191e"); tile.paste(show, (0, 24))
                ImageDraw.Draw(tile).text((3, 4), f"{idx}: {rows} / fg {inside.sum() if mask is not None else '?'}", fill="white")
                feature_tiles.append(tile)
                full = im.copy(); full_draw = ImageDraw.Draw(full)
                for n,(x,y) in enumerate(xy):
                    full_draw.ellipse((x-2,y-2,x+2,y+2), fill="#33ff66" if inside[n] else "#ff8844")
                full.save(detail_dir / f"{idx:06}_features.jpg", quality=94)
                for record in report["frame_statuses"]:
                    if record["input_index"] == idx: record["detail_features"] = f"frames/{idx:06}_features.jpg"
            if feature_tiles:
                sheet = Image.new("RGB", (244*6, 452*((len(feature_tiles)+5)//6)), "#15191e")
                for n, tile in enumerate(feature_tiles): sheet.paste(tile, (n%6*244, n//6*452))
                sheet.save(out / "feature_centres.jpg", quality=92)
                report["visualizations"].append("feature_centres.jpg")
            for table in ("matches", "two_view_geometries"):
                pairs = list(db.execute(f"SELECT pair_id,rows,cols,data FROM {table} WHERE rows>0 ORDER BY rows DESC,pair_id"))
                report["counts"][table] = dict(nonempty_pairs=len(pairs), correspondence_rows=sum(p[1] for p in pairs))
                rendered = 0
                for pid, rows, cols, blob in pairs:
                    a, b = divmod(pid, 2147483647)
                    if a not in image_by_id or b not in image_by_id: continue
                    report["correspondences"].append(dict(table=table, pair_id=pid, image_ids=[a,b], rows=rows))
                    if rendered >= 8: continue
                    if cols != 2 or rows > 1000000: raise ValueError("invalid match shape")
                    matches = np.frombuffer(blob, np.uint32).reshape(rows, cols)
                    left, right = image_by_id[a][1].copy(), image_by_id[b][1].copy()
                    left.thumbnail((288,512)); right.thumbnail((288,512))
                    canvas = Image.new("RGB", (left.width+right.width,max(left.height,right.height)+28), "#15191e")
                    canvas.paste(left,(0,28));canvas.paste(right,(left.width,28));draw=ImageDraw.Draw(canvas)
                    # Deterministic evenly spaced subset; counts retain all stored rows.
                    sample = np.linspace(0, rows-1, min(120,rows), dtype=int)
                    for m in matches[sample]:
                        if m[0] >= len(keypoints[a]) or m[1] >= len(keypoints[b]): raise ValueError("match index out of range")
                        x,y=keypoints[a][m[0]];u,v=keypoints[b][m[1]]
                        li,ri=image_by_id[a][1],image_by_id[b][1]
                        draw.line((x*left.width/li.width,y*left.height/li.height+28,left.width+u*right.width/ri.width,v*right.height/ri.height+28),fill="#52ff87",width=1)
                    draw.text((3,5),f"{table} {image_by_id[a][0]}-{image_by_id[b][0]} rows={rows} shown={len(sample)}",fill="white")
                    filename=f"{table}_{rendered:02}.jpg";canvas.save(out/filename,quality=92)
                    report["visualizations"].append(filename);rendered+=1
        finally:
            db.close()
    if summary.get("surface_run"):
        surface = Path(summary["surface_run"]) / "RESULT.json"
        report["surface"] = _surface(Path(summary["surface_run"]))
        work = Path(summary["surface_run"]) / "surface" / "work"
        depth_files = sorted(work.glob("depth*.png"), key=lambda p:p.name)
        depth_dir = out / "native_depth"
        depth_dir.mkdir(exist_ok=True)
        report["native_depth_visualizations"] = dict(available_files=len(depth_files), limit=64, files=[],
            interpretation="Native exported PNG visualization only; filename is not mapped to an input frame; physical depth units and confidence calibration are not inferred.")
        for source in depth_files[:64]:
            source_sha = _sha(source)
            target = depth_dir / source.name
            shutil.copyfile(source,target)
            with Image.open(target) as opened: opened.verify()
            if _sha(target) != source_sha: raise ValueError("native depth copy differs: "+str(source))
            report["native_depth_visualizations"]["files"].append(dict(source_path=str(source), filename=source.name,
                report_copy="native_depth/"+source.name, sha256=source_sha,
                kind="native_confidence_png" if source.name.endswith(".conf.png") else "native_depth_png"))
        from integrations.viewer.ply_data import read_ply
        report["geometry_readback"] = []
        for label, filename in (("dense", "__mvs_dense.ply"), ("mesh", "__mvs_surface_raw.ply")):
            p = Path(summary["surface_run"]) / "surface" / "work" / filename
            if not p.is_file(): continue
            data = read_ply(p)
            xyz = np.asarray(data["vertices"], dtype=float)
            report["geometry_readback"].append(dict(kind=label, path=str(p), sha256=data["sha256"], vertices=len(xyz), faces=len(data["faces"])))
            if not len(xyz): continue
            canvas = Image.new("RGB", (1200,430), "#15191e");draw=ImageDraw.Draw(canvas)
            for n,(a,b) in enumerate(((0,1),(0,2),(1,2))):
                xy=xyz[:,[a,b]]; low=xy.min(axis=0);span=np.ptp(xy,axis=0);scale=360/max(float(span.max()),1e-12)
                xy=(xy-low)*scale
                for x,y in xy: draw.point((n*400+20+x,410-y),fill="#81c8ff")
                draw.text((n*400+12,8),label+" "+"XYZ"[a]+"/"+"XYZ"[b]+" native axes",fill="white")
            file=label+"_axis_projections.png";canvas.save(out/file);report["visualizations"].append(file)
    if summary.get("baseline_run"):
        baseline = Path(summary["baseline_run"])
        camera = baseline / "camera_recovery" / "RESULT.json"
        if camera.is_file():
            old = _json(camera).get("original_sfm")
            if old and Path(old).is_file(): report["baseline"]["sfm"] = _native(Path(old).parent)
        old_surface = baseline / "geometry" / "surface_pipeline" / "RESULT.json"
        if old_surface.is_file(): report["baseline"]["surface"] = _surface(old_surface.parent)
    report["counts"]["keypoint_centres"] = sum(r["centres"] for r in report["features"])
    report["counts"]["foreground_keypoint_centres"] = sum(r["foreground_centres"] or 0 for r in report["features"])
    report["counts"]["foreground_feature_views_evaluated"] = sum(r["foreground_evaluated"] for r in report["features"])
    report["limitations"] = ["Visual agreement and feature counts are not accuracy acceptance.",
        "Raw matches and geometrically verified matches are shown separately.",
        "Pair pictures show at most 120 evenly spaced correspondences for the 8 strongest available pairs per type.",
        "Axis projections use all existing vertices and native axes, not standard physical object views; no points or faces are added.",
        "No solver, CAD check, physical scale check, or mask-consumption verification is performed by this report."]
    (out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    def esc(x): return html.escape(str(x))
    def url(value):
        value=str(value)
        if value.startswith(("https://", "http://", "file:///")): return value
        path=Path(value)
        if path.is_absolute():
            try: return quote(os.path.relpath(path,out).replace("\\","/"),safe="/:")
            except ValueError: return path.as_uri()
        return quote(value.replace("\\","/"),safe="/:")
    def details(title,data):
        return '<details><summary>'+esc(title)+'</summary><pre>'+esc(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details>'
    def picture(filename,title):
        if filename not in report['visualizations']: return '<p>本阶段尚无可读图像。</p>'
        return '<figure><figcaption>'+esc(title)+'（点击查看大图）</figcaption><a href="'+url(filename)+'"><img loading="lazy" src="'+url(filename)+'"></a></figure>'
    counts=report['counts'];states=counts.get('frame_statuses',{})
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>手套重建逐阶段诊断</title><style>body{background:#15191e;color:#eee;font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 18px}img{max-width:100%}pre{white-space:pre-wrap;max-height:600px;overflow:auto}a{color:#7cceff}table{border-collapse:collapse;width:100%}td,th{border:1px solid #555;padding:8px}.bad{color:#ffba81}.good{color:#85efa0}section{padding:18px 0;border-top:1px solid #555}nav{position:sticky;top:0;background:#212a33;padding:14px;z-index:2}nav a{margin-right:24px}summary{cursor:pointer;margin:12px 0}figure{margin:20px 0}</style><h1>手套重建：每一步的可检查结果</h1>'
    body += '<p>运行状态：<b>'+esc(report['status'])+'</b>。几何质量仍为 <b>NOT_EVALUATED（未验收）</b>；本页读取现有证据，不执行重建，不据返回码宣称质量合格。</p>'
    if report['reused_stages']:
        body += '<p class="bad"><b>阶段沿用：</b>本轮沿用 union_model01 的 SfM 结果；本轮表面重建不代表重新执行 SfM。沿用阶段：'+esc('、'.join(report['reused_stages']))+'。</p>'+details('展开沿用阶段原始记录',report['reused_stages'])
    body += '<nav>'+''.join('<a href="#s'+str(n)+'">'+label+'</a>' for n,label in enumerate(('输入/掩膜','特征','对应','相机/稀疏','稠密','网格/Blender'),1))+'</nav>'
    body += '<section id="s1"><h2>1 · 输入与融合掩膜</h2><p>输入清单 '+str(counts['manifest_frames'])+' 帧。先检查绿色叠加是否覆盖整个手套、是否误含手或背景；叠图只能展示范围，不能证明分割准确。</p>'
    body += picture('input_overlay_contact.jpg','全部输入：左侧原图、右侧融合掩膜叠图')
    body += '<details><summary>展开逐帧状态与原尺寸检查链接（未注册视角橙色标记）</summary><table><tr><th>输入帧</th><th>注册状态</th><th>逐帧放大检查</th></tr>'
    translations={'REGISTERED':'已注册','UNREGISTERED':'未注册','EXCLUDED_FOREGROUND':'未进入前景重建'}
    for r in report['frame_statuses']:
        status=r['status'];cl='good' if status=='REGISTERED' else 'bad'
        links=' · '.join('<a href="'+url(r[key])+'">'+label+'</a>' for key,label in (('detail_original','原图'),('detail_overlay','掩膜叠图'),('detail_features','特征中心')) if key in r)
        body += '<tr><td>'+esc(r['input_index'])+'</td><td class="'+cl+'">'+esc(translations.get(status,status))+' ('+esc(status)+')</td><td>'+links+'</td></tr>'
    body += '</table></details></section>'
    body += '<section id="s2"><h2>2 · 特征中心分布</h2><p>数据库实际特征中心 '+str(counts['keypoint_centres'])+' 个；已检查前景的 '+str(counts['foreground_feature_views_evaluated'])+' 个视角中，前景内 '+str(counts['foreground_keypoint_centres'])+' 个。绿色为掩膜内，橙色为掩膜外。检查特征是否只集中于红色花纹或边缘、黑色布面是否缺少支撑。</p>'
    body += picture('feature_centres.jpg','真实数据库全部特征中心')+details('展开每帧计数与前景包围框4×4分布',report['features'])+'</section>'
    body += '<section id="s3"><h2>3 · 图像间对应</h2><p>原始匹配：'+esc(counts.get('matches',{}).get('nonempty_pairs',0))+' 对 / '+esc(counts.get('matches',{}).get('correspondence_rows',0))+' 行；几何验证后：'+esc(counts.get('two_view_geometries',{}).get('nonempty_pairs',0))+' 对 / '+esc(counts.get('two_view_geometries',{}).get('correspondence_rows',0))+' 行。每类展示最强8对，每对最多等距显示120条线；总计数保留全部数据库行。连线通过验证仍不等于三维形状准确。</p>'
    for prefix,title in (('matches_','原始匹配'),('two_view_geometries_','几何验证后的对应')):
        body += '<details><summary>展开'+title+'连线图</summary>'
        for filename in report['visualizations']:
            if filename.startswith(prefix): body += picture(filename,title+' '+filename)
        body += '</details>'
    body += details('展开全部对应计数',report['correspondences'])+'</section>'
    body += '<section id="s4"><h2>4 · 相机注册与稀疏点云</h2><p>已注册 '+str(states.get('REGISTERED',0))+'；未注册 '+str(states.get('UNREGISTERED',0))+'；前景排除 '+str(states.get('EXCLUDED_FOREGROUND',0))+'。注册表示解算得到相机姿态；未注册帧未对该稀疏模型提供相机视角。</p><table><tr><th>模型</th><th>注册视角</th><th>真实稀疏点</th></tr>'
    for model in report['models']: body += '<tr><td>'+esc(model['model_id'])+'</td><td>'+esc(model['registered_images'])+'</td><td>'+esc(model['points3D'])+'</td></tr>'
    body += '</table>'+details('展开旧流程对照（原证据，不修改原状态）',report['baseline'])+'</section>'
    geometry={r['kind']:r for r in report.get('geometry_readback',[])}
    dense=geometry.get('dense',{});mesh=geometry.get('mesh',{})
    body += '<section id="s5"><h2>5 · 稠密点云</h2><p>PLY实际读回 '+str(dense.get('vertices','尚无产物'))+' 个点。下图使用全部现有顶点，按模型自身X/Y/Z轴正交投影；不是有物理标定的标准三视图，没有补点或封洞。</p>'+picture('dense_axis_projections.png','稠密点云原坐标轴投影')+'</section>'
    depth=report.get('native_depth_visualizations',{})
    depth_html='<h3>原生深度 / 置信可视化</h3><p>发现 '+str(depth.get('available_files',0))+' 个原生PNG，按原生文件名排序，最多展示64个。仅展示工具已导出的颜色/灰度图；不猜测帧映射、物理深度单位或置信分数含义。报告副本SHA与原文件一致，原文件只读。</p><details><summary>展开原生深度与置信图（文件名原样保留）</summary>'
    for item in depth.get('files',[]):
        link=esc(url(item['report_copy']))
        depth_html+='<figure><figcaption>'+esc(item['filename'])+'</figcaption><a href="'+link+'"><img loading="lazy" style="max-height:520px" src="'+link+'"></a></figure>'
    depth_html+='</details>'
    pos=body.index('</section>',body.index('<section id="s5">'));body=body[:pos]+depth_html+body[pos:]
    body += '<section id="s6"><h2>6 · 网格与 Blender</h2><p>原始网格实际读回 '+str(mesh.get('vertices','尚无产物'))+' 顶点、'+str(mesh.get('faces','尚无产物'))+' 三角面。请检查手指是否分离、轮廓是否完整、是否有漂浮点/尖刺；可打开不代表几何验收通过。</p>'+picture('mesh_axis_projections.png','原始网格全部顶点轴向投影')
    links=dict(report['viewer_links']) if isinstance(report['viewer_links'],dict) else {}
    blender=summary.get('geometry',{}).get('blender',{}).get('path')
    if blender: links['Blender 文件']=blender
    for label,link in links.items():
        if isinstance(link,str): body += '<p><a href="'+esc(url(link))+'">'+esc(label)+'</a></p>'
    body += '</section><section><h2>阶段状态与完整证据</h2>'+details('展开阶段状态、错误及警告',{k:report.get(k) for k in ('stages','errors','warnings')})+details('展开完整机器证据',report)+'<p><a href="REPORT.json">下载完整 REPORT.json</a></p><p>质量、物理尺度、CAD拓扑与原生掩膜是否实际消费仍需各自证据；此报告不代替验收。</p></section></html>'
    # Link only existing saved reports/logs, including original SfM sources when reused.
    sources={1:[manifest_path,run/'SOURCE_BINDINGS.json'],2:[],3:[],4:[],5:[],6:[]}
    if sfm:
        sources[2]=[sfm/'run.json',sfm/'foreground'/'FOREGROUND_SELECTION_LEDGER.json']
        sources[3]=[sfm/'run.json',sfm/'frames_map.jsonl']
        sources[4]=[sfm/'run.json',sfm/'frames_map.jsonl']
        sfm_logs=sorted((sfm/'logs').glob('*')) if (sfm/'logs').is_dir() else []
        for n in (2,3,4): sources[n]+=sfm_logs
    if summary.get('surface_run'):
        surface_root=Path(summary['surface_run'])
        sources[5]=[surface_root/'surface'/'result.json',surface_root/'RESULT.json']
        sources[6]=[surface_root/'surface'/'result.json',surface_root/'blender'/'RESULT.json']
    log_stages={1:(),2:('sfm',),3:('sfm',),4:('sfm','camera_domain_original','view_geometry','native_subset'),5:('mvs_bridge','mvs_masks','surface_plan','surface_reconstruction'),6:('surface_reconstruction','blender_export')}
    for n,names in log_stages.items():
        for name in names:
            folder=run/'logs'/name
            if folder.is_dir(): sources[n]+=sorted(folder.glob('*'))
    report['stage_source_links']={str(n):[str(p.resolve()) for p in values if p.is_file()] for n,values in sources.items()}
    for n,values in report['stage_source_links'].items():
        foot='<details><summary>本步骤源报告 / 阶段日志</summary>'
        for p in values: foot+='<p><a href="'+esc(url(p))+'">'+esc(Path(p).name)+' · '+esc(p)+'</a></p>'
        foot+='</details>'
        pos=body.index('</section>',body.index('<section id="s'+n+'">'));body=body[:pos]+foot+body[pos:]
    analysis_path=run/'analysis'/'page_diagnostic.json'
    if analysis_path.is_file() and analysis_path.stat().st_size < 1024*1024:
        diagnosis=_json(analysis_path)
        text=diagnosis.get('page_diagnostic_zh',{})
        report['supplemental_diagnosis']={'path':str(analysis_path),'sha256':_sha(analysis_path),
                                         'annotation_source':'separate recorded read-only analysis'}
        note='<section><h2>本轮排查：问题在哪一步出现</h2><p><b>'+esc(text.get('headline',''))+'</b></p>'
        for title,key in [('证据支持的事实','facts'),('图像观察与可能原因','visual_inference'),('解释边界','limits')]:
            note+='<h3>'+title+'</h3><ul>'+''.join('<li>'+esc(item)+'</li>' for item in text.get(key,[]))+'</ul>'
        note+='<p><a href="'+esc(url(analysis_path))+'">排查记录及来源范围</a></p></section>'
        pos=body.index('<section id="s1">');body=body[:pos]+note+body[pos:]
    (out / 'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out / "index.html").write_text(body, encoding="utf-8")
    return dict(report_json=str(out / "REPORT.json"), report_html=str(out / "index.html"), counts=report["counts"], warnings=report["warnings"])
