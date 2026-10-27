"""CPU-only report for hash-bound temporal mask proposals; never draws points."""
from pathlib import Path
import hashlib
import json


def render(output):
    """Render INPUTS.json and GAP_RESULT.json frames (path/sha256 mask objects)."""
    from PIL import Image, ImageDraw, ImageFont
    out = Path(output).resolve(strict=True)
    products = ('index.html', 'motion.gif', 'contact.png', 'report_assets', 'REPORT_BINDINGS.json')
    if any((out / name).exists() for name in products):
        raise ValueError('fresh temporal mask report required')
    bindings = {}

    def bind(path, digest):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('SHA256 binding required')
        p = Path(path)
        p = (p if p.is_absolute() else out / p).resolve(strict=True)
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError('hash mismatch: ' + str(p))
        bindings[str(p)] = actual
        return p

    def read(name):
        p = out / name
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(bind(p, digest).read_text(encoding='utf-8'))

    inputs, result = read('INPUTS.json'), read('GAP_RESULT.json')
    frames, rows = inputs['frames'], result['frames']
    indices = [f['decode_index'] for f in frames]
    if not frames or len(set(indices)) != len(indices):
        raise ValueError('empty or duplicate frames')
    if [r['decode_index'] for r in rows] != indices:
        raise ValueError('input/result frame identity mismatch')
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
        small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    except OSError:
        font = small = ImageFont.load_default()

    def text(draw, pos, value, fill='white', use_font=None):
        draw.text(pos, value, font=use_font or font, fill=fill)

    def mask(spec, size):
        if spec is None:
            return None
        with Image.open(bind(spec['path'], spec['sha256'])) as im:
            if im.size != size or im.mode not in ('1', 'L'):
                raise ValueError('mask size or mode mismatch')
            im = im.convert('L')
            if set(im.getdata()) - {0, 1, 255}:
                raise ValueError('non-binary mask')
            return im.point(lambda v: 255 if v else 0)

    def overlay(raw, layers):
        im = raw.convert('RGBA')
        for binary, color in layers:
            if binary is not None:
                tint = Image.new('RGBA', im.size, color)
                tint.putalpha(binary.point(lambda v: 105 if v else 0))
                im = Image.alpha_composite(im, tint)
        return im.convert('RGB')

    # Validate every source before publishing any report product.
    prepared = []
    for source, row in zip(frames, rows):
        with Image.open(bind(source['path'], source['sha256'])) as im:
            raw = im.convert('RGB')
        original = mask({'path': source['mask_path'], 'sha256': source['mask_sha256']}
                        if source.get('mask_path') is not None else None, raw.size)
        forward = mask(row.get('forward_mask'), raw.size)
        backward = mask(row.get('backward_mask'), raw.size)
        # Optional validity domains are bound too; predictions outside them are hidden.
        forward_valid = mask(row.get('forward_valid'), raw.size)
        backward_valid = mask(row.get('backward_valid'), raw.size)
        from PIL import ImageChops
        if forward is not None and forward_valid is not None:
            forward = ImageChops.multiply(forward, forward_valid)
        if backward is not None and backward_valid is not None:
            backward = ImageChops.multiply(backward, backward_valid)
        support = mask(row.get('support_mask'), raw.size)
        uncertain = mask(row.get('uncertain_mask'), raw.size)
        allowed = row.get('support_allowed') is True
        if support is not None and not allowed:
            raise ValueError('support mask present without permission')
        prepared.append((source, row, raw, original, forward, backward, support, uncertain, allowed))
    assets = out / 'report_assets'
    assets.mkdir()
    panels, descriptions = [], []
    for source, row, raw, original, forward, backward, support, uncertain, allowed in prepared:
        titles = ['原 SAM：绿色', '双向预测：青色 / 紫色', '临时支持：黄色；不确定：红色']
        images = [overlay(raw, [(original, (30, 255, 80))]),
                  overlay(raw, [(forward, (0, 230, 255)), (backward, (210, 80, 255))]),
                  overlay(raw, [(support, (255, 220, 0)), (uncertain, (255, 30, 40))])]
        panel = Image.new('RGB', (960, 660), '#121826')
        d = ImageDraw.Draw(panel)
        for col, image in enumerate(images):
            image.thumbnail((312, 554))
            x = col * 320
            text(d, (x + 6, 8), titles[col], use_font=small)
            panel.paste(image, (x + (320 - image.width) // 2, 72))
            if col == 0 and original is None:
                text(d, (x + 8, 36), 'NO_MATCH：无原 SAM mask', '#ff7c7c', small)
            if col == 2 and not allowed:
                text(d, (x + 8, 36), '拒绝临时支持', '#ff7c7c', small)
        caption = f"raw {source['decode_index']} | {source.get('time_seconds', '?')} s | {row.get('status', 'UNKNOWN')}"
        text(d, (8, 634), caption, use_font=small)
        name = f"frame_{source['decode_index']:06d}.png"
        panel.save(assets / name)
        panels.append(panel)
        descriptions.append({'image': 'report_assets/' + name, 'caption': caption,
                             'reason': row.get('rejection_reason') or ''})
    panels[0].save(out / 'motion.gif', save_all=True, append_images=panels[1:], duration=300, loop=0)
    contact = Image.new('RGB', (960, len(panels) * 330), '#121826')
    for n, panel in enumerate(panels):
        contact.paste(panel.resize((480, 330)), ((n % 2) * 480, (n // 2) * 330))
    # Two columns retain every frame, including all six boundary/gap views.
    contact = contact.crop((0, 0, 960, ((len(panels) + 1) // 2) * 330))
    contact.save(out / 'contact.png')
    payload = json.dumps(descriptions, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><meta charset="utf-8"><title>时间掩膜缺口审阅</title>
<style>body{background:#121826;color:#eee;font:16px sans-serif;max-width:1100px;margin:24px auto}img{width:100%}input{width:75%}button{padding:8px}p{line-height:1.6}</style>
<h1>时间掩膜缺口审阅</h1><p>旧 NO_MATCH 保留。TEMPORAL 是由前后帧推导的临时区域，不是 SAM 观测，也不是点身份。本页不画点。黄色仅表示获准临时支持；红色是不确定区域。未获准时明确拒绝补充。</p>
<button id="play">播放</button> <input id="slider" type="range" min="0" value="0"><p id="caption"></p><p id="reason"></p><img id="frame"><p><a href="motion.gif">动画</a> · <a href="contact.png">全部帧接触表</a> · <a href="REPORT_BINDINGS.json">文件哈希绑定</a></p>
<script>const rows=PAYLOAD,s=document.querySelector('#slider');s.max=rows.length-1;function show(){const r=rows[+s.value];document.querySelector('#frame').src=r.image;document.querySelector('#caption').textContent=r.caption;document.querySelector('#reason').textContent=r.reason;}s.oninput=show;let timer;document.querySelector('#play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='播放';}else{this.textContent='暂停';timer=setInterval(()=>{s.value=(+s.value+1)%rows.length;show()},300)}};show();</script>'''.replace('PAYLOAD', payload)
    (out / 'index.html').write_text(page, encoding='utf-8')
    generated = {str(out / name): hashlib.sha256((out / name).read_bytes()).hexdigest()
                 for name in ('index.html', 'motion.gif', 'contact.png')}
    for path in assets.iterdir():
        generated[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'REPORT_BINDINGS.json').write_text(json.dumps({'schema': 'temporal-mask-report-bindings-v1',
        'source_run': inputs.get('source_run'), 'sources': bindings, 'products': generated},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return {'index': str(out / 'index.html'), 'frames': len(frames)}
