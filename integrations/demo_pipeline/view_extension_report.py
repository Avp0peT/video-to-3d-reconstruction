"""Task-local append-only evidence and inspectable query correspondences."""
import hashlib
import html
import json
import os
from pathlib import Path
import time
from PIL import Image, ImageDraw


def _write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def _binding(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class ExtensionEvidence:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = self.output/'EVENTS.jsonl'
        self.sequence = 0
        self.status = 'NOT_PLANNED'
        self._hash = hashlib.sha256()
        self._bytes = 0
        self._plan_binding = None

    def __call__(self, event):
        record = dict(seq=self.sequence, monotonic=time.monotonic(), payload=event)
        data = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        with self.events.open('ab' if self.sequence else 'xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        self._hash.update(data); self._bytes += len(data); self.sequence += 1
        if event['event'] == 'PLAN':
            _write(self.output/'PLAN.json', event)
            self._plan_binding = _binding(self.output/'PLAN.json')
            self.status = 'RUNNING'
        elif event['event'] == 'FAILED':
            _write(self.output/'FAILURE.json', event)
            self.status = 'FAILED'

    def snapshot(self):
        return dict(status=self.status, event_records=self.sequence,
                    record='view_extension/RESULT.json' if self.status == 'COMPLETE' else None,
                    events='view_extension/EVENTS.jsonl' if self.sequence else None)

    def finish(self, result, frames, budget=lambda: None):
        try:
            return self._finish(result, frames, budget)
        except BaseException as exc:
            if self.status != 'FAILED':
                try:
                    self(dict(event='FAILED', stage='delivery', partial_files_not_publishable=True,
                        error=dict(type=type(exc).__name__, message=str(exc))))
                except BaseException as log_error:
                    exc.add_note('view extension delivery failure record: '+str(log_error))
                self.status='FAILED'
            raise

    def _finish(self, result, frames, budget):
        if self.status != 'RUNNING' or result['status'] != 'VIEW_EXTENSION_LEDGER_COMPLETE':
            raise ValueError('view extension has no completed ledger')
        actual = _binding(self.events)
        if actual['bytes'] != self._bytes or actual['sha256'] != self._hash.hexdigest():
            raise ValueError('view extension events changed after write')
        if _binding(self.output/'PLAN.json') != self._plan_binding:
            raise ValueError('view extension plan changed after freeze')
        frozen=json.loads((self.output/'PLAN.json').read_text(encoding='utf-8'))
        if frozen['plan'] != result['plan'] or frozen.get('profile') != result.get('profile'):
            raise ValueError('view extension result plan differs from frozen plan')
        frame_by_id = {f['input_index']: f for f in frames}
        overlays = []
        for number, row in enumerate(result['query_rows']):
            budget()
            if row['status'] != 'CORRESPONDENCES_MEASURED':
                continue
            frame = frame_by_id[row['query_input_index']]
            with Image.open(frame['path']) as original:
                if original.size != (frame['width'], frame['height']):
                    raise ValueError('query image dimensions changed')
                image = original.convert('RGB')
            draw = ImageDraw.Draw(image)
            for point in row['points']:
                accepted = point['status'] == 'ACCEPTED_CONDITIONAL_CORRESPONDENCE'
                for route in point['routes']:
                    xy = route.get('query_raw_xy')
                    if xy is None:
                        continue
                    x, y = xy[0]-.5, xy[1]-.5
                    color = '#16ed88' if accepted else '#ff674f'
                    draw.ellipse((x-3,y-3,x+3,y+3), outline=color, width=2)
            name = f'query_{number:04d}.png'
            image.thumbnail((640, 640))
            image.save(self.output/name)
            overlays.append(dict(model_key=row['model_key'], input_index=row['query_input_index'],
                                 path=name, binding=_binding(self.output/name)))
        _write(self.output/'RESULT.json', result)
        esc = html.escape
        rows = []
        for q in result['query_rows']:
            if q['status'] == 'ALREADY_REGISTERED':
                continue
            summary = q.get('summary') or {}
            rows.append('<tr><td>'+esc(q['model_key'])+'</td><td>'+str(q['query_input_index'])+'</td><td>'
                +esc(q['status'])+'</td><td>'+str(summary.get('correspondence_count', 0))+'</td><td>'
                +esc(', '.join(summary.get('registration_reasons', q['reasons'])))+'</td></tr>')
        pictures = ''.join('<figure><img loading="lazy" src="'+o['path']+'"><figcaption>模型 '
            +esc(o['model_key'])+' · 帧 '+str(o['input_index'])+'</figcaption></figure>' for o in overlays)
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动相邻视角对应扩展</title>
<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:16px;color:#24334a}aside{background:#fff2cc;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}.images{display:flex;flex-wrap:wrap}figure{margin:10px}img{max-width:420px;width:100%}</style>
<h1>自动相邻视角对应扩展</h1><aside>条件性2D–3D对应；没有新增相机或原生模型点。缺少查询帧的可靠内参，位姿尚未计算。两条路线可能共享帧，不是独立身份真值。</aside>
<p>从两个不同的已有源视图分别逐帧跟踪。每步检查实际正反光流及原始物体掩膜；仅在最终位置相差不超过1像素时保留。未知或歧义掩膜继续留空。</p>
<p>绿圈：两条路线通过；红圈：有终点但最终对应未通过。每条路线的实测终点分别显示；不是物体完整度或标定质量图。</p>'''
        page += '<pre>'+esc(json.dumps(result['counts'], ensure_ascii=False, indent=2))+'</pre><div class="images">'+pictures+'</div>'
        page += '<table><tr><th>独立模型</th><th>帧</th><th>状态</th><th>保留对应</th><th>原因</th></tr>'+''.join(rows)+'</table>'
        page += '<p><a href="RESULT.json">全量测量账本</a> · <a href="PLAN.json">计算前冻结计划</a> · <a href="EVENTS.jsonl">逐步事件</a> · <a href="../index.html">自动流程</a></p></html>'
        (self.output/'index.html').write_text(page, encoding='utf-8')
        self.status = 'COMPLETE'
        files = [self.events, self.output/'PLAN.json', self.output/'RESULT.json', self.output/'index.html']
        return dict(status=result['status'], counts=result['counts'], flow_calls=result['flow_calls'],
            record='view_extension/RESULT.json', report='view_extension/index.html',
            pose_fits=0, new_cameras=0, geometry_accepted=False, identity_verified=False,
            source_bindings=[_binding(p) for p in files]+[o['binding'] for o in overlays])
