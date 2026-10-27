"""Export partial search proposals from existing flow evidence, never full masks."""
from pathlib import Path
import os,sys,argparse,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from integrations.demo_pipeline.balanced_point_run import read,save,binding,check_bindings


def prepare(source,out):
    if out.exists(): raise FileExistsError(out)
    plan=read(source/'PLAN.json')
    if check_bindings(plan['source_bindings']+plan['implementation']):raise ValueError('full experiment drift')
    data=read(source/'GAP_RESULT.json');inputs=read(source/'INPUTS.json')
    refs=[binding(source/n) for n in ['GAP_RESULT.json','INPUTS.json','CONTROL.json','FLOW_DIAGNOSTICS.json','RESULT.json']]
    for f,r in zip(inputs['frames'],data['frames']):
        if f['decode_index']!=r['decode_index']:raise ValueError('frame identity mismatch')
        actual=binding(f['path'])
        if actual['sha256']!=f['sha256']:raise ValueError('raw hash drift')
        refs.append(actual)
        if f['mask_path']:
            actual=binding(f['mask_path'])
            if actual['sha256']!=f['mask_sha256']:raise ValueError('SAM hash drift')
            refs.append(actual)
        for key in ['forward_mask','backward_mask','forward_valid','backward_valid']:
            if binding(r[key]['path'])!=r[key]:raise ValueError('flow artifact drift')
            refs.append(r[key])
    out.mkdir(parents=True)
    save(out/'PLAN.json',dict(full_run=str(source),source_bindings=refs,implementation=[binding(ROOT/n) for n in [
         'scripts/export_temporal_search_support.py','integrations/demo_pipeline/temporal_search_support.py',
         'tests/test_temporal_search_support.py']],semantics='SEARCH_PROPOSAL_ONLY',full_mask_accepted=False,
         point_identity_recovered=False,allowed_downstream=['search_proposal_generation'],
         forbidden_downstream=['SAM_manifest','MVS_mask','3D_observation_admission']))


def work(out):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from integrations.demo_pipeline.temporal_search_support import derive
    p=read(out/'PLAN.json');source=Path(p['full_run']);inputs=read(source/'INPUTS.json');data=read(source/'GAP_RESULT.json');control=read(source/'CONTROL.json')
    if check_bindings(p['source_bindings']+p['implementation']):raise ValueError('input/code drift')
    assets=out/'assets';assets.mkdir();rows=[];panels=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    def array(ref):
        with Image.open(ref['path']) as im:return np.array(im)
    def png(name,a):
        path=assets/name;Image.fromarray(a.astype(np.uint8)*255).save(path)
        if not np.array_equal(array({'path':path})>0,a):raise ValueError('PNG readback failed')
        return binding(path)
    for f,r in zip(inputs['frames'],data['frames']):
        idx=f['decode_index'];isgap=f['mask_path'] is None
        with Image.open(f['path']) as im:raw=im.convert('RGB')
        if isgap:
            metrics=dict(r['metrics'],full_shape_status=r['status'])
            result=derive(*[array(r[k]) for k in ['forward_mask','backward_mask','forward_valid','backward_valid']],
                          leaveout_passed=control['passed'],metrics=metrics)
            search=result.pop('search');unknown=result.pop('uncertain')
            result.update(decode_index=idx,search_pixels=int(search.sum()),uncertain_pixels=int(unknown.sum()),
                          search_union_fraction=float(search.sum()/max(1,(search|unknown).sum())),
                          search_path=png(f'{idx}_search.png',search),uncertain_path=png(f'{idx}_unknown.png',unknown),
                          original_status=f['status'])
        else:
            search=array({'path':f['mask_path']})>0;unknown=np.zeros(search.shape,bool)
            result=dict(decode_index=idx,original_status=f['status'],usage='ACTUAL_SAM_ANCHOR',allowed=True,
                        full_mask_accepted=False,search_pixels=None,uncertain_pixels=None)
        rows.append(result)
        panel=Image.new('RGB',(640,640),'#121826');d=ImageDraw.Draw(panel)
        for col in range(2):
            im=raw.convert('RGBA')
            if col==1:
                for region,color in [(search,(255,215,0) if isgap else (0,255,80)),(unknown,(255,25,40))]:
                    tint=Image.new('RGBA',raw.size,(*color,0));tint.putalpha(Image.fromarray(region.astype(np.uint8)*105));im=Image.alpha_composite(im,tint)
            im=im.convert('RGB');im.thumbnail((310,550));panel.paste(im,(col*320+(320-im.width)//2,65))
        d.text((8,6),f'raw {idx} 原图 / {f["status"]}',font=font,fill='white')
        d.text((328,6),'黄色仅供搜点；红色未知' if isgap else '绿色：原始SAM锚',font=font,fill='white')
        d.text((8,32),'完整轮廓：未恢复；旧点身份：未恢复' if isgap else '前后锚；未改原始掩膜',font=font,fill='#ffcc77')
        if isgap:d.text((8,612),f'局部区域占预测并集 {result["search_union_fraction"]:.1%}（不是实际物体覆盖率）',font=font,fill='white')
        name=f'{idx}_panel.png';panel.save(assets/name);result['panel']='assets/'+name;panels.append(panel)
    save(out/'SEARCH_SUPPORT.json',dict(status='PARTIAL_SEARCH_PROPOSALS_ONLY',frames=rows,
         original_no_match_unchanged=True,full_masks_recovered=0,old_point_identity_recovered=False,
         sam_calls=0,flow_calls=0,source_flow_calls=20,source_control_passed=control['passed']))
    panels[0].save(out/'motion.gif',save_all=True,append_images=panels[1:],duration=350,loop=0,disposal=2)
    sheet=Image.new('RGB',(1280,640*3),'#121826')
    for j,im in enumerate(panels):sheet.paste(im,((j%2)*640,(j//2)*640))
    sheet.save(out/'contact.png')
    payload=json.dumps([dict(decode_index=r['decode_index'],panel=r['panel']) for r in rows]).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>短时缺失的局部物体约束</title><style>body{font:17px system-ui;max-width:1000px;margin:30px auto;background:#111827;color:white;line-height:1.7}img{max-width:100%}a{color:#8ecbff}input{width:60%}aside{padding:18px;background:#323044}</style>
<h1>四帧分割缺失：保留局部运动支持</h1><aside>黄色仅是前后双向运动一致的局部搜索范围；红色是仍不确定的区域。完整轮廓门控未通过，原576–579的NO_MATCH保持原样。此范围不能作为SAM掩膜、MVS掩膜或物理点身份凭据。没有将上一轮失联点恢复为有效点。</aside>
<p>有掩膜的570–575段做隐藏中间掩膜对照，IoU约87%–88%；缺口段完整端点支持不足，只保留双向一致且循环检查通过的局部像素。局部范围占预测并集约数以机器结果为准，不是实际物体覆盖率。</p>
<p><a href="SEARCH_SUPPORT.json">逐帧数据</a> · <a href="motion.gif">动画</a> · <a href="contact.png">六帧总览</a></p><button id="play">播放/暂停</button> <input id="slide" type="range" min="0" max="5" value="1"><span id="label"></span><p><img id="image"></p><script>const rows=PAYLOAD,s=document.querySelector('#slide');function show(){let r=rows[+s.value];document.querySelector('#image').src=r.panel;document.querySelector('#label').textContent='原视频帧 '+r.decode_index;}s.oninput=show;let timer;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{s.value=(+s.value+1)%rows.length;show()},350)};show();</script>'''.replace('PAYLOAD',payload)
    (out/'index.html').write_text(page,encoding='utf-8')
    if check_bindings(p['source_bindings']+p['implementation']):raise ValueError('post-read drift')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','tests','work']);p.add_argument('--full-run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.action=='run':
        prepare(a.full_run.resolve(strict=True),out)
        from integrations.demo_pipeline.masked_trace_run import invoke
        for action in ['tests','work']:
            e=invoke(out,action,[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),action,'--output',str(out)],120,4<<30)
            if e['root_exit_code']!=0 or not e['reclaim_confirmed']:raise RuntimeError(action+' failed')
    elif a.action=='tests':
        import pytest
        raise SystemExit(pytest.main([str(ROOT/'tests/test_temporal_search_support.py'),'-q','-p','no:cacheprovider',
                                     '--junitxml='+str(out/'tests.xml'),'--basetemp='+str(out/'pytest_tmp')]))
    else:work(out)
    return 0


if __name__=='__main__':raise SystemExit(main())
