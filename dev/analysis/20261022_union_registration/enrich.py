exec((__import__('pathlib').Path(__file__).parent/'analyze.py').read_text(encoding='utf-8-sig').split("canvas=Image.new")[0])
# The labels below are explicit visual annotations of the saved contact sheet, not model semantics.
logo=set(range(13,26))|set(range(43,55))|{62,63}
opposite={0,1,2,3,4,5,6,7,8,9,10,28,29,30,31,32,33,34,55,56,57}
registered={r['input_index'] for r in rows if r['registered']}
parts={}; unseen={r['input_index'] for r in rows}
while unseen:
 start=next(iter(unseen));seen={start};todo=[start]
 while todo:
  n=todo.pop()
  for p in pairs:
   if n in (p['a'],p['b']):
    q=p['b'] if p['a']==n else p['a']
    if q not in seen:seen.add(q);todo.append(q)
 unseen-=seen;parts[str(start)]=sorted(seen)
summary['verified_graph_components']=sorted(parts.values(),key=lambda a:(-len(a),a))
summary['match_concentration']={'registered_registered_pairs':sum(p['a'] in registered and p['b'] in registered for p in pairs),'registered_registered_correspondences':sum(p['matches'] for p in pairs if p['a'] in registered and p['b'] in registered),'other_pairs':sum(not(p['a'] in registered and p['b'] in registered) for p in pairs),'other_correspondences':sum(p['matches'] for p in pairs if not(p['a'] in registered and p['b'] in registered))}
summary['visual_annotations']={'method':'Human-readable visual inspection by assistant of frames_contact.jpg; side labels are visual inference, not independently calibrated physical normals.','sport_label_side_indices':sorted(logo),'clear_opposite_side_indices':sorted(opposite),'registered_clear_opposite_side':sorted(opposite&registered),'registered_sport_label_side':sorted(logo&registered),'no_claim':'All registered sampled images visibly expose SPORT-label side; this does not prove zero opposite surface is visible at oblique edges or assign physical camera hemisphere.'}
summary['group_statistics']={}
for name,indices in [('registered',registered),('unregistered',set(range(64))-registered),('clear_opposite_visual',opposite),('sport_label_visual',logo)]:
 rr=[r for r in rows if r['input_index'] in indices]
 summary['group_statistics'][name]={'frames':len(rr),'registered':sum(r['registered'] for r in rr),'feature_total':sum(r['features'] for r in rr),'feature_median':statistics.median(r['features'] for r in rr),'no_verified_neighbor':sum(r['verified_neighbors']==0 for r in rr)}
track_lengths=collections.Counter(len(p['track']) for p in doc['points3D'].values());summary['native_track_lengths']=dict(sorted(track_lengths.items()))
summary['mapper_log_facts']=['Database loaded 64 frames; only 38 connected images and 116 match pairs.','Initial image IDs 19/18 (input indices18/17) succeeded.','Remaining models: repeated no-good-initial-pair; ID15/14 (indices14/13) and ID17/25 (indices16/24) discarded as bad initial pair.','The native mapper itself relaxes initialization constraints within this single invocation; this is not an additional orchestration/SfM retry.']
summary['page_diagnostic_zh']={'headline':'64张图都进入特征提取，但只有带SPORT字样侧的15张进入同一个三维分量。','facts':['64图共14849特征；5140原始对应经几何验证剩4233对应，分布于116图对。对应数可在多个图对重复使用同一特征，不能当作4233个三维点。','38图具有至少一组验证配对，26图无验证配对；38图中仍有23图未注册，说明有两图匹配也不足以完成稳定初始化/多视图注册。','最终注册零基索引：17–23、45–46、51–54、62–63；其余49图未注册。','最终300条原生三维轨迹；现有几何审计只覆盖前256条，74条检测到观测不一致，44条未审计；不能据此宣布形状正确。'],'visual_inference':['接触表显示两侧都已拍摄；清楚展示另一侧的0–10、28–34、55–57均未注册。所有15张注册图都可见SPORT字样所在侧。因此当前模型的侧面覆盖不完整。','无字样侧大块黑色区域的可重复纹理较少，图像间局部形变、遮挡和视角变化也可见；这些是匹配不稳的合理视觉解释，尚未通过受控实验分离因果。'],'limits':['不能把未注册统一归因于漏拍、SAM失败或某一个单独参数；掩膜非空且进入特征提取不保证可辨识对应。','SPORT侧进入同一component不是两侧闭环或真实几何验收。','已核验的来源仅为列出的源文件与既有采用掩膜验证子集，未声称逐项复核所有生产者START_BINDINGS冻结来源。']}
(out/'registration.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=[summary['page_diagnostic_zh']['headline'],'']
for k in ('facts','visual_inference','limits'):
 lines.append({'facts':'硬事实','visual_inference':'视觉推断','limits':'解释边界'}[k]);lines.extend('- '+s for s in summary['page_diagnostic_zh'][k]);lines.append('')
lines.append('匹配集中统计：'+json.dumps(summary['match_concentration'],ensure_ascii=False));lines.append('分组统计：'+json.dumps(summary['group_statistics'],ensure_ascii=False));lines.append('原生映射日志：'+'；'.join(summary['mapper_log_facts']));lines.append('读取前后所列文件SHA256一致：'+str(summary['source_unchanged']))
(out/'原因说明.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['match_concentration','group_statistics','verified_graph_components','native_track_lengths']},ensure_ascii=False,indent=2))
