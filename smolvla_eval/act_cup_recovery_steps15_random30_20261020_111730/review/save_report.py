import json,csv
from pathlib import Path
R=Path(__file__).resolve().parent;ms=json.loads((R/'metrics.json').read_text());groups=[('接近偏差或碰推，全程没有闭爪指令',[3,6,7,8,9,12,14,15,17,18,24,27,29,30]),('偏位闭爪，未形成有效夹持',[4,11,19,22,23,25]),('反复开闭、接触后未稳定抬起',[2,13,26]),('有效抓取较晚，仍在搬运中超时',[5])];lookup={ep:name for name,eps in groups for ep in eps}
assert len(ms)==30 and len(lookup)==24
assert {m['episode'] for m in ms if not m['final_box_success']}==set(lookup)
assert all(not m['ever_verified'] for m in ms if m['episode'] in lookup and m['episode']!=5)
assert all(m['close_events']==0 and m['chunks_predict_close']==0 and m['max_pred']<.6 for m in ms if m['episode'] in groups[0][1])
assert len((R/'frames.log').read_text().splitlines())==48
assert all(line.split()[-3]==line.split()[-1] for line in (R/'frames.log').read_text().splitlines())
with (R/'逐次失败复核.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['次数','种子','最终入盒成功','原分类','复核现象','闭爪次数','开爪次数','首次闭爪秒','预测夹爪最大值','杯根最高米','原始记录'])
 for m in ms:w.writerow([m['episode'],m['seed'],m['final_box_success'],m['category'],lookup.get(m['episode'],'最终入盒并释放成功'),m['close_events'],m['open_events'],m['first_close_s'],m['max_pred'],m['max_root_z'],m['path']])
report='''# 新纠偏杯子 ACT 50k：失败过程复核

30 次、执行 15 步；最终入盒并释放成功 6 次，失败 24 次。已检查全部预测、轨迹和结果，并抽查 16 次失败的三视角事件画面；48 条视频流的帧数与轨迹帧序列一致。第 21 次按用户最终入盒口径成功，不计入本报告失败数。

下面是对失败过程的互斥归类，不是已经证明的训练根因。原始结果文件保持不变。

| 复核现象 | 次数 | 编号 |
|---|---:|---|
'''
for name,eps in groups:report+=f'| {name} | {len(eps)} | '+ '、'.join(map(str,eps))+' |\n'
report+='''
## 1. 接近偏离伴随全程不闭爪：14 次

3、6、7、8、9、12、14、15、17、18、24、27、29、30 全程没有闭爪指令，所有 16 步预测也都低于 0.6 的闭爪阈值。各次最大预测值约 0.005～0.409，不是闭爪仅落在未执行的第 16 步。

相对于示范预闭爪局部位置区间，最接近时仍有约 18～173 mm 的距离。该数值是杯中心到夹爪局部区间的诊断距离，不是世界坐标误差，也不是完整抓取充分条件。

视频中，3 在杯子以外的位置持续活动；6 接近时推倒杯子；18、29、30 接近过程中杯子被碰推，随后滚落或离开桌面。24 曾短暂接触杯子，却没有产生闭爪指令，之后停留在偏离的位置。杯子掉出桌面后轨迹中的巨大负 Z 是持续落体结果，不应解释为模型主动执行了那样的运动。

## 2. 闭爪时仍未对准杯子：6 次

4、11、19、22、23、25 会发出闭爪指令，但未完成有效夹持。

- 4 在约 24.3 秒首次闭爪时，杯中心仍明显在夹取区域之外；腕相机和全局图像可见它在瓶子处操作，接触记录也有大量瓶子接触。它没有夹住杯子。该观察不能单独证明模型在语义上把杯子认成瓶子。
- 11、19 到约 51.7、50.2 秒才首次闭爪，仍没有杯子的双指接触；首次闭爪时局部位置区间误差约 66、69 mm。
- 22、23 在位置/高度未稳定对齐时闭爪，虽有短暂双指接触，却把杯子继续推离，没有形成稳定抬升。
- 25 在 13.55 秒闭爪，接近示范位置区间，但随后杯子被推出夹持区域；14.57 秒开爪时已经失去杯子接触。不能将它单纯解释为开爪导致掉杯。

## 3. 接触到杯子，闭爪—抬升衔接仍不稳定：2、13、26

2 有 6 次闭爪/6 次开爪，杯根最高约 2.8 cm。13 有 7 次闭爪/7 次开爪，杯根最高约 0.8 cm。两次都曾进入示范预闭爪局部位置区间并形成短暂双指接触，但多次在尚未稳定抬起时重新开爪。

可直接核对的例子：

- 2：14.83 秒关闭，15.85 秒重新打开，开爪时仍为双指杯子接触。对应原始夹爪预测由约 0.709 转为 0.024。
- 13：22.78 秒关闭，23.82 秒重新打开，开爪时双指接触；预测由约 1.003 转为 0.120。27.45 秒再关闭，28.48 秒又打开，后一开爪预测约 0.003。

这些不是 0.5 附近的小幅阈值抖动：模型本身确实重新输出了明确开爪意图。已有 1 秒最短闭爪保持约束生效，但保持结束后仍会打开。

26 有 5 次闭爪/5 次开爪，杯子在首次闭爪前已被碰倒，后续反复接触和闭合仍未形成正常稳定抓取，杯根最高约 4.0 cm；杯子侧躺时根高度本身就约 2.5 cm，不能将它当作可靠抬升。

因此，不能把本版所有失败都归因于“没到位所以不闭爪”。2、13 至少还暴露了接触后的闭爪保持与抬升动作衔接问题。是否由状态依赖、示范停顿、采样或其他机制导致，目前没有足够对照证据确定。

## 4. 夹起太晚，未完成搬运：5

5 前几次闭合均未稳定夹持，最终在约 50.05 秒再次闭爪，54.48 秒杯根才超过 5 cm。随后仍在向上抬升和移动，60 秒结束时杯根约 (0.423,-0.235,0.390) m，仍双指接触、保持闭爪，明显不在蓝盒上方。

这是晚抓取加搬运轨迹未完成，不是到达蓝盒后不会松爪，也不是抓起后停在原地。仅延长时间是否有效尚未验证。

## 其他结论

按最终入盒口径，24 次失败里 23 次没有通过有效抓取验证，1 次通过抓取但未完成搬运。本次没有手指撞桌终止，也没有推理/通信/清理报错。

成功轨迹本身也可能有不稳定动作：例如成功的第 10 次同样有 10 次开闭循环，说明最终成功并不意味着全程动作已经稳定。上一版和本版使用不同种子，不能依据各类次数的变化单独认定纠偏能力提升或退步。

逐次编号、预测与原始记录路径保存在同目录逐次失败复核.csv，事件统计见 metrics.json，关键画面见 frames_s15_epXX.jpg。
'''
(R/'具体失败原因.md').write_text(report)
(R/'failure_review_verified.json').write_text(json.dumps({'episodes':30,'successes':6,'failures':24,'failure_categories':{name:len(eps) for name,eps in groups},'all_predictions_and_traces_checked':True,'reviewed_video_episodes':16,'video_frame_mappings_verified':48,'original_result_files_modified':False},ensure_ascii=False,indent=2))
print(json.dumps({'successes':6,'failures':24,'counts':{name:len(eps) for name,eps in groups},'report':str(R/'具体失败原因.md')},ensure_ascii=False))
