from pathlib import Path
import json,numpy as np,os
w=Path(__file__).resolve().parent;out={}
for stage in ['before','after']:
 p=next((w/stage).rglob('result.json'));r=json.loads(p.read_text());rows=[json.loads(l) for l in (p.parent/'trace.jsonl').read_text().splitlines()]
 stats=dict(success=r['success'],failure_category=r['failure_category'],physical_grasp_ever=any(x['physical_grasp_ok'] for x in rows),was_grasped=r['final_status']['was_grasped'],final_failure_reason=r['final_status']['failure_reason'],final_position=r['final_status']['object_position_m'],result=str(p))
 if stage=='after':
  dt=[abs(x['contact_status']['gripper_tf_stamp_s']-x['contact_status']['object_pose_stamp_s']) for x in rows if x['contact_status'].get('gripper_tf_stamp_s') is not None and x['contact_status'].get('object_pose_stamp_s') is not None]
  assert dt and max(dt)<=.100001
  stats.update(trace_rows=len(rows),rows_with_tf_stamps=len(dt),maximum_tf_object_delta_s=max(dt),median_tf_object_delta_s=float(np.median(dt)))
 out[stage]=stats
rs=[json.loads(x) for x in (w/'after_live_probe.jsonl').read_text().splitlines()];errs=[np.linalg.norm(np.array(x['status_relative'])-np.array(x['fk_relative'])) for x in rs if x.get('status_relative')]
out['online_fk_check']=dict(samples=len(errs),median_error_mm=float(np.median(errs)*1000),p95_error_mm=float(np.quantile(errs,.95)*1000),maximum_error_mm=float(max(errs)*1000),samples_above100mm=int(sum(e>.1 for e in errs)))
out['tests']=json.loads((w/'test_summary.json').read_text());out['root_cause']='Simulation clock rewind left future dynamic TF in evaluator buffer; fallback accepted old latest TF.'
out['fixes']=['Clear dynamic TF and object history on clock rewind','Reject TF/object time gaps exceeding 0.1s','Persist evaluated gripper pose and TF/object timestamps in rollout trace']
pid=json.loads((w/'restart.json').read_text())['new_pid'];os.kill(pid,0);out['active_evaluator_pid']=pid
(w/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
old=Path('/home/lp531/smolvla_eval/act_gentle_15vs8_random30_20261015_113836')
(old/'evaluation_validity.json').write_text(json.dumps(dict(status='grasp_and_success_scoring_affected_by_stale_tf',reason=out['root_cause'],do_not_treat_recorded_zero_success_as_reliable_model_success_rate=True,original_results_preserved=True,repair_verification=str(w/'verification.json')),indent=2))
(w/'修复说明.md').write_text('''# 评估链路修复

根因已由运行日志确认：Isaac重启使仿真时间回退，Humble Python TF Buffer未自动清空，评估器仍持有82,229秒之后的动态TF；当前物体约57,000秒。带时间查询失败后无条件使用latest，把历史夹爪位姿与当前瓶子位置配对。

修改：时钟回退时清空动态TF及物体位姿历史；拒绝与物体时间相差超过0.1秒的夹爪TF（包括未来TF）；在评估trace保存夹爪世界位姿、TF时间和物体时间。评分条件未放宽。已仅重启评估器节点。

4个回归测试使用真实tf2 Buffer，覆盖仿真时间回退后重新接收TF、拒收未来TF、拒收陈旧TF及允许正常小延迟。全套464通过、13失败；13个失败名称与此前运行完全一致，详见test_summary.json和tests.log。

同种子1628616970，ACT温和50k、执行15步、最短闭爪保持1秒：修复前grasp_failed、was_grasped=false；修复后真实抓取确认恢复，was_grasped=true，最终为placement_failed。本次不是成功任务，说明评分现在可以区分抓取与放置失败。

在线FK与评分位姿数值、TF时差和前后报告路径见verification.json。之前15/8步60次评分受旧TF影响，已另写evaluation_validity.json作出标记，原报告不修改；预测与实际指令仍可分析，基于错误相对位置的到位判断和最终成功率需重新评估。
''')
