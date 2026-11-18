"""Summarize real episodes and audit initial-state matching across checkpoints."""
from pathlib import Path
import json,math,datetime
ROOT=Path(__file__).resolve().parent

def main():
 manifest=json.loads((ROOT/'manifest.json').read_text())
 metrics=[];summaries={};raw={}
 for name,model in manifest['models'].items():
  episodes=[]
  for p in sorted((ROOT/name).glob('*/episodes/*/result.json')):
   result=json.loads(p.read_text());raw[(name,result.get('seed'))]=result
   trace=p.parent/'trace.jsonl'
   rows=[json.loads(s) for s in trace.read_text().splitlines() if s.strip()] if trace.exists() else []
   closed=[r['execution_target'][-1]>-.45 for r in rows]
   transitions=[i for i in range(1,len(rows)) if closed[i]!=closed[i-1]]
   starts=([0] if closed and closed[0] else [])+[i for i in transitions if closed[i]]
   ends=[i for i in transitions if not closed[i]]
   durations=[]
   for i in starts:
    j=next((x for x in ends if x>i),len(rows)-1)
    durations.append(round(rows[j]['simulation_elapsed_s']-rows[i]['simulation_elapsed_s'],3))
   both=[r for r in rows if r['contact_status'].get('left_finger_contact_object')=='beverage_bottle' and r['contact_status'].get('right_finger_contact_object')=='beverage_bottle']
   pos=[r['contact_status']['object_position_m'] for r in rows]
   physical=result.get('physical_predicates',{})
   sync_file=p.parent/'input_sync.jsonl'
   sync=[json.loads(line) for line in sync_file.read_text().splitlines() if line.strip()] if sync_file.exists() else []
   d=dict(model=name,seed=result.get('seed'),success=bool(result.get('success')),failure=result.get('failure_category'),
     physical_grasp=bool(physical.get('physical_grasp_ok')),lifted=bool(physical.get('lifted') and physical.get('physical_grasp_ok')),raw_lifted_flag=bool(physical.get('lifted')),fell_below_table=bool(pos and min(p[2] for p in pos)<-.05),bilateral=bool(both) or bool(physical.get('bilateral_contact')),
     close_attempts=len(starts),open_close_transitions=len(transitions),closed_intervals_s=durations,
     max_object_reference_z_m=max((p[2] for p in pos),default=0),
     net_xy_displacement_m=math.dist(pos[0][:2],pos[-1][:2]) if pos else None,
     initial_position=result.get('target_position_m'),initial_joint_state=result.get('initial_joint_state'),
     simulation_duration_s=result.get('simulation_duration_s'),path=str(p.parent),
     table_contact=result.get('failure_category')=='finger_table_contact',
     opens_at_chunk_boundary=sum(rows[i]['policy_action_index']==0 for i in ends),
     all_trace_unfrozen=bool(rows) and all(not row['contact_status'].get('frozen',True) for row in rows),
     all_trace_collisions_enabled=bool(rows) and all(row['contact_status'].get('target_collisions_enabled',False) for row in rows),
     max_sensor_skew_s=max((row['sensor_skew_s'] for row in sync),default=None),
     max_sensor_age_s=max((row['oldest_sensor_age_s'] for row in sync),default=None))
   episodes.append(d);metrics.append(d)
  summaries[name]=dict(complete_episodes=len(episodes),successes=sum(d['success'] for d in episodes),
    lifted=sum(d['lifted'] for d in episodes),bilateral=sum(d['bilateral'] for d in episodes),
    never_close=sum(d['close_attempts']==0 for d in episodes),table_contacts=sum(d['table_contact'] for d in episodes),
    close_attempts=sum(d['close_attempts'] for d in episodes))
 audits=[]
 for scene in manifest['scenes']:
  seed=scene['seed'];base=raw.get(('old_080000',seed))
  if not base:continue
  for name in list(manifest['models'])[1:]:
   other=raw.get((name,seed))
   if not other:continue
   b=base.get('initial_scene_randomization',{});o=other.get('initial_scene_randomization',{})
   bp=b.get('object_positions_m',{});op=o.get('object_positions_m',{})
   errs={k:math.dist(bp[k],op[k]) for k in bp.keys()&op.keys()}
   angles={}
   for k in b.get('object_orientations_xyzw',{}).keys()&o.get('object_orientations_xyzw',{}).keys():
    q=b['object_orientations_xyzw'][k];r=o['object_orientations_xyzw'][k]
    dot=abs(sum(x*y for x,y in zip(q,r)))/math.sqrt(sum(x*x for x in q)*sum(x*x for x in r))
    angles[k]=2*math.degrees(math.acos(min(1,dot)))
   bj=base.get('initial_joint_state',[]);oj=other.get('initial_joint_state',[])
   joint=max((abs(x-y) for x,y in zip(bj,oj)),default=float('inf'))
   audits.append(dict(seed=seed,model=name,max_object_position_difference_m=max(errs.values(),default=float('inf')),
      max_object_orientation_difference_deg=max(angles.values(),default=float('inf')),max_initial_joint_difference_rad=joint,
      matched=bool(len(errs)==5 and len(bj)==len(oj)==7 and max(errs.values())<=.001 and max(angles.values(),default=999)<=.5 and joint<=.01)))
 report=dict(updated_at=datetime.datetime.now().astimezone().isoformat(),expected_episodes=32,
  complete_episodes=len(metrics),models=summaries,episodes=metrics,initial_state_audit=audits,
  all_available_pairs_matched=all(a['matched'] for a in audits),selection=manifest['selection'])
 (ROOT/'comparison_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 text='# PI05 固定场景模型对照\n\n'
 text+=f"已记录 {len(metrics)}/32 场；更新时间 {report['updated_at']}。\n\n"
 text+='训练差异说明：旧版余弦衰减总长度80000微步，新版为120000微步；新旧80k对比同时包含阶段采样与学习率调度差异，不是加权单因素消融。新版三个检查点用于观察训练进程中的表现变化。\n\n'
 text+='四个模型各评估同一组 8 个场景；推理种子 20261011；执行10帧，动作10 Hz，插值60 Hz，每场最多60秒仿真时间；二值夹爪，物体自由物理状态。场景选自历史成功与失败案例，用于诊断，不代表随机总体成功率。\n\n'
 text+='|模型|完成|成功|稳定抬升|双指接触|全程未闭爪|撞桌|闭爪尝试总数|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for name,s in summaries.items():text+='|'+name+'|'+'|'.join(str(s[k]) for k in ['complete_episodes','successes','lifted','bilateral','never_close','table_contacts','close_attempts'])+'|\n'
 text+='\n|场景种子|标签|旧80k|新80k|新96k|新120k|\n|---|---|---|---|---|---|\n'
 for scene in manifest['scenes']:
  vals=[]
  for name in manifest['models']:
   d=next((d for d in metrics if d['model']==name and d['seed']==scene['seed']),None)
   vals.append(('成功' if d['success'] else d['failure']) if d else '待完成')
  text+='|'+str(scene['seed'])+'|'+scene['label']+'|'+'|'.join(vals)+'|\n'
 text+=f"\n初始状态核验：已有 {len(audits)} 对记录；全部符合物体位置差≤1 mm、姿态差≤0.5°、初始关节差≤0.01 rad：{report['all_available_pairs_matched']}。推理种子已固定，但不承诺整个GPU物理仿真逐位确定。\n"
 text+='\n物体最大 z 为参考点高度，倒瓶或碰飞也会改变该值；稳定抬升同时要求 physical_grasp_ok 与 lifted 成立。单独 lifted 标志可能在瓶子被碰飞后变真，不能据此当作成功抓起。\n\n原始录像、轨迹和结果位于各模型子目录；量化详情见 comparison_metrics.json；配置与权重哈希见 manifest.json。\n'
 (ROOT/'对照结果.md').write_text(text)
 print(json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
