"""Mutually exclusive failure groups from actual commands and contact traces."""
import collections,csv,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
D=json.loads((R/'comparison.json').read_text())

def inbox(p):return p is not None and abs(p[0]-.32)<=.055 and abs(p[1]-.14)<=.055
rows=[]
for e in D['episodes']:
 trace=[json.loads(l) for l in (Path(e['path'])/'trace.jsonl').read_text().splitlines()]
 final=e['final_status'];pos=final['object_position_m'];held=bool(final.get('holding_observed'))
 after_physical=[t for t in trace if e['first_physical_time'] is not None and t['simulation_elapsed_s']>=e['first_physical_time']]
 above=[t for t in after_physical if inbox(t['contact_status'].get('object_position_m'))]
 first_box=above[0]['simulation_elapsed_s'] if above else None
 after_box=[t for t in trace if first_box is not None and t['simulation_elapsed_s']>=first_box]
 open_after_phys=[x for x in e['events'] if x['kind']=='open' and e['first_physical_time'] is not None and x['time']>=e['first_physical_time']]
 evidence=dict(close_count=e['close_count'],ever_capture=e['ever_capture'],ever_capture_before_close=e['ever_capture_before_close'],ever_physical=e['ever_physical'],bilateral_seen=e['bilateral_seen'],first_close=e['first_close'],max_predicted_close=e['max_predicted_close'],max_cup_root_z_m=e['max_cup_z'],cup_shift_before_close_m=e['cup_shift_before_first_close_m'],final_position_m=pos,final_holding=held,final_released=final.get('released'),first_above_box_s=first_box,time_after_first_box_s=trace[-1]['simulation_elapsed_s']-first_box if first_box is not None else None,open_command_after_box=any(t['execution_target'][-1]<-.895 for t in after_box),min_policy_jaw_after_box=min((t['policy_action'][-1] for t in after_box),default=None),open_events_after_physical=open_after_phys)
 if e['success']:category='成功'
 elif e['category']=='finger_table_contact':category='夹爪碰桌，安全终止'
 elif e['close_count']==0:
  category='未进入抓取范围，未触发闭爪' if not e['ever_capture'] else '曾进入抓取范围，但未触发闭爪'
 elif not e['ever_physical']:
  category='闭爪后未形成可靠夹持'
 elif held:
  if inbox(pos) and pos[2]>.12 and not evidence['open_command_after_box']:category='夹稳后停在盒上，未下降松爪'
  elif inbox(pos) and not evidence['open_command_after_box']:category='已到蓝盒，但未完成松爪'
  elif not inbox(pos):category='夹稳后搬运轨迹偏离'
  else:category='发出开爪后仍未解除夹持'
 elif open_after_phys:
  released_at=open_after_phys[0]['object_position']
  category='未到蓝盒便高处松爪' if not inbox(released_at) and released_at[2]>.12 else '松爪后未完成有效放置'
 else:category='搬运中失去可靠夹持'
 rows.append(dict(episode=e['episode'],seed=e['seed'],success=e['success'],failure_group=category,legacy_category=e['category'],path=e['path'],evidence=evidence))
groups=collections.defaultdict(list)
for row in rows:groups[row['failure_group']].append(row['episode'])
assert sum(len(ids) for name,ids in groups.items() if name!='成功')==30-sum(e['success'] for e in D['episodes'])
result=dict(total=30,successes=sum(e['success'] for e in D['episodes']),groups=dict(groups),cross_cutting=dict(repeated_close=D['summary']['repeated_close'],preclose_cup_shift_over5mm=D['summary']['preclose_cup_shift_over5mm'],never_entered_capture=D['summary']['never_entered_capture'],ever_reliable_physical_grasp=D['summary']['ever_physical']),episodes=rows)
(R/'failure_categories.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (R/'failure_categories.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['编号','种子','最终成功','失败分类','原过程分类','闭爪次数','曾进捕获范围','曾通过可靠夹持','最终持握','最终松爪','最终位置'])
 for x in rows:
  ev=x['evidence'];w.writerow([x['episode'],x['seed'],x['success'],x['failure_group'],x['legacy_category'],ev['close_count'],ev['ever_capture'],ev['ever_physical'],ev['final_holding'],ev['final_released'],ev['final_position_m']])
print(json.dumps({k:v for k,v in result.items() if k!='episodes'},ensure_ascii=False,indent=2))
