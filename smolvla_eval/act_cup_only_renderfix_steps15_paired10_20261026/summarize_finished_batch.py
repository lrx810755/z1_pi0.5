"""Read traces only after the complete ten-trial batch has finished."""
from pathlib import Path
import json,collections,subprocess
R=Path(__file__).parent
summary=json.loads((R/'completed.json').read_text());assert summary['completed'] and summary['total']==10
rows=[]
for e in summary['episodes']:
 ep=Path(e['source_result']).parent;tr=[json.loads(l) for l in (ep/'trace.jsonl').read_text().splitlines()];ever_physical=any(t.get('physical_grasp_ok') for t in tr);closed=[float(t['execution_target'][-1])>=-.1 for t in tr];count=sum(b and (i==0 or not closed[i-1]) for i,b in enumerate(closed));bilateral=any(t['contact_status'].get('left_finger_contact_object')=='cup' and t['contact_status'].get('right_finger_contact_object')=='cup' for t in tr)
 if e['success']:reason='成功放入蓝盒并松爪'
 elif e['legacy_failure_category']=='finger_table_contact':reason='手指碰桌触发终止'
 elif ever_physical:reason='抓稳后搬运或放置松爪未完成'
 elif count==0:reason='未执行闭爪'
 elif bilateral:reason='双指接触后未形成稳定抓取'
 else:reason='闭爪未形成双指抓取'
 video=[]
 for name in ['global_third_person.mp4','global_right.mp4','wrist.mp4']:
  p=ep/'videos'/name;meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames,width,height,r_frame_rate:format=duration','-of','json',str(p)],text=True));assert float(meta['format']['duration'])>0 and int(meta['streams'][0]['nb_frames'])>0;video.append(dict(path=str(p),**meta))
 rows.append(dict(episode=e['episode'],seed=e['seed'],success=e['success'],reason=reason,close_count=count,ever_physical_grasp=ever_physical,bilateral_contact_seen=bilateral,max_reported_finger_penetration_mm=1000*max((t['contact_status'].get('target_finger_max_penetration_m') or 0) for t in tr),videos=video))
counts=dict(collections.Counter(x['reason'] for x in rows));out=dict(total=10,successes=summary['successes'],success_rate=summary['success_rate'],failure_counts=counts,episodes=rows,all_30_videos_verified=True,analysis_started_after_batch_completed=True)
(R/'结果与失败分类.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(dict(total=10,successes=summary['successes'],failure_counts=counts),ensure_ascii=False))
