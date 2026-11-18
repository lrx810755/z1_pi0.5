import json
from pathlib import Path
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent
read=lambda name:[json.loads(l) for l in (P/name).read_text().splitlines() if l.strip()]
def stats(values):
    a=np.array(values,dtype=float)
    return {} if not len(a) else dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))
m=read('messages.jsonl');pairs=read('shadow_pairs.jsonl');frames=read('recorded_frames_snapshot.jsonl');result={}
streams={}
for name in ('wrist','global','global_right','state','action'):
    group=[x for x in m if x['kind']==name];ss=[x['stamp'] for x in group]
    streams[name]=dict(count=len(group),receipt_sim_age_s=stats([x['age'] for x in group if x['age'] is not None]),stamp_intervals_s=stats(np.diff(ss)),non_increasing=sum(b<=a for a,b in zip(ss,ss[1:])))
result['streams']=streams
camera_sets=[{round(x['stamp'],6) for x in m if x['kind']==k} for k in ('wrist','global','global_right')]
result['common_camera_stamps']=len(set.intersection(*camera_sets))
result['shadow_pairs']=dict(count=len(pairs),spread_s=stats([x['spread'] for x in pairs]),state_bracket_gap_s=stats([x['state_bracket'][1]-x['state_bracket'][0] for x in pairs]),action_age_s=stats([x['action_age'] for x in pairs]))
status=[x['data'] for x in m if x['kind']=='recording_status']
result['recorder_status']=dict(count=len(status),episodes=sorted({x['episode_index'] for x in status if x['episode_index'] is not None}),max_dropped=max([x['dropped_frames'] for x in status],default=0),errors=sorted({x.get('last_error','') for x in status if x.get('last_error')}),sync_failures=sorted({x.get('sync_failure','') for x in status if x.get('sync_failure')}))
def audit(rows):
    spreads=[];gaps=[];ages=[];errors=Counter();last={}
    for r in rows:
        if not all(k in r for k in ('global_right_image_stamp','joint_state_source_stamps','simulation_time')):errors['missing_sync_metadata']+=1;continue
        t=r['simulation_time'];cam=[r[k] for k in ('wrist_image_stamp','global_image_stamp','global_right_image_stamp')]
        spread=max([t,*cam])-min([t,*cam]);a,b=r['joint_state_source_stamps'];age=t-r['action_stamp']
        spreads.append(spread);gaps.append(b-a);ages.append(age)
        if spread>r.get('sync_tolerance_s',.045)+1e-8:errors['camera_spread']+=1
        if not a-1e-8<=t<=b+1e-8:errors['state_not_bracketed']+=1
        if b-a>.1+1e-8:errors['state_gap']+=1
        if age < -1e-8:errors['future_action']+=1
        if age>.4+1e-8:errors['stale_action']+=1
        for k,v in zip(('wrist','global','global_right'),cam):
            key=(r['source_episode_dir'],k)
            if key in last and v<=last[key]:errors['reused_image']+=1
            last[key]=v
    return dict(frames=len(rows),episodes=len({r['source_episode_dir'] for r in rows}),camera_spread_s=stats(spreads),state_bracket_gap_s=stats(gaps),action_age_s=stats(ages),violations=errors)
result['stored_frame_audit']=audit(frames)
live_dirs={f'episode_{i:06d}' for i in result['recorder_status']['episodes']}|{f'.recording_{i:06d}' for i in result['recorder_status']['episodes']}
result['monitored_episode_frame_audit']=audit([r for r in frames if r['source_episode_dir'] in live_dirs])
(P/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2,ensure_ascii=False))
