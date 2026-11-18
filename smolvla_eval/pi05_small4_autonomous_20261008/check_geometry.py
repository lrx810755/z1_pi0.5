import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/lp531/smolvla_eval/pi05_action_contract_audit_20261007')
from geometry import height,audit
OUT=Path(__file__).resolve().parent;rows=[]
for p in sorted(OUT.glob('repeat*/episode*/episodes/*/result.json')):
 r=json.loads(p.read_text());dst=p.parent/'prediction_geometry.json'
 if dst.exists():item=json.loads(dst.read_text())
 else:
  pr=[json.loads(x) for x in (p.parent/'predictions.jsonl').read_text().splitlines()];tr=[json.loads(x) for x in (p.parent/'trace.jsonl').read_text().splitlines()];records=[]
  for pred in pr:
   raw=np.array(pred['full_actions']);raw[:,-1]=np.where(raw[:,-1]>=.5,0,-.9);clip=np.clip(raw,audit.limits[:,0],audit.limits[:,1]);rh=[height(q) for q in raw[:5]];ch=[height(q) for q in clip[:5]]
   records.append(dict(prediction=pred['prediction'],raw_height_min=min(rh),clipped_height_min=min(ch),limit_change_max_rad=float(np.max(abs(raw[:5,:6]-clip[:5,:6]))),joint_exceeds=np.any(abs(raw[:5,:6]-clip[:5,:6])>.001,axis=0).tolist()))
  limited=[]
  for t in tr:
   if t['execution_substep']!=0:continue
   target=np.clip(t['execution_target'],audit.limits[:,0],audit.limits[:,1]);current=np.array(t['joint_state']);limited.append(float(max(0,np.max(abs(target[:6]-current[:6]))-.1)))
  item=dict(episode=p.parts[-4],repeat=p.parts[-5],raw_risk_chunks=sum(x['raw_height_min']<.004 for x in records),clipped_risk_chunks=sum(x['clipped_height_min']<.004 for x in records),first_raw_risk=next((x for x in records if x['raw_height_min']<.004),None),first_clipped_risk=next((x for x in records if x['clipped_height_min']<.004),None),limit_changed_chunks=sum(x['limit_change_max_rad']>.001 for x in records),step_limited_actions=sum(x>.001 for x in limited),max_step_change_rad=max(limited),joint_exceed_chunk_counts=np.sum([x['joint_exceeds'] for x in records],axis=0).tolist(),records=records)
  dst.write_text(json.dumps(item,indent=2))
 rows.append({k:v for k,v in item.items() if k!='records'})
(OUT/'geometry_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
