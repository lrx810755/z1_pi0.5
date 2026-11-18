import json,sys,collections,datetime
from pathlib import Path
import numpy as np
R=Path(__file__).parent
from lift_metrics import proof_lift
from analyze_progress import category

def collect(prefix):
 a=[]
 for p in sorted((R/'cases').glob(prefix+'*/result.json')):
  d=json.loads(p.read_text());d['lift_audit']=proof_lift(p.parent/'trace.jsonl');d['failure_category']=category(d);a.append(d)
 return a

def summary(ds):
 return dict(n=len(ds),workflow_pass=sum(d['passed'] for d in ds),stable_lift_pass=sum(d['lift_audit']['proof_lift_pass'] for d in ds),failure_categories=dict(collections.Counter(d['failure_category'] for d in ds)))

if __name__=='__main__':
 names=['original_grid_','optimized_grid_h40_','optimized_grid_h40z17_','optimized_grid_h34z15_','original_range_','range_flexible_','screen_','deployed_smoke_'];data={k:collect(k) for k in names};summ={k:summary(v) for k,v in data.items()};(R/'最终实验明细.json').write_text(json.dumps(dict(summary=summ,experiments=data),ensure_ascii=False,indent=2));print(json.dumps(summ,ensure_ascii=False,indent=2))
