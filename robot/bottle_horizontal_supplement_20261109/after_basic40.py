"""Enable the next approach only after all20 cells have2 saved successes."""
import json,os
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/bottle_horizontal_supplement_20261109')
def completion_counts(data):
 counts=Counter()
 for p in data.glob('episode_*/episode_summary.json'):
  try:
   d=json.loads(p.read_text());grid=d.get('scene_randomization',{}).get('target_grid',[])
   if d.get('target_object')=='beverage_bottle' and d.get('task_success') is True and len(grid)==2:counts[tuple(grid)]+=1
  except (OSError,ValueError,TypeError):continue
 return counts
if __name__=='__main__':
 counts=completion_counts(DATA);complete=all(counts[c,r]>=2 for c in range(5) for r in range(4))
 print(json.dumps(dict(basic_complete=complete,saved=sum(counts.values())),ensure_ascii=False))
 if complete:
  target=R/'horizontal50_ready.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(dict(basic_data=str(DATA),basic_target=40,approach='column0 original; columns1..4 fixed50mm horizontal entry',cells={f'{c},{r}':counts[c,r] for c in range(5) for r in range(4)}),ensure_ascii=False,indent=2));os.replace(tmp,target)

  plan=R/'horizontal50_next_quota.json'
  if plan.exists():
   import subprocess
   quota=json.loads(plan.read_text())['quota_per_cell']
   if not isinstance(quota,int) or quota<1:raise ValueError('invalid next quota')
   subprocess.run(['systemd-run','--user','--unit=bottle-horizontal50-after-basic-20261109','--property=Restart=no','--setenv=DISPLAY=:1','--setenv=XAUTHORITY=/run/user/1000/gdm/Xauthority','/bin/bash','/home/lp531/SH/start_bottle_horizontal50_after_basic_20261109.sh',str(quota)],check=True)
  else:print('Horizontal50 enabled; next-stage quota awaits user choice, no extra collection started')
