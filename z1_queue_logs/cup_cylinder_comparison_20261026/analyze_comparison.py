import json,sys,importlib.util,collections,math
from pathlib import Path
R=Path(__file__).parent
OLD=Path('/home/lp531/z1_queue_logs/cup_frustum_grasp_validation_20261026')
from lift_metrics import proof_lift as cylinder_proof
from analyze_progress import category
spec=importlib.util.spec_from_file_location('frustum_metrics',OLD/'lift_metrics.py');fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)

def read_group(root,prefix,metric):
 out={}
 for p in sorted((root/'cases').glob(prefix+'*/result.json')):
  d=json.loads(p.read_text());grid=tuple(d['case']['target_grid']);d['independent_lift_audit']=metric(p.parent/'trace.jsonl');d['failure_category']=category(d);out[grid]=d
 return out

def summary(ds):
 a=list(ds.values());return dict(completed=len(a),workflow_pass=sum(d['passed'] for d in a),stable_lift_pass=sum(d['independent_lift_audit']['proof_lift_pass'] for d in a),workflow_and_stability_pass=sum(d['passed'] and d['independent_lift_audit']['proof_lift_pass'] for d in a),categories=dict(collections.Counter(d['failure_category'] for d in a)))

def analyze():
 a=read_group(OLD,'original_grid_',fm.proof_lift);b=read_group(R,'cylinder_grid_',cylinder_proof);pairs=[]
 for grid in sorted(b):
  f=a[grid];c=b[grid];fp=OLD/'cases'/f['case']['id'];cp=R/'cases'/c['case']['id'];fc=json.loads((fp/'configured.json').read_text()) if (fp/'configured.json').exists() else None;cc=json.loads((cp/'configured.json').read_text()) if (cp/'configured.json').exists() else None
  assert {k:v for k,v in c['case'].items() if k!='id'}=={k:v for k,v in f['case'].items() if k!='id'}
  if fc and cc:assert fc['physical']==cc['physical'] and fc['profile']==cc['profile'],('Mismatched control config',grid)
  pairs.append(dict(grid=list(grid),position_m=c['case']['cup_position_m'],frustum_workflow=f['passed'],cylinder_workflow=c['passed'],frustum_strict=f['passed'] and f['independent_lift_audit']['proof_lift_pass'],cylinder_strict=c['passed'] and c['independent_lift_audit']['proof_lift_pass'],frustum_category=f['failure_category'],cylinder_category=c['failure_category'],control_config_equal=bool(fc and cc),cylinder_error=c.get('error')))
 sa=summary(a);sb=summary(b);wins=sum(p['cylinder_workflow'] and not p['frustum_workflow'] for p in pairs);losses=sum(p['frustum_workflow'] and not p['cylinder_workflow'] for p in pairs);discordant=wins+losses;pval=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
 out=dict(frustum=sa,cylinder=sb,paired_cases=pairs,cylinder_only_successes=wins,frustum_only_successes=losses,paired_exact_binomial_p=pval,not_act=True,internal_planning_seed_fixed=False,shape_validation=json.loads((R/'scene_validation.json').read_text()))
 (R/'对照结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(dict(frustum=sa,cylinder=sb,paired_cylinder_only=wins,paired_frustum_only=losses),ensure_ascii=False,indent=2));return out
if __name__=='__main__':analyze()
