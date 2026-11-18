import time,json,subprocess
from pathlib import Path
from lift_metrics import proof_lift
r=Path(__file__).parent
while not (r/'radius_compensation3_results.json').exists():time.sleep(5)
ds=json.loads((r/'radius_compensation3_results.json').read_text());n=sum(d['passed'] and proof_lift(r/'cases'/d['case']['id']/'trace.jsonl')['proof_lift_pass'] for d in ds)
if n<2:print('Radius compensation did not pass screening; combined candidate skipped',flush=True)
else:
 print('Radius screening passed '+str(n)+'/3; validating height40+Z17 on identical 20 grids',flush=True)
 with (r/'optimized_grid_h40z17_20.log').open('w') as f:result=subprocess.run(['bash',str(r/'run_client.sh'),str(r/'optimized_grid_h40z17_20.json')],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Combined validation failed to finish')
 print('Combined validation completed',flush=True)
