import os,time,subprocess,json
from pathlib import Path
r=Path(__file__).parent
while True:
 try:os.kill(3033940,0)
 except ProcessLookupError:break
 time.sleep(5)
if not (r/'original_grid20_results.json').exists():raise RuntimeError('Original workspace sweep did not finish')
for batch in ['original_range9','optimization_screen9']:
 with (r/(batch+'.log')).open('w') as f:
  result=subprocess.run(['bash',str(r/'run_client.sh'),str(r/(batch+'.json'))],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(batch+' client failed')
print('Batches complete',flush=True)
