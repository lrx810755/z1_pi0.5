import time,json,subprocess
from pathlib import Path
r=Path(__file__).parent
while not (r/'optimization_screen9_results.json').exists():time.sleep(5)
for batch in ['optimized_grid_h40_20','range_flexible9','radius_compensation3']:
 with (r/(batch+'.log')).open('w') as f:result=subprocess.run(['bash',str(r/'run_client.sh'),str(r/(batch+'.json'))],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(batch+' failed')
print('All validation batches finished',flush=True)
