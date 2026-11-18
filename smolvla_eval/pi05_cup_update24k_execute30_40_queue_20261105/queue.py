from pathlib import Path
import json,subprocess,datetime
R=Path(__file__).resolve().parent
roots=['/home/lp531/smolvla_eval/pi05_cup20hz_chunk50_update24k_execute30_paired30_20261105', '/home/lp531/smolvla_eval/pi05_cup20hz_chunk50_update24k_execute40_paired30_20261105']
for steps,root in zip([30,40],roots):
    (R/'status.json').write_text(json.dumps(dict(stage='running',execute_steps=steps,root=root,updated=datetime.datetime.now().astimezone().isoformat()),indent=2))
    subprocess.run(['/bin/bash',str(Path(root)/'start.sh')],cwd=root,check=True)
    s=json.loads((Path(root)/'supervisor_status.json').read_text())
    assert s['stage']=='completed',s
(R/'status.json').write_text(json.dumps(dict(stage='completed',total=60,roots=roots),indent=2))
