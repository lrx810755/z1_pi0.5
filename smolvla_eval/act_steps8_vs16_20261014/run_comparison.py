import subprocess,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
seeds=[437216237,1126247665,1328542479]
for steps in [8,16]:
 command=['bash','/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh','--model','9','--execute-steps',str(steps),'--episodes-per-object','3','--eval-seeds',*[str(x) for x in seeds],'--output',str(ROOT/f'steps{steps}')]
 (ROOT/f'command_steps{steps}.json').write_text(json.dumps(command,indent=2))
 with (ROOT/f'steps{steps}.log').open('w') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
  (ROOT/'status.json').write_text(json.dumps(dict(steps=steps,pid=p.pid,status='running',seeds=seeds),indent=2))
  code=p.wait()
 (ROOT/f'finished_steps{steps}.json').write_text(json.dumps(dict(returncode=code,finished=datetime.datetime.now().isoformat()),indent=2))
 if code:raise SystemExit(code)
(ROOT/'status.json').write_text(json.dumps(dict(status='complete',seeds=seeds),indent=2))
