from pathlib import Path
import subprocess,os,json
r=Path(__file__).parent
for trial,time,name in [(2,56.983,'multiview_trial02'),(8,56.967,'multiview_trial08'),(12,27.4,'multiview_trial12_contact'),(12,56.9,'multiview_trial12_late'),(1,17.883,'multiview_trial01')]:
 env=dict(os.environ,CUP_TRIAL=str(trial),CUP_TIME=str(time),PROBE_OUT=name)
 with (r/(name+'.log')).open('w') as log:subprocess.run(['/home/lp531/isaac-sim/python.sh',str(r/'multi_view.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert (r/name/'after_wrist.png').exists(),name
 print('VERIFIED_RENDER',name,flush=True)
