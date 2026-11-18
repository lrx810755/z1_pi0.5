"""Wait for real scene/recorder readiness and tolerate ROS discovery delay."""
import argparse,json,subprocess,time,urllib.request
COMPONENTS=['recording','episode_reset','collision_scene','object_collisions']
def ready_status(s):return bool(s.get('connected') and all(s.get(k,{}).get('ready') for k in COMPONENTS))
def read_parameter(name,attempts=20):
 last=''
 for _ in range(attempts):
  result=subprocess.run(['ros2','param','get','/smolvla_episode_recorder',name,'--hide-type'],text=True,capture_output=True,timeout=8)
  if result.returncode==0 and result.stdout.strip():return result.stdout.strip()
  last=(result.stderr or result.stdout).strip();time.sleep(.5)
 raise RuntimeError('Recorder parameter unavailable after discovery retries: '+name+': '+last)
def wait_scene(url,timeout):
 deadline=time.monotonic()+timeout;last='waiting for scene'
 while time.monotonic()<deadline:
  try:
   with urllib.request.urlopen(url,timeout=2) as f:s=json.load(f)
   if ready_status(s):
    root=read_parameter('dataset_root');right=read_parameter('global_right_rgb_topic')
    print(json.dumps(dict(ready=True,dataset_root=root,right_topic=right)),flush=True);return
   last=json.dumps({k:s.get(k) for k in ['connected',*COMPONENTS]})
  except Exception as e:last=str(e)
  time.sleep(1)
 raise RuntimeError('Full startup readiness timeout: '+last)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parameter');p.add_argument('--timeout',type=float,default=300);p.add_argument('--url',default='http://127.0.0.1:8835/api/status');a=p.parse_args()
 if a.parameter:print(read_parameter(a.parameter))
 else:wait_scene(a.url,a.timeout)
