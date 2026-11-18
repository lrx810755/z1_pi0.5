#!/usr/bin/python3
"""Three finite passes over apple column 3, with durable per-attempt audit."""
import sys,json,time,threading,signal,datetime
from pathlib import Path
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge')
from continuous_training import JsonApiClient,ContinuousTrainingRunner
ROOT=Path('/home/lp531/z1_queue_logs/apple_column3_three_rounds_20261016')
DATA=Path('/home/lp531/smolvla_data/z1_tabletop_raw_apple_preplanned_three_camera_20261016')

def counts():
    result={row:0 for row in range(5)}
    for p in DATA.glob('episode_*/episode_summary.json'):
        d=json.loads(p.read_text()); g=d.get('scene_randomization',{}).get('target_grid')
        if d.get('task_success') and d.get('target_object')=='apple' and g and g[0]==3:
            result[g[1]]+=1
    return result

def append(path,data):
    with path.open('a') as f:
        f.write(json.dumps(data,ensure_ascii=False)+'\n');f.flush()

class AuditApi(JsonApiClient):
    def __init__(self,round_no,row):
        super().__init__('http://127.0.0.1:8767')
        self.round=round_no; self.row=row;self.attempt=0
    def post(self,path,payload,timeout_s):
        if path!='/api/demonstration/auto':return super().post(path,payload,timeout_s)
        self.attempt+=1
        key=f'round{self.round}_grid3_{self.row}_attempt{self.attempt:03d}'
        trace=ROOT/(key+'.jsonl'); done=threading.Event(); latest={}; start_meta={}
        def sample():
            nonlocal latest,start_meta
            try:
                s=self.get('/api/status',timeout_s=2)
                latest=s
                r=s.get('recording',{})
                if r.get('recording'):
                    ep=DATA/f'.recording_{r["episode_index"]:06d}'/'episode_start.json'
                    if ep.exists():start_meta=json.loads(ep.read_text())
                append(trace,{'time':time.time(),'status':s})
            except Exception as e:
                append(trace,{'time':time.time(),'poll_error':str(e)})
        def monitor():
            while not done.is_set():sample();done.wait(.5)
        worker=threading.Thread(target=monitor,daemon=True); worker.start()
        result=None;error=None;began=time.time()
        try:
            result=super().post(path,payload,timeout_s)
            return result
        except BaseException as exc:
            error=str(exc);raise
        finally:
            done.set();worker.join(4);sample()
            success=bool(result and result.get('success'))
            record={'id':key,'round':self.round,'grid':[3,self.row],
                    'started_at':datetime.datetime.fromtimestamp(began).astimezone().isoformat(),
                    'duration_wall_s':time.time()-began,'success':success,'failure_reason':error or (None if success else str(result)),
                    'initial_apple_position_m':start_meta.get('scene_randomization',{}).get('object_positions_m',{}).get('apple'),
                    'randomization_seed':start_meta.get('scene_randomization',{}).get('seed'),
                    'final_observed_apple_position_m':latest.get('object_position_m'),
                    'initial_position_source':'episode_start' if start_meta else 'unavailable_before_recording',
                    'last_phase':latest.get('auto_demonstration'), 'trace':trace.name}
            append(ROOT/'attempts.jsonl',record)
            if not success:append(ROOT/'failures.jsonl',record)
            print(json.dumps(record,ensure_ascii=False),flush=True)

active=None
stop=False
def on_signal(sig,frame):
    global stop
    stop=True
    if active:active.request_stop()
signal.signal(signal.SIGINT,on_signal);signal.signal(signal.SIGTERM,on_signal)
for round_no in range(1,4):
    if stop:break
    for row in range(5):
        if stop:break
        saved=counts()[row]
        if saved>=8:continue
        print(f'第{round_no}/3轮，网格(3,{row})，已保存{saved}/8',flush=True)
        active=ContinuousTrainingRunner(AuditApi(round_no,row),'apple','full',8-saved,
                                         target_grid=(3,row),max_failures=50)
        code=active.run();active=None
        (ROOT/'status.json').write_text(json.dumps({'round':round_no,'last_grid':[3,row],
                        'runner_code':code,'saved':counts(),'stopped':stop},ensure_ascii=False,indent=2))
        if code not in (0,3,4,130):raise RuntimeError(f'Unexpected exit {code}')
(ROOT/'completed.json').write_text(json.dumps({'stopped':stop,'saved':counts(),'round_limit':3},indent=2))
