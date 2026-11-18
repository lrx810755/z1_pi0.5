"""Sequential20Hz collection; final manifest exists only after exact513 accepted episodes."""
import json,os,sys,time,signal,subprocess,threading,datetime,fcntl,hashlib
from pathlib import Path
from basic.audit import audit_episode
from basic.fault_latch import check_fault
ROOT=Path(__file__).resolve().parent
DATA={k:Path(v) for k,v in dict(basic='/home/lp531/smolvla_data/z1_cup_basic400_20hz_20261031',correction='/home/lp531/smolvla_data/z1_cup_correction100_20hz_20261031',recovery='/home/lp531/smolvla_data/z1_cup_push_recovery13_20hz_20261031').items()}
QUOTAS=dict(basic=400,correction=100,recovery=13)
def save(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def counts():
 result={}
 for k in DATA:
  p=ROOT/k/'episode_audits.json';result[k]=len(json.loads(p.read_text())) if p.exists() else 0
 return result
def state(stage,**kwargs):
 c=counts();save('status.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),stage=stage,completed=sum(c.values()),total=513,by_source=c,quotas=QUOTAS,**kwargs))
def stop(p):
 if p is None:return
 for sig,delay in [(signal.SIGTERM,35),(signal.SIGKILL,3)]:
  try:os.killpg(p.pid,sig)
  except ProcessLookupError:break
  try:p.wait(timeout=delay)
  except subprocess.TimeoutExpired:continue
def run(command,stage,component=None):
 env=os.environ.copy()
 if component:env['Z1_ISAAC_GRASP_CONFIG']=str(ROOT/component/'grasp_physics_minus90.json')
 save('active_command.json',dict(command=command,stage=stage,component=component))
 with (ROOT/(stage+'.log')).open('a') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
  try:
   while p.poll() is None:
    for k in DATA:check_fault(ROOT/k/'sync_fault.json')
    state(stage,pid=p.pid);time.sleep(2)
   if p.returncode:raise RuntimeError(f'{stage} failed with exit {p.returncode}; inspect {stage}.log')
  finally:stop(p)
def finalize():
 episodes=[]
 for key,quota in QUOTAS.items():
  folders=sorted(DATA[key].glob('episode_[0-9]*'));assert len(folders)==quota,(key,len(folders))
  assert not list(DATA[key].glob('.recording_*')),'unfinished recording at completion'
  for p in folders:
   report=audit_episode(p);meta=json.loads((p/'correction.json').read_text()) if (p/'correction.json').exists() else {}
   episodes.append(dict(episode_index=len(episodes),source=key,source_path=str(p),fps=20,grid=report['grid'],seed=report['seed'],frames=report['frames'],task_id=meta.get('task_id'),kind=meta.get('task',{}).get('kind'),source_identity=meta.get('source_identity'),split=meta.get('split','validation' if tuple(report['grid']) in [(1,1),(2,3)] else 'train'),audit=report))
 basic=[x for x in episodes if x['source']=='basic'];from collections import Counter
 assert Counter(tuple(x['grid']) for x in basic)=={(c,r):20 for r in range(4) for c in range(5)}
 original=json.loads((ROOT/'original513_sources.json').read_text())['sources']
 assert {x['task_id'] for x in episodes if x['source']=='correction'}=={x['task_id'] for x in original['correction']}
 assert {x['source_identity'] for x in episodes if x['source']=='recovery'}=={x['source_identity'] for x in original['actual_policy_push_recovery']}
 for path,h in json.loads((ROOT/'source_snapshot.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 manifest=dict(completed=True,total=513,fps=20,quotas=QUOTAS,episodes=episodes,all_timestamps_audited=True,finished=datetime.datetime.now().astimezone().isoformat())
 save('dataset_manifest.json',manifest);save('completed.json',{k:v for k,v in manifest.items() if k!='episodes'});state('completed')
def main():
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 def watch():
  while True:
   if any((ROOT/k/'sync_fault.json').exists() for k in DATA):os.kill(os.getpid(),signal.SIGTERM);return
   time.sleep(.1)
 with (ROOT/'.pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for k in DATA:check_fault(ROOT/k/'sync_fault.json')
  threading.Thread(target=watch,daemon=True).start()
  try:
   for key,data in DATA.items():
    for pending in data.glob('.recording_*'):
     archive=ROOT/key/'interrupted'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');archive.mkdir(parents=True,exist_ok=True)
     pending.rename(archive/pending.name)
   # Re-audit every inherited20Hz basic episode before counting it.
   reports=[audit_episode(p) for p in sorted(DATA['basic'].glob('episode_[0-9]*'))]
   (ROOT/'basic/episode_audits.json').write_text(json.dumps(reports,indent=2))
   state('preflight')
   run(['/usr/bin/python3',str(ROOT/'recovery/collector/audit_saved.py'),'--freeze-source-pins'],'pin_recovery_sources','recovery')
   marker=DATA['recovery']/'completion_verified.json'
   if not (marker.exists() and json.loads(marker.read_text()).get('passed')):
    run(['/usr/bin/python3','-u',str(ROOT/'recovery/collector/supervise.py')],'collect_recovery13','recovery')
   else:
    run(['/usr/bin/python3',str(ROOT/'recovery/collector/audit_saved.py')],'audit_existing_recovery13','recovery')
   assert json.loads((DATA['recovery']/'completion_verified.json').read_text())['passed']
   if counts()['correction']<100:
    command=['/usr/bin/python3','-u',str(ROOT/'correction/collector/run.py')]
    # A hard correction task must not block the independent basic quota.
    if counts()['basic']<400:command+=['--max-attempts','12']
    run(command,'collect_correction100','correction')
   if counts()['basic']<400:
    run(['/usr/bin/python3','-u',str(ROOT/'basic/supervise.py')],'collect_basic400','basic')
   if counts()['correction']<100:
    run(['/usr/bin/python3','-u',str(ROOT/'correction/collector/run.py')],'finish_correction100','correction')
   run(['/usr/bin/python3',str(ROOT/'correction/collector/audit_saved.py')],'audit_correction100','correction')
   integrity=json.loads((ROOT/'correction/saved_integrity.json').read_text());assert integrity['complete'] and integrity['accepted']==100
   state('final_audit');finalize()
  except BaseException as e:
   state('stopped' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
if __name__=='__main__':main()
