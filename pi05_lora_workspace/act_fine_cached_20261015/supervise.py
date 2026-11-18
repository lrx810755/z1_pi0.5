from pathlib import Path
import subprocess,json,datetime,traceback
W=Path(__file__).resolve().parent
def save(name,data):
    p=W/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
try:
    for name in ['smoke_completed.json','checkpoint_reload_verified.json','cache_reuse_verified.json','sampling_audit.json']:
        assert (W/name).is_file(),name
    with (W/'training.log').open('ab') as log:
        process=subprocess.Popen(['bash',str(W/'run_training.sh')],cwd=W,stdout=log,stderr=subprocess.STDOUT)
        info=dict(pid=process.pid,started=datetime.datetime.now().isoformat(),target_optimizer_updates=50000,batch_size=16,gradient_accumulation=1,fresh_training=True,cache_reused=True,seed=20261013,output='/home/lp531/smolvla_training/act_fine_cached_chunk16_20261015')
        save('launch.json',info);save('pipeline_status.json',dict(stage='training',**info))
        code=process.wait()
    if code:raise RuntimeError(f'Training exited {code}; see training.log')
    assert (W/'completed.json').exists()
    save('pipeline_status.json',dict(stage='complete',optimizer_updates=50000,finished=datetime.datetime.now().isoformat()))
except Exception as e:
    save('pipeline_status.json',dict(stage='failed',error=str(e),traceback=traceback.format_exc()));raise
