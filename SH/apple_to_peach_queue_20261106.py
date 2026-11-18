"""Start peach basic demonstrations only after all apple correction IDs are saved."""
import importlib.util,json,os,subprocess,time,urllib.request
from pathlib import Path
APPLE=Path('/home/lp531/smolvla_data/apple_scale110_correction60_20261106')
CODE=Path('/home/lp531/robot/apple_scale110_correction60_20261106')
LOG=Path('/home/lp531/z1_queue_logs/apple_to_peach_20261106')
APPLE_UNIT='apple-scale110-correction60-20261106.service'
PEACH_UNIT='peach-scale120-demo400-20261106.service'
spec=importlib.util.spec_from_file_location('apple_completion',CODE/'core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def write(stage,**values):
    record=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**values)
    core.atomic_json(LOG/'status.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)

def active(unit):
    p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True,capture_output=True)
    return p.stdout.strip()

def complete():
    plan=json.loads((APPLE/'correction_plan.json').read_text())
    if plan.get('target_object')!='apple' or len(plan['tasks'])!=60:raise RuntimeError('苹果纠偏计划不符合预期')
    required={t['id'] for t in plan['tasks']}
    done=core.completed_ids(APPLE,plan['hash'])
    return required.issubset(done),len(required & done)

def main():
    LOG.mkdir(parents=True,exist_ok=True)
    last_count=None
    while True:
        done,count=complete()
        if done:break
        if count!=last_count:
            write('WAITING_FOR_APPLE',apple_completed=count,apple_quota=60,peach_started=False)
            last_count=count
        time.sleep(10)
    write('STOPPING_APPLE',apple_completed=count,apple_quota=60)
    if active(APPLE_UNIT) in ('active','activating','deactivating'):
        subprocess.run(['systemctl','--user','stop','--no-block',APPLE_UNIT],check=True)
        for _ in range(20):
            if active(APPLE_UNIT) not in ('active','activating','deactivating'):break
            time.sleep(1)
        if active(APPLE_UNIT) in ('active','activating','deactivating'):
            subprocess.run(['systemctl','--user','kill','--signal=SIGKILL','--kill-who=all',APPLE_UNIT],check=True)
            time.sleep(2)
    # A failed/inactive unit must not conceal a surviving old simulation.
    config=str(CODE/'grasp_physics.json').encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if config in args and any(a.endswith(b'run_z1_ros2_scene.py') for a in args):
            raise RuntimeError('苹果Sim仍存活，拒绝启动第二个Sim：PID '+proc.name)
    if active(PEACH_UNIT) not in ('active','activating'):
        write('STARTING_PEACH',apple_completed=count,peach_quota=400,quota_per_cell=20)
        subprocess.run(['systemd-run','--user','--unit='+PEACH_UNIT,'--property=KillMode=control-group',
                        '--setenv=DISPLAY=:1','--setenv=XAUTHORITY=/run/user/1000/gdm/Xauthority',
                        '--setenv=PYTHONUNBUFFERED=1','--setenv=Z1_PEACH_QUOTA_PER_CELL=20',
                        '/bin/bash','/home/lp531/SH/start_peach_scale120_demo.sh'],check=True)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if active(PEACH_UNIT) not in ('active','activating'):
            raise RuntimeError('桃子采集启动退出，请检查桃子服务日志')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8835/api/status',timeout=3) as response:s=json.load(response)
            if (s.get('connected') and all(s.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions'])
                and s['recording'].get('dataset_root')=='/home/lp531/smolvla_data/peach_scale120_demo_20261106'):
                write('PEACH_RUNNING',apple_completed=count,peach_quota=400,peach_unit=PEACH_UNIT)
                return
        except (OSError,ValueError):pass
        time.sleep(2)
    raise RuntimeError('桃子场景300秒内未就绪')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('ERROR',error=str(exc),peach_unit=PEACH_UNIT)
        raise
