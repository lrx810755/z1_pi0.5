"""Run cup after the requested three peach rounds, including incomplete completion."""
import importlib.util,json,subprocess,time,urllib.request
from pathlib import Path
LOG=Path('/home/lp531/z1_queue_logs/peach_to_cup_20261108')
PEACH_DATA=Path('/home/lp531/smolvla_data/peach_scale120_correction60_20261106')
PEACH_CODE=Path('/home/lp531/robot/peach_scale120_correction60_20261106')
RUNTIME=PEACH_DATA/'collection_logs/20261108_003700'
PEACH_UNIT='peach-correction-three-rounds-20261108.service'
CUP_UNIT='cup-horizontal-demo400-20261108.service'
spec=importlib.util.spec_from_file_location('peach_queue_completion',PEACH_CODE/'core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def write(stage,**data):
    value=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**data)
    core.atomic_json(LOG/'status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)

def active(unit):
    p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],capture_output=True,text=True)
    return p.stdout.strip()

def progress():
    plan=json.loads((PEACH_DATA/'correction_plan.json').read_text())
    required={t['id'] for t in plan['tasks']}
    done=core.completed_ids(PEACH_DATA,plan['hash'])
    return len(done & required),sorted(required-done)

def main():
    LOG.mkdir(parents=True,exist_ok=True)
    last=None
    while True:
        completed,missing=progress();live={}
        p=RUNTIME/'live_progress.json'
        if p.exists():live=json.loads(p.read_text())
        state=(completed,live.get('round',0),active(PEACH_UNIT))
        if state!=last:
            write('WAITING_FOR_PEACH_THREE_ROUNDS',peach_completed=completed,peach_round=live.get('round',0),requested_rounds=3,missing_tasks=missing,cup_quota=400)
            last=state
        summary_path=RUNTIME/'run_summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            if summary.get('exit_code')!=0 or not summary.get('scene_stopped'):
                raise RuntimeError('桃子补录异常停止，未确认三轮结束')
            if completed<60 and live.get('round',0)<3:
                raise RuntimeError('桃子尚未完成三轮，拒绝提前切换')
            if active(PEACH_UNIT) not in ('active','activating','deactivating'):break
        elif state[2] not in ('active','activating','deactivating'):
            raise RuntimeError('桃子服务停止但没有正常结束记录')
        time.sleep(5)
    config=str(PEACH_CODE/'grasp_physics.json').encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if config in args and any(a.endswith(b'run_z1_ros2_scene.py') for a in args):
            raise RuntimeError('桃子Sim尚未退出，拒绝重复启动仿真')
    core.atomic_json(PEACH_DATA/'stopped_after_three_additional_rounds_20261108.json',dict(completed=completed,missing_tasks=missing,rounds=live.get('round',0),user_requested_switch=True))
    write('STARTING_CUP',peach_completed=completed,missing_peach_tasks=missing,cup_quota=400)
    if active(CUP_UNIT) not in ('active','activating'):
        subprocess.run(['systemd-run','--user','--unit='+CUP_UNIT,'--property=KillMode=control-group',
                        '--setenv=DISPLAY=:1','--setenv=XAUTHORITY=/run/user/1000/gdm/Xauthority',
                        '--setenv=PYTHONUNBUFFERED=1','--setenv=Z1_CUP_QUOTA_PER_CELL=20',
                        '/bin/bash','/home/lp531/SH/start_cup_horizontal_20261108.sh'],check=True)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if active(CUP_UNIT) not in ('active','activating'):raise RuntimeError('杯子入口启动退出')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8837/api/status',timeout=3) as response:s=json.load(response)
            if (s.get('connected') and all(s.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions'])
                and s['recording'].get('dataset_root')=='/home/lp531/smolvla_data/cup_horizontal_demo_20261108'):
                write('CUP_RUNNING',peach_completed=completed,missing_peach_tasks=missing,cup_quota=400,cup_unit=CUP_UNIT)
                return
        except (OSError,ValueError):pass
        time.sleep(2)
    raise RuntimeError('杯子场景300秒内未就绪')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('ERROR',error=str(exc));raise
