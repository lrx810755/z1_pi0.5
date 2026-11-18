"""One-command scene lifecycle and disk-authoritative resumable collection."""
import argparse
import fcntl
import json
import hashlib
import os
import shlex
import signal
import subprocess
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path
from core import atomic_json,completed_ids,QUOTAS

HERE=Path(__file__).resolve().parent
DEFAULT=Path('/home/lp531/smolvla_data/z1_bottle_corrections_20261009')

def api(port,path,payload=None,timeout=10):
    data=None if payload is None else json.dumps(payload).encode()
    request=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            result=json.load(response)
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode('utf-8',errors='replace')
        raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc
    if result.get('success') is False:raise RuntimeError(result.get('message',str(result)))
    return result

def resume_command(args):
    return 'bash '+shlex.quote(str(HERE.parent/'collect_pi05_corrections.sh'))+' --output '+shlex.quote(str(args.output))

def show_progress(root,plan):
    done=completed_ids(root,plan['hash'])
    for kind,n in QUOTAS:
        print(f'{kind}: {sum(t["id"] in done for t in plan["tasks"] if t["kind"]==kind)}/{n}',flush=True)
    print(f'TOTAL {len(done)}/{len(plan["tasks"])}',flush=True)
    return done

def stop_scene(process):
    # Always signal our own group, including Isaac orphaned by its shell wrapper.
    for sig,delay in [(signal.SIGINT,8),(signal.SIGTERM,3),(signal.SIGKILL,0)]:
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:break
        if delay:time.sleep(delay)
    try:process.wait(timeout=3)
    except subprocess.TimeoutExpired:pass

def main():
    def terminate(_signum,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    parser=argparse.ArgumentParser(description='启动仿真并顺序补录80条纠正示范；重复同一命令续录。')
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--rounds',type=int,default=1,help='每轮每个缺失配额尝试一次，默认一轮')
    parser.add_argument('--max-attempts',type=int,default=80)
    parser.add_argument('--only',default='',help='可选类别，如 low,tilt；仍按预定顺序')
    parser.add_argument('--per-kind-limit',type=int,default=0,help='冒烟测试时每类最多尝试多少条')
    parser.add_argument('--port',type=int,default=8769)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();args.output=args.output.expanduser().resolve()
    if args.rounds<1 or args.max_attempts<1 or args.per_kind_limit<0:parser.error('轮数/尝试数应为正，类别限制非负')
    kinds=set(filter(None,args.only.split(',')))
    if kinds-set(dict(QUOTAS)):parser.error('未知类别')
    if args.output.name=='z1_tabletop_raw_left_wrist_three_camera_v1':parser.error('补录必须使用独立目录')
    if args.status and (args.output/'correction_plan.json').exists():
        show_progress(args.output,json.loads((args.output/'correction_plan.json').read_text()));return 0
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.collection.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('此目录已有补录进程，拒绝重复启动')
        plan_path=args.output/'correction_plan.json'
        if not plan_path.exists():
            if list(args.output.glob('episode_*')) or (args.output/'dataset_info.json').exists():
                raise SystemExit('现有目录没有纠正计划，拒绝混入其他数据')
            from build_plan import build
            plan=build(plan_path)
        else:plan=json.loads(plan_path.read_text())
        done=show_progress(args.output,plan)
        if args.status or args.dry_run:
            print('偏差幅度：',plan['amplitudes']);print('续录：'+resume_command(args));return 0
        if len(done)==len(plan['tasks']):return 0
        try:api(args.port,'/api/status',timeout=1)
        except Exception:pass
        else:raise SystemExit(f'端口 {args.port} 已有服务，拒绝连接未知场景')
        runtime=args.output/'collection_logs'/time.strftime('%Y%m%d_%H%M%S');runtime.mkdir(parents=True)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
        code_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        atomic_json(runtime/'collector_manifest.json',dict(code_hash=code_hash,files=hashes,plan_hash=plan['hash'],ros_domain_id=os.environ.get('ROS_DOMAIN_ID')))
        current=runtime/'current.json';os.environ['Z1_CORRECTION_CURRENT']=str(current)
        # Interrupted temporary episodes are retained for diagnosis, never counted.
        for pending in args.output.glob('.recording_*'):
            quarantine=runtime/'interrupted';quarantine.mkdir(exist_ok=True);pending.rename(quarantine/pending.name)
        command=['ros2','launch',str(HERE/'collection.launch.py'),'external_camera_layout:=dual',
                 'target_object:=beverage_bottle','rviz:=false','task_gui:=true','run_smolvla_policy:=false','freeze_target_keep_collisions:=true',
                 f'task_gui_port:={args.port}',f'dataset_root:={args.output}',f'headless:={str(args.headless).lower()}']
        print('启动场景：'+shlex.join(command),flush=True)
        process=None;attempts=0;exit_code=0
        with (runtime/'scene.log').open('w') as log:
            try:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+240;last_notice=0
                while time.monotonic()<deadline:
                    if process.poll() is not None:raise RuntimeError('场景启动退出，请查看 '+str(runtime/'scene.log'))
                    try:
                        status=api(args.port,'/api/status',timeout=2)
                        if (status.get('connected') and all(status.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions'])
                            and Path(status['recording']['dataset_root']).resolve()==args.output):break
                    except Exception:pass
                    if time.monotonic()-last_notice>20:print('等待仿真、三相机和规划服务就绪…',flush=True);last_notice=time.monotonic()
                    time.sleep(2)
                else:raise RuntimeError('240秒内场景未就绪')
                for round_index in range(args.rounds):
                    counts={}
                    for task in plan['tasks']:
                        done=completed_ids(args.output,plan['hash'])
                        if task['id'] in done or (kinds and task['kind'] not in kinds):continue
                        if args.per_kind_limit and counts.get(task['kind'],0)>=args.per_kind_limit:continue
                        if attempts>=args.max_attempts:break
                        attempts+=1;counts[task['kind']]=counts.get(task['kind'],0)+1
                        ctx=dict(task=task,plan_hash=plan['hash'],attempt_id=uuid.uuid4().hex,collector_code_hash=code_hash)
                        atomic_json(current,ctx)
                        print(f'ROUND {round_index+1} ATTEMPT {attempts} {task["id"]} grid={task["grid"]}',flush=True)
                        row=dict(**ctx)
                        try:
                            row['result']=api(args.port,'/api/demonstration/auto',{'target_object':'beverage_bottle','curriculum_level':'full','target_grid':task['grid']},timeout=600)
                            if task['id'] not in completed_ids(args.output,plan['hash']):raise RuntimeError('API返回成功但有效episode未落盘')
                            row['saved']=True;print('SAVED '+task['id'],flush=True)
                        except Exception as exc:
                            row.update(saved=False,error=str(exc));print('FAILED '+task['id']+': '+str(exc),flush=True)
                            for path in ['/api/force_stop','/api/recording/discard']:
                                try:api(args.port,path,{},timeout=20)
                                except Exception:pass
                            deadline=time.monotonic()+30
                            while api(args.port,'/api/status').get('auto_demonstration',{}).get('running'):
                                if time.monotonic()>deadline:raise RuntimeError('前一尝试未停止，拒绝覆盖任务上下文')
                                time.sleep(.5)
                        with (runtime/'attempts.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    if attempts>=args.max_attempts:break
            except KeyboardInterrupt:exit_code=130;print('收到停止信号；已保存数据保留。',flush=True)
            except Exception as exc:exit_code=1;print('ERROR '+str(exc),flush=True)
            finally:
                if process:
                    try:api(args.port,'/api/force_stop',{},timeout=8)
                    except Exception:pass
                    stop_scene(process)
                done=show_progress(args.output,plan)
                atomic_json(runtime/'run_summary.json',dict(attempts=attempts,completed=len(done),quota=80,exit_code=exit_code,scene_stopped=True))
                if len(done)<len(plan['tasks']):
                    print('尚未补齐，继续执行以下 SH 命令（自动跳过已保存样本）：\n'+resume_command(args),flush=True)
        return exit_code

if __name__=='__main__':raise SystemExit(main())
